"""Heston 参数校准：对 raw_option_chain 的每个快照日，用虚值期权的中间价拟合 (v0, kappa, theta, xi, rho)。

流程
1. 取快照日的 S（close_raw）、r（rate_1y）、q（dividend_yield），来自 feature_daily。
2. 筛选合约：有效报价、期限与价内外程度在范围内、只留相对远期的虚值期权、价格和 vega 不能太小。
3. 用中间价和同一组 r、q 反算 BSM 隐含波动率，并以 1/vega 加权：
       残差 = (模型价 - 市场价) / vega  ≈  隐含波动率误差
   深度虚值期权的价格很小，按相对价格误差拟合会被它们主导；1/vega 加权则近似于直接拟合隐含波动率。
4. 差分进化（全局）→ least_squares（局部精修）。参数取优化器返回的 x，而不是模型对象里的最后状态。
5. 用拟合参数的价格反算隐含波动率，报告 RMSE（单位：波动率点）。

局限：个股期权是美式，这里按欧式定价；只使用虚值期权，提前行权溢价较小。
r 用 1 年期利率的平坦曲线，q 用最近一次分红的年化股息率。

用法：python -m co_heston.calibrate [--force] [--dates 2018-01-02 ...] [--feller]
"""

import argparse
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

from co_bsm.implied_vol import bsm_vega, implied_vol
from co_bsm.pricing import HestonPricer
from common.logger import get_logger
from common.paths import get_outputs_path

lg = get_logger("co_heston")

PARAM_NAMES = ["v0", "kappa", "theta", "xi", "rho"]
BOUNDS = [(0.001, 1.0), (0.1, 25.0), (0.001, 1.0), (0.01, 4.0), (-0.999, 0.999)]

FILTERS = dict(
    min_days=14, max_days=730,     # 期限范围
    max_abs_logm=0.35,             # |ln(K/F)| 上限，约 0.70 ~ 1.42 倍远期
    min_mid=0.10,                  # 中间价下限
    max_rel_spread=0.5,            # (ask - bid) / mid 上限
    min_vega_ratio=0.02,           # vega / (S sqrt(T)) 下限，平值约 0.4
)
MIN_OPTIONS = 30
MIN_MATURITIES = 3


# ---------------------------------------------------------------------------
# 合约筛选
# ---------------------------------------------------------------------------

def select_options(chain: pd.DataFrame, S: float, r: float, q: float, **overrides):
    """返回 (筛选后的合约表, 各步剔除数量)。合约表列：K, T, mid, is_call, iv, vega。"""
    f = {**FILTERS, **overrides}
    df = chain.copy()
    df["K"] = df["strike"].astype(float)
    df["T"] = (df["expiration_date"] - df["date"]).dt.days / 365.0
    df["is_call"] = df["option_type"].str.lower().eq("call")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["F"] = S * np.exp((r - q) * df["T"])

    dropped = {"总数": len(df)}

    def keep(name, mask):
        nonlocal df
        dropped[name] = int((~mask).sum())
        df = df[mask].copy()

    keep("无效报价", (df["bid"] > 0) & (df["ask"] > df["bid"]))
    keep("期限超范围", df["T"].between(f["min_days"] / 365.0, f["max_days"] / 365.0))
    keep("非虚值", (df["is_call"] & (df["K"] >= df["F"])) | (~df["is_call"] & (df["K"] < df["F"])))
    keep("价内外程度超范围", np.log(df["K"] / df["F"]).abs() <= f["max_abs_logm"])
    keep("价格过低或价差过大",
         (df["mid"] >= f["min_mid"]) & ((df["ask"] - df["bid"]) / df["mid"] <= f["max_rel_spread"]))

    df["iv"] = implied_vol(df["mid"], S, df["K"], r, df["T"], q, df["is_call"])
    keep("隐含波动率无解", df["iv"].notna())

    df["vega"] = bsm_vega(S, df["K"], r, df["T"], df["iv"], q)
    keep("vega 过小", df["vega"] >= f["min_vega_ratio"] * S * np.sqrt(df["T"]))

    cols = ["K", "T", "mid", "is_call", "iv", "vega"]
    return df[cols].reset_index(drop=True), dropped


# ---------------------------------------------------------------------------
# 校准
# ---------------------------------------------------------------------------

def calibrate_snapshot(opts: pd.DataFrame, S: float, r: float, q: float,
                       x0: Optional[np.ndarray] = None, seed: int = 1,
                       enforce_feller: bool = False, de_maxiter: int = 300, popsize: int = 12):
    """对一个快照拟合 Heston 参数，返回结果字典。"""
    pricer = HestonPricer(S, r, q, opts["T"].to_numpy(), opts["K"].to_numpy())
    is_call = opts["is_call"].to_numpy()
    price = opts["mid"].to_numpy()
    vega = opts["vega"].to_numpy()

    def residuals(x):
        res = (pricer.prices(is_call, *x) - price) / vega
        if enforce_feller:
            res = np.append(res, 10.0 * max(0.0, x[3] ** 2 - 2.0 * x[1] * x[2]))
        return res

    def cost(x):
        return float(np.mean(residuals(x) ** 2))

    lo, hi = np.array(BOUNDS).T
    start = None if x0 is None else np.clip(np.asarray(x0, float), lo, hi)
    de = differential_evolution(cost, BOUNDS, seed=seed, maxiter=de_maxiter, popsize=popsize,
                                tol=1e-4, mutation=(0.5, 1.0), recombination=0.7,
                                polish=False, updating="immediate", x0=start)
    ls = least_squares(residuals, de.x, bounds=(lo, hi), x_scale="jac")
    x = ls.x if cost(ls.x) <= de.fun else de.x

    model = pricer.prices(is_call, *x)
    iv_model = implied_vol(model, S, opts["K"], r, opts["T"], q, is_call)
    err = (iv_model - opts["iv"].to_numpy()) * 100.0          # 波动率点
    v0, kappa, theta, xi, rho = x
    return dict(
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        feller_ratio=2.0 * kappa * theta / xi**2,
        rmse_iv=float(np.sqrt(np.nanmean(err**2))),
        mae_iv=float(np.nanmean(np.abs(err))),
        max_iv_err=float(np.nanmax(np.abs(err))),
        n_fit_failed=int(np.isnan(err).sum()),
        cost=cost(x), de_evals=int(de.nfev),
    )


# ---------------------------------------------------------------------------
# 批量运行
# ---------------------------------------------------------------------------

def load_inputs():
    from co_data_center.data_center import DataCenter

    dc = DataCenter()
    chain = dc.get("raw_option_chain").df
    feat = dc.get("feature_daily").df
    for col in ("close_raw", "rate_1y", "dividend_yield"):
        if col not in feat.columns:
            raise KeyError(f"feature_daily 缺少 {col}，请先重新运行 FeatureEngine")
    feat = feat[["date", "close_raw", "rate_1y", "dividend_yield"]].copy()
    feat["date"] = pd.to_datetime(feat["date"])
    return chain, feat.sort_values("date").set_index("date")


def market_inputs(feat: pd.DataFrame, date: pd.Timestamp):
    """取 date 当天（或之前最近一个交易日）的 S、r、q。"""
    row = feat.loc[:date].iloc[-1]
    S, r, q = float(row["close_raw"]), float(row["rate_1y"]) / 100.0, float(row["dividend_yield"])
    if not np.all(np.isfinite([S, r, q])):
        raise ValueError(f"{date.date()}: S、r、q 存在缺失值 {S, r, q}")
    return S, r, q


def default_path():
    return get_outputs_path() / "heston_params.parquet"


def run(force=False, dates=None, enforce_feller=False, seed=1, out_path=None, filters=None):
    out_path = out_path or default_path()
    chain, feat = load_inputs()

    ok_rows = []
    if out_path.exists():
        done = pd.read_parquet(out_path)
        ok_rows = [r for r in done.to_dict("records") if r["status"] == "ok"]

    target = [pd.Timestamp(d) for d in sorted(pd.to_datetime(chain["date"]).unique())]
    if dates:
        wanted = {pd.Timestamp(d) for d in dates}
        target = [d for d in target if d in wanted]
    ok_dates = set() if force else {r["date"] for r in ok_rows}
    todo = [d for d in target if d not in ok_dates]
    lg.info(f"Heston 校准：{len(target)} 个快照，已完成 {len(target) - len(todo)}，待校准 {len(todo)}")

    rows = [r for r in ok_rows if r["date"] not in set(todo)]
    earlier = [r for r in rows if todo and r["date"] < todo[0]]
    prev = np.array([max(earlier, key=lambda r: r["date"])[k] for k in PARAM_NAMES]) if earlier else None

    for date in todo:
        t0 = time.perf_counter()
        S, r, q = market_inputs(feat, date)
        opts, dropped = select_options(chain[chain["date"] == date], S, r, q, **(filters or {}))
        base = dict(date=date, S=S, r=r, q=q, n_options=len(opts), n_maturities=int(opts["T"].nunique()))

        if len(opts) < MIN_OPTIONS or base["n_maturities"] < MIN_MATURITIES:
            lg.warning(f"{date.date()}: 可用合约 {len(opts)} 个、期限 {base['n_maturities']} 个，不足，跳过。剔除情况：{dropped}")
            rows.append({**base, "status": f"skipped: {len(opts)} options, {base['n_maturities']} maturities"})
        else:
            res = calibrate_snapshot(opts, S, r, q, x0=prev, seed=seed, enforce_feller=enforce_feller)
            prev = np.array([res[k] for k in PARAM_NAMES])
            rows.append({**base, **res, "status": "ok", "seconds": time.perf_counter() - t0})
            lg.info(f"{date.date()}: {len(opts)} 个合约，RMSE {res['rmse_iv']:.2f} 个波动率点，"
                    f"v0={res['v0']:.3f} kappa={res['kappa']:.2f} theta={res['theta']:.3f} "
                    f"xi={res['xi']:.2f} rho={res['rho']:.2f}，Feller 比 {res['feller_ratio']:.2f}，"
                    f"{time.perf_counter() - t0:.0f}s")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("date").reset_index(drop=True).to_parquet(out_path, index=False)  # 每个快照写一次，中断后可续

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Heston 参数校准")
    parser.add_argument("--force", action="store_true", help="重新校准所有快照")
    parser.add_argument("--dates", nargs="*", help="只校准指定日期，如 2018-01-02")
    parser.add_argument("--feller", action="store_true", help="强制满足 Feller 条件")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    run(force=args.force, dates=args.dates, enforce_feller=args.feller, seed=args.seed)


if __name__ == "__main__":
    main()