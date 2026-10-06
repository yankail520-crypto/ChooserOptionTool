"""每日滚动定价：对每个交易日，用 BSM 与 Heston 给同一份 chooser 定价，输出对比表。

每天、每份合约：
- bsm_hist  ：BSM，sigma = rolling_vol_20d（历史波动率）
- bsm_impl  ：BSM，sigma = Heston 在 T2 期限、平值（K = 远期价格）的隐含波动率
- heston_exact：Heston 半解析（Call(K, T2) + e^{-q tau} Put(K', T1)）
- heston_mc ：Heston 蒙特卡罗（QE，方式 A）及标准误；MC 与精确值之差应落在标准误范围内

合约：atm（K = 当日 S，固定 moneyness）与 k150（K = 150，与论文设定一致）。
输入：feature_daily（close_raw、rate_1y、dividend_yield、rolling_vol_20d）与每日 Heston 参数表。

用法：python -m co_bsm.daily_pricing [--mc-paths 100000] [--steps 32] [--start ...] [--end ...]
"""

import argparse
import time

import numpy as np
import pandas as pd

from co_bsm.chooser import ChooserParams, chooser_price
from co_bsm.heston_mc import heston_chooser_exact, heston_chooser_mc
from co_bsm.implied_vol import implied_vol
from co_bsm.pricing import heston_call
from common.logger import get_logger
from common.paths import get_outputs_path

lg = get_logger("co_bsm")

HESTON_COLS = ["v0", "kappa", "theta", "xi", "rho"]
FIXED_K = 150.0


def heston_atm_vol(S, r, q, T, params):
    """Heston 在期限 T、K = 远期价格 处的 BSM 隐含波动率。"""
    F = S * np.exp((r - q) * T)
    call = heston_call(S, F, r, q, T, **params)[0]
    return float(implied_vol(call, S, F, r, T, q, True))


def _bsm(S, K, r, q, T1, T2, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return float(chooser_price(ChooserParams(S=S, K=K, r=r, T1=T1, T2=T2, sigma=sigma, q=q)))


def price_day(S, r, q, sigma_hist, params, K, T1, T2, mc_paths=0, steps=32, seed=None):
    exact, call_leg, put_leg = heston_chooser_exact(S, K, r, q, T1, T2, **params)
    sigma_impl = heston_atm_vol(S, r, q, T2, params)
    out = dict(
        sigma_hist=sigma_hist, sigma_impl=sigma_impl,
        bsm_hist=_bsm(S, K, r, q, T1, T2, sigma_hist),
        bsm_impl=_bsm(S, K, r, q, T1, T2, sigma_impl),
        heston_exact=exact, heston_call_leg=call_leg, heston_put_leg=put_leg,
        heston_mc=np.nan, heston_mc_se=np.nan,
    )
    if mc_paths:
        res = heston_chooser_mc(S, K, r, q, T1, T2, **params, n_paths=mc_paths,
                                steps1=steps, steps2=steps, scheme="qe", seed=seed)
        out["heston_mc"], out["heston_mc_se"] = res["price_a"], res["se_a"]
    return out


def build_prices(feat: pd.DataFrame, params: pd.DataFrame, T1=0.5, T2=1.0, mc_paths=0, steps=32,
                 start=None, end=None, contracts=("atm", "k150")) -> pd.DataFrame:
    """feat：以 date 为索引；params：每日 Heston 参数表（含 date 列）。"""
    df = params.set_index("date").join(feat[["close_raw", "rate_1y", "dividend_yield", "rolling_vol_20d"]], how="left")
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]

    rows, t0 = [], time.perf_counter()
    for i, (date, row) in enumerate(df.iterrows()):
        S, r, q = float(row["close_raw"]), float(row["rate_1y"]) / 100.0, float(row["dividend_yield"])
        if not np.all(np.isfinite([S, r, q])):
            lg.warning(f"{date.date()}: S、r、q 缺失，跳过")
            continue
        heston = {k: float(row[k]) for k in HESTON_COLS}
        for j, name in enumerate(contracts):
            K = S if name == "atm" else FIXED_K
            res = price_day(S, r, q, float(row["rolling_vol_20d"]), heston, K, T1, T2,
                            mc_paths=mc_paths, steps=steps, seed=1_000_000 + 10 * i + j)
            rows.append(dict(date=date, contract=name, S=S, K=K, r=r, q=q, T1=T1, T2=T2, **heston,
                             low_conf=bool(row["low_conf"]), days_since_snapshot=int(row["days_since_snapshot"]), **res))
        if (i + 1) % 100 == 0:
            lg.info(f"已完成 {i + 1}/{len(df)} 天，用时 {time.perf_counter() - t0:.0f}s")
    return pd.DataFrame(rows)


def run(mc_paths=100_000, steps=32, start=None, end=None, T1=0.5, T2=1.0, out_path=None):
    from co_bsm.daily_params import daily_path
    from co_data_center.data_center import DataCenter

    params = pd.read_parquet(daily_path())
    feat = DataCenter().get("feature_daily").df
    feat = feat.assign(date=pd.to_datetime(feat["date"])).set_index("date")

    out = build_prices(feat, params, T1=T1, T2=T2, mc_paths=mc_paths, steps=steps, start=start, end=end)
    path = out_path or (get_outputs_path() / "chooser_daily_prices.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    lg.info(f"每日定价表：{len(out)} 行，已写入 {path}")
    if mc_paths:
        z = ((out["heston_mc"] - out["heston_exact"]) / out["heston_mc_se"]).abs()
        lg.info(f"MC 与半解析之差 / 标准误：中位数 {z.median():.2f}，超过 3 的占比 {(z > 3).mean():.1%}")
    return out


def main():
    p = argparse.ArgumentParser(description="每日滚动定价：BSM 与 Heston")
    p.add_argument("--mc-paths", type=int, default=100_000, help="蒙特卡罗路径数，0 表示不跑 MC")
    p.add_argument("--steps", type=int, default=32, help="[0,T1] 与 [T1,T2] 各自的步数")
    p.add_argument("--start"), p.add_argument("--end")
    p.add_argument("--T1", type=float, default=0.5), p.add_argument("--T2", type=float, default=1.0)
    a = p.parse_args()
    run(mc_paths=a.mc_paths, steps=a.steps, start=a.start, end=a.end, T1=a.T1, T2=a.T2)


if __name__ == "__main__":
    main()