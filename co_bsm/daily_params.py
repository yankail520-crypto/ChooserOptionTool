"""每日 Heston 参数表：把月度快照的校准结果扩展到每个交易日。

规则
- kappa、theta、xi、rho：沿用最近一个"不晚于当天"的快照，不使用未来的快照。
- v0：按 VIX 的变化缩放，v0_t = v0_snap * (VIX_t / VIX_snap)^2。
  假设 JPM 的瞬时方差随 VIX 同比例变动；快照日当天比值为 1，v0 即校准值。
- 快照之间的日子 Heston 并没有重新拟合市场，只是按快照参数外推。
- low_conf：快照的参数撞到搜索边界（kappa 或 xi），或拟合 RMSE 超过 2 个波动率点。

用法：python -m co_heston.daily_params
"""

import numpy as np
import pandas as pd

from co_bsm.calibrate import BOUNDS
from common.logger import get_logger
from common.paths import get_outputs_path

lg = get_logger("co_heston")

RMSE_LIMIT = 2.0
EDGE = 0.1     # 距离搜索上界不到 EDGE 视为撞界


def snapshot_path():
    return get_outputs_path() / "heston_params.parquet"


def daily_path():
    return get_outputs_path() / "heston_daily_params.parquet"


def flag_low_confidence(snap: pd.DataFrame) -> pd.Series:
    return (snap["kappa"] >= BOUNDS[1][1] - EDGE) | (snap["xi"] >= BOUNDS[3][1] - EDGE) | (snap["rmse_iv"] > RMSE_LIMIT)


def build_daily_params(snap: pd.DataFrame, feat: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """snap：校准结果表；feat：以 date 为索引、含 vix 列的日频表。"""
    snap = snap[snap["status"] == "ok"].sort_values("date").reset_index(drop=True)
    snap = snap.assign(low_conf=flag_low_confidence(snap))

    vix = feat["vix"]
    days = feat.index[feat.index >= snap["date"].iloc[0]]
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]

    pos = np.searchsorted(snap["date"].to_numpy(), days.to_numpy(), side="right") - 1   # 最近一个 <= 当天的快照
    s = snap.iloc[pos].reset_index(drop=True)

    vix_t = vix.reindex(days).to_numpy()
    vix_s = vix.asof(pd.DatetimeIndex(s["date"])).to_numpy()
    ratio = vix_t / vix_s
    missing = ~np.isfinite(ratio)
    if missing.any():
        lg.warning(f"{int(missing.sum())} 天缺少 VIX，v0 取快照值")
        ratio = np.where(missing, 1.0, ratio)

    lo, hi = BOUNDS[0]
    out = pd.DataFrame({
        "date": days,
        "snapshot_date": s["date"],
        "days_since_snapshot": (days - pd.DatetimeIndex(s["date"])).days,
        "v0": np.clip(s["v0"].to_numpy() * ratio**2, lo, hi),
        "kappa": s["kappa"], "theta": s["theta"], "xi": s["xi"], "rho": s["rho"],
        "vix": vix_t, "vix_snapshot": vix_s,
        "rmse_iv_snapshot": s["rmse_iv"], "low_conf": s["low_conf"],
    })
    return out


def run(start="2018-01-01", end="2024-12-31", out_path=None):
    snap = pd.read_parquet(snapshot_path())
    from co_data_center.data_center import DataCenter
    feat = DataCenter().get("feature_daily").df[["date", "vix"]].copy()
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.sort_values("date").set_index("date")

    out = build_daily_params(snap, feat, start, end)
    path = out_path or daily_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    lg.info(f"每日参数表：{len(out)} 个交易日（{out['date'].min().date()} ~ {out['date'].max().date()}），已写入 {path}")
    lg.info(f"距最近快照的天数：中位数 {int(out['days_since_snapshot'].median())}，最大 {int(out['days_since_snapshot'].max())}")
    lg.info(f"低可信度的交易日占比：{out['low_conf'].mean():.1%}（{out['snapshot_date'][out['low_conf']].nunique()} 个快照）")
    lg.info(f"v0 范围：{out['v0'].min():.3f} ~ {out['v0'].max():.3f}")
    return out


if __name__ == "__main__":
    run()