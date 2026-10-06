"""每日 Heston 参数表的测试。运行：python -m pytest co_test/ts_heston_daily.py -v"""

import numpy as np
import pandas as pd
import pytest

from co_bsm.daily_params import build_daily_params, flag_low_confidence


def _snap():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-01"]),
        "status": ["ok", "ok", "ok"],
        "v0": [0.04, 0.06, 0.05], "kappa": [8.0, 25.0, 6.0], "theta": [0.06, 0.055, 0.07],
        "xi": [1.5, 3.4, 1.2], "rho": [-0.4, -0.39, -0.5], "rmse_iv": [1.0, 1.2, 2.5],
    })


def _feat():
    days = pd.bdate_range("2023-12-01", "2024-03-29")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"vix": 14 + rng.uniform(0, 8, len(days))}, index=days)


def test_no_lookahead():
    out = build_daily_params(_snap(), _feat())
    assert (out["snapshot_date"] <= out["date"]).all()
    assert (out["days_since_snapshot"] >= 0).all()


def test_snapshot_day_uses_calibrated_v0():
    out = build_daily_params(_snap(), _feat()).set_index("date")
    for d, v0 in zip(_snap()["date"], _snap()["v0"]):
        assert out.loc[d, "v0"] == pytest.approx(v0)
        assert out.loc[d, "days_since_snapshot"] == 0


def test_v0_scales_with_vix():
    feat = _feat()
    out = build_daily_params(_snap(), feat).set_index("date")
    d = pd.Timestamp("2024-01-17")
    expected = 0.04 * (feat.loc[d, "vix"] / feat.loc[pd.Timestamp("2024-01-02"), "vix"]) ** 2
    assert out.loc[d, "v0"] == pytest.approx(expected)


def test_other_parameters_are_constant_within_month():
    out = build_daily_params(_snap(), _feat())
    jan = out[(out["date"] >= "2024-01-02") & (out["date"] < "2024-02-01")]
    assert jan["kappa"].nunique() == 1 and jan["kappa"].iloc[0] == 8.0


def test_start_and_end():
    out = build_daily_params(_snap(), _feat(), start="2024-01-10", end="2024-02-20")
    assert out["date"].min() >= pd.Timestamp("2024-01-10") and out["date"].max() <= pd.Timestamp("2024-02-20")


def test_low_confidence_flag():
    flags = flag_low_confidence(_snap())
    assert list(flags) == [False, True, True]      # kappa 撞界；rmse > 2