"""每日滚动定价的测试。运行：python -m pytest co_test/ts_daily_pricing.py -v"""

import numpy as np
import pandas as pd
import pytest

from co_bsm.daily_pricing import build_prices, heston_atm_vol


def _inputs(xi=0.01, n=6):
    days = pd.bdate_range("2024-03-01", periods=n)
    feat = pd.DataFrame({"close_raw": np.linspace(150, 160, n), "rate_1y": 4.5, "dividend_yield": 0.022,
                         "rolling_vol_20d": 0.25}, index=days)
    var = 0.28**2
    params = pd.DataFrame({"date": days, "v0": var, "kappa": 3.0, "theta": var, "xi": xi, "rho": 0.0,
                           "low_conf": False, "days_since_snapshot": np.arange(n)})
    return feat, params


def test_shape_and_columns():
    feat, params = _inputs()
    out = build_prices(feat, params, mc_paths=0)
    assert len(out) == 2 * len(feat)
    assert set(out["contract"]) == {"atm", "k150"}
    assert (out.loc[out.contract == "atm", "K"].to_numpy() == feat["close_raw"].to_numpy()).all()
    assert (out.loc[out.contract == "k150", "K"] == 150.0).all()


def test_bsm_limit():
    """xi 很小、rho = 0 时 Heston 退化为 BSM，两个 BSM 版本中 bsm_impl 应与 Heston 一致。"""
    feat, params = _inputs(xi=0.01)
    out = build_prices(feat, params, mc_paths=0)
    assert (out["bsm_impl"] - out["heston_exact"]).abs().max() < 0.02
    assert out["sigma_impl"].between(0.27, 0.29).all()


def test_smile_makes_bsm_differ():
    """有偏斜时，同一个平值隐含波动率的 BSM 与 Heston 的价格不再一致。"""
    feat, params = _inputs()
    params = params.assign(xi=1.5, rho=-0.7)
    out = build_prices(feat, params, mc_paths=0)
    assert (out["bsm_impl"] - out["heston_exact"]).abs().max() > 0.1


def test_mc_matches_exact():
    feat, params = _inputs(n=2)
    params = params.assign(v0=0.06, kappa=8.0, theta=0.07, xi=1.8, rho=-0.5)
    out = build_prices(feat, params, mc_paths=200_000, steps=32)
    assert ((out["heston_mc"] - out["heston_exact"]).abs() < 4 * out["heston_mc_se"]).all()


def test_atm_vol_matches_bsm():
    var = 0.3**2
    sigma = heston_atm_vol(150.0, 0.03, 0.02, 1.0, dict(v0=var, kappa=3.0, theta=var, xi=0.005, rho=0.0))
    assert sigma == pytest.approx(0.3, abs=1e-3)