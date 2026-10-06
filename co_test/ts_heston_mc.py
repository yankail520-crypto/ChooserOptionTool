"""Heston 蒙特卡罗的测试。运行：python -m pytest co_test/ts_heston_mc.py -v"""

import numpy as np
import pytest

from co_bsm.chooser import ChooserParams, chooser_price
from co_bsm.heston_mc import heston_chooser_exact, heston_chooser_mc, simulate_heston

S, K, R, Q, T1, T2 = 156.7, 150.0, 0.0015, 0.0233, 0.5, 1.0    # 论文 Table 2

# 校准结果中的几组真实参数
PARAMS = {
    "median": dict(v0=0.054, kappa=7.514, theta=0.077, xi=1.855, rho=-0.503),
    "xi_at_bound": dict(v0=0.141, kappa=17.518, theta=0.096, xi=4.000, rho=-0.384),   # 2021-12-01
    "kappa_at_bound": dict(v0=0.032, kappa=25.0, theta=0.055, xi=3.478, rho=-0.389),  # 2024-06-03
    "high_vol": dict(v0=0.721, kappa=4.034, theta=0.280, xi=4.000, rho=-0.682),       # 2020-04-01
}


@pytest.mark.parametrize("name", list(PARAMS))
def test_qe_is_martingale(name):
    """带 martingale 修正的 QE：折现股价的期望等于初始股价。"""
    s1, s2 = simulate_heston(S, R, Q, **PARAMS[name], T1=T1, T2=T2, n_paths=300_000, steps1=8, steps2=8, scheme="qe", seed=3)
    for s, t in ((s1, T1), (s2, T2)):
        target = S * np.exp((R - Q) * t)
        assert abs(s.mean() - target) < 4 * s.std(ddof=1) / np.sqrt(len(s))


@pytest.mark.parametrize("name", list(PARAMS))
def test_qe_matches_semi_analytic(name):
    exact, call, put = heston_chooser_exact(S, K, R, Q, T1, T2, **PARAMS[name])
    res = heston_chooser_mc(S, K, R, Q, T1, T2, **PARAMS[name], n_paths=300_000, steps1=32, steps2=32, scheme="qe", seed=5)
    assert abs(res["price_a"] - exact) < 4 * res["se_a"]
    assert abs(res["price_b"] - exact) < 4 * res["se_b"]
    assert abs(res["call"] - call) < 4 * res["call_se"]
    assert abs(res["put"] - put) < 4 * res["put_se"]


def test_full_truncation_converges_slowly():
    """同样的步数下，Full Truncation 的偏差远大于 QE。"""
    p = PARAMS["median"]
    exact = heston_chooser_exact(S, K, R, Q, T1, T2, **p)[0]
    ft = heston_chooser_mc(S, K, R, Q, T1, T2, **p, n_paths=200_000, steps1=8, steps2=8, scheme="ft", seed=7)
    qe = heston_chooser_mc(S, K, R, Q, T1, T2, **p, n_paths=200_000, steps1=8, steps2=8, scheme="qe", seed=7)
    assert abs(ft["price_a"] - exact) > 10 * ft["se_a"]
    assert abs(qe["price_a"] - exact) < 4 * qe["se_a"]


def test_bsm_limit():
    """xi 很小、v0 = theta = sigma^2、rho = 0 时，Heston 退化为 BSM，chooser 价格应接近 29.13。"""
    var = 0.282**2
    p = dict(v0=var, kappa=3.0, theta=var, xi=0.01, rho=0.0)
    res = heston_chooser_mc(S, K, R, Q, T1, T2, **p, n_paths=400_000, steps1=16, steps2=16, scheme="qe", seed=9)
    bsm = chooser_price(ChooserParams(S=S, K=K, r=R, T1=T1, T2=T2, sigma=0.282, q=Q))
    assert abs(res["price_a"] - bsm) < 4 * res["se_a"]
    assert heston_chooser_exact(S, K, R, Q, T1, T2, **p)[0] == pytest.approx(bsm, abs=2e-3)