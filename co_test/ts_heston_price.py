"""Heston 定价与校准的测试。运行：pytest co_test/test_heston.py -v"""
""" python -m pytest co_test/ts_heston_price.py -v """
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from co_bsm.black_scholes import BSMParams, bsm_call
from co_bsm.implied_vol import implied_vol
from co_bsm.calibrate import PARAM_NAMES, calibrate_snapshot, select_options
from co_bsm.pricing import HestonPricer, heston_call, heston_put

S, R, Q = 150.0, 0.03, 0.025
PARAM_SETS = [
    dict(v0=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
    dict(v0=0.09, kappa=1.0, theta=0.06, xi=0.8, rho=-0.9),
    dict(v0=0.02, kappa=5.0, theta=0.05, xi=1.5, rho=-0.3),
    dict(v0=0.04, kappa=0.5, theta=0.04, xi=0.4, rho=0.4),
    dict(v0=0.50, kappa=10.0, theta=0.30, xi=3.0, rho=-0.95),
]


def gil_pelaez_call(S, K, r, q, T, v0, kappa, theta, xi, rho):
    """Heston (1993) 的 P1 / P2 形式，对 ln S_T 的特征函数积分。与 Lewis 形式相互独立。"""
    def P(j):
        uj = 0.5 if j == 1 else -0.5
        b = kappa - rho * xi if j == 1 else kappa

        def f(u):
            iu = 1j * u
            d = np.sqrt((rho * xi * iu - b) ** 2 - xi**2 * (2 * uj * iu - u * u))
            g = (b - rho * xi * iu - d) / (b - rho * xi * iu + d)
            e = np.exp(-d * T)
            C = (r - q) * iu * T + kappa * theta / xi**2 * ((b - rho * xi * iu - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
            D = (b - rho * xi * iu - d) / xi**2 * (1 - e) / (1 - g * e)
            return (np.exp(-iu * np.log(K)) * np.exp(C + D * v0 + iu * np.log(S)) / iu).real

        return 0.5 + quad(f, 0, np.inf, limit=1000, epsabs=1e-13, epsrel=1e-13)[0] / np.pi

    return S * np.exp(-q * T) * P(1) - K * np.exp(-r * T) * P(2)


@pytest.mark.parametrize("p", PARAM_SETS)
def test_matches_gil_pelaez(p):
    for T in (0.04, 0.25, 1.0, 2.0):
        for K in (105, 130, 150, 170, 210):
            assert heston_call(S, K, R, Q, T, **p)[0] == pytest.approx(gil_pelaez_call(S, K, R, Q, T, **p), abs=1e-6)


def test_bsm_limit():
    """xi 很小、v0 = theta、rho = 0 时退化为 BSM（sigma^2 = theta）。"""
    p = dict(v0=0.09, kappa=3.0, theta=0.09, xi=0.003, rho=0.0)
    for T in (0.1, 0.5, 2.0):
        for K in (110, 150, 200):
            assert heston_call(S, K, R, Q, T, **p)[0] == pytest.approx(bsm_call(BSMParams(S, K, R, T, 0.3, Q)), abs=1e-4)


def test_put_call_parity():
    p = PARAM_SETS[0]
    c, pu = heston_call(S, 140.0, R, Q, 0.5, **p)[0], heston_put(S, 140.0, R, Q, 0.5, **p)[0]
    assert c - pu == pytest.approx(S * np.exp(-Q * 0.5) - 140.0 * np.exp(-R * 0.5), abs=1e-10)


def test_monte_carlo():
    """Full Truncation Euler 蒙特卡罗，满足 Feller 条件的参数。"""
    p = dict(v0=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    K, T, n, steps = 150.0, 1.0, 400_000, 250
    rng = np.random.default_rng(7)
    dt = T / steps
    lnS, v = np.full(n, np.log(S)), np.full(n, p["v0"])
    for _ in range(steps):
        z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
        vp = np.maximum(v, 0.0)
        lnS += (R - Q - 0.5 * vp) * dt + np.sqrt(vp * dt) * (p["rho"] * z1 + np.sqrt(1 - p["rho"] ** 2) * z2)
        v += p["kappa"] * (p["theta"] - vp) * dt + p["xi"] * np.sqrt(vp * dt) * z1
    payoff = np.exp(-R * T) * np.maximum(np.exp(lnS) - K, 0.0)
    se = payoff.std(ddof=1) / np.sqrt(n)
    assert abs(payoff.mean() - heston_call(S, K, R, Q, T, **p)[0]) < 3 * se + 0.02


def test_implied_vol_roundtrip():
    K = np.array([120.0, 150.0, 180.0])
    is_call = np.array([False, True, True])
    sigma = np.array([0.35, 0.28, 0.22])
    p = BSMParams(S, K, R, 0.5, sigma, Q)
    from co_bsm.black_scholes import bsm_put
    price = np.where(is_call, bsm_call(p), bsm_put(p))
    assert pytest.approx(sigma, abs=1e-7) == implied_vol(price, S, K, R, 0.5, Q, is_call)


def _synthetic_chain(true, rng, date=pd.Timestamp("2018-01-02")):
    rows = []
    for d in (21, 49, 84, 126, 182, 273, 365, 547):
        T = d / 365.0
        F = S * np.exp((R - Q) * T)
        Ks = F * np.exp(np.linspace(-0.3, 0.3, 25))
        pr = HestonPricer(S, R, Q, np.full(len(Ks), T), Ks)
        call = pr.call_prices(**true)
        put = call - pr.disc * (pr.F - Ks)
        for K, c, p in zip(Ks, call, put):
            for typ, px in (("call", c), ("put", p)):
                px *= 1 + rng.normal(0, 0.002)
                half = max(0.01, 0.015 * px)
                rows.append(dict(date=date, expiration_date=date + pd.Timedelta(days=d), strike=round(K, 1),
                                 option_type=typ, bid=max(px - half, 0.0), ask=px + half))
    return pd.DataFrame(rows)


def test_calibration_recovers_parameters():
    true = dict(v0=0.06, kappa=2.5, theta=0.05, xi=0.7, rho=-0.7)
    chain = _synthetic_chain(true, np.random.default_rng(0))
    opts, _ = select_options(chain, S, R, Q)
    res = calibrate_snapshot(opts, S, R, Q)
    assert res["rmse_iv"] < 0.2
    for k in ("v0", "theta", "xi", "rho"):
        assert res[k] == pytest.approx(true[k], rel=0.05, abs=0.02)
    assert res["kappa"] == pytest.approx(true["kappa"], rel=0.15)


def test_quantlib_crosscheck():
    """与 QuantLib 的 AnalyticHestonEngine 对比（只在安装了 QuantLib 的环境里运行）。"""
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(2, 1, 2018)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, R, dc))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, Q, dc))
    for p in PARAM_SETS[:4]:
        process = ql.HestonProcess(r_ts, q_ts, ql.QuoteHandle(ql.SimpleQuote(S)), p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"])
        engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
        for days in (30, 180, 365):
            for K in (120.0, 150.0, 180.0):
                opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(today + days))
                opt.setPricingEngine(engine)
                assert heston_call(S, K, R, Q, days / 365.0, **p)[0] == pytest.approx(opt.NPV(), abs=1e-4)