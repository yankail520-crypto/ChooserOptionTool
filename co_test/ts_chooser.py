"""Chooser 定价的单元测试。运行：pytest tests/test_chooser.py -v"""

import numpy as np
import pytest

from co_bsm.black_scholes import BSMParams, bsm_call, bsm_put
from co_bsm.chooser import ChooserParams, chooser_price, chooser_price_rubinstein

# 贴近 JPM 的一组参数：带股息，K=150，T2=1
BASE = dict(S=145.0, K=150.0, r=0.04, T1=0.5, T2=1.0, sigma=0.25, q=0.028)


def _params(**overrides) -> ChooserParams:
    return ChooserParams(**{**BASE, **overrides})


def _call_put_at_t2(p: ChooserParams):
    kw = dict(S=p.S, K=p.K, r=p.r, T=p.T2, sigma=p.sigma, q=p.q)
    return bsm_call(BSMParams(**kw)), bsm_put(BSMParams(**kw))


# ---- 1. 分解形式 vs Rubinstein 原公式 -------------------------------------

@pytest.mark.parametrize("q", [0.0, 0.028, 0.06])
@pytest.mark.parametrize("S", [100.0, 150.0, 200.0])
@pytest.mark.parametrize("T1", [0.05, 0.25, 0.5, 0.9, 1.0])
def test_decomposition_matches_rubinstein(q, S, T1):
    p = _params(q=q, S=S, T1=T1)
    assert chooser_price(p) == pytest.approx(chooser_price_rubinstein(p), rel=1e-10, abs=1e-10)


def test_old_decomposition_is_wrong_with_dividends():
    """回归测试：旧写法 Put(K*e^{-r*tau}, T1) 在 q>0 时与闭式解不一致。"""
    p = _params()
    tau = p.T2 - p.T1
    old = bsm_call(BSMParams(S=p.S, K=p.K, r=p.r, T=p.T2, sigma=p.sigma, q=p.q)) + bsm_put(
        BSMParams(S=p.S, K=p.K * np.exp(-p.r * tau), r=p.r, T=p.T1, sigma=p.sigma, q=p.q)
    )
    assert abs(old - chooser_price_rubinstein(p)) > 1e-3


# ---- 2. 边界条件 ------------------------------------------------------------

def test_t1_equals_t2_is_straddle():
    p = _params(T1=1.0, T2=1.0)
    call, put = _call_put_at_t2(p)
    assert chooser_price(p) == pytest.approx(call + put, rel=1e-12)


def test_t1_to_zero_is_max_call_put():
    p = _params(T1=1e-10)
    call, put = _call_put_at_t2(p)
    assert chooser_price(p) == pytest.approx(max(call, put), rel=1e-6)


# ---- 3. 单调性与上下界 -----------------------------------------------------

def test_monotone_in_t1_and_bounded():
    t1_grid = np.linspace(0.01, 1.0, 100)
    prices = np.array([chooser_price(_params(T1=t)) for t in t1_grid])
    call, put = _call_put_at_t2(_params())

    assert np.all(np.diff(prices) > 0), "价格应随 T1 严格递增"
    assert np.all(prices >= max(call, put) - 1e-12)
    assert np.all(prices <= call + put + 1e-12)


# ---- 4. 独立的 GBM 蒙特卡罗验证（不依赖分解） -----------------------------

def test_monte_carlo_gbm():
    """模拟 S_{T1}，在 T1 用 BSM 算出剩余期限的 C、P 并取较大者，再折现到 0。"""
    p = _params()
    rng = np.random.default_rng(20260923)
    n = 400_000
    tau = p.T2 - p.T1

    z = rng.standard_normal(n)
    s_t1 = p.S * np.exp((p.r - p.q - 0.5 * p.sigma**2) * p.T1 + p.sigma * np.sqrt(p.T1) * z)

    kw = dict(S=s_t1, K=p.K, r=p.r, T=tau, sigma=p.sigma, q=p.q)
    value_t1 = np.maximum(bsm_call(BSMParams(**kw)), bsm_put(BSMParams(**kw)))
    discounted = np.exp(-p.r * p.T1) * value_t1

    mc_price = discounted.mean()
    std_err = discounted.std(ddof=1) / np.sqrt(n)
    assert abs(mc_price - chooser_price(p)) < 4 * std_err


# ---- 5. 向量化 ------------------------------------------------------------

def test_vectorized_matches_scalar():
    S = np.array([120.0, 145.0, 150.0, 180.0])
    sigma = np.array([0.20, 0.25, 0.30, 0.45])
    r = np.array([0.01, 0.02, 0.04, 0.05])

    vec = chooser_price(_params(S=S, sigma=sigma, r=r))
    scalar = [chooser_price(_params(S=s, sigma=v, r=rr)) for s, v, rr in zip(S, sigma, r)]
    np.testing.assert_allclose(vec, scalar, rtol=1e-12)


# ---- 6. 参数校验 ------------------------------------------------------------

@pytest.mark.parametrize(
    "overrides",
    [
        dict(T1=0.0),
        dict(T1=-0.1),
        dict(T1=1.2, T2=1.0),
        dict(sigma=0.0),
        dict(S=-1.0),
        dict(K=0.0),
        dict(r=np.nan),
        dict(S=np.array([150.0, np.nan])),
    ],
)
def test_invalid_inputs_raise(overrides):
    with pytest.raises(ValueError):
        chooser_price(_params(**overrides))