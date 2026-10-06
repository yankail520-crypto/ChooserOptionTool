from __future__ import annotations

import numpy as np

from co_bsm.black_scholes import BSMParams, bsm_call, bsm_put


def implied_vol(price, S, K, r, T, q, is_call, lo: float = 1e-4, hi: float = 5.0,
                n_iter: int = 60, tol: float = 1e-8):
    """BSM 隐含波动率，向量化二分法。

    价格低于 sigma=lo 时的价格、或高于 sigma=hi 时的价格（无解）返回 NaN。
    is_call 为布尔，可以是数组，用来在同一批合约里混合 Call 与 Put。
    """
    price, S, K, r, T, q, is_call = np.broadcast_arrays(
        *[np.asarray(a, dtype=float) for a in (price, S, K, r, T, q)],
        np.asarray(is_call, dtype=bool),
    )

    def bsm(sigma):
        p = BSMParams(S, K, r, T, sigma, q)
        return np.where(is_call, bsm_call(p), bsm_put(p))

    low = np.full(price.shape, lo)
    high = np.full(price.shape, hi)
    price_lo, price_hi = bsm(low), bsm(high)

    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        below = bsm(mid) < price
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)

    iv = 0.5 * (low + high)
    no_solution = (price < price_lo - tol) | (price > price_hi + tol) | ~np.isfinite(price)
    return np.where(no_solution, np.nan, iv)


def bsm_vega(S, K, r, T, sigma, q):
    """BSM vega（价格对 sigma 的导数，sigma 变动 1.0 的价格变化）。"""
    from scipy.stats import norm

    S, K, r, T, sigma, q = [np.asarray(a, dtype=float) for a in (S, K, r, T, sigma, q)]
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)