from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from co_bsm.black_scholes import ArrayLike, BSMParams, bsm_call, bsm_put
from co_bsm.black_scholes import _validate as _validate_bsm


@dataclass
class ChooserParams:
    """Simple Chooser Option 定价所需的全部输入参数。

    T1: 选择日（投资者决定这是 Call 还是 Put 的时刻）
    T2: 最终到期日
    K:  行权价（Call 和 Put 共用同一个，这是 "Simple" Chooser 的定义）
    q:  连续股息率

    所有字段均可为标量或可广播的 numpy 数组。
    """

    S: ArrayLike
    K: ArrayLike
    r: ArrayLike
    T1: ArrayLike
    T2: ArrayLike
    sigma: ArrayLike
    q: ArrayLike = 0.0


def _validate(p: ChooserParams) -> None:
    """要求 0 < T1 <= T2。

    T1 = 0 时 Put 腿的 sigma*sqrt(T1) 为 0，会出现除零；其极限 max(C, P)
    作为测试用例单独验证，主函数不支持。S、K、sigma 的正性由 BSM 层校验。
    """
    t1 = np.asarray(p.T1, dtype=float)
    t2 = np.asarray(p.T2, dtype=float)
    if np.any(t1 <= 0):
        raise ValueError(f"T1 必须严格为正，最小值为 {t1.min()}")
    if np.any(t1 > t2):
        raise ValueError("T1 不能晚于 T2")


def chooser_price(p: ChooserParams) -> ArrayLike:
    """Simple Chooser Option 定价（静态复制 / 分解形式）。

    记 tau = T2 - T1。T1 时刻带股息的看跌-看涨平价为
        P - C = K*e^{-r*tau} - S_{T1}*e^{-q*tau}
    因此
        max(C, P) = C + e^{-q*tau} * max(0, K*e^{-(r-q)*tau} - S_{T1})
    即
        Chooser = Call(K, T2) + e^{-q*tau} * Put(K*e^{-(r-q)*tau}, T1)

    这是精确的复制关系，不是数值近似。注意 Put 腿有两处股息调整：
    行权价用 (r-q) 折现，数量乘以 e^{-q*tau}。q=0 时退化为
    Call(K, T2) + Put(K*e^{-r*tau}, T1)。
    """
    _validate(p)
    tau = p.T2 - p.T1

    call_leg = bsm_call(BSMParams(S=p.S, K=p.K, r=p.r, T=p.T2, sigma=p.sigma, q=p.q))

    put_strike = p.K * np.exp(-(p.r - p.q) * tau)
    put_leg = bsm_put(BSMParams(S=p.S, K=put_strike, r=p.r, T=p.T1, sigma=p.sigma, q=p.q))

    return call_leg + np.exp(-p.q * tau) * put_leg


def chooser_price_rubinstein(p: ChooserParams) -> ArrayLike:
    """Rubinstein (1991) 闭式解的直接实现，用于和分解形式交叉验证。

    V = S e^{-qT2} N(d1) - K e^{-rT2} N(d2) - S e^{-qT2} N(-y1) + K e^{-rT2} N(-y2)

    d1 = [ln(S/K) + (r - q + sigma^2/2) T2] / (sigma sqrt(T2)),  d2 = d1 - sigma sqrt(T2)
    y1 = [ln(S/K) + (r - q) T2 + sigma^2 T1 / 2] / (sigma sqrt(T1)), y2 = y1 - sigma sqrt(T1)
    """
    _validate(p)
    _validate_bsm(BSMParams(S=p.S, K=p.K, r=p.r, T=p.T2, sigma=p.sigma, q=p.q))  # S、K、sigma 正性

    log_sk = np.log(p.S / p.K)
    vol_t2 = p.sigma * np.sqrt(p.T2)
    vol_t1 = p.sigma * np.sqrt(p.T1)

    d1 = (log_sk + (p.r - p.q + 0.5 * p.sigma**2) * p.T2) / vol_t2
    d2 = d1 - vol_t2
    y1 = (log_sk + (p.r - p.q) * p.T2 + 0.5 * p.sigma**2 * p.T1) / vol_t1
    y2 = y1 - vol_t1

    s_disc = p.S * np.exp(-p.q * p.T2)
    k_disc = p.K * np.exp(-p.r * p.T2)

    return (
        s_disc * norm.cdf(d1) - k_disc * norm.cdf(d2)
        - s_disc * norm.cdf(-y1) + k_disc * norm.cdf(-y2)
    )