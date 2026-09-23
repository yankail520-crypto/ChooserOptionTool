from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.stats import norm

# 所有字段既可以是标量，也可以是同形状（或可广播）的 numpy 数组——
# 每日滚动定价时直接传入整条时间序列，一次算出全部结果。
ArrayLike = Union[float, np.ndarray]


@dataclass
class BSMParams:
    """BSM 定价所需的全部输入参数。"""

    S: ArrayLike       # 标的现价
    K: ArrayLike       # 行权价
    r: ArrayLike       # 无风险利率（连续复利）
    T: ArrayLike       # 到期时间（年）
    sigma: ArrayLike   # 波动率（年化）
    q: ArrayLike = 0.0  # 连续股息率，默认 0（不分红）


def _validate(p: BSMParams) -> None:
    """边界情况下 numpy 只会报 RuntimeWarning 并返回 inf/NaN，不会报错，这里显式拦截。"""
    checks = {"S": p.S, "K": p.K, "T": p.T, "sigma": p.sigma}
    for name, value in checks.items():
        arr = np.asarray(value, dtype=float)
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} 含有 NaN 或 inf")
        if np.any(arr <= 0):
            raise ValueError(f"{name} 必须严格为正，最小值为 {arr.min()}")
    for name, value in {"r": p.r, "q": p.q}.items():
        if np.any(~np.isfinite(np.asarray(value, dtype=float))):
            raise ValueError(f"{name} 含有 NaN 或 inf")


def _d1_d2(p: BSMParams) -> tuple:
    vol_sqrt_t = p.sigma * np.sqrt(p.T)
    d1 = (np.log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bsm_call(p: BSMParams) -> ArrayLike:
    """标准欧式看涨期权的 BSM 定价。"""
    _validate(p)
    d1, d2 = _d1_d2(p)
    return p.S * np.exp(-p.q * p.T) * norm.cdf(d1) - p.K * np.exp(-p.r * p.T) * norm.cdf(d2)


def bsm_put(p: BSMParams) -> ArrayLike:
    """标准欧式看跌期权的 BSM 定价。"""
    _validate(p)
    d1, d2 = _d1_d2(p)
    return p.K * np.exp(-p.r * p.T) * norm.cdf(-d2) - p.S * np.exp(-p.q * p.T) * norm.cdf(-d1)