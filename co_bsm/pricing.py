"""Heston 模型欧式期权的半解析定价（Lewis 单积分形式，Albrecher 等的特征函数写法）。

记号：v0 当前方差，kappa 均值回复速度，theta 长期方差，xi 波动率的波动率，rho 相关系数。
X_T = ln(S_T / F)，F = S e^{(r-q)T}，特征函数取 z = u - i/2：

    Call = e^{-rT} [ F - sqrt(F K) / pi * ∫_0^∞ Re( e^{iuk} phi(u - i/2) ) / (u^2 + 1/4) du ],  k = ln(F / K)

积分用 Gauss-Legendre 求积。对一组固定的合约，积分节点和 exp(iuk) 只算一次，
参数变化时只重算特征函数，供校准反复调用。
"""

import numpy as np

_SIGN = 1.0      # exp(SIGN * i u k) 中的符号，由与 Gil-Pelaez 独立实现的对比确定


def char_fn(u, T, v0, kappa, theta, xi, rho):
    """phi(u - i/2)，phi 为 X_T = ln(S_T/F) 的特征函数。u、T 可以广播。"""
    iz = 1j * u + 0.5
    b = kappa - rho * xi * iz
    d = np.sqrt(b * b + xi * xi * (u * u + 0.25))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    D = (b - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e)
    C = kappa * theta / (xi * xi) * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    return np.exp(C + D * v0)


class HestonPricer:
    """一组固定 (T, K) 合约的 Heston 定价器。

    S, r, q 为标量；T、K 为等长的一维数组（每个合约一个）。
    """

    def __init__(self, S, r, q, T, K, n_nodes=384, var_guess=0.01, u_cap=2000.0, z=20.0):
        self.S, self.r, self.q = float(S), float(r), float(q)
        self.T = np.asarray(T, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.F = self.S * np.exp((self.r - self.q) * self.T)
        self.disc = np.exp(-self.r * self.T)
        k = np.log(self.F / self.K)

        self.T_unique, self._inv = np.unique(self.T, return_inverse=True)
        # 积分上限只依赖期限（与参数无关）：特征函数大致按 exp(-w u^2 / 2) 衰减
        upper = np.minimum(z / np.sqrt(var_guess * self.T_unique), u_cap)
        x, w = np.polynomial.legendre.leggauss(n_nodes)
        x, w = 0.5 * (x + 1.0), 0.5 * w                      # 映射到 [0, 1]
        self._u = upper[:, None] * x[None, :]                # (期限数, 节点数)
        self._w = upper[:, None] * w[None, :] / (self._u**2 + 0.25)
        self._E = np.exp(_SIGN * 1j * self._u[self._inv] * k[:, None])   # (合约数, 节点数)
        self._sqrtFK = np.sqrt(self.F * self.K)

    def call_prices(self, v0, kappa, theta, xi, rho):
        phi = char_fn(self._u, self.T_unique[:, None], v0, kappa, theta, xi, rho)
        integral = (self._E * (self._w * phi)[self._inv]).sum(axis=1).real
        return self.disc * (self.F - self._sqrtFK / np.pi * integral)

    def prices(self, is_call, v0, kappa, theta, xi, rho):
        """按 is_call 返回 Call 或 Put 价格（Put 由平价关系得到）。"""
        c = self.call_prices(v0, kappa, theta, xi, rho)
        return np.where(is_call, c, c - self.disc * (self.F - self.K))


def heston_call(S, K, r, q, T, v0, kappa, theta, xi, rho, **kw):
    """单个或一组 (K, T) 的 Call 价格（便捷接口）。"""
    K, T = np.broadcast_arrays(np.atleast_1d(np.asarray(K, float)), np.atleast_1d(np.asarray(T, float)))
    return HestonPricer(S, r, q, T.ravel(), K.ravel(), **kw).call_prices(v0, kappa, theta, xi, rho).reshape(K.shape)


def heston_put(S, K, r, q, T, v0, kappa, theta, xi, rho, **kw):
    K, T = np.broadcast_arrays(np.atleast_1d(np.asarray(K, float)), np.atleast_1d(np.asarray(T, float)))
    p = HestonPricer(S, r, q, T.ravel(), K.ravel(), **kw)
    return p.prices(False, v0, kappa, theta, xi, rho).reshape(K.shape)