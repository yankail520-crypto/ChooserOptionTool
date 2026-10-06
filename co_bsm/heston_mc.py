"""Heston 模型的蒙特卡罗：路径模拟（QE 与 Full Truncation 两种格式）和 chooser 定价。

路径：同时模拟 (ln S, v)，只记录选择日 T1 与到期日 T2 的股价。[0, T1] 与 [T1, T2] 各自等分为若干步，
因此 T1 一定落在网格点上。

格式
- "qe"：Andersen (2008) 的 Quadratic-Exponential 格式，带 martingale 修正（中心离散，gamma1 = gamma2 = 0.5）。
- "ft"：Full Truncation Euler（Lord 等, 2010），作为对照。

chooser：分解形式 Call(K, T2) + e^{-q tau} Put(K', T1) 对股价只依赖 S_T1 与 S_T2，不需要嵌套模拟。
    方式 A：两条腿的 payoff 之和（方差较小，主结果）
    方式 B：路径上直接选择，S_T1 >= K' 选 Call，否则选 Put
半解析定价（co_bsm.pricing）给出精确值，用来检验蒙特卡罗。
"""

import numpy as np

from co_bsm.pricing import heston_call, heston_put

PSI_C = 1.5


# ---------------------------------------------------------------------------
# 单步推进
# ---------------------------------------------------------------------------

def _qe_step(lnS, v, dt, mu, kappa, theta, xi, rho, rng, psi_c=PSI_C):
    e = np.exp(-kappa * dt)
    m = theta + (v - theta) * e
    s2 = v * xi**2 * e * (1.0 - e) / kappa + theta * xi**2 * (1.0 - e) ** 2 / (2.0 * kappa)
    psi = s2 / (m * m)

    zv = rng.standard_normal(v.shape)
    u = rng.random(v.shape)
    z = rng.standard_normal(v.shape)

    k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = 0.5 * dt * (1.0 - rho**2)
    k4 = 0.5 * dt * (1.0 - rho**2)
    A = k2 + 0.5 * k4

    quad = psi <= psi_c
    v_new = np.empty_like(v)
    k0 = np.empty_like(v)

    # 二次型分支
    inv = 2.0 / psi[quad]
    b2 = inv - 1.0 + np.sqrt(inv * (inv - 1.0))
    a = m[quad] / (1.0 + b2)
    d = 1.0 - 2.0 * A * a
    if np.any(d <= 0):
        raise ValueError("QE martingale 修正条件不满足（1 - 2Aa <= 0），请减小步长")
    v_new[quad] = a * (np.sqrt(b2) + zv[quad]) ** 2
    k0[quad] = -A * b2 * a / d + 0.5 * np.log(d) - (k1 + 0.5 * k3) * v[quad]

    # 指数分支
    ex = ~quad
    p = (psi[ex] - 1.0) / (psi[ex] + 1.0)
    beta = (1.0 - p) / m[ex]
    if np.any(beta <= A):
        raise ValueError("QE martingale 修正条件不满足（beta <= A），请减小步长")
    uu = u[ex]
    v_new[ex] = np.where(uu <= p, 0.0, np.log((1.0 - p) / (1.0 - uu)) / beta)
    k0[ex] = -np.log(p + beta * (1.0 - p) / (beta - A)) - (k1 + 0.5 * k3) * v[ex]

    lnS_new = lnS + mu * dt + k0 + k1 * v + k2 * v_new + np.sqrt(k3 * v + k4 * v_new) * z
    return lnS_new, v_new


def _ft_step(lnS, v, dt, mu, kappa, theta, xi, rho, rng):
    z1 = rng.standard_normal(v.shape)
    z2 = rng.standard_normal(v.shape)
    vp = np.maximum(v, 0.0)
    sq = np.sqrt(vp * dt)
    lnS_new = lnS + (mu - 0.5 * vp) * dt + sq * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)
    v_new = v + kappa * (theta - vp) * dt + xi * sq * z1
    return lnS_new, v_new


_STEPS = {"qe": _qe_step, "ft": _ft_step}


# ---------------------------------------------------------------------------
# 路径模拟
# ---------------------------------------------------------------------------

def simulate_heston(S, r, q, v0, kappa, theta, xi, rho, T1, T2, n_paths, steps1, steps2,
                    scheme="qe", seed=None):
    """返回 (S_T1, S_T2)。[0, T1] 走 steps1 步，[T1, T2] 走 steps2 步。"""
    step = _STEPS[scheme]
    rng = np.random.default_rng(seed)
    mu = r - q
    lnS = np.full(n_paths, np.log(S))
    v = np.full(n_paths, float(v0))

    dt1 = T1 / steps1
    for _ in range(steps1):
        lnS, v = step(lnS, v, dt1, mu, kappa, theta, xi, rho, rng)
    S_T1 = np.exp(lnS)

    dt2 = (T2 - T1) / steps2
    for _ in range(steps2):
        lnS, v = step(lnS, v, dt2, mu, kappa, theta, xi, rho, rng)
    return S_T1, np.exp(lnS)


# ---------------------------------------------------------------------------
# chooser
# ---------------------------------------------------------------------------

def heston_chooser_exact(S, K, r, q, T1, T2, v0, kappa, theta, xi, rho):
    """半解析的 Heston chooser 价格：Call(K, T2) + e^{-q tau} Put(K', T1)。返回 (总价, Call 腿, Put 腿)。"""
    tau = T2 - T1
    Kp = K * np.exp(-(r - q) * tau)
    call = float(heston_call(S, K, r, q, T2, v0, kappa, theta, xi, rho)[0])
    put = float(np.exp(-q * tau) * heston_put(S, Kp, r, q, T1, v0, kappa, theta, xi, rho)[0])
    return call + put, call, put


def heston_chooser_mc(S, K, r, q, T1, T2, v0, kappa, theta, xi, rho, n_paths=200_000,
                      steps1=64, steps2=64, scheme="qe", seed=None, batch=250_000):
    """蒙特卡罗 chooser 价格，方式 A（分解）与方式 B（路径选择），都带标准误。"""
    tau = T2 - T1
    Kp = K * np.exp(-(r - q) * tau)
    rng_seed = np.random.SeedSequence(seed)
    pay_a, pay_b, call_pay, put_pay = [], [], [], []

    done = 0
    for child in rng_seed.spawn(int(np.ceil(n_paths / batch))):
        n = min(batch, n_paths - done)
        s1, s2 = simulate_heston(S, r, q, v0, kappa, theta, xi, rho, T1, T2, n, steps1, steps2,
                                 scheme=scheme, seed=child)
        call = np.exp(-r * T2) * np.maximum(s2 - K, 0.0)
        put = np.exp(-q * tau) * np.exp(-r * T1) * np.maximum(Kp - s1, 0.0)
        path = np.exp(-r * T2) * np.where(s1 >= Kp, np.maximum(s2 - K, 0.0), np.maximum(K - s2, 0.0))
        call_pay.append(call); put_pay.append(put); pay_a.append(call + put); pay_b.append(path)
        done += n

    def est(x):
        x = np.concatenate(x)
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))

    (a, a_se), (b, b_se), (c, c_se), (p, p_se) = est(pay_a), est(pay_b), est(call_pay), est(put_pay)
    return dict(price_a=a, se_a=a_se, price_b=b, se_b=b_se, call=c, call_se=c_se, put=p, put_se=p_se)