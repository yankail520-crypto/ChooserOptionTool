"""Heston-MC 的步长收敛检验：在校准得到的四组真实参数上，对比 QE 与 Full Truncation 相对半解析精确值的偏差。
运行：python -m co_test.ts_heston_convergence [参数集名称 ...]
"""
import sys

import numpy as np

from co_bsm.heston_mc import heston_chooser_exact, heston_chooser_mc
S,K,r,q,T1,T2 = 156.7,150.0,0.0015,0.0233,0.5,1.0
sets = {
 "全样本中位数":       dict(v0=0.054, kappa=7.514, theta=0.077, xi=1.855, rho=-0.503),
 "2021-12-01 ξ=4":     dict(v0=0.141, kappa=17.518, theta=0.096, xi=4.000, rho=-0.384),
 "2024-06-03 κ=25":    dict(v0=0.032, kappa=25.000, theta=0.055, xi=3.478, rho=-0.389),
 "2020-04-01 高波动":  dict(v0=0.721, kappa=4.034, theta=0.280, xi=4.000, rho=-0.682),
}
which = sys.argv[1:] or list(sets)
N = 300_000
for name in which:
    p = sets[name]
    ex, c, pu = heston_chooser_exact(S,K,r,q,T1,T2,**p)
    print(f"\n[{name}]  精确 chooser = {ex:.4f}（Call 腿 {c:.4f}，Put 腿 {pu:.4f}）  Feller 比 {2*p['kappa']*p['theta']/p['xi']**2:.2f}")
    print(f"{'每段步数':>8} | {'QE 方式A':>10} {'偏差':>8} {'/SE':>6} | {'FT 方式A':>10} {'偏差':>8} {'/SE':>6}")
    for n in (4, 8, 16, 32, 64):
        row = []
        for scheme in ("qe", "ft"):
            try:
                res = heston_chooser_mc(S,K,r,q,T1,T2,**p, n_paths=N, steps1=n, steps2=n, scheme=scheme, seed=11)
                row.append((res["price_a"], res["price_a"]-ex, (res["price_a"]-ex)/res["se_a"]))
            except ValueError as e:
                row.append(None)
        f = lambda x: f"{x[0]:10.4f} {x[1]:+8.4f} {x[2]:+6.1f}" if x else f"{'条件不满足':>26}"
        print(f"{n:>8} | {f(row[0])} | {f(row[1])}")