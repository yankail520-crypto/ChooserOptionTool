"""敏感性分析检查脚本：复现论文 4.1-4.4 节的四张敏感性分析图
（Volatility / Strike Price / Risk-free Rate / Dividend Yield），
用我们自己的 BSM 实现验证与论文报告的方向性趋势是否一致。

基准参数取自论文 Table 2：S=156.7, K=150, r=0.15%, sigma=28.2%,
q=2.33%, T=1 年（论文 4.1-4.4 节做的是普通欧式 Call/Put 的敏感性分析，
不涉及 Chooser 的 T1）。

BSMParams 支持数组输入：把被扫描的参数直接传成数组，一次调用算出整条曲线。
"""

import numpy as np
import matplotlib.pyplot as plt

from co_bsm.black_scholes import BSMParams, bsm_call, bsm_put

# 按顺序尝试中文字体，第一个可用的生效（Windows: SimHei / Microsoft YaHei；
# macOS: PingFang SC；Linux: Noto Sans CJK / 文泉驿）
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "WenQuanYi Zen Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 论文 Table 2 的基准参数
BASE = dict(S=156.7, K=150.0, r=0.0015, T=1.0, sigma=0.282, q=0.0233)

N_POINTS = 200

# (参数名, 扫描区间, x 轴标签, x 轴缩放, 子图标题, Call 预期方向, Put 预期方向)
# 预期方向：+1 表示随参数增大而上升，-1 表示下降。
# sigma 和 K 必须严格为正（BSM 层会校验）；r 和 q 允许为 0。
SWEEPS = [
    ("sigma", np.linspace(0.001, 1.0, N_POINTS), "波动率 σ (%)", 100, "4.1 波动率敏感性", +1, +1),
    ("K", np.linspace(1.0, 500.0, N_POINTS), "行权价 K ($)", 1, "4.2 行权价敏感性", -1, +1),
    ("r", np.linspace(0.0, 0.10, N_POINTS), "无风险利率 r (%)", 100, "4.3 利率敏感性", +1, -1),
    ("q", np.linspace(0.0, 0.10, N_POINTS), "股息率 q (%)", 100, "4.4 股息率敏感性", -1, +1),
]


def price_curve(param_name: str, values: np.ndarray) -> tuple:
    """固定其余参数，把 param_name 替换为数组，一次向量化调用返回 (call, put)。"""
    p = BSMParams(**{**BASE, param_name: values})
    return bsm_call(p), bsm_put(p)


def check_direction(prices: np.ndarray, expected: int, tol: float = 1e-9) -> bool:
    """检查曲线是否在整个区间上单调（允许浮点噪声和深度虚值区间的平坦段）。"""
    diffs = np.diff(prices)
    return bool(np.all(diffs * expected >= -tol))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    results = []

    for ax, (name, values, xlabel, scale, title, call_dir, put_dir) in zip(axes.flat, SWEEPS):
        calls, puts = price_curve(name, values)

        ax.plot(values * scale, calls, label="Call")
        ax.plot(values * scale, puts, label="Put")
        ax.axvline(BASE[name] * scale, color="gray", linestyle="--", alpha=0.6, label="基准值")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("期权价格 ($)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        results.append((title, "Call", call_dir, check_direction(calls, call_dir)))
        results.append((title, "Put", put_dir, check_direction(puts, put_dir)))

    plt.tight_layout()
    plt.savefig("sensitivity_analysis.png", dpi=150)
    print("图已保存: sensitivity_analysis.png\n")

    # 方向性趋势检查（对应论文 4.1-4.4 节的结论）
    print("方向性趋势检查：")
    arrow = {+1: "↑", -1: "↓"}
    for title, kind, expected, ok in results:
        print(f"  [{'通过' if ok else '失败'}] {title:<12} {kind:<4} 预期随参数增大 {arrow[expected]}")

    # 基准点数值，方便和论文图中的位置目测对照
    base_params = BSMParams(**BASE)
    print(f"\n基准参数下: Call={bsm_call(base_params):.2f}, Put={bsm_put(base_params):.2f}")

    if not all(ok for *_, ok in results):
        raise SystemExit("存在与预期方向不一致的曲线，请检查实现或参数")

    plt.show()


if __name__ == "__main__":
    main()