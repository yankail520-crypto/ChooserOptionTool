"""股息率检查脚本：绘制 dividend_yield 时间序列，并打印关键数值供人工核对。

用法：
    python ts_dividend_yield.py                       # 全部历史
    python ts_dividend_yield.py --start 2018-01-01    # 项目研究区间
    python ts_dividend_yield.py --start 2018-01-01 --out dy.png

图分三栏（共享时间轴）：
    1. 股息率（%），圆点标出除息日
    2. 最近一次每股分红金额（阶梯线），用于核对股息率的变化来自分红调整还是股价波动
    3. 未复权收盘价 close_raw（股息率的分母）
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from co_data_center.data_center import DataCenter
from co_features.features.dividend_yield import DividendYieldFeature

plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "WenQuanYi Zen Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

SPLIT_DATE = pd.Timestamp("2000-06-12")   # 大通曼哈顿 3 拆 2
SANE_RANGE = (0.005, 0.10)                # 超出 0.5%–10% 的股息率打印出来人工核对


def load_data() -> pd.DataFrame:
    df = DataCenter().get("feature_daily").df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "dividend_yield" not in df.columns:
        print("feature_daily 中还没有 dividend_yield（FeatureEngine 尚未重新运行），临时计算\n")
        df = DividendYieldFeature(df).df
    return df.sort_values("date").reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    dy = df["dividend_yield"]
    valid = df.loc[dy.notna()]
    if valid.empty:
        print("没有有效的股息率数据")
        return

    print(f"有效区间：{valid['date'].min().date()} ~ {valid['date'].max().date()}，共 {len(valid)} 个交易日")
    print(f"最新股息率：{dy.iloc[-1]:.2%}（{df['date'].iloc[-1].date()}）")
    print(f"最小值：{dy.min():.2%}（{df.loc[dy.idxmin(), 'date'].date()}）")
    print(f"最大值：{dy.max():.2%}（{df.loc[dy.idxmax(), 'date'].date()}）\n")

    yearly = valid.groupby(valid["date"].dt.year)["dividend_yield"].agg(["mean", "min", "max"])
    print("按年统计：")
    print(yearly.map(lambda x: f"{x:.2%}").to_string(), "\n")

    ex = df.loc[df["amount"].notna(), ["date", "amount", "close_raw", "dividend_yield"]]
    print("除息日记录（最近 8 次）：")
    print(ex.tail(8).to_string(index=False), "\n")

    # 拆股前后的分红金额：用于判断 Alpha Vantage 的分红金额是否经过拆股调整
    around_split = ex[(ex["date"] >= "1999-10-01") & (ex["date"] <= "2001-01-31")]
    if not around_split.empty:
        print("2000 年拆股前后的分红金额（拆股前季度分红为 0.48 美元；若显示 0.32，")
        print("说明分红金额经过了拆股调整，拆股前的股息率会偏低）：")
        print(around_split[["date", "amount"]].to_string(index=False), "\n")

    lo, hi = SANE_RANGE
    odd = valid[(valid["dividend_yield"] < lo) | (valid["dividend_yield"] > hi)]
    if odd.empty:
        print(f"所有股息率都在 {lo:.1%}–{hi:.0%} 之间")
    else:
        print(f"{len(odd)} 个交易日的股息率超出 {lo:.1%}–{hi:.0%}，示例：")
        print(odd[["date", "amount", "close_raw", "dividend_yield"]].head(10).to_string(index=False))


def plot(df: pd.DataFrame, out: str) -> None:
    ex = df[df["amount"].notna()]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})

    ax = axes[0]
    ax.plot(df["date"], df["dividend_yield"] * 100, lw=1)
    ax.scatter(ex["date"], ex["dividend_yield"] * 100, s=12, color="C3", zorder=3, label="除息日")
    ax.set_ylabel("股息率 (%)")
    ax.set_title("JPM 股息率 = 最近一次分红 × 4 / close_raw")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.step(df["date"], df["amount"].ffill(), where="post", lw=1)
    ax.scatter(ex["date"], ex["amount"], s=12, color="C3", zorder=3)
    ax.set_ylabel("每股分红 ($)")

    ax = axes[2]
    ax.plot(df["date"], df["close_raw"], lw=1, color="C2")
    ax.set_ylabel("close_raw ($)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        if df["date"].min() <= SPLIT_DATE <= df["date"].max():
            ax.axvline(SPLIT_DATE, color="gray", ls="--", alpha=0.6)
    if df["date"].min() <= SPLIT_DATE <= df["date"].max():
        axes[0].annotate("2000-06-12 拆股", (SPLIT_DATE, axes[0].get_ylim()[1]),
                         xytext=(5, -15), textcoords="offset points", color="gray")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\n图已保存：{out}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 JPM 股息率时间序列")
    parser.add_argument("--start", type=str, default=None, help="起始日期，如 2018-01-01")
    parser.add_argument("--end", type=str, default=None, help="结束日期")
    parser.add_argument("--out", type=str, default="dividend_yield.png", help="图片保存路径")
    args = parser.parse_args()

    df = load_data()
    if args.start:
        df = df[df["date"] >= args.start]
    if args.end:
        df = df[df["date"] <= args.end]

    print_summary(df)
    plot(df, args.out)


if __name__ == "__main__":
    main()