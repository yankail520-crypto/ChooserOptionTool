"""BSM检查脚本：用 feature_daily 的历史数据，逐日计算 Chooser Option
理论价格（K=150, T1=0.5, T2=1年，固定参数），画出2018年至今的价格走势。
"""

import matplotlib.pyplot as plt

from co_bsm.chooser import ChooserParams, chooser_price
from co_data_center.data_center import DataCenter

# 中文显示，Windows环境常见的必要配置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

K = 150.0
T1 = 0.5
T2 = 1.0


def main():
    dc = DataCenter()
    feature_obj = dc.get("feature_daily")

    subset = feature_obj.filter(date__gte="2018-01-01")
    df = subset.df.copy()

    # BSM需要的核心字段不能有缺失，直接丢弃这几行，不做填充
    before = len(df)
    df = df.dropna(subset=["close", "rate_1y", "rolling_vol_20d"]).reset_index(drop=True)
    print(f"过滤缺失值：{before} -> {len(df)} 行（丢弃 {before - len(df)} 行）")

    # 股息率粗略估计：最近一次单次分红(前向填充) * 4(按季度年化) / 当前股价
    dividend_yield = (df["amount"].ffill().fillna(0) * 4) / df["close"]

    prices = []
    for idx, row in df.iterrows():
        params = ChooserParams(
            S=row["close"],
            K=K,
            r=row["rate_1y"] / 100,       # FRED数据是百分比，转成小数
            T1=T1,
            T2=T2,
            sigma=row["rolling_vol_20d"],  # 已经是年化小数
            q=dividend_yield.iloc[idx],
        )
        prices.append(chooser_price(params))

    df["chooser_price"] = prices

    print()
    print(df[["close", "chooser_price"]].describe())
    print()
    print("最新10行:")
    print(df[["date", "close", "chooser_price"]].tail(10))

    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["chooser_price"], label="Chooser Option价格 (BSM闭式解)")
    plt.plot(df["date"], df["close"], label="JPM股价", alpha=0.5)
    plt.xlabel("日期")
    plt.ylabel("价格 ($)")
    plt.title(f"JPM Chooser Option理论价格 (K={K}, T1={T1}, T2={T2})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chooser_price_2018_now.png", dpi=150)
    print("\n图已保存: chooser_price_2018_now.png")
    plt.show()


if __name__ == "__main__":
    main()