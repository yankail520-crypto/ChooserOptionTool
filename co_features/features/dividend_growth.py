import pandas as pd

from co_features.feature_base import FeatureBase


class DividendGrowthFeature(FeatureBase):
    """股息增长率：相邻两次实际分红之间的环比增长率。

    依赖合并阶段已并入的 amount 列（来自 raw_dividends，经 DividendObject
    重命名 ex_date->date）。amount 是稀疏的（只有除息日当天有值，其余
    为 NaN），先前向填充成"最近一次分红金额在每天的状态"，再对这个
    已填充的稠密序列做逐行 pct_change——没变化的日子增长率为0，
    真正发生变化的那天正确算出实际增长率。
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().compute(df)
        df = df.copy()

        filled = df["amount"].ffill()
        df["dividend_growth"] = filled.pct_change(fill_method=None)
        return df

    def get_feature_name(self) -> str:
        return "dividend_growth"