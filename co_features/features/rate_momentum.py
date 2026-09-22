import pandas as pd

from co_features.feature_base import FeatureBase


class InterestRateMomentumFeature(FeatureBase):
    """利率动量：20个交易日内 1年期国债利率(rate_1y) 的变化量（不是比率，
    因为利率本身是百分比数值，直接做差比做比值更符合利率数据的习惯用法）。
    """

    WINDOW = 20

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().compute(df)
        df = df.copy()
        df["rate_momentum_20d"] = df["rate_1y"].diff(self.WINDOW)
        return df

    def get_feature_name(self) -> str:
        return "rate_momentum_20d"