import pandas as pd

from co_features.features.daily_return import DailyReturnFeature


class VixCorrelationFeature(DailyReturnFeature):
    """
    VIX-JPM滚动相关性：JPM日收益率 与 VIX日度变化率 的20日滚动相关系数。

    继承 DailyReturnFeature——依赖 daily_return 列已存在。用收益率而
    不是价格水平本身做相关性，是因为价格水平通常非平稳（两者都有
    长期趋势），直接相关容易产生虚假的高相关性；用变化率/收益率
    做相关性是更标准、更有实际意义的做法。
    """

    WINDOW = 20

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().compute(df)  # 确保 daily_return 列存在
        df = df.copy()

        vix_change = df["vix"].pct_change(fill_method=None)
        df["vix_jpm_correlation_20d"] = df["daily_return"].rolling(
            window=self.WINDOW
        ).corr(vix_change)
        return df

    def get_feature_name(self) -> str:
        return "vix_jpm_correlation_20d"