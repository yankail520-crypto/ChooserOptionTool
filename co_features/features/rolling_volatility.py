import numpy as np
import pandas as pd

from co_features.features.daily_return import DailyReturnFeature


class RollingVolatility20DFeature(DailyReturnFeature):
    """
    20日滚动年化波动率。

    继承 DailyReturnFeature 而不是 FeatureBase——依赖它已经算出
    daily_return 列，compute() 里先调用 super().compute(df) 保证这一点，
    不再需要靠文件名前缀人为控制执行顺序。
    """

    WINDOW = 20
    TRADING_DAYS_PER_YEAR = 252

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().compute(df)  # 先确保 daily_return 列存在
        df = df.copy()
        df["rolling_vol_20d"] = df["daily_return"].rolling(
            window=self.WINDOW
        ).std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return df

    def get_feature_name(self) -> str:
        return "rolling_vol_20d"