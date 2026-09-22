import numpy as np
import pandas as pd

from co_features.feature_base import FeatureBase


class DailyReturnFeature(FeatureBase):
    """日对数收益率：ln(close_t / close_{t-1})。"""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().compute(df)  # 无条件先调用，保证多继承时MRO链路完整
        df = df.copy()
        df["daily_return"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def get_feature_name(self) -> str:
        return "daily_return"