import pandas as pd

from co_features.feature_base import FeatureBase


class DividendYieldFeature(FeatureBase):
    """股息率 q：定价模型（BSM / Heston）所需的连续股息率输入。

        dividend_yield_t = 最近一次已除息的每股分红 × 4 / close_raw_t

    口径选择
    --------
    - 分子用"最近一次分红 × 4"（指示性年化分红），而不是过去 12 个月的实际
      分红合计。期权定价需要的是期权存续期内**预期**支付的分红：JPM 每年
      调整一次分红，调整后最新一次分红最能代表未来水平；过去 12 个月合计会
      在分红上调后连续三个季度偏低。按日历 365 天滚动求和还有一个技术问题：
      季度除息日间隔不是严格的 91.25 天，窗口两端偶尔会同时包含 5 次分红，
      使股息率在几天内虚高 25%。
    - 分母必须用 close_raw（未复权价格）。amount 是未复权的分红金额，
      用前复权的 close 会因早期历史价格被分红调整压低，使股息率系统性偏高。
    - 严格的连续股息率为 -ln(1 - D/S)，与 D/S 在 JPM 2%–3% 的股息率水平下
      相差不到 0.01 个百分点，这里直接用 D/S。

    时间归属
    --------
    分红在除息日之前就已公告，以除息日作为"已知"时点是保守的，不存在前视偏差。
    第一次分红之前的日期为缺失值。

    依赖
    ----
    - amount：来自 raw_dividends（除息日有值，其余日期为缺失）
    - close_raw：来自 raw_stock_prices

    注意：如果 Alpha Vantage 返回的分红金额经过了拆股调整，而 close_raw 未调整，
    则 2000-06-12 拆股之前的股息率会偏低三分之一。项目研究区间（2018 年起）
    内 JPM 没有拆股，不受影响。
    """

    PAYMENTS_PER_YEAR = 4

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().compute(df)
        df = df.copy()

        latest_dividend = df["amount"].ffill()
        df["dividend_yield"] = latest_dividend * self.PAYMENTS_PER_YEAR / df["close_raw"]
        return df

    def get_feature_name(self) -> str:
        return "dividend_yield"