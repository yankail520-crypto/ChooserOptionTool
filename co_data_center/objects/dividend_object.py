import pandas as pd

from co_data_center.data_center_base import DataObject


class DividendObject(DataObject):
    """对应 raw_dividends 表的特化数据对象。

    raw_dividends 的主键列叫 ex_date（除息日），而不是其他表统一使用的
    date——为了能和其他表一样按 date 合并进 feature_daily 宽表，
    to_daily() 把 ex_date 重命名为 date，其余保持原样（本身就是
    稀疏的、一天最多一条的数据，不需要额外聚合）。
    """

    SOURCE_TABLE = "raw_dividends"

    def to_daily(self) -> pd.DataFrame:
        df = self._df.copy()
        df = df.rename(columns={"ex_date": "date"})
        return df[["date", "amount"]]