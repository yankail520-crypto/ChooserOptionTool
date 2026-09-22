import pandas as pd

from co_data_pipeline.pipeline_base import DataSourceBase
from common.decorators import data_clear
from common.paths import get_raw_path
from common.logger import get_logger

lg = get_logger("co_data_pipeline")


def _clean_vix(df: pd.DataFrame) -> pd.DataFrame:
    """
    VIX数据专属清洗：日期排序、前向填充缺失值（和利率数据同样的处理方式，
    VIX是每日收盘的单一数值。
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "vix" in df.columns:
        df["vix"] = df["vix"].ffill().bfill()

    return df


class VixSource(DataSourceBase):
    """
    VIX（CBOE波动率指数）日频收盘数据，数据源：FRED（代码 VIXCLS，
    CBOE官方数据）
    """

    SERIES_ID = "VIXCLS"
    START_DATE = "2018-01-01"

    @data_clear(_clean_vix)
    def fetch(self, **params) -> pd.DataFrame:
        import pandas_datareader.data as web  # 延迟导入：避免引擎扫描/实例化阶段就强依赖该库

        series_id = params.get("series_id", self.SERIES_ID)
        start = params.get("start_date", self.START_DATE)

        raw = web.DataReader(series_id, "fred", start=start)
        raw = raw.reset_index()
        raw.columns = ["date", "vix"]

        expected_cols = list(self.get_fields().keys())
        return raw[[c for c in expected_cols if c in raw.columns]]

    def get_fields(self) -> dict:
        return {
            "date": "datetime64[ns]",
            "vix": "float64",
        }

    def get_table_name(self) -> str:
        return "raw_vix"

    def is_calendar_anchor(self) -> bool:
        return False  # 交易日历基准由 raw_stock_prices 承担

    def save(self, df: pd.DataFrame) -> None:
        """按 date 主键 upsert：新数据覆盖旧数据里同日期的行，其余保留。"""
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
        else:
            combined = df

        combined = combined.sort_values("date").reset_index(drop=True)
        combined.to_parquet(path, index=False)
        lg.info(f"{self.get_table_name()}: 已写入 {path}，共 {len(combined)} 行")

    def load(self) -> pd.DataFrame:
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        return pd.read_parquet(path)