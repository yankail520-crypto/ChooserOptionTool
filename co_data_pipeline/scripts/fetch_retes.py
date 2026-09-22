import pandas as pd

from co_data_pipeline.pipeline_base import DataSourceBase
from common.decorators import data_clear
from common.paths import get_raw_path
from common.logger import get_logger

lg = get_logger("co_data_pipeline")


def _clean_treasury_rates(df: pd.DataFrame) -> pd.DataFrame:
    """利率数据专属清洗：日期排序、前向填充缺失值（利率变化平缓，前值仍具参考性）。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "rate_1y" in df.columns:
        df["rate_1y"] = df["rate_1y"].ffill().bfill()

    return df


class TreasuryRateSource(DataSourceBase):
    """1年期美国国债利率（FRED 代码 DGS1），用作 BSM 定价中与 T2=1年 期权到期
    期限匹配的无风险利率。数据源：FRED，经 pandas_datareader 读取，无需 API key。
    """

    SERIES_ID = "DGS1"
    START_DATE = "2018-01-01"

    @data_clear(_clean_treasury_rates)
    def fetch(self, **params) -> pd.DataFrame:
        import pandas_datareader.data as web  # 延迟导入：避免引擎扫描/实例化阶段就强依赖该库

        series_id = params.get("series_id", self.SERIES_ID)
        start = params.get("start_date", self.START_DATE)

        raw = web.DataReader(series_id, "fred", start=start)
        raw = raw.reset_index()
        raw.columns = ["date", "rate_1y"]

        expected_cols = list(self.get_fields().keys())
        return raw[[c for c in expected_cols if c in raw.columns]]

    def get_fields(self) -> dict:
        return {
            "date": "datetime64[ns]",
            "rate_1y": "float64",
        }

    def get_table_name(self) -> str:
        return "raw_treasury_rates"

    def is_calendar_anchor(self) -> bool:
        return False  # 交易日历基准由 raw_stock_prices 承担，本数据源仅被对齐

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