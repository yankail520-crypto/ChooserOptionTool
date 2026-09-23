import os

import pandas as pd
import requests

from co_data_pipeline.pipeline_base import DataSourceBase
from common.decorators import data_clear
from common.paths import get_raw_path
from common.logger import get_logger

lg = get_logger("co_data_pipeline")

_ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def _clean_dividends(df: pd.DataFrame) -> pd.DataFrame:
    """股息数据专属清洗：日期格式统一、按除息日排序。

    股息本身是季度性的稀疏数据，某个季度没有分红是正常状态，
    不是"缺失"，因此不做插值/填充，只做基础的格式规整。
    """
    df = df.copy()
    for col in ["ex_date", "declaration_date", "record_date", "payment_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.sort_values("ex_date").reset_index(drop=True)

    return df


class DividendSource(DataSourceBase):
    """JPM 历史分红记录，数据源：Alpha Vantage DIVIDENDS。"""

    TICKER = "JPM"

    @data_clear(_clean_dividends)
    def fetch(self, **params) -> pd.DataFrame:
        ticker = params.get("ticker", self.TICKER)

        api_key = os.environ.get("ALPHA_VANTAGE_KEY")
        if not api_key:
            raise RuntimeError("环境变量 ALPHA_VANTAGE_KEY 未设置")

        resp = requests.get(
            _ALPHA_VANTAGE_URL,
            params={
                "function": "DIVIDENDS",
                "symbol": ticker,
                "apikey": api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()

        data = payload.get("data")
        if data is None:
            raise RuntimeError(f"Alpha Vantage 未返回有效股息数据：{payload}")

        records = []
        for item in data:
            records.append(
                {
                    "ex_date": item.get("ex_dividend_date"),
                    "declaration_date": item.get("declaration_date"),
                    "record_date": item.get("record_date"),
                    "payment_date": item.get("payment_date"),
                    "amount": item.get("amount"),
                }
            )

        df = pd.DataFrame(records)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        expected_cols = list(self.get_fields().keys())
        return df[[c for c in expected_cols if c in df.columns]]

    def get_fields(self) -> dict:
        return {
            "ex_date": "datetime64[ns]",
            "declaration_date": "datetime64[ns]",
            "record_date": "datetime64[ns]",
            "payment_date": "datetime64[ns]",
            "amount": "float64",
        }

    def get_table_name(self) -> str:
        return "raw_dividends"

    def is_calendar_anchor(self) -> bool:
        return False  # 交易日历基准由 raw_stock_prices 承担

    def save(self, df: pd.DataFrame) -> None:
        """按 ex_date 主键 upsert：新数据覆盖旧数据里同除息日的行，其余保留。"""
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["ex_date"], keep="last")
        else:
            combined = df

        combined = combined.sort_values("ex_date").reset_index(drop=True)
        combined.to_parquet(path, index=False)
        lg.info(f"{self.get_table_name()}: 已写入 {path}，共 {len(combined)} 行")

    def load(self) -> pd.DataFrame:
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        return pd.read_parquet(path)