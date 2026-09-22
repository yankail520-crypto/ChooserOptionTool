import os

import pandas as pd
import requests

from co_data_pipeline.pipeline_base import DataSourceBase
from common.decorators import data_clear
from common.paths import get_raw_path
from common.logger import get_logger

lg = get_logger("co_data_pipeline")

_ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def _clean_stock_prices(df: pd.DataFrame) -> pd.DataFrame:
    """股价数据专属清洗：日期排序、价格列线性插值、成交量缺失填0。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear")

    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype("int64")

    return df


class StockPriceSource(DataSourceBase):
    """JPM 日频股价（OHLCV），数据源：Alpha Vantage TIME_SERIES_DAILY_ADJUSTED
    （付费层接口，返回后复权收盘价，需要 Premium 订阅）。

    close 列直接取"5. adjusted close"（后复权收盘价），而不是原始未复权
    的 "4. close"——后复权价格不会因未来新的分红而回溯性改变历史值，
    更适合做固定的历史数据集（不同于前复权，每次新增分红都会让
    历史价格整体重新调整）。
    """

    TICKER = "JPM"

    @data_clear(_clean_stock_prices)
    def fetch(self, **params) -> pd.DataFrame:
        ticker = params.get("ticker", self.TICKER)

        api_key = os.environ.get("ALPHA_VANTAGE_KEY")
        if not api_key:
            raise RuntimeError("环境变量 ALPHA_VANTAGE_KEY 未设置")

        resp = requests.get(
            _ALPHA_VANTAGE_URL,
            params={
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": "full",
                "apikey": api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()

        series = payload.get("Time Series (Daily)")
        if series is None:
            raise RuntimeError(f"Alpha Vantage 未返回有效数据：{payload}")

        records = []
        for date_str, values in series.items():
            records.append(
                {
                    "date": date_str,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["5. adjusted close"]),  # 后复权收盘价
                    "volume": int(values["6. volume"]),
                }
            )

        df = pd.DataFrame(records)
        expected_cols = list(self.get_fields().keys())
        return df[[c for c in expected_cols if c in df.columns]]

    def get_fields(self) -> dict:
        return {
            "date": "datetime64[ns]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
        }

    def get_table_name(self) -> str:
        return "raw_stock_prices"

    def is_calendar_anchor(self) -> bool:
        return True

    def save(self, df: pd.DataFrame) -> None:
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