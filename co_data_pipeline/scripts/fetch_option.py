import os

import pandas as pd
import requests

from co_data_pipeline.pipeline_base import DataSourceBase
from common.decorators import data_clear
from common.paths import get_raw_path
from common.logger import get_logger

lg = get_logger("co_data_pipeline")

_ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Alpha Vantage 返回字段 -> 项目内统一字段名
_FIELD_MAP = {
    "date": "date",
    "symbol": "underlying",
    "expiration": "expiration_date",
    "strike": "strike",
    "type": "option_type",
    "bid": "bid",
    "ask": "ask",
    "last": "last_price",
    "volume": "volume",
    "open_interest": "open_interest",
    "implied_volatility": "implied_volatility",
    "delta": "delta",
    "gamma": "gamma",
    "theta": "theta",
    "vega": "vega",
    "rho": "rho",
}

_NUMERIC_COLS = [
    "strike", "bid", "ask", "last_price", "volume", "open_interest",
    "implied_volatility", "delta", "gamma", "theta", "vega", "rho",
]


def _clean_option_chain(df: pd.DataFrame) -> pd.DataFrame:
    """期权链数据专属清洗：日期/到期日格式统一、数值列类型转换。
    不做跨记录的插值/填充——期权链每条记录是独立的合约报价，
    不存在"前后相邻记录可以互相代表"这种关系。
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["expiration_date"] = pd.to_datetime(df["expiration_date"])

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["date", "expiration_date", "strike", "option_type"]).reset_index(drop=True)
    return df


class OptionChainSource(DataSourceBase):
    """JPM 期权链快照，数据源：Alpha Vantage HISTORICAL_OPTIONS。

    用途仅限于 Heston 模型的参数校准，不用于日频跟踪或对标 OTC 成交价，
    因此只在少数几个时间点（默认每年年初一次）拉取当天完整期权链，
    而不是逐日采集——这也大幅降低了免费层限流下的调用压力。
    """

    TICKER = "JPM"
    # 默认校准时间点：每年年初一次，覆盖 2018-2024（若非交易日，
    # Alpha Vantage 通常会返回最近一个交易日的数据，具体行为需实测确认）
    CALIBRATION_DATES = [
        "2018-01-02", "2019-01-02", "2020-01-02", "2021-01-04",
        "2022-01-03", "2023-01-03", "2024-01-02",
    ]

    @data_clear(_clean_option_chain)
    def fetch(self, **params) -> pd.DataFrame:
        ticker = params.get("ticker", self.TICKER)
        dates = params.get("dates", self.CALIBRATION_DATES)

        api_key = os.environ.get("ALPHA_VANTAGE_KEY")
        if not api_key:
            raise RuntimeError("环境变量 ALPHA_VANTAGE_KEY 未设置")

        all_records = []
        for snapshot_date in dates:
            resp = requests.get(
                _ALPHA_VANTAGE_URL,
                params={
                    "function": "HISTORICAL_OPTIONS",
                    "symbol": ticker,
                    "date": snapshot_date,
                    "apikey": api_key,
                },
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()

            data = payload.get("data")
            if data is None:
                # 常见于限流("Note"/"Information")或当天无数据
                lg.warning(f"{snapshot_date}: 未返回有效期权链数据：{payload}")
                continue

            for record in data:
                mapped = {
                    target: record.get(source)
                    for source, target in _FIELD_MAP.items()
                }
                # 部分响应可能不含独立的 date 字段，兜底用请求的 snapshot_date
                if not mapped.get("date"):
                    mapped["date"] = snapshot_date
                all_records.append(mapped)

        if not all_records:
            raise RuntimeError("所有校准日期均未获取到期权链数据")

        df = pd.DataFrame(all_records)
        expected_cols = list(self.get_fields().keys())
        return df[[c for c in expected_cols if c in df.columns]]

    def get_fields(self) -> dict:
        return {
            "date": "datetime64[ns]",
            "underlying": "object",
            "expiration_date": "datetime64[ns]",
            "strike": "float64",
            "option_type": "object",
            "bid": "float64",
            "ask": "float64",
            "last_price": "float64",
            "volume": "float64",
            "open_interest": "float64",
            "implied_volatility": "float64",
            "delta": "float64",
            "gamma": "float64",
            "theta": "float64",
            "vega": "float64",
            "rho": "float64",
        }

    def get_table_name(self) -> str:
        return "raw_option_chain"

    def is_calendar_anchor(self) -> bool:
        return False  # 交易日历基准由 raw_stock_prices 承担

    def save(self, df: pd.DataFrame) -> None:
        """按 date+expiration_date+strike+option_type 联合主键 upsert。"""
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        key_cols = ["date", "expiration_date", "strike", "option_type"]

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            combined = df

        combined = combined.sort_values(key_cols).reset_index(drop=True)
        combined.to_parquet(path, index=False)
        lg.info(f"{self.get_table_name()}: 已写入 {path}，共 {len(combined)} 行")

    def load(self) -> pd.DataFrame:
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        return pd.read_parquet(path)