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
    """日期排序，并生成与 close 同口径的复权开高低价和复权成交量。

    接口返回的每个值都经过 float()/int() 解析，不会出现缺失，无需插值。
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 开高低价：乘以复权因子，与 close 的口径（拆股 + 分红）一致
    factor = df["close"] / df["close_raw"]
    for col in ["open", "high", "low"]:
        df[col] = df[f"{col}_raw"] * factor

    # 成交量：只按拆股调整（分红不改变股数），乘以该日之后所有拆股系数的乘积
    later_splits = df["split_coef"][::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    df["volume"] = (df["volume_raw"] * later_splits).round().astype("int64")

    return df.drop(columns="split_coef")


class StockPriceSource(DataSourceBase):
    """JPM 日频行情，数据源：Alpha Vantage TIME_SERIES_DAILY_ADJUSTED（需要 Premium 订阅）。

    复权口径
    --------
    Alpha Vantage 只提供复权收盘价（adjusted close），开高低价和成交量都是
    未调整的原始值。本表同时保存两套：

    - open / high / low / close：前复权价格（拆股 + 分红）。close 取自接口，
      开高低价乘以复权因子 close / close_raw 得到，四者口径一致。用于收益率、
      波动率（包括基于开高低价的区间波动率）等计算。
    - volume：按拆股调整后的成交量，拆股前后可比。
    - open_raw / high_raw / low_raw / close_raw / volume_raw：未调整的原始值。
      与未调整的分红金额做运算时必须用原始价格，例如股息率 = 分红金额 / close_raw。

    前复权以最新价格为基准向过去调整，每次新的分红或拆股都会使所有历史复权值
    整体变化，用于最终报告的结果应记录数据抓取日期。

    全量获取、全量覆盖写入：新旧数据的复权基准不同，不能合并。
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

        return pd.DataFrame(
            {
                "date": date_str,
                "open_raw": float(values["1. open"]),
                "high_raw": float(values["2. high"]),
                "low_raw": float(values["3. low"]),
                "close_raw": float(values["4. close"]),
                "close": float(values["5. adjusted close"]),  # 前复权收盘价
                "volume_raw": int(values["6. volume"]),
                "split_coef": float(values["8. split coefficient"]),  # 仅用于调整成交量，清洗后删除
            }
            for date_str, values in series.items()
        )

    def get_fields(self) -> dict:
        return {
            "date": "datetime64[ns]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
            "open_raw": "float64",
            "high_raw": "float64",
            "low_raw": "float64",
            "close_raw": "float64",
            "volume_raw": "int64",
        }

    def get_table_name(self) -> str:
        return "raw_stock_prices"

    def is_calendar_anchor(self) -> bool:
        return True

    def save(self, df: pd.DataFrame) -> None:
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df = df[list(self.get_fields())].sort_values("date").reset_index(drop=True)
        df.to_parquet(path, index=False)
        lg.info(f"{self.get_table_name()}: 已全量覆盖写入 {path}，共 {len(df)} 行")

    def load(self) -> pd.DataFrame:
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        return pd.read_parquet(path)