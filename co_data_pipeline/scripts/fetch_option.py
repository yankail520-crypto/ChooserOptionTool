import os
import time

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

_KEY_COLS = ["date", "expiration_date", "strike", "option_type"]


def _clean_option_chain(df: pd.DataFrame) -> pd.DataFrame:
    """日期/到期日格式统一、数值列类型转换。不做跨记录的插值或填充。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["expiration_date"] = pd.to_datetime(df["expiration_date"])

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(_KEY_COLS).reset_index(drop=True)


class OptionChainSource(DataSourceBase):
    """JPM 期权链快照，数据源：Alpha Vantage HISTORICAL_OPTIONS。

    用途是 Heston 参数校准，不逐日采集，而是每个月取当月第一个交易日的完整期权链。

    - 增量抓取：历史快照不会改变，已经有数据的月份直接跳过，所以全部抓完之后，
      日常运行不再产生任何 API 调用。需要重抓某些日期时，通过 dates 参数指定。
    - 交易日历：优先使用 raw_stock_prices 中的交易日；文件不存在时退化为工作日，
      接口当天没有数据（节假日）就顺延到后面最多 MAX_FALLBACK 个交易日。
    - 限流：接口没有返回数据时等待后重试；重试仍失败的月份记入日志并跳过，
      下次运行时会自动补抓。
    """

    TICKER = "JPM"
    START_DATE = "2018-01-01"
    END_DATE = "2024-12-31"        # 项目研究区间

    REQUEST_INTERVAL = 1.0         # 每次请求后的间隔（秒）
    MAX_RETRIES = 3
    RETRY_WAIT = 60                # 限流后等待（秒）
    MAX_FALLBACK = 3               # 当天无数据时最多顺延的交易日数

    # ------------------------------------------------------------------
    # 快照日期
    # ------------------------------------------------------------------

    def _path(self):
        return get_raw_path() / f"{self.get_table_name()}.parquet"

    def _trading_days(self) -> pd.DatetimeIndex:
        path = get_raw_path() / "raw_stock_prices.parquet"
        if path.exists():
            days = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
        else:
            lg.warning("raw_stock_prices 不存在，用工作日代替交易日历")
            days = pd.bdate_range(self.START_DATE, self.END_DATE)
        days = pd.DatetimeIndex(days).sort_values().unique()
        return days[(days >= self.START_DATE) & (days <= self.END_DATE)]

    def _existing_months(self) -> set:
        path = self._path()
        if not path.exists():
            return set()
        dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
        return set(dates.dt.to_period("M"))

    def _plan(self) -> dict:
        """返回 {月份: [候选交易日, ...]}，每个缺失的月份取当月前 1 + MAX_FALLBACK 个交易日。"""
        days = self._trading_days()
        have = self._existing_months()
        plan = {}
        for month, group in pd.Series(days, index=days).groupby(days.to_period("M")):
            if month not in have:
                plan[month] = [d.date().isoformat() for d in group.iloc[: 1 + self.MAX_FALLBACK]]
        return plan

    # ------------------------------------------------------------------
    # 请求
    # ------------------------------------------------------------------

    def _request(self, ticker: str, snapshot_date: str, api_key: str):
        """返回合约记录列表；当天确实没有数据返回空列表；多次重试仍失败返回 None。"""
        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": ticker,
            "date": snapshot_date,
            "apikey": api_key,
        }
        for attempt in range(1, self.MAX_RETRIES + 1):
            resp = requests.get(_ALPHA_VANTAGE_URL, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            time.sleep(self.REQUEST_INTERVAL)

            if "data" in payload:
                return payload["data"] or []

            if "Error Message" in payload:       # 参数错误，重试没有意义
                lg.warning(f"{snapshot_date}: 请求被拒绝：{payload['Error Message']}")
                return None

            message = payload.get("Note") or payload.get("Information") or payload
            lg.warning(f"{snapshot_date}: 第 {attempt} 次请求未返回数据：{message}")
            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_WAIT)
        return None

    def _fetch_month(self, ticker: str, candidates: list, api_key: str):
        """依次尝试当月的候选交易日，取到第一个有数据的。"""
        for snapshot_date in candidates:
            records = self._request(ticker, snapshot_date, api_key)
            if records is None:
                return None                      # 限流或请求失败，不再往后试
            if records:
                return snapshot_date, records
            lg.info(f"{snapshot_date}: 当天没有期权链数据，顺延到下一个交易日")
        return None

    # ------------------------------------------------------------------
    # 抓取与写入
    # ------------------------------------------------------------------

    @data_clear(_clean_option_chain)
    def fetch(self, **params) -> pd.DataFrame:
        ticker = params.get("ticker", self.TICKER)

        api_key = os.environ.get("ALPHA_VANTAGE_KEY")
        if not api_key:
            raise RuntimeError("环境变量 ALPHA_VANTAGE_KEY 未设置")

        if params.get("dates"):
            plan = {d: [d] for d in params["dates"]}   # 手动指定，不做增量过滤
        else:
            plan = self._plan()

        columns = [c for c in self.get_fields()]
        if not plan:
            lg.info("raw_option_chain: 所有月份都已有快照，无需抓取")
            return pd.DataFrame(columns=columns)

        lg.info(f"raw_option_chain: 需要抓取 {len(plan)} 个月的快照")

        all_records, failed = [], []
        for key, candidates in plan.items():
            result = self._fetch_month(ticker, candidates, api_key)
            if result is None:
                failed.append(str(key))
                continue

            snapshot_date, records = result
            for record in records:
                mapped = {target: record.get(source) for source, target in _FIELD_MAP.items()}
                if not mapped.get("date"):
                    mapped["date"] = snapshot_date
                all_records.append(mapped)

        if failed:
            lg.warning(f"raw_option_chain: {len(failed)} 个月份未获取到数据，下次运行会补抓：{failed}")

        if not all_records:
            if failed:
                raise RuntimeError("所有月份均未获取到期权链数据")
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(all_records)
        return df[[c for c in columns if c in df.columns]]

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
        """按 date+expiration_date+strike+option_type 联合主键增量追加。"""
        if df.empty:
            lg.info(f"{self.get_table_name()}: 没有新数据，跳过写入")
            return

        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            combined = pd.concat([pd.read_parquet(path), df], ignore_index=True)
            combined = combined.drop_duplicates(subset=_KEY_COLS, keep="last")
        else:
            combined = df

        combined = combined.sort_values(_KEY_COLS).reset_index(drop=True)
        combined.to_parquet(path, index=False)
        lg.info(f"{self.get_table_name()}: 已写入 {path}，共 {len(combined)} 行")

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self._path())