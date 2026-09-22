import os
from datetime import date, datetime

import pandas as pd
import requests

from co_data_pipeline.pipeline_base import DataSourceBase
from common.decorators import data_clear
from common.paths import get_raw_path
from common.logger import get_logger

lg = get_logger("co_data_pipeline")

_ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def _month_chunks(start: str, end: str) -> list:
    """把 [start, end] 拆成按月的 (time_from, time_to) 区间列表，
    格式符合 Alpha Vantage 要求的 YYYYMMDDTHHMM。"""
    periods = pd.period_range(start=start, end=end, freq="M")
    chunks = []
    for p in periods:
        month_start = p.start_time.strftime("%Y%m%dT0000")
        month_end = p.end_time.strftime("%Y%m%dT2359")
        chunks.append((month_start, month_end))
    return chunks


def _clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """新闻情绪数据专属清洗：日期解析、情绪分从[-1,1]映射到文档要求的[0,1]区间。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Alpha Vantage 情绪区间映射
    if "sentiment_score" in df.columns:
        df["sentiment_score"] = (df["sentiment_score"] + 1) / 2

    return df


class NewsSentimentSource(DataSourceBase):
    """JPM 相关新闻及情绪分数，数据源：Alpha Vantage NEWS_SENTIMENT。

    一次调用同时返回新闻内容和已计算好的情绪分数，因此 raw_news 和
    raw_sentiment 不再拆成两张表，合并为本表，情绪分作为其中一列。
    仅保留必要的文章元信息（不存全文），情绪分基于文章中与JPM
    相关的 ticker_sentiment 片段，而非整篇文章的整体情绪。
    """

    TICKER = "JPM"
    START_DATE = "2018-01-01"
    END_DATE = date.today().isoformat()
    LIMIT_PER_CALL = 1000  # Alpha Vantage 单次调用最大返回条数

    @data_clear(_clean_news)
    def fetch(self, **params) -> pd.DataFrame:
        ticker = params.get("ticker", self.TICKER)
        start = params.get("start_date", self.START_DATE)
        end = params.get("end_date", self.END_DATE)
        limit = params.get("limit", self.LIMIT_PER_CALL)

        api_key = os.environ.get("ALPHA_VANTAGE_KEY")
        if not api_key:
            raise RuntimeError("环境变量 ALPHA_VANTAGE_KEY 未设置")

        all_records = []
        for time_from, time_to in _month_chunks(start, end):
            resp = requests.get(
                _ALPHA_VANTAGE_URL,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "time_from": time_from,
                    "time_to": time_to,
                    "sort": "EARLIEST",
                    "limit": limit,
                    "apikey": api_key,
                },
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()

            feed = payload.get("feed")
            if feed is None:
                lg.warning(f"{time_from}~{time_to}: 未返回有效新闻数据：{payload}")
                continue

            for article in feed:
                # 从 ticker_sentiment 列表里找到本标的(JPM)对应的那一条
                ticker_score = None
                relevance = None
                for ts in article.get("ticker_sentiment", []):
                    if ts.get("ticker") == ticker:
                        ticker_score = ts.get("ticker_sentiment_score")
                        relevance = ts.get("relevance_score")
                        break

                all_records.append(
                    {
                        "date": article.get("time_published"),
                        "title": article.get("title"),
                        "source": article.get("source"),
                        "url": article.get("url"),
                        "relevance_score": relevance,
                        "sentiment_score": ticker_score,
                    }
                )

        if not all_records:
            raise RuntimeError("所有时间区间均未获取到新闻数据")

        df = pd.DataFrame(all_records)
        # time_published 格式类似 "20240115T093000"，需要专门解析
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")

        for col in ["relevance_score", "sentiment_score"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        expected_cols = list(self.get_fields().keys())
        return df[[c for c in expected_cols if c in df.columns]]

    def get_fields(self) -> dict:
        return {
            "date": "datetime64[ns]",
            "title": "object",
            "source": "object",
            "url": "object",
            "relevance_score": "float64",
            "sentiment_score": "float64",
        }

    def get_table_name(self) -> str:
        return "raw_news"

    def is_calendar_anchor(self) -> bool:
        return False  # 交易日历基准由 raw_stock_prices 承担

    def save(self, df: pd.DataFrame) -> None:
        """按 date+url 联合主键 upsert（同一天可能有多篇不同新闻）。"""
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        key_cols = ["date", "url"]

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            combined = df

        combined = combined.sort_values("date").reset_index(drop=True)
        combined.to_parquet(path, index=False)
        lg.info(f"{self.get_table_name()}: 已写入 {path}，共 {len(combined)} 行")

    def load(self) -> pd.DataFrame:
        path = get_raw_path() / f"{self.get_table_name()}.parquet"
        return pd.read_parquet(path)