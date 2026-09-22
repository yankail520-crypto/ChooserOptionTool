import pandas as pd
from co_data_center.data_center_base import DataObject


class NewsObject(DataObject):
    """
    对应 raw_news 表的特化数据对象。

    raw_news 是逐条新闻（同一天可能有多篇），to_daily() 按天聚合成
    一行：情绪分取当天所有文章按 relevance_score 加权平均（相关性越高
    的文章，对当天情绪分的影响权重越大），同时附带当天的新闻篇数，
    用于下游区分"真正中性"和"当天没有新闻"这两种不同情况。
    """

    SOURCE_TABLE = "raw_news"

    def to_daily(self) -> pd.DataFrame:
        df = self._df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()  # 只保留日期部分，去掉时分秒

        def _weighted_avg(group: pd.DataFrame) -> pd.Series:
            weights = group["relevance_score"]
            if weights.sum() == 0:
                score = group["sentiment_score"].mean()
            else:
                score = (group["sentiment_score"] * weights).sum() / weights.sum()
            return pd.Series(
                {
                    "sentiment_score": score,
                    "news_count": len(group),
                }
            )

        daily = df.groupby("date", as_index=False).apply(_weighted_avg, include_groups=False)
        return daily.sort_values("date").reset_index(drop=True)