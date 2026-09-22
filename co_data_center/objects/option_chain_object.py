import pandas as pd

from co_data_center.data_center_base import DataObject


class OptionChainObject(DataObject):
    """
    对应 raw_option_chain 表的特化数据对象。

    期权链不是"一天一行"的结构——每个快照日期对应大量（约上千条）
    不同 strike/expiration/option_type 的记录，"聚合成日频"这个操作
    本身对这张表没有明确定义的语义，因此 to_daily() 主动拦截并报错，
    而不是沿用默认的恒等返回（那样会悄悄吐出未经聚合的原始快照，
    造成"看起来能用但语义是错的"这种更难排查的问题）。

    本表的用途是 Heston 参数校准，正确的取数方式是用 filter() 按
    快照日期（date）取出某一天的完整期权链，而不是走 to_daily()。
    """

    SOURCE_TABLE = "raw_option_chain"

    def to_daily(self) -> pd.DataFrame:
        raise NotImplementedError(
            "raw_option_chain 不支持按天聚合（每个快照日期对应大量不同"
            "strike/expiration/option_type的记录，聚合语义未定义）。"
            "请改用 filter(date=...) 取出某个快照日期的完整期权链。"
        )