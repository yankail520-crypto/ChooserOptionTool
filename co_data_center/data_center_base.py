from abc import ABCMeta, abstractmethod, ABC

import pandas as pd


class DataObjectAbstract(metaclass=ABCMeta):
    """
    数据对象的纯契约层。

    只声明"每个具体表的数据对象必须能做什么"，不含任何实现——
    因表而异的业务逻辑（如何聚合成日频）完全交给子类自己决定。
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        """只读方式暴露底层 DataFrame，外部不应直接修改。"""
        return self._df

    @abstractmethod
    def to_daily(self) -> pd.DataFrame:
        """把本对象持有的数据聚合/转换成日频（一天一行）。

        不同表的聚合含义完全不同（新闻要按天取均值，股价本身已是
        日频、恒等返回即可，期权链这类表可能根本不适用该操作），
        因此没有通用实现，必须由具体子类各自实现。
        """
        ...


class DataObjectBase(DataObjectAbstract, ABC):
    """
    提供通用、不因表而异的机制性操作。

    to_daily() 仍未实现，本类依然是抽象的——filter/merge 这类纯机制性
    操作和"如何聚合"这种业务判断是两回事，不该混在同一层。
    """

    _LOOKUP_OPS = {
        "eq": lambda s, v: s == v,
        "ne": lambda s, v: s != v,
        "gt": lambda s, v: s > v,
        "gte": lambda s, v: s >= v,
        "lt": lambda s, v: s < v,
        "lte": lambda s, v: s <= v,
        "in": lambda s, v: s.isin(v),
        "contains": lambda s, v: s.astype(str).str.contains(str(v), na=False),
    }

    def filter(self, **kwargs) -> "DataObjectBase":
        """
        条件筛选，例如：
            obj.filter(date__gte="2018-01-01", date__lte="2024-12-31")
            obj.filter(option_type="call")
        不修改原对象，返回同一具体类型的新对象（不可变、链式调用）。
        """
        mask = pd.Series(True, index=self._df.index)

        for key, value in kwargs.items():
            if "__" in key:
                col, op = key.rsplit("__", 1)
            else:
                col, op = key, "eq"

            if op not in self._LOOKUP_OPS:
                raise ValueError(f"不支持的查询操作符: '{op}'（来自参数 '{key}'）")
            if col not in self._df.columns:
                raise KeyError(f"字段 '{col}' 不存在，可用字段：{list(self._df.columns)}")

            mask &= self._LOOKUP_OPS[op](self._df[col], value)

        filtered_df = self._df[mask].reset_index(drop=True)
        return self.__class__(filtered_df)

    def merge(self, other: "DataObjectBase", on: str, how: str = "left") -> "DataObject":
        """
        和另一个数据对象合成，返回一个新的 DataObject。

        合并结果不天然属于任一操作数原本的具体类型（例如股价对象
        和VIX对象合并后，结果既不是股价对象也不是VIX对象），因此
        统一包装成通用的 DataObject，而不是尝试保留某一方的类型。
        """
        merged_df = pd.merge(self._df, other.df, on=on, how=how)
        return DataObject(merged_df)


class DataObject(DataObjectBase):
    """
    具体、可直接实例化的默认数据对象。

    to_daily() 默认恒等返回——适用于本身已经是日频（一天一行）的表，
    如股价、利率、VIX，这类表没有特殊聚合需求，直接 DataObject(df)
    实例化即可使用。

    需要自定义聚合逻辑的表（例如新闻数据要按天取情绪分均值），
    继续往下继承一个子类并覆写 to_daily() 即可，不需要改动这一层。
    """

    def to_daily(self) -> pd.DataFrame:
        return self._df