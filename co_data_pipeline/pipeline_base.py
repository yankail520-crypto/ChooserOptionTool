from abc import abstractmethod, ABCMeta
import pandas as pd


class DataSourceBase(metaclass=ABCMeta):
    """
    数据基类
    """
    @abstractmethod
    def fetch(self, **params) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_fields(self) -> dict:
        """
        字段及其类型
        :return:
        """
        ...

    @abstractmethod
    def get_table_name(self) -> str:
        """
        对应表名
        :return:
        """
        ...

    @abstractmethod
    def save(self, df: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def is_calendar_anchor(self) -> bool:
        """
        基准数据源 时间对齐表
        :return: 是否为基准数据源
        """
        ...