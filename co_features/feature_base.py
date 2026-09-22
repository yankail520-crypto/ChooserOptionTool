import functools
from abc import ABCMeta, abstractmethod

import pandas as pd


def apply_computation(init_func):
    """装饰器：__init__ 存好原始 df 后，自动调用 compute() 覆盖为处理后的结果。

    依赖 self.compute 的动态派发——子类只需要覆写 compute()，不需要
    重新实现或装饰 __init__，继承链上的 super().compute() 调用自然
    表达了特征之间的先后依赖，不再需要靠文件名前缀人为控制顺序。
    """

    @functools.wraps(init_func)
    def wrapper(self, df: pd.DataFrame, *args, **kwargs):
        init_func(self, df, *args, **kwargs)
        self._df = self.compute(self._df)

    return wrapper


class FeatureBase(metaclass=ABCMeta):
    """特征的纯契约层，兼具"依赖关系表达"的机制。

    实例化时自动执行计算（__init__ 被 apply_computation 装饰），
    结果存入 self._df，通过 .df 属性取出。

    特征之间的先后依赖用类继承表达：B 依赖 A 的计算结果，就让 B 继承
    A，并在 B.compute() 里先调用 super().compute(df)，天然保证 A 的
    逻辑先跑完，再在其结果基础上追加 B 自己的列。
    """

    @apply_computation
    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """在 df 基础上计算并添加本特征对应的列，返回新的 DataFrame。

        必须无条件先调用 super().compute(df)，再在其结果基础上追加
        自己的列——单继承场景下这只是安全的恒等透传（FeatureBase 自身
        的 compute() 原样返回 df），但多继承场景下，Python 的 MRO 会
        依次沿着每一条继承路径调用 super()，只有每一环都遵守这个约定，
        多条特征链才能被正确、完整地串联起来。
        """
        return df

    @abstractmethod
    def get_feature_name(self) -> str:
        """本特征对应的列名，用于日志/校验，需要和 compute() 实际新增的列名一致。"""
        ...