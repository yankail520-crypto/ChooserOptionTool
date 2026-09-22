"""FeatureEngine：特征工程的总入口（对应 DataEngine 之于原始数据源）。

职责：
1. 通过 DataCenter 拿到基础宽表（以 raw_stock_prices 为基准，
   left join 其余原始表，稀疏表已通过各自的 to_daily() 聚合）
2. 扫描 co_features/features/ 目录，发现所有 FeatureBase 子类
3. 识别"叶子"类——继承链最末端、在已发现集合里没有被其他类继承过的类，
   每一个叶子代表一整条完整的特征依赖链（实例化叶子时，继承链会
   通过 super().compute() 自动补全所有祖先的计算，不需要单独实例化祖先类）
4. 对每个叶子类用同一份基础宽表实例化，取得该叶子的完整结果；
   多个叶子的结果按列名去重合并（同名列只保留第一次出现的，
   避免共享祖先导致的重复合并）
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import List, Type

import pandas as pd

from co_data_center.data_center_base import DataObject
from co_features.feature_base import FeatureBase
from common.logger import get_logger

lg = get_logger("co_features")

_FEATURES_DIR = Path(__file__).resolve().parent / "features"


def _discover_feature_classes() -> List[Type[FeatureBase]]:
    """扫描 features/ 目录，发现所有具体的 FeatureBase 子类（只发现类，不实例化）。"""
    classes: List[Type[FeatureBase]] = []

    if not _FEATURES_DIR.exists():
        lg.warning(f"特征目录不存在：{_FEATURES_DIR}")
        return classes

    for py_file in sorted(_FEATURES_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"co_features.features.{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception:
            lg.exception(f"加载特征文件失败：{py_file.name}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            is_concrete_subclass = (
                issubclass(obj, FeatureBase)
                and obj is not FeatureBase
                and obj.__module__ == module_name
            )
            if is_concrete_subclass:
                classes.append(obj)

    return classes


def _filter_leaf_classes(classes: List[Type[FeatureBase]]) -> List[Type[FeatureBase]]:
    """
    筛出"叶子"类——在已发现的集合里，没有被其他任何一个类继承过。

    如果 B 继承 A，说明 B 的继承链已经完整覆盖了 A 的计算逻辑，
    A 就不再需要单独实例化（否则会和 B 重复计算、产生重复列）。
    """
    leaves = []
    for cls in classes:
        has_discovered_descendant = any(
            other is not cls and issubclass(other, cls) for other in classes
        )
        if not has_discovered_descendant:
            leaves.append(cls)
    return leaves


class FeatureEngine:
    def build(self) -> pd.DataFrame:
        from co_data_center.data_center import DataCenter  # 延迟导入，避免循环依赖

        dc = DataCenter()

        base = DataObject(dc.get("raw_stock_prices").to_daily())
        for table_name in dc.list_tables():
            if table_name in ("raw_stock_prices", "raw_option_chain", "feature_daily"):
                continue
            try:
                other_daily = DataObject(dc.get(table_name).to_daily())
            except NotImplementedError:
                lg.info(f"{table_name}: 不支持 to_daily()，跳过并入宽表")
                continue
            base = base.merge(other_daily, on="date", how="left")

        base_df = base.df

        all_classes = _discover_feature_classes()
        leaf_classes = _filter_leaf_classes(all_classes)
        lg.info(
            f"发现 {len(all_classes)} 个特征类，其中 {len(leaf_classes)} 个叶子类："
            f"{[c.__name__ for c in leaf_classes]}"
        )

        result = base_df.copy()
        known_columns = set(result.columns)

        for cls in leaf_classes:
            leaf_df = cls(base_df).df  # __init__ 自动触发整条继承链的计算
            new_columns = [c for c in leaf_df.columns if c not in known_columns]
            if not new_columns:
                lg.warning(f"{cls.__name__}: 未产出任何新列，跳过合并")
                continue
            result = result.merge(leaf_df[["date"] + new_columns], on="date", how="left")
            known_columns.update(new_columns)
            lg.info(f"{cls.__name__}: 已合并新增列 {new_columns}")

        return result

    def run(self) -> None:
        """跑一遍完整流程并落盘：build() 计算 -> 通过 FeatureDailySource.save()
        写入 data/features/feature_daily.parquet。跑完之后，DataCenter.get(
        "feature_daily") 才能读到真正的结果（DataCenter 只负责读，不负责算）。
        """
        from co_features.sources.feature_daily_source import FeatureDailySource

        df = self.build()
        FeatureDailySource().save(df)
        lg.info(f"FeatureEngine 运行完成，共 {len(df)} 行 {len(df.columns)} 列")


if __name__ == "__main__":
    FeatureEngine().run()