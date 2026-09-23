"""FeatureEngine：特征工程的总入口（对应 DataEngine 之于原始数据源）。

职责：
1. 通过 DataCenter 拿到基础宽表（以 raw_stock_prices 为基准，
   left join 其余原始表，稀疏表已通过各自的 to_daily() 聚合）
2. 扫描 co_features/features/ 目录，发现所有具体的 FeatureBase 子类
3. 识别"叶子"类——继承链最末端、在已发现集合里没有被其他类继承过的类，
   每一个叶子代表一整条完整的特征依赖链（实例化叶子时，继承链会
   通过 super().compute() 自动补全所有祖先的计算，不需要单独实例化祖先类）
4. 对每个叶子类用同一份基础宽表实例化，取得该叶子的完整结果；
   多个叶子的结果按列名去重合并。同名列只有在"值完全一致"时才视为
   共享祖先产出的重复列并跳过；值不一致说明两个特征撞了列名，直接报错。

边界约定：
- FeatureEngine 只负责"无状态、只看过去"的特征（shift/diff/rolling 等）。
- 任何需要拟合参数的处理（缩尾阈值、标准化均值方差、IQR 界限等）
  不放在这里，必须在 70/15/15 时间序列切分之后只用训练集拟合，
  否则会引入前视偏差。

数据源约定：
- 数据源类可声明类属性 include_in_wide_table = False 来表示不并入宽表；
  未声明的默认并入。_DEFAULT_EXCLUDED_TABLES 作为兼容旧数据源的兜底。
"""

import importlib
import inspect
from pathlib import Path
from typing import List, Tuple, Type

import numpy as np
import pandas as pd

from co_data_center.data_center_base import DataObject
from co_features.feature_base import FeatureBase
from common.logger import get_logger

lg = get_logger("co_features")

_FEATURES_DIR = Path(__file__).resolve().parent / "features"
_FEATURES_PACKAGE = "co_features.features"

_BASE_TABLE = "raw_stock_prices"
_DEFAULT_EXCLUDED_TABLES = {"raw_stock_prices", "raw_option_chain", "feature_daily"}

# 数值列一致性比较的容差（共享祖先重复计算的结果应当逐位相同，这里只留浮点噪声余量）
_CONSISTENCY_ATOL = 1e-12


# ---------------------------------------------------------------------------
# 校验工具
# ---------------------------------------------------------------------------

def _assert_date_column(df: pd.DataFrame, name: str) -> None:
    """断言 date 列存在、无缺失、无重复。"""
    if "date" not in df.columns:
        raise KeyError(f"{name}: 缺少 date 列")
    if df["date"].isna().any():
        raise ValueError(f"{name}: date 列存在缺失值")
    dup = df["date"][df["date"].duplicated()]
    if not dup.empty:
        raise ValueError(
            f"{name}: date 列存在 {len(dup)} 个重复日期，"
            f"示例：{dup.head(5).tolist()}（检查 to_daily() 的聚合逻辑）"
        )


def _sort_and_check(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """按日期升序排序并确认单调递增——所有 shift/diff/rolling 都依赖这个前提。"""
    _assert_date_column(df, name)
    df = df.sort_values("date").reset_index(drop=True)
    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"{name}: 排序后 date 仍非单调递增")
    return df


def _merge_without_overlap(left: pd.DataFrame, right: pd.DataFrame,
                           right_name: str) -> pd.DataFrame:
    """
    按 date 做 left join，合并前检查列名冲突。

    pandas 在列名冲突时会静默加 _x/_y 后缀，导致下游按原列名取数时
    报 KeyError 或者读到错误的列，所以这里直接报错，要求在数据源层面改名。
    """
    _assert_date_column(right, right_name)

    overlap = (set(left.columns) & set(right.columns)) - {"date"}
    if overlap:
        raise ValueError(
            f"{right_name}: 与宽表存在同名列 {sorted(overlap)}，"
            f"请在该数据源的 to_daily() 中给列加上明确前缀后再并入"
        )

    return left.merge(right, on="date", how="left", validate="one_to_one")


def _check_duplicate_columns_consistent(result: pd.DataFrame, leaf_df: pd.DataFrame,
                                        cols: List[str], leaf_name: str) -> None:
    """
    叶子结果中与已有列同名的列，必须与已有值一致。

    一致：来自共享祖先（例如多个叶子都继承 DailyReturnFeature），可以安全跳过。
    不一致：两个不相关的特征撞了列名，或者某个特征篡改了基础列，直接报错。
    """
    if not cols:
        return

    left = result.set_index("date")[cols]
    right = leaf_df.set_index("date")[cols].reindex(left.index)

    for col in cols:
        a, b = left[col], right[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.allclose(
                a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                rtol=0.0, atol=_CONSISTENCY_ATOL, equal_nan=True,
            )
        else:
            same = a.equals(b)

        if not same:
            raise ValueError(
                f"{leaf_name}: 列 '{col}' 与已存在的同名列取值不一致——"
                f"可能是两个特征撞了列名，或者该特征修改了基础宽表中的原有列"
            )


# ---------------------------------------------------------------------------
# 特征类发现
# ---------------------------------------------------------------------------

def _discover_feature_classes() -> Tuple[List[Type[FeatureBase]], List[str]]:
    """
    扫描 features/ 目录，发现所有具体的 FeatureBase 子类（只发现类，不实例化）。

    使用 importlib.import_module 而不是 spec_from_file_location + exec_module：
    后者每次都会新建模块对象并覆盖 sys.modules，如果某个特征文件已经被
    其他文件正常 import 过，就会出现"同名不同类"的两个类对象，导致
    issubclass 判断失效、父类被误判为叶子。import_module 会复用缓存。

    返回 (发现的类列表, 加载失败的文件名列表)。
    """
    classes: List[Type[FeatureBase]] = []
    failures: List[str] = []

    if not _FEATURES_DIR.exists():
        lg.warning(f"特征目录不存在：{_FEATURES_DIR}")
        return classes, failures

    if not (_FEATURES_DIR / "__init__.py").exists():
        raise FileNotFoundError(
            f"{_FEATURES_DIR} 下缺少 __init__.py，无法作为包导入特征模块"
        )

    for py_file in sorted(_FEATURES_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"{_FEATURES_PACKAGE}.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            lg.exception(f"加载特征文件失败：{py_file.name}")
            failures.append(py_file.name)
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            is_concrete_subclass = (
                issubclass(obj, FeatureBase)
                and obj is not FeatureBase
                and obj.__module__ == module_name   # 排除 import 进来的类
                and not inspect.isabstract(obj)      # 排除抽象中间类
            )
            if is_concrete_subclass:
                classes.append(obj)

    return classes, failures


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


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------

class FeatureEngine:
    def __init__(self, strict: bool = True):
        """
        strict=True：任何特征文件加载失败或特征类计算失败都会让整个任务失败。
        适合 GitHub Actions 定时任务——宁可任务报红，也不要静默产出缺列的数据。
        strict=False：记录失败并跳过，适合本地调试。
        """
        self.strict = strict

    # ---- 基础宽表 ----------------------------------------------------------

    @staticmethod
    def _should_include(table_name: str, source) -> bool:
        flag = getattr(source, "include_in_wide_table", None)
        if flag is not None:
            return bool(flag)
        return table_name not in _DEFAULT_EXCLUDED_TABLES

    def _build_base(self) -> pd.DataFrame:
        from co_data_center.data_center import DataCenter  # 延迟导入，避免循环依赖

        dc = DataCenter()

        base_df = DataObject(dc.get(_BASE_TABLE).to_daily()).df
        base_df = _sort_and_check(base_df, _BASE_TABLE)

        for table_name in dc.list_tables():
            if table_name == _BASE_TABLE:
                continue

            source = dc.get(table_name)
            if not self._should_include(table_name, source):
                lg.info(f"{table_name}: 声明为不并入宽表，跳过")
                continue

            try:
                other_df = DataObject(source.to_daily()).df
            except NotImplementedError:
                lg.info(f"{table_name}: 不支持 to_daily()，跳过并入宽表")
                continue

            rows_before = len(base_df)
            base_df = _merge_without_overlap(base_df, other_df, table_name)
            assert len(base_df) == rows_before, f"{table_name}: 合并后行数发生变化"

            new_cols = [c for c in other_df.columns if c != "date"]
            lg.info(f"{table_name}: 已并入宽表，新增列 {new_cols}")

        return base_df

    # ---- 特征计算 ----------------------------------------------------------

    def build(self) -> pd.DataFrame:
        base_df = self._build_base()

        all_classes, load_failures = _discover_feature_classes()
        leaf_classes = _filter_leaf_classes(all_classes)
        lg.info(
            f"发现 {len(all_classes)} 个特征类，其中 {len(leaf_classes)} 个叶子类："
            f"{[c.__name__ for c in leaf_classes]}"
        )

        result = base_df.copy()
        known_columns = set(result.columns)
        compute_failures: List[str] = []

        for cls in leaf_classes:
            name = cls.__name__
            try:
                # 传入副本：防止某个特征就地修改 base_df，污染后续叶子的输入
                leaf_df = cls(base_df.copy()).df
            except Exception:
                lg.exception(f"{name}: 特征计算失败")
                compute_failures.append(name)
                continue

            _assert_date_column(leaf_df, name)
            if len(leaf_df) != len(base_df):
                raise ValueError(
                    f"{name}: 输出行数 {len(leaf_df)} 与基础宽表 {len(base_df)} 不一致，"
                    f"特征计算不应增删行"
                )

            dup_columns = [c for c in leaf_df.columns if c in known_columns and c != "date"]
            _check_duplicate_columns_consistent(result, leaf_df, dup_columns, name)

            new_columns = [c for c in leaf_df.columns if c not in known_columns]
            if not new_columns:
                lg.warning(f"{name}: 未产出任何新列，跳过合并")
                continue

            result = result.merge(
                leaf_df[["date"] + new_columns], on="date", how="left",
                validate="one_to_one",
            )
            known_columns.update(new_columns)
            lg.info(f"{name}: 已合并新增列 {new_columns}")

        self._report_failures(load_failures, compute_failures)
        return _sort_and_check(result, "feature_daily")

    def _report_failures(self, load_failures: List[str],
                         compute_failures: List[str]) -> None:
        if not load_failures and not compute_failures:
            return

        msg = (
            f"特征构建存在失败项——加载失败文件：{load_failures or '无'}；"
            f"计算失败类：{compute_failures or '无'}"
        )
        if self.strict:
            raise RuntimeError(msg)
        lg.error(msg + "（strict=False，已跳过）")

    # ---- 落盘 --------------------------------------------------------------

    def run(self) -> None:
        """跑一遍完整流程并落盘：build() 计算 -> 通过 FeatureDailySource.save()
        写入 data/features/feature_daily.parquet。跑完之后，DataCenter.get(
        "feature_daily")。
        """
        from co_features.sources.feature_daily_source import FeatureDailySource

        df = self.build()
        FeatureDailySource().save(df)
        lg.info(f"FeatureEngine 运行完成，共 {len(df)} 行 {len(df.columns)} 列")


if __name__ == "__main__":
    FeatureEngine().run()