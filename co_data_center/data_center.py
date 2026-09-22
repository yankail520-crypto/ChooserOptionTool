"""DataCenter：统一管理所有数据对象的入口。

职责：
1. 依次扫描已登记的来源目录（co_data_pipeline/scripts/ 存放原始数据
   源，co_features/sources/ 存放 feature_daily 这类内部计算结果的
   DataSourceBase 包装），发现所有 DataSourceBase 子类
   （复用引擎同一套动态导入+反射机制），一视同仁地纳入统一管辖——
   任何模块想要数据，都应通过 DataCenter.get()，不直接读取文件
2. 扫描 co_data_center/objects/，发现所有特化的 DataObject 子类，
   建立 {表名: 特化子类} 的映射；没有特化子类的表，用通用 DataObject 包装
3. 懒加载：get(table_name) 第一次被调用时才真正执行
   DataSourceBase.load()，读取结果后缓存，之后重复 get() 直接命中缓存
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, Optional, Type

from co_data_pipeline.pipeline_base import DataSourceBase
from co_data_center.data_center_base import DataObject, DataObjectBase
from common.logger import get_logger

lg = get_logger("co_data_center")


class DataCenter:
    def __init__(self):
        # (脚本目录, 用于构造 module_name 的前缀)
        self._source_scan_targets = [
            (
                Path(__file__).resolve().parent.parent / "co_data_pipeline" / "scripts",
                "co_data_pipeline.scripts",
            ),
            (
                Path(__file__).resolve().parent.parent / "co_features" / "sources",
                "co_features.sources",
            ),
        ]
        self._objects_dir = Path(__file__).resolve().parent / "objects"

        # {table_name: DataSourceBase 实例}
        self._sources: Dict[str, DataSourceBase] = self._discover_sources()
        # {table_name: 特化 DataObject 子类}（没有特化的表不在这里出现）
        self._object_classes: Dict[str, Type[DataObjectBase]] = self._discover_objects()

        # 懒加载缓存：{table_name: 已加载的 DataObjectBase 实例}
        self._cache: Dict[str, DataObjectBase] = {}

    def __repr__(self) -> str:
        return f"<DataCenter tables={list(self._sources.keys())}>"

    def _discover_sources(self) -> Dict[str, DataSourceBase]:
        """依次扫描所有已登记的来源目录，发现 DataSourceBase 子类并实例化，
        建立 {表名: 实例} 映射。不同目录之间如出现同名表，视为配置错误，
        以最后扫描到的为准，并打印警告提醒排查。"""
        sources: Dict[str, DataSourceBase] = {}

        for scan_dir, module_prefix in self._source_scan_targets:
            if not scan_dir.exists():
                lg.warning(f"数据源脚本目录不存在：{scan_dir}")
                continue

            for py_file in sorted(scan_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue

                module_name = f"{module_prefix}.{py_file.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                except Exception:
                    lg.exception(f"加载脚本文件失败：{py_file.name}")
                    continue

                for _, obj in inspect.getmembers(module, inspect.isclass):
                    is_concrete_subclass = (
                        issubclass(obj, DataSourceBase)
                        and obj is not DataSourceBase
                        and obj.__module__ == module_name
                    )
                    if not is_concrete_subclass:
                        continue
                    try:
                        instance = obj()
                        table_name = instance.get_table_name()
                        if table_name in sources:
                            lg.warning(
                                f"表名 '{table_name}' 在多个来源目录中重复定义，"
                                f"以最后扫描到的为准（来自 {scan_dir}）"
                            )
                        sources[table_name] = instance
                    except Exception:
                        lg.exception(f"实例化数据源类失败：{obj.__name__}")

        return sources

    def _discover_objects(self) -> Dict[str, Type[DataObjectBase]]:
        """扫描 co_data_center/objects/，发现所有特化的 DataObject 子类，
        依据类属性 SOURCE_TABLE 建立 {表名: 类} 映射。没有该目录或
        没有特化子类的表，在这里不会出现，get() 时会退回通用 DataObject。
        """
        object_classes: Dict[str, Type[DataObjectBase]] = {}

        if not self._objects_dir.exists():
            return object_classes

        for py_file in sorted(self._objects_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            module_name = f"co_data_center.objects.{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception:
                lg.exception(f"加载数据对象文件失败：{py_file.name}")
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                is_concrete_subclass = (
                    issubclass(obj, DataObjectBase)
                    and obj is not DataObject
                    and obj is not DataObjectBase
                    and obj.__module__ == module_name
                )
                if not is_concrete_subclass:
                    continue

                table_name = getattr(obj, "SOURCE_TABLE", None)
                if table_name is None:
                    lg.warning(f"{obj.__name__} 未声明 SOURCE_TABLE，跳过")
                    continue
                object_classes[table_name] = obj

        return object_classes

    def get(self, table_name: str) -> DataObjectBase:
        """
        获取指定表的数据对象。首次调用时才真正 load() 并缓存。
        """
        if table_name in self._cache:
            return self._cache[table_name]

        if table_name not in self._sources:
            raise KeyError(
                f"未找到数据源 '{table_name}'，可用表：{list(self._sources.keys())}"
            )

        df = self._sources[table_name].load()
        object_cls = self._object_classes.get(table_name, DataObject)
        instance = object_cls(df)

        self._cache[table_name] = instance
        lg.info(f"{table_name}: 已加载（{object_cls.__name__}，{len(df)} 行）")
        return instance

    def list_tables(self) -> list:
        """列出所有可用的表名（不触发加载）。"""
        return list(self._sources.keys())