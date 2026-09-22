"""数据管线引擎
职责仅限于调度，不涉及任何具体数据源的采集/字段/存储细节：
1. 扫描 scripts/ 目录，自动发现所有 DataSourceBase 的具体子类
2. 依据 configs/enabled_sources.json 决定哪些数据源本次要跑
   （未在配置中出现的数据源，默认视为未启用）
3. 对每个启用的数据源依次执行 fetch -> 字段校验（仅告警）-> save
4. 单个数据源失败不影响其他数据源，最后打印一份汇总报告
"""

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from co_data_pipeline import pipeline_base as base
from common import logger as logger_module

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")  # 找到env

lg = logger_module.get_logger("co_data_pipeline")


class DataEngine:
    def __init__(self):
        self._scripts_dir = Path(__file__).resolve().parent / "scripts"
        self._enabled_config_path = (
            Path(__file__).resolve().parent.parent / "configs" / "enabled_sources.json"
        )
        self._sources: List[base.DataSourceBase] = self.find_scripts()

    def __repr__(self) -> str:
        table_names = [s.get_table_name() for s in self._sources]
        return f"<DataEngine sources={table_names}>"

    def find_scripts(self) -> List[base.DataSourceBase]:
        """扫描 scripts/ 目录下所有 .py 文件，找出 DataSourceBase 的具体子类并实例化。"""
        instances: List[base.DataSourceBase] = []

        if not self._scripts_dir.exists():
            lg.warning(f"数据源脚本目录不存在：{self._scripts_dir}")
            return instances

        for py_file in sorted(self._scripts_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue  # 跳过 __init__.py 等文件

            module_name = f"co_data_pipeline.scripts.{py_file.stem}"
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
                    issubclass(obj, base.DataSourceBase)
                    and obj is not base.DataSourceBase
                    and obj.__module__ == module_name
                )
                if not is_concrete_subclass:
                    continue
                try:
                    instances.append(obj())
                except Exception:
                    lg.exception(f"实例化数据源类失败：{obj.__name__}")

        return instances

    def _load_enabled_config(self) -> dict:
        """读取启用开关配置。文件不存在或解析失败则视为空配置（不运行任何数据源）。"""
        if not self._enabled_config_path.exists():
            lg.warning(f"未找到启用配置文件：{self._enabled_config_path}")
            return {}
        try:
            with open(self._enabled_config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            lg.exception(f"解析启用配置文件失败：{self._enabled_config_path}")
            return {}

    def _validate_fields(self, source: base.DataSourceBase, df) -> None:
        """比对 fetch() 返回的列名与 get_fields() 声明，不一致仅告警，不阻断流程。"""
        table_name = source.get_table_name()
        try:
            declared = set(source.get_fields().keys())
        except Exception:
            lg.exception(f"{table_name}: 获取字段声明失败，跳过本次校验")
            return

        actual = set(df.columns)
        missing = declared - actual
        extra = actual - declared

        if missing:
            lg.warning(f"{table_name}: 缺少声明字段 {sorted(missing)}")
        if extra:
            lg.warning(f"{table_name}: 出现未声明字段 {sorted(extra)}")

    @staticmethod
    def _log_summary(results: Dict[str, str]) -> None:
        success = [k for k, v in results.items() if v == "success"]
        failed = [k for k, v in results.items() if v == "failed"]

        lg.info(f"运行汇总：成功 {len(success)} 个，失败 {len(failed)} 个")
        if success:
            lg.info(f"成功：{success}")
        if failed:
            lg.error(f"失败：{failed}")

    def run_all(self) -> Dict[str, str]:
        """运行所有已启用的数据源，返回 {table_name: 'success' | 'failed'}。"""
        enabled_config = self._load_enabled_config()

        if not self._sources:
            lg.warning("未发现任何数据源脚本")
            return {}

        results: Dict[str, str] = {}

        for source in self._sources:
            table_name = source.get_table_name()

            if not enabled_config.get(table_name, False):
                lg.info(f"{table_name}: 未启用，跳过")
                continue

            lg.info(f"{table_name}: 开始采集")
            try:
                df = source.fetch()
            except Exception:
                lg.exception(f"{table_name}: fetch() 失败")
                results[table_name] = "failed"
                continue

            self._validate_fields(source, df)

            try:
                source.save(df)
            except Exception:
                lg.exception(f"{table_name}: save() 失败")
                results[table_name] = "failed"
                continue

            lg.info(f"{table_name}: 完成，共 {len(df)} 行")
            results[table_name] = "success"

        self._log_summary(results)
        return results


if __name__ == "__main__":
    engine = DataEngine()
    print(engine)
    engine.run_all()