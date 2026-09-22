# logger.py

import logging
from datetime import datetime
from pathlib import Path


def _get_logs_root() -> Path:
    """项目顶层 logs/ 目录，与 common/ 平级的项目根目录下。"""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "logs"


def get_logger(module_name: str, level: int = logging.INFO) -> logging.Logger:
    """获取（或创建）指定模块的 logger。

    日志按天切分，存放在 logs/<module_name>/YYYY-MM-DD.log，
    同时输出到控制台。同一 module_name 重复调用不会重复添加 handler。

    参数：
        module_name: 模块标识，例如 "co_data_pipeline"、"co_bsm"，
                      用于区分日志文件夹和日志前缀。
        level: 日志级别，默认 INFO。
    """
    logger = logging.getLogger(module_name)

    if logger.handlers:
        # 已经配置过，直接复用，避免重复添加 handler 导致日志重复打印
        return logger

    logger.setLevel(level)
    logger.propagate = False  # 不向 root logger 传播，避免重复输出

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件 handler：按天切分，写到 logs/<module_name>/YYYY-MM-DD.log
    log_dir = _get_logs_root() / module_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger