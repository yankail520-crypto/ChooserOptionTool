"""数据质量检查：自动遍历 DataCenter 中的所有表和数值列，做 IQR 异常值检测。

只打警告、不修改数据。建议在 GitHub Actions 中按
    抓取 -> 质量检查 -> FeatureEngine
的顺序执行，方便在特征计算之前发现问题。

自动化逻辑：
1. 表：遍历 dc.list_tables()。支持 to_daily() 的表先聚合为日频
   （例如新闻按天聚合）；聚合后日期仍有重复的表（如期权链，一天多行）
   不是单一时间序列，IQR 没有意义，自动跳过。
2. 列：只检查数值列（排除 date 和布尔列）。
3. 变换：对原始水平直接做 IQR 没有意义（有趋势的序列会被整段标记），
   所以按以下规则自动选择：
     - 样本太少、取值太离散或大部分是同一个值（如稀疏的股息列） -> 跳过
     - 始终为正且没有贴近 0 的值（股价、VIX、成交量）         -> 对数收益率
     - 其余（利率这类接近 0 或可正可负的序列）                  -> 差分
   特征表里的列本身多是收益率、波动率等已处理过的量，默认直接检测水平值。
   自动规则判断不准的列，在 OVERRIDES 里手动指定。
4. 特征表只检查 FeatureEngine 新算出的列；从原始表透传过来的列（价格、
   利率、VIX 等）已经在原始表里按合适的变换检查过，不再用水平值重复检查。
5. 滚动窗口按日历时间（默认 365 天）。稀疏表（日期间隔中位数超过
   SPARSE_TABLE_GAP_DAYS，如季度分红）窗口内观测太少，改用全样本 IQR。
6. 每一列实际使用的变换和窗口都会写进汇总报告，方便核对自动判断是否合理。
"""

import argparse
import sys
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from common.logger import get_logger
from common.outliers import warn_outliers_iqr  # 按 outliers.py 实际所在位置调整

lg = get_logger("data_quality")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TRANSFORMS: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "level": lambda s: s,
    "diff": lambda s: s.diff(),
    "log_return": lambda s: np.log(s / s.shift(1)),
}

# 整张表的默认变换（优先于自动判断）
TABLE_DEFAULTS: Dict[str, str] = {
    "feature_daily": "level",
}

# 单列的手动指定（优先级最高）。值为 None 表示跳过该列。
OVERRIDES: Dict[Tuple[str, str], Optional[str]] = {
    # 情绪分数是有界的得分，不是价格，对它做对数收益率没有经济含义
    ("raw_news", "sentiment_score"): "level",
    # daily_return 就是 close 的对数收益率，原始表里已经检查过
    ("feature_daily", "daily_return"): None,
    # 未复权原始值：拆股日的跳变是真实的，不是错误；只检查复权后的列
    ("raw_stock_prices", "open_raw"): None,
    ("raw_stock_prices", "high_raw"): None,
    ("raw_stock_prices", "low_raw"): None,
    ("raw_stock_prices", "close_raw"): None,
    ("raw_stock_prices", "volume_raw"): None,
}

EXCLUDED_TABLES = set()          # 需要整表跳过的，在这里声明
MIN_OBS = 60                     # 有效样本少于这个数不检测
SPARSE_MODE_SHARE = 0.5          # 超过一半是同一个值，视为稀疏列
MIN_UNIQUE = 10                  # 不同取值少于这个数，视为离散列
NEAR_ZERO_RATIO = 0.05           # 最小值 / 中位数低于这个比例，视为"贴近 0"
FEATURE_TABLE = "feature_daily"
SPARSE_TABLE_GAP_DAYS = 5        # 日期间隔中位数超过这个天数，视为稀疏表，用全样本 IQR


# ---------------------------------------------------------------------------
# 自动判断
# ---------------------------------------------------------------------------

def infer_transform(series: pd.Series) -> Tuple[Optional[str], str]:
    """返回 (变换名, 判断理由)。变换名为 None 表示跳过。"""
    s = series.dropna()
    if len(s) < MIN_OBS:
        return None, f"有效样本仅 {len(s)} 个"
    if s.nunique() < MIN_UNIQUE:
        return None, f"仅 {s.nunique()} 个不同取值，属于离散列"
    if s.value_counts(normalize=True).iloc[0] > SPARSE_MODE_SHARE:
        return None, "超过一半为同一取值，属于稀疏列"

    median = s.median()
    if (s > 0).all() and median > 0 and s.min() / median > NEAR_ZERO_RATIO:
        return "log_return", "始终为正且不贴近 0"
    return "diff", "存在 0/负值或贴近 0"


def resolve_transform(table: str, column: str, series: pd.Series) -> Tuple[Optional[str], str]:
    if (table, column) in OVERRIDES:
        return OVERRIDES[(table, column)], "手动指定"
    inferred, reason = infer_transform(series)
    if inferred is None:            # 样本不足、离散、稀疏的列，表默认也不检测
        return None, reason
    if table in TABLE_DEFAULTS:
        return TABLE_DEFAULTS[table], "表默认"
    return inferred, reason


# ---------------------------------------------------------------------------
# 检查流程
# ---------------------------------------------------------------------------

def _load_daily(dc, table: str) -> pd.DataFrame:
    obj = dc.get(table)
    try:
        return obj.to_daily()
    except NotImplementedError:
        return obj.df


def check_table(table: str, df: pd.DataFrame, k: float, window: Optional[str],
                skip_columns: frozenset = frozenset()) -> list:
    rows = []

    if "date" not in df.columns:
        lg.info(f"{table}: 没有 date 列，跳过")
        return rows
    if df["date"].duplicated().any():
        lg.info(f"{table}: 日期存在重复（非单一时间序列），跳过 IQR 检测")
        return rows

    df = df.sort_values("date").reset_index(drop=True)

    table_window = window
    median_gap = pd.to_datetime(df["date"]).diff().dt.days.median()
    if window is not None and median_gap > SPARSE_TABLE_GAP_DAYS:
        lg.info(f"{table}: 日期间隔中位数 {median_gap:.0f} 天，属于稀疏表，改用全样本 IQR")
        table_window = None

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != "date" and not pd.api.types.is_bool_dtype(df[c])
    ]

    for column in numeric_cols:
        if column in skip_columns:
            transform, reason = None, "原始表透传列，已在原始表检查"
        else:
            transform, reason = resolve_transform(table, column, df[column])
        row = {"table": table, "column": column, "transform": transform or "跳过",
               "window": ("全样本" if table_window is None else table_window) if transform else "",
               "reason": reason, "n_obs": int(df[column].notna().sum()), "n_outliers": None}

        if transform is not None:
            checked = TRANSFORMS[transform](df[column])
            # 变换后大部分为同一个值（如分红金额多数季度不变，对数收益率为 0），
            # IQR 为 0，无法检测。如实报告"无法检测"，而不是报告 0 个异常值
            share = checked.dropna().value_counts(normalize=True)
            if not share.empty and share.iloc[0] > SPARSE_MODE_SHARE:
                row["transform"] = f"{transform}（无法检测）"
                row["reason"] = f"变换后 {share.iloc[0]:.0%} 为同一取值，IQR 为 0"
                rows.append(row)
                continue
            frame = pd.DataFrame({"date": df["date"], column: checked})
            label = f"{table}[{transform}]"
            row["n_outliers"] = warn_outliers_iqr(frame, column, lg, label, k=k, window=table_window)

        rows.append(row)

    return rows


def run(k: float = 3.0, window: Optional[str] = "365D",
        report_path: Optional[str] = None) -> pd.DataFrame:
    from co_data_center.data_center import DataCenter

    dc = DataCenter()
    tables = {}
    for table in dc.list_tables():
        if table in EXCLUDED_TABLES:
            continue
        try:
            tables[table] = _load_daily(dc, table)
        except Exception:
            lg.exception(f"{table}: 读取失败")

    # 特征表中从原始表透传过来的列，不在特征表里重复检查
    raw_columns = frozenset(
        c for name, df in tables.items() if name != FEATURE_TABLE for c in df.columns
    ) - {"date"}

    rows = []
    for table, df in tables.items():
        skip = raw_columns if table == FEATURE_TABLE else frozenset()
        rows.extend(check_table(table, df, k, window, skip_columns=skip))

    report = pd.DataFrame(rows)
    if report.empty:
        lg.warning("没有检查到任何列")
        return report

    lg.info("数据质量检查汇总：\n" + report.to_string(index=False))
    if report_path:
        report.to_csv(report_path, index=False)
        lg.info(f"汇总报告已保存：{report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="IQR 数据质量检查")
    parser.add_argument("--k", type=float, default=3.0, help="IQR 倍数，默认 3.0")
    parser.add_argument("--window", type=str, default="365D",
                        help="按日历时间的滚动窗口（如 365D），0 表示全样本")
    parser.add_argument("--report", type=str, default=None, help="汇总报告 CSV 保存路径")
    args = parser.parse_args()

    window = None if args.window in ("0", "") else args.window
    run(k=args.k, window=window, report_path=args.report)
    sys.exit(0)  # IQR 只做提醒，不让任务失败


if __name__ == "__main__":
    main()