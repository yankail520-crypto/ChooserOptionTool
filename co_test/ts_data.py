"""
数据质量检查脚本 —— 对 data/raw/ 下的表做基础校验：
- 时间范围、行数
- 2018-2024区间覆盖情况
- 重复日期、缺失值统计
- 交易日缺口检测（相邻两条记录间隔是否异常大）
"""

import pandas as pd

from common.paths import get_raw_path


def check_table(table_name: str, date_col: str = "date") -> None:
    path = get_raw_path() / f"{table_name}.parquet"
    if not path.exists():
        print(f"[{table_name}] 文件不存在：{path}")
        return

    df = pd.read_parquet(path)
    print(f"\n{'=' * 60}")
    print(f"表名: {table_name}")
    print(f"{'=' * 60}")
    print(f"形状: {df.shape}")
    print(f"字段类型:\n{df.dtypes}")

    if date_col not in df.columns:
        print(f"警告: 未找到日期列 '{date_col}'")
        return

    df[date_col] = pd.to_datetime(df[date_col])
    print(f"\n时间范围: {df[date_col].min()} 至 {df[date_col].max()}")

    # 2018-2024 覆盖情况
    subset = df[(df[date_col] >= "2018-01-01") & (df[date_col] <= "2024-12-31")]
    print(f"2018-2024 区间行数: {len(subset)}")

    # 重复日期
    dup_count = df[date_col].duplicated().sum()
    print(f"重复日期数: {dup_count}")
    if dup_count > 0:
        print(f"  重复的日期示例: {df[date_col][df[date_col].duplicated()].head().tolist()}")

    # 缺失值统计（按列）
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts) > 0:
        print(f"\n缺失值统计:\n{na_counts}")
    else:
        print("\n无缺失值")

    # 交易日缺口检测：相邻记录间隔超过7天视为可疑缺口（覆盖长假但排查异常长缺口）
    sorted_dates = df[date_col].sort_values().reset_index(drop=True)
    gaps = sorted_dates.diff().dt.days
    suspicious_gaps = gaps[gaps > 7]
    if len(suspicious_gaps) > 0:
        print(f"\n发现 {len(suspicious_gaps)} 处超过7天的日期缺口:")
        for idx in suspicious_gaps.index:
            print(f"  {sorted_dates[idx - 1].date()} -> {sorted_dates[idx].date()} "
                  f"(间隔 {int(gaps[idx])} 天)")
    else:
        print("\n未发现异常日期缺口")

    # 数值列基础统计
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        print(f"\n数值列统计:\n{df[numeric_cols].describe()}")


if __name__ == "__main__":
    for table in ["raw_stock_prices", "raw_treasury_rates", "raw_vix", "raw_option_chain",
                  "raw_news", "raw_dividends"]:
        check_table(table)