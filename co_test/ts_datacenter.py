"""DataCenter 测试脚本：验证六张原始表 + feature_daily 的发现、
懒加载、缓存机制，并对最新合成的 feature_daily 宽表做详细检查。
"""

import pandas as pd

from co_data_center.data_center import DataCenter


def inspect_all_tables(dc: DataCenter) -> None:
    print(repr(dc))
    print(f"可用表：{dc.list_tables()}")
    print()

    for table_name in dc.list_tables():
        print("=" * 60)
        print(f"表名: {table_name}")
        print("=" * 60)

        obj = dc.get(table_name)
        print(f"数据对象类型: {type(obj).__name__}")
        print(f"原始形状: {obj.df.shape}")
        print(f"字段: {list(obj.df.columns)}")

        try:
            daily = obj.to_daily()
            print(f"to_daily() 形状: {daily.shape}")
        except NotImplementedError as e:
            print(f"to_daily() 按设计拒绝调用: {e}")

        print()


def verify_cache(dc: DataCenter) -> None:
    print("=" * 60)
    print("缓存验证")
    print("=" * 60)
    first_table = dc.list_tables()[0]
    obj_a = dc.get(first_table)
    obj_b = dc.get(first_table)
    print(f"两次 get('{first_table}') 是否命中同一缓存对象: {obj_a is obj_b}")
    print()


def inspect_feature_daily(dc: DataCenter) -> None:
    """对最新合成的 feature_daily 宽表做详细检查。"""
    print("=" * 60)
    print("feature_daily 详细检查")
    print("=" * 60)

    feature_obj = dc.get("feature_daily")
    df = feature_obj.df

    print(f"形状: {df.shape}")
    print(f"字段类型:\n{df.dtypes}")
    print()

    print(f"时间范围: {df['date'].min()} 至 {df['date'].max()}")
    subset_2018_2024 = feature_obj.filter(date__gte="2018-01-01", date__lte="2026-09-22")
    print(f"2018-2026 区间行数: {len(subset_2018_2024.df)}")
    print()

    print("缺失值统计:")
    na_counts = df.isna().sum()
    print(na_counts[na_counts > 0] if (na_counts > 0).any() else "无缺失值")
    print()

    print("数值列统计:")
    print(df.describe())
    print()

    print("最新5行:")
    print(df.tail(5))
    print()

    print("2018-2026区间前5行:")
    print(subset_2018_2024.df.head(5))


def main():
    dc = DataCenter()
    inspect_all_tables(dc)
    verify_cache(dc)
    inspect_feature_daily(dc)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet(r"E:\华泰证券\ChooserOptionTool\data\features\feature_daily.parquet")
    has_news = df[df['sentiment_score'].notna()]
    print("有新闻数据的时间范围:", has_news['date'].min(), "至", has_news['date'].max())

    subset = df[(df['date'] >= '2018-01-01') & (df['date'] <= '2024-12-31')]
    print("2018-2024区间内,有新闻的天数:", subset['sentiment_score'].notna().sum(), "/", len(subset))
    main()