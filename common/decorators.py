import functools


def data_clear(clear_func):
    """装饰器工厂：让被装饰函数（通常是某个数据源的 fetch()）的返回值
    自动经过 clear_func 处理后再返回。

    用法：
        @data_clear(clean_stock_prices)
        def fetch(self, **params) -> pd.DataFrame:
            ...
            return raw_df

    clear_func 的签名应为 (df: pd.DataFrame) -> pd.DataFrame，
    每个数据源可以传入各自专属的清洗函数（填充方式、异常值处理等互不相同）。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return clear_func(func(*args, **kwargs))

        return wrapper

    return decorator