import pandas as pd


def detect_outliers_iqr(
    series: pd.Series,
    k: float = 3.0,
    window: int | str | None = None,
    dates: pd.Series | None = None,
    min_periods: int = 60,
) -> pd.Series:
    """返回布尔 Series，标记落在 IQR 上下界之外的值。只做判断，不修改数据。

    k: 默认 3.0 而不是 1.5——金融收益率是厚尾分布，1.5 倍会把大量
       真实的极端行情标出来，警告多到没人看。
    window: None 表示全样本 IQR；整数（如 252）表示按行数滚动；
       字符串（如 "365D"）表示按日历时间滚动，此时必须传入 dates。
       按时间滚动对稀疏表（如季度分红）更合理：按行数的 252 行对季度数据
       相当于 63 年，窗口永远填不满，等于什么都没检查。
    min_periods: 滚动窗口内至少需要的观测数，不足的位置不做判断。

    IQR 为 0 的位置（稀疏或离散列，如大部分为 0 的股息增长率）无法
    定义异常，一律不标记。
    """
    if window is None:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
    elif isinstance(window, str):
        if dates is None:
            raise ValueError("按时间滚动（window 为字符串）时必须传入 dates")
        timed = pd.Series(series.to_numpy(), index=pd.DatetimeIndex(dates))
        rolling = timed.rolling(window, min_periods=min_periods)
        q1 = pd.Series(rolling.quantile(0.25).to_numpy(), index=series.index)
        q3 = pd.Series(rolling.quantile(0.75).to_numpy(), index=series.index)
    else:
        rolling = series.rolling(window, min_periods=min(min_periods, window))
        q1 = rolling.quantile(0.25)
        q3 = rolling.quantile(0.75)

    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = ((series < lower) | (series > upper)) & (iqr > 0)
    return mask.fillna(False).astype(bool)


def warn_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    logger,
    table_name: str = "",
    date_column: str = "date",
    k: float = 3.0,
    window: int | str | None = None,
    n_show: int = 10,
    min_periods: int = 60,
) -> int:
    """对 df[column] 做 IQR 异常值检测并打印警告日志，不修改数据。返回异常值个数。"""
    if column not in df.columns:
        logger.warning(f"{table_name}: 字段 '{column}' 不存在，跳过异常值检测")
        return 0

    valid = df[column].dropna()
    if valid.empty:
        return 0

    dates = df.loc[valid.index, date_column] if date_column in df.columns else None
    mask = detect_outliers_iqr(
        valid, k=k, window=window, dates=dates, min_periods=min_periods
    ).reindex(df.index, fill_value=False)
    n_outliers = int(mask.sum())
    if n_outliers == 0:
        return 0

    # 展示偏离中位数最远的样本，而不是时间上最早的几条
    outliers = df.loc[mask].copy()
    outliers["_dev"] = (outliers[column] - valid.median()).abs()
    cols = [date_column, column] if date_column in df.columns else [column]
    sample = outliers.nlargest(n_show, "_dev")[cols]

    logger.warning(
        f"{table_name}: 字段 '{column}' 检测到 {n_outliers} 个 IQR 异常值"
        f"（占 {n_outliers / len(valid):.1%}，k={k}，"
        f"{'全样本' if window is None else f'滚动窗口 {window}'}；未自动处理，仅提醒排查）。"
        f"偏离最大的 {len(sample)} 条：\n{sample.to_string(index=False)}"
    )
    return n_outliers