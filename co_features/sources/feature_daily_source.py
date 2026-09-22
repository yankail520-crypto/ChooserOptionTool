import pandas as pd

from co_data_pipeline.pipeline_base import DataSourceBase
from common.paths import get_features_path
from common.logger import get_logger

lg = get_logger("co_features")


class FeatureDailySource(DataSourceBase):
    """
    feature_daily 的 DataSourceBase 包装。

    让"特征工程算出的衍生宽表"和其他六张 raw 表一样，被 DataCenter
    统一管辖——上层模块一律通过 DataCenter.get("feature_daily") 取用，
    不需要关心它是从外部API拉的，还是内部计算出来的。
    """

    def fetch(self, **params) -> pd.DataFrame:
        from co_features.feature_engine import FeatureEngine  # 延迟导入，避免循环依赖

        return FeatureEngine().build()

    def get_fields(self) -> dict:
        # 特征列会随着 co_features/features/ 下新增文件动态增长，
        # 这里不做强校验，实际字段以 fetch() 的真实返回为准。
        return {"date": "datetime64[ns]"}

    def get_table_name(self) -> str:
        return "feature_daily"

    def is_calendar_anchor(self) -> bool:
        return False

    def save(self, df: pd.DataFrame) -> None:
        path = get_features_path() / "feature_daily.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        lg.info(f"feature_daily: 已写入 {path}，共 {len(df)} 行 {len(df.columns)} 列")

    def load(self) -> pd.DataFrame:
        path = get_features_path() / "feature_daily.parquet"
        return pd.read_parquet(path)