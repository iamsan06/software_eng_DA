"""Lag-Aware tabular machine learning models (Random Forest and XGBoost) using 16-week pre-harvest sequences."""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

from src.crop_calendar import get_pre_harvest_lag_weeks
from src.lag_construction import LAG_COLUMN_NAMES
from src.evaluation import calculate_metrics, evaluate_splits


class LagAwareTabularPipeline:
    """Feature preparation and modeling pipeline using 16 pre-harvest rainfall lags."""

    def __init__(
        self,
        lag_features: Sequence[str] = LAG_COLUMN_NAMES,
        secondary_num_features: Sequence[str] = ("IMD_Tmax",),
        cat_features: Sequence[str] = ("State_Clean",),
    ):
        self.lag_features = list(lag_features)
        self.secondary_num_features = list(secondary_num_features)
        self.cat_features = list(cat_features)
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names: List[str] = []

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fit encoder strictly on train split and transform all lag, thermal, and spatial features."""
        num_cols = self.lag_features + self.secondary_num_features
        X_num = train_df[num_cols].values

        if self.cat_features:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_cat = self.encoder.fit_transform(train_df[self.cat_features])
            cat_names = list(self.encoder.get_feature_names_out(self.cat_features))
            self.feature_names = num_cols + cat_names
            return np.hstack([X_num, X_cat])
        else:
            self.feature_names = num_cols
            return X_num

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform validation or test splits using fitted encoder."""
        num_cols = self.lag_features + self.secondary_num_features
        X_num = df[num_cols].values
        if self.cat_features:
            if self.encoder is None:
                raise RuntimeError("Encoder must be fitted before transformation.")
            X_cat = self.encoder.transform(df[self.cat_features])
            return np.hstack([X_num, X_cat])
        return X_num


def extract_lag_importance_profile(
    model: Any,
    pipeline: LagAwareTabularPipeline,
) -> pd.DataFrame:
    """Extract and normalize feature importance specifically for the 16 pre-harvest rainfall weeks."""
    all_importances = model.feature_importances_
    records = []
    for lag_name in pipeline.lag_features:
        idx = pipeline.feature_names.index(lag_name)
        imp = float(all_importances[idx])
        week_num = int(lag_name.split("_")[-1])  # e.g. 16 for Rain_Lag_16
        records.append({
            "Lag_Name": lag_name,
            "Weeks_Before_Harvest": -week_num,
            "Feature_Importance": imp,
        })
    imp_df = pd.DataFrame(records).sort_values(by="Weeks_Before_Harvest", ascending=True).reset_index(drop=True)
    # Normalize lag importances so they sum to 1.0 for response profile comparison
    total_imp = imp_df["Feature_Importance"].sum()
    if total_imp > 0:
        imp_df["Normalized_Lag_Response"] = imp_df["Feature_Importance"] / total_imp
    else:
        imp_df["Normalized_Lag_Response"] = 0.0
    return imp_df
