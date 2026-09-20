"""Model training, validation tuning, and inference for Random Forest and XGBoost."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

from src.config import load_config
from src.validation import validate_feature_exclusion


class BaselinePipeline:
    """Wrapper pipeline managing feature preparation, encoder fitting on train, and model training."""

    def __init__(
        self,
        num_features: Sequence[str] = ("IMD_RF", "IMD_Tmax", "SD_RF_jun_nov"),
        cat_features: Sequence[str] = ("State_Clean",),
    ):
        self.num_features = list(num_features)
        self.cat_features = list(cat_features)
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names: List[str] = []

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fit one-hot encoder on train split only, and transform numerical & categorical features."""
        validate_feature_exclusion(self.num_features + self.cat_features)

        X_num = train_df[self.num_features].values

        if self.cat_features:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_cat = self.encoder.fit_transform(train_df[self.cat_features])
            cat_names = list(self.encoder.get_feature_names_out(self.cat_features))
            self.feature_names = self.num_features + cat_names
            return np.hstack([X_num, X_cat])
        else:
            self.feature_names = self.num_features
            return X_num

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform validation or test features using fitted encoder."""
        if self.encoder is None and self.cat_features:
            raise RuntimeError("Pipeline must be fitted before transforming.")

        X_num = df[self.num_features].values
        if self.cat_features:
            X_cat = self.encoder.transform(df[self.cat_features])
            return np.hstack([X_num, X_cat])
        return X_num


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """Train RandomForestRegressor with validation tuning over candidate max_depths."""
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }

    # Validation tuning over max_depth candidates
    candidate_depths = [8, 12, 16, 20]
    best_depth = params.get("max_depth", 15)
    best_val_mae = float("inf")

    for d in candidate_depths:
        p = params.copy()
        p["max_depth"] = d
        rf = RandomForestRegressor(**p)
        rf.fit(X_train, y_train)
        val_preds = rf.predict(X_val)
        val_mae = float(np.mean(np.abs(y_val - val_preds)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_depth = d

    final_params = params.copy()
    final_params["max_depth"] = best_depth
    best_rf = RandomForestRegressor(**final_params)
    best_rf.fit(X_train, y_train)

    return best_rf, final_params


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Train XGBoost Regressor with validation tuning over learning_rate and max_depth."""
    if params is None:
        params = {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

    # Validation tuning grid
    tuning_candidates = [
        {"max_depth": 4, "learning_rate": 0.05},
        {"max_depth": 6, "learning_rate": 0.05},
        {"max_depth": 6, "learning_rate": 0.03},
    ]

    best_cand = tuning_candidates[0]
    best_val_mae = float("inf")

    for cand in tuning_candidates:
        p = params.copy()
        p.update(cand)
        model = xgb.XGBRegressor(**p)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_preds = model.predict(X_val)
        val_mae = float(np.mean(np.abs(y_val - val_preds)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_cand = cand

    final_params = params.copy()
    final_params.update(best_cand)
    best_xgb = xgb.XGBRegressor(**final_params)
    best_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return best_xgb, final_params


def build_prediction_dataframe(
    df: pd.DataFrame,
    predictions: np.ndarray,
    target_col: str = "RICE.YIELD..Kg.per.ha.",
) -> pd.DataFrame:
    """Construct standard prediction and residual dataframe."""
    actuals = df[target_col].values
    residuals = actuals - predictions

    state_col = "State_Clean" if "State_Clean" in df.columns else "State"
    dist_col = "District_Clean" if "District_Clean" in df.columns else "District"
    geo_id_col = "stable_geographic_id" if "stable_geographic_id" in df.columns else "Dist.Code"

    pred_df = pd.DataFrame({
        "stable_geographic_id": df[geo_id_col].values,
        "State": df[state_col].values,
        "District": df[dist_col].values,
        "Year": df["Year"].values,
        "Actual": actuals,
        "Predicted": np.round(predictions, 2),
        "Residual": np.round(residuals, 2),
    })
    return pred_df
