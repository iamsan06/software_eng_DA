"""Evaluation metrics, disaggregated error analysis, and metric table exports."""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # For tiny sample sizes or zero variance, r2_score can be degenerate
    if len(y_true) > 1 and np.var(y_true) > 1e-6:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4) if not np.isnan(r2) else np.nan,
    }


def evaluate_splits(
    model_name: str,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> List[Dict[str, any]]:
    """Evaluate model performance across train, val, and test partitions."""
    splits = [
        ("train", y_train, y_train_pred),
        ("validation", y_val, y_val_pred),
        ("test", y_test, y_test_pred),
    ]

    records = []
    for split_name, yt, yp in splits:
        metrics = calculate_metrics(yt, yp)
        records.append({
            "model": model_name,
            "split": split_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "n_samples": len(yt),
        })
    return records


def compute_group_metrics(
    pred_df: pd.DataFrame,
    group_cols: List[str],
    min_samples: int = 3,
) -> pd.DataFrame:
    """Compute disaggregated MAE, RMSE, and R2 across geographic groups (district or state)."""
    rows = []
    for group_key, sub_df in pred_df.groupby(group_cols):
        n = len(sub_df)
        if n < min_samples:
            continue

        yt = sub_df["Actual"].values
        yp = sub_df["Predicted"].values
        m = calculate_metrics(yt, yp)

        row = {}
        if isinstance(group_key, tuple):
            for col_name, val in zip(group_cols, group_key):
                row[col_name] = val
        else:
            row[group_cols[0]] = group_key

        row["n_samples"] = n
        row["MAE"] = m["MAE"]
        row["RMSE"] = m["RMSE"]
        row["R2"] = m["R2"]
        rows.append(row)

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values(by="MAE", ascending=True).reset_index(drop=True)
    return res_df
