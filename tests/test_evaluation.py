"""Tests for evaluation metrics and prediction schema."""
import pytest
import numpy as np
import pandas as pd
from src.evaluation import calculate_metrics, evaluate_splits, compute_group_metrics
from src.models import build_prediction_dataframe


def test_calculate_metrics_values():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    res = calculate_metrics(y_true, y_pred)
    assert res["MAE"] == pytest.approx(6.667, abs=1e-2)
    assert res["RMSE"] > 0
    assert not np.isnan(res["R2"])
    assert np.isfinite(res["MAE"])
    assert np.isfinite(res["RMSE"])
    assert np.isfinite(res["R2"])


def test_build_prediction_dataframe():
    df = pd.DataFrame({
        "stable_geographic_id": ["PUNJAB_1", "KERALA_2"],
        "State_Clean": ["Punjab", "Kerala"],
        "District_Clean": ["Ludhiana", "Palakkad"],
        "Year": [2015, 2016],
        "RICE.YIELD..Kg.per.ha.": [3500.0, 2400.0],
    })
    preds = np.array([3450.0, 2350.0])

    pred_df = build_prediction_dataframe(df, preds)

    expected_cols = ["stable_geographic_id", "State", "District", "Year", "Actual", "Predicted", "Residual"]
    for c in expected_cols:
        assert c in pred_df.columns

    assert len(pred_df) == 2
    assert pred_df.loc[0, "Residual"] == pytest.approx(50.0, abs=1e-2)
    assert pred_df.loc[1, "Residual"] == pytest.approx(50.0, abs=1e-2)


def test_compute_group_metrics():
    pred_df = pd.DataFrame({
        "State": ["StateA", "StateA", "StateA", "StateB", "StateB", "StateB"],
        "Actual": [1000, 1100, 1200, 2000, 2100, 2200],
        "Predicted": [1050, 1080, 1220, 1950, 2120, 2180],
    })

    group_metrics = compute_group_metrics(pred_df, group_cols=["State"], min_samples=2)
    assert len(group_metrics) == 2
    assert "MAE" in group_metrics.columns
    assert "RMSE" in group_metrics.columns
    assert "R2" in group_metrics.columns
    assert "n_samples" in group_metrics.columns
    assert np.isfinite(group_metrics["MAE"].values).all()
