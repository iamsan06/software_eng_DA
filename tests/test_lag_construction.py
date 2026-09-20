"""Unit tests for 16-week pre-harvest rainfall lag extraction and tabular lag pipeline."""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.lag_construction import extract_16week_lag_sequence, LAG_COLUMN_NAMES
from src.lag_models import LagAwareTabularPipeline, extract_lag_importance_profile


@pytest.fixture
def sample_data():
    yield_df = pd.DataFrame({
        "stable_geographic_id": ["PUNJAB_1", "KERALA_2"],
        "State_Clean": ["Punjab", "Kerala"],
        "District_Clean": ["Ludhiana", "Palakkad"],
        "Year": [2010, 2010],
        "RICE.YIELD..Kg.per.ha.": [3500.0, 2200.0],
        "IMD_Tmax": [32.5, 30.1],
    })

    # Weekly rainfall spanning weeks 20 to 50 for both districts
    rf_records = []
    for geo_id in ["PUNJAB_1", "KERALA_2"]:
        for wk in range(20, 52):
            rf_records.append({
                "stable_geographic_id": geo_id,
                "year": 2010,
                "iso_week": wk,
                "cumulative_rainfall_mm": float(wk * 1.5),
            })
    weekly_df = pd.DataFrame(rf_records)
    return yield_df, weekly_df


def test_extract_16week_lag_sequence(sample_data):
    yield_df, weekly_df = sample_data
    lag_df = extract_16week_lag_sequence(yield_df, weekly_df)

    assert len(lag_df) == 2
    for lag_col in LAG_COLUMN_NAMES:
        assert lag_col in lag_df.columns
        assert (lag_df[lag_col] >= 0).all()

    # Check Punjab (T_ref = 42): Lag 1 should be week 41 (41 * 1.5 = 61.5)
    punjab_row = lag_df[lag_df["stable_geographic_id"] == "PUNJAB_1"].iloc[0]
    assert punjab_row["harvest_reference_week"] == 42
    assert punjab_row["Rain_Lag_1"] == pytest.approx(41 * 1.5, abs=1e-1)
    assert punjab_row["Rain_Lag_16"] == pytest.approx((42 - 16) * 1.5, abs=1e-1)


def test_lag_aware_tabular_pipeline(sample_data):
    yield_df, weekly_df = sample_data
    lag_df = extract_16week_lag_sequence(yield_df, weekly_df)

    pipeline = LagAwareTabularPipeline()
    X = pipeline.fit_transform(lag_df)

    assert X.shape[0] == 2
    # 16 lag columns + 1 IMD_Tmax + 2 states one-hot
    assert X.shape[1] == 16 + 1 + 2

    # Fit a simple model and extract lag importance profile
    y = lag_df["RICE.YIELD..Kg.per.ha."].values
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)

    imp_df = extract_lag_importance_profile(rf, pipeline)
    assert len(imp_df) == 16
    assert "Normalized_Lag_Response" in imp_df.columns
    assert "Weeks_Before_Harvest" in imp_df.columns
