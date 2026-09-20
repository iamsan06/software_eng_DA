"""Tests for validation module ensuring data integrity and preventing leakage."""
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data
from src.validation import (
    validate_raw_schema,
    validate_non_negative,
    validate_key_integrity,
    validate_temporal_split,
    validate_feature_exclusion,
)


def test_validate_raw_schema():
    df = load_raw_data()
    assert validate_raw_schema(df) is True


def test_validate_non_negative_yield_and_rainfall():
    df = load_raw_data()
    # Rice yield must be non-negative (strictly positive)
    assert validate_non_negative(df, "RICE.YIELD..Kg.per.ha.", allow_zero=False) is True
    # Rainfall must be non-negative
    assert validate_non_negative(df, "IMD_RF", allow_zero=True) is True


def test_validate_key_integrity_raw():
    df = load_raw_data()
    # District + Year alone has collisions (76 duplicates) due to duplicate district names across states
    unique_dist_yr, dupe_count = validate_key_integrity(df, key_columns=["District", "Year"])
    assert not unique_dist_yr
    assert dupe_count == 76

    # State + Dist.Code + Year is completely unique (0 duplicates)
    unique_state_dist_yr, dupe_zero = validate_key_integrity(df, key_columns=["State", "Dist.Code", "Year"])
    assert unique_state_dist_yr
    assert dupe_zero == 0


def test_validate_temporal_split_order():
    train_df = pd.DataFrame({"Year": [1990, 1995, 2000, 2009]})
    val_df = pd.DataFrame({"Year": [2010, 2011, 2012, 2013]})
    test_df = pd.DataFrame({"Year": [2014, 2015, 2016, 2017]})

    assert validate_temporal_split(train_df, val_df, test_df) is True

    # Leaking train year into val should fail
    leaky_val = pd.DataFrame({"Year": [2009, 2010, 2011]})
    with pytest.raises(ValueError):
        validate_temporal_split(train_df, leaky_val, test_df)


def test_validate_feature_exclusion():
    safe_features = ["IMD_RF", "IMD_Tmax", "SD_RF_jun_nov", "State_Clean_Kerala"]
    assert validate_feature_exclusion(safe_features) is True

    unsafe_features = ["IMD_RF", "RICE.PRODUCTION..1000.tons.", "IMD_Tmax"]
    with pytest.raises(ValueError):
        validate_feature_exclusion(unsafe_features)

    target_leaked = ["IMD_RF", "RICE.YIELD..Kg.per.ha."]
    with pytest.raises(ValueError):
        validate_feature_exclusion(target_leaked)
