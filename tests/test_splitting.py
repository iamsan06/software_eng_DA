"""Tests for chronological splitting module."""
import pytest
import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.splitting import compute_temporal_splits


def test_chronological_splitting_properties():
    raw_df = load_raw_data()
    clean_df, _ = preprocess_data(raw_df, save_processed=False)

    train_df, val_df, test_df, split_summary = compute_temporal_splits(
        clean_df,
        custom_year_bounds={
            "train_years": [1990, 2009],
            "val_years": [2010, 2013],
            "test_years": [2014, 2017],
        },
    )

    # Check non-empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(val_df) + len(test_df) == len(clean_df)

    # Check chronological boundary conditions
    assert train_df["Year"].max() < val_df["Year"].min()
    assert val_df["Year"].max() < test_df["Year"].min()

    # Exact expected boundaries
    assert train_df["Year"].min() == 1990
    assert train_df["Year"].max() == 2009
    assert val_df["Year"].min() == 2010
    assert val_df["Year"].max() == 2013
    assert test_df["Year"].min() == 2014
    assert test_df["Year"].max() == 2017


def test_split_summary_structure():
    raw_df = load_raw_data()
    clean_df, _ = preprocess_data(raw_df, save_processed=False)

    _, _, _, split_summary = compute_temporal_splits(clean_df)

    expected_cols = ["split", "start_year", "end_year", "n_rows", "n_states", "n_districts"]
    for c in expected_cols:
        assert c in split_summary.columns
    assert set(split_summary["split"]) == {"train", "validation", "test"}
