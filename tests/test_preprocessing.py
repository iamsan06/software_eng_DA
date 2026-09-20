"""Tests for preprocessing pipeline."""
import pytest
import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data, build_stable_geographic_id


def test_preprocess_data_execution():
    raw_df = load_raw_data()
    clean_df, audit_df = preprocess_data(raw_df, save_processed=False)

    assert isinstance(clean_df, pd.DataFrame)
    assert isinstance(audit_df, pd.DataFrame)
    assert len(clean_df) == 10279  # all raw observations valid
    assert "stable_geographic_id" in clean_df.columns
    assert "State_Clean" in clean_df.columns
    assert "District_Clean" in clean_df.columns


def test_stable_geographic_id_uniqueness():
    raw_df = load_raw_data()
    clean_df, _ = preprocess_data(raw_df, save_processed=False)

    # Check that stable_geographic_id + Year has 0 duplicates
    dupes = clean_df.duplicated(subset=["stable_geographic_id", "Year"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (stable_geographic_id, Year) combinations."


def test_audit_table_completeness():
    raw_df = load_raw_data()
    _, audit_df = preprocess_data(raw_df, save_processed=False)

    assert "step" in audit_df.columns
    assert "reason" in audit_df.columns
    assert "rows_removed" in audit_df.columns
    assert "remaining_rows" in audit_df.columns
    assert len(audit_df) >= 4
