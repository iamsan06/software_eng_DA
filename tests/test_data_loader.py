"""Tests for data loader module."""
import pytest
import pandas as pd
from src.data_loader import load_raw_data, get_raw_data_path


def test_get_raw_data_path():
    path = get_raw_data_path()
    assert path.is_file(), f"Resolved path {path} does not exist."


def test_load_raw_data_schema():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "RICE.YIELD..Kg.per.ha." in df.columns
    assert "IMD_RF" in df.columns
    assert "Dist.Code" in df.columns
    assert "Year" in df.columns


def test_load_raw_data_row_count():
    df = load_raw_data()
    # Verified 10,279 rows in actual CSV
    assert len(df) == 10279
    assert df["Year"].min() == 1990
    assert df["Year"].max() == 2017
