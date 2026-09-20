"""Chronological splitting module strictly enforcing temporal causality."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.config import load_config
from src.validation import validate_temporal_split


def compute_temporal_splits(
    df: pd.DataFrame,
    year_col: str = "Year",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    custom_year_bounds: Optional[Dict[str, List[int]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically based on distinct observation years.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset.
    year_col : str
        Name of temporal year column.
    train_ratio, val_ratio, test_ratio : float
        Target proportions for train, validation, and test periods.
    custom_year_bounds : Optional[Dict[str, List[int]]]
        Optional explicit [start_year, end_year] for each split.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Chronologically partitioned datasets.
    split_summary : pd.DataFrame
        Summary table recording split, year boundaries, rows, states, districts.
    """
    years = sorted(df[year_col].unique())
    n_years = len(years)

    if custom_year_bounds is not None:
        train_start, train_end = custom_year_bounds["train_years"]
        val_start, val_end = custom_year_bounds["val_years"]
        test_start, test_end = custom_year_bounds["test_years"]
    else:
        # Calculate year cutoffs based on total number of unique years
        n_train_years = int(round(n_years * train_ratio))
        n_val_years = int(round(n_years * val_ratio))
        
        train_years_list = years[:n_train_years]
        val_years_list = years[n_train_years : n_train_years + n_val_years]
        test_years_list = years[n_train_years + n_val_years :]

        train_start, train_end = train_years_list[0], train_years_list[-1]
        val_start, val_end = val_years_list[0], val_years_list[-1]
        test_start, test_end = test_years_list[0], test_years_list[-1]

    train_df = df[(df[year_col] >= train_start) & (df[year_col] <= train_end)].copy()
    val_df = df[(df[year_col] >= val_start) & (df[year_col] <= val_end)].copy()
    test_df = df[(df[year_col] >= test_start) & (df[year_col] <= test_end)].copy()

    # Enforce strict validation
    validate_temporal_split(train_df, val_df, test_df, year_col=year_col)

    state_col = "State_Clean" if "State_Clean" in df.columns else "State"
    dist_col = "stable_geographic_id" if "stable_geographic_id" in df.columns else "District"

    summary_records = [
        {
            "split": "train",
            "start_year": int(train_start),
            "end_year": int(train_end),
            "n_rows": int(len(train_df)),
            "n_states": int(train_df[state_col].nunique()),
            "n_districts": int(train_df[dist_col].nunique()),
        },
        {
            "split": "validation",
            "start_year": int(val_start),
            "end_year": int(val_end),
            "n_rows": int(len(val_df)),
            "n_states": int(val_df[state_col].nunique()),
            "n_districts": int(val_df[dist_col].nunique()),
        },
        {
            "split": "test",
            "start_year": int(test_start),
            "end_year": int(test_end),
            "n_rows": int(len(test_df)),
            "n_states": int(test_df[state_col].nunique()),
            "n_districts": int(test_df[dist_col].nunique()),
        },
    ]

    split_summary = pd.DataFrame(summary_records)
    return train_df, val_df, test_df, split_summary
