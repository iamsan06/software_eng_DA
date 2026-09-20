"""Data and model validation suite to prevent leakage, invalid ranges, and key collisions."""
from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd

REQUIRED_RAW_COLUMNS = [
    "Dist.Code",
    "Sh_Dist_code",
    "FID_1",
    "State",
    "AgroClimate",
    "District",
    "Year",
    "RICE.AREA..1000.ha.",
    "RICE.PRODUCTION..1000.tons.",
    "RICE.YIELD..Kg.per.ha.",
    "RICE.SHARE.IN.TOTAL..Percent.",
    "Crops",
    "IMD_RF",
    "IMD_Tmax",
    "SD_RF_jun_nov",
]


def validate_raw_schema(df: pd.DataFrame) -> bool:
    """Validate that all expected raw columns are present."""
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    return True


def validate_non_negative(df: pd.DataFrame, column: str, allow_zero: bool = True) -> bool:
    """Validate that numeric values in a column are non-negative."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    vals = pd.to_numeric(df[column], errors="coerce")
    if vals.isna().any():
        raise ValueError(f"Column '{column}' contains non-numeric or NaN values.")

    if allow_zero:
        invalid = vals < 0
        op_str = "< 0"
    else:
        invalid = vals <= 0
        op_str = "<= 0"

    if invalid.any():
        num_invalid = invalid.sum()
        raise ValueError(f"Column '{column}' contains {num_invalid} values {op_str}.")

    return True


def validate_key_integrity(
    df: pd.DataFrame,
    key_columns: Sequence[str] = ("State", "Dist.Code", "Year"),
) -> Tuple[bool, int]:
    """Validate uniqueness of geographic-temporal keys.
    
    Returns (is_unique, num_duplicates).
    """
    for col in key_columns:
        if col not in df.columns:
            raise KeyError(f"Key column '{col}' not in DataFrame.")

    dupes = df.duplicated(subset=list(key_columns)).sum()
    if dupes > 0:
        return False, int(dupes)
    return True, 0


def validate_temporal_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    year_col: str = "Year",
) -> bool:
    """Ensure strict chronological separation without temporal leakage.
    
    max(train_year) < min(val_year) and max(val_year) < min(test_year).
    """
    train_years = set(train_df[year_col])
    val_years = set(val_df[year_col])
    test_years = set(test_df[year_col])

    # Disjointness
    if not train_years.isdisjoint(val_years):
        overlap = train_years.intersection(val_years)
        raise ValueError(f"Temporal leakage: Train and Val share years {overlap}")
    if not val_years.isdisjoint(test_years):
        overlap = val_years.intersection(test_years)
        raise ValueError(f"Temporal leakage: Val and Test share years {overlap}")
    if not train_years.isdisjoint(test_years):
        overlap = train_years.intersection(test_years)
        raise ValueError(f"Temporal leakage: Train and Test share years {overlap}")

    # Strict chronological order
    if max(train_years) >= min(val_years):
        raise ValueError(
            f"Chronological violation: max(train)={max(train_years)} >= min(val)={min(val_years)}"
        )
    if max(val_years) >= min(test_years):
        raise ValueError(
            f"Chronological violation: max(val)={max(val_years)} >= min(test)={min(test_years)}"
        )

    return True


def validate_feature_exclusion(
    feature_names: Sequence[str],
    forbidden_features: Sequence[str] = (
        "RICE.YIELD..Kg.per.ha.",
        "RICE.PRODUCTION..1000.tons.",
        "RICE.AREA..1000.ha.",
        "RICE.SHARE.IN.TOTAL..Percent.",
        "FID_1",
        "Dist.Code",
        "Sh_Dist_code",
    ),
) -> bool:
    """Verify that target-derived and raw numeric identifiers are not passed into models as features."""
    for feat in feature_names:
        if feat in forbidden_features:
            raise ValueError(f"Forbidden feature '{feat}' detected in feature set!")
    return True
