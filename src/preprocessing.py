"""Preprocessing pipeline for India-wide crop yield and meteorological data."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data
from src.config import load_config
from src.validation import validate_raw_schema, validate_non_negative, validate_key_integrity


COLUMN_MAPPING = {
    "Dist.Code": "dist_code",
    "Sh_Dist_code": "sh_dist_code",
    "FID_1": "fid_1",
    "State": "state",
    "AgroClimate": "agro_climate",
    "District": "district",
    "Year": "year",
    "RICE.AREA..1000.ha.": "rice_area_1000ha",
    "RICE.PRODUCTION..1000.tons.": "rice_production_1000tons",
    "RICE.YIELD..Kg.per.ha.": "rice_yield_kg_per_ha",
    "RICE.SHARE.IN.TOTAL..Percent.": "rice_share_percent",
    "Crops": "crop",
    "IMD_RF": "rainfall_imd",
    "IMD_Tmax": "temp_max_imd",
    "SD_RF_jun_nov": "rainfall_sd_jun_nov",
}

# Reverse mapping for compatibility with original naming
REVERSE_COLUMN_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}


def standardize_text(series: pd.Series, mode: str = "title") -> pd.Series:
    """Standardize string formatting: strip whitespace and apply consistent casing."""
    s = series.astype(str).str.strip()
    if mode == "title":
        return s.str.title()
    elif mode == "upper":
        return s.str.upper()
    elif mode == "lower":
        return s.str.lower()
    return s


def build_stable_geographic_id(state: pd.Series, dist_code: pd.Series) -> pd.Series:
    """Create globally stable, unique spatial identifier: STATE_DISTCODE."""
    clean_state = state.astype(str).str.strip().str.upper().str.replace(r"\s+", "_", regex=True)
    clean_code = dist_code.astype(str).str.strip()
    return clean_state + "_" + clean_code


def preprocess_data(
    raw_df: Optional[pd.DataFrame] = None,
    save_processed: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete leakage-safe data preprocessing.
    
    Returns
    -------
    clean_df : pd.DataFrame
        Cleaned, typed, standardized dataset with stable keys.
    audit_summary : pd.DataFrame
        Accounting of every filtering step and record retention count.
    """
    config = load_config()
    if raw_df is None:
        raw_df = load_raw_data()

    validate_raw_schema(raw_df)

    audit_records = []
    current_count = len(raw_df)
    audit_records.append({
        "step": "Raw Ingestion",
        "reason": "Initial raw observations loaded from source CSV",
        "rows_removed": 0,
        "remaining_rows": current_count,
    })

    # Copy to avoid mutating original
    df = raw_df.copy()

    # Check exact duplicates
    exact_dupes = df.duplicated().sum()
    if exact_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    current_count = len(df)
    audit_records.append({
        "step": "Deduplication",
        "reason": "Exact row duplicate removal",
        "rows_removed": int(exact_dupes),
        "remaining_rows": current_count,
    })

    # Validate and standardize types
    df["Year"] = pd.to_numeric(df["Year"], errors="raise").astype(int)
    df["Dist.Code"] = pd.to_numeric(df["Dist.Code"], errors="raise").astype(int)
    df["Sh_Dist_code"] = pd.to_numeric(df["Sh_Dist_code"], errors="raise").astype(int)
    df["FID_1"] = pd.to_numeric(df["FID_1"], errors="raise").astype(int)
    df["RICE.YIELD..Kg.per.ha."] = pd.to_numeric(df["RICE.YIELD..Kg.per.ha."], errors="raise").astype(float)
    df["IMD_RF"] = pd.to_numeric(df["IMD_RF"], errors="raise").astype(float)
    df["IMD_Tmax"] = pd.to_numeric(df["IMD_Tmax"], errors="raise").astype(float)
    df["SD_RF_jun_nov"] = pd.to_numeric(df["SD_RF_jun_nov"], errors="raise").astype(float)

    # Standardize State and District names
    df["State_Clean"] = standardize_text(df["State"], mode="title")
    df["District_Clean"] = standardize_text(df["District"], mode="title")

    # Construct stable geographic identifier
    df["stable_geographic_id"] = build_stable_geographic_id(df["State_Clean"], df["Dist.Code"])

    # Validate non-negative yield and rainfall
    invalid_yield = (df["RICE.YIELD..Kg.per.ha."] <= 0) | df["RICE.YIELD..Kg.per.ha."].isna()
    yield_removed = int(invalid_yield.sum())
    if yield_removed > 0:
        df = df[~invalid_yield].reset_index(drop=True)
    current_count = len(df)
    audit_records.append({
        "step": "Target Validation",
        "reason": "Removal of non-positive or missing rice yield observations",
        "rows_removed": yield_removed,
        "remaining_rows": current_count,
    })

    invalid_rf = (df["IMD_RF"] < 0) | df["IMD_RF"].isna()
    rf_removed = int(invalid_rf.sum())
    if rf_removed > 0:
        df = df[~invalid_rf].reset_index(drop=True)
    current_count = len(df)
    audit_records.append({
        "step": "Rainfall Validation",
        "reason": "Removal of negative or missing aggregate rainfall records",
        "rows_removed": rf_removed,
        "remaining_rows": current_count,
    })

    audit_df = pd.DataFrame(audit_records)

    if save_processed:
        processed_path = Path(config["paths"]["processed_data"])
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)

        tables_dir = Path(config["paths"]["tables_dir"])
        tables_dir.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(tables_dir / "preprocessing_summary.csv", index=False)

    return df, audit_df
