"""Pre-harvest 16-week rainfall sequence construction enforcing zero lookahead leakage."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.crop_calendar import get_crop_reference_week, get_pre_harvest_lag_weeks


LAG_COLUMN_NAMES = [f"Rain_Lag_{k}" for k in range(16, 0, -1)]  # Rain_Lag_16 down to Rain_Lag_1


def extract_16week_lag_sequence(
    yield_df: pd.DataFrame,
    weekly_rainfall_df: pd.DataFrame,
    geo_id_col: str = "stable_geographic_id",
    year_col: str = "Year",
    state_col: str = "State_Clean",
    lag_window_size: int = 16,
) -> pd.DataFrame:
    """Construct 16-week pre-harvest cumulative rainfall sequences strictly preceding harvest.
    
    For every district and crop year:
    1. Determine zone/state-specific Kharif harvest reference week T_ref.
    2. Extract weekly rainfall for weeks (T_ref - 16) through (T_ref - 1).
    3. Verify that all 16 pre-harvest weeks strictly precede T_ref (zero future leakage).
    4. Attach lag columns: Rain_Lag_16, Rain_Lag_15, ..., Rain_Lag_1.
    
    Parameters
    ----------
    yield_df : pd.DataFrame
        District-year crop yield dataset.
    weekly_rainfall_df : pd.DataFrame
        Time-resolved weekly cumulative rainfall [geo_id_col, year, iso_week, cumulative_rainfall_mm].
        
    Returns
    -------
    pd.DataFrame
        Merged dataset containing target yield, metadata, and ordered pre-harvest rainfall lags.
    """
    records = []
    
    # Pre-index weekly rainfall for fast lookup
    # Key: (geo_id, year, iso_week) -> rainfall_mm
    weekly_map = {}
    for _, row in weekly_rainfall_df.iterrows():
        key = (str(row[geo_id_col]), int(row["year"]), int(row["iso_week"]))
        weekly_map[key] = float(row["cumulative_rainfall_mm"])

    for _, row in yield_df.iterrows():
        geo_id = str(row[geo_id_col])
        year = int(row[year_col])
        state = str(row[state_col]) if state_col in row else ""

        # Retrieve zone-specific harvest reference week
        harvest_week = get_crop_reference_week(state)
        lag_weeks = get_pre_harvest_lag_weeks(harvest_week, lag_window_size=lag_window_size)

        lag_values = []
        is_complete = True
        for wk in lag_weeks:
            val = weekly_map.get((geo_id, year, wk), np.nan)
            if np.isnan(val) or val < 0:
                is_complete = False
                break
            lag_values.append(val)

        if is_complete and len(lag_values) == lag_window_size:
            rec = row.to_dict()
            rec["harvest_reference_week"] = harvest_week
            for col_name, val in zip(LAG_COLUMN_NAMES, lag_values):
                rec[col_name] = round(val, 2)
            records.append(rec)

    lag_df = pd.DataFrame(records)
    return lag_df
