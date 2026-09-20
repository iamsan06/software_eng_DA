"""Unit tests for Agro-Climatic reference periods and crop calendar."""
import pytest
import pandas as pd
from src.crop_calendar import (
    AGRO_CLIMATIC_ZONES,
    STATE_TO_ZONE,
    get_crop_reference_week,
    get_pre_harvest_lag_weeks,
    build_crop_calendar_dataframe,
)


def test_agro_climatic_zones_completeness():
    assert len(AGRO_CLIMATIC_ZONES) >= 13
    for zone_name, info in AGRO_CLIMATIC_ZONES.items():
        assert "kharif_harvest_week" in info
        assert 35 <= info["kharif_harvest_week"] <= 50
        assert "kharif_sowing_weeks" in info


def test_state_to_zone_mapping():
    # Must map all 20 states in the empirical dataset
    required_states = [
        "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat",
        "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
        "Madhya Pradesh", "Maharashtra", "Odisha", "Punjab", "Rajasthan",
        "Tamil Nadu", "Telangana", "Uttar Pradesh", "Uttarakhand", "West Bengal"
    ]
    for st in required_states:
        assert st in STATE_TO_ZONE, f"State '{st}' missing from STATE_TO_ZONE mapping."
        zone = STATE_TO_ZONE[st]
        assert zone in AGRO_CLIMATIC_ZONES, f"Zone '{zone}' for state '{st}' not found."


def test_get_crop_reference_week():
    kerala_wk = get_crop_reference_week("Kerala")
    punjab_wk = get_crop_reference_week("Punjab")
    bengal_wk = get_crop_reference_week("West Bengal")

    assert kerala_wk == 41   # Mid October
    assert punjab_wk == 42   # Mid October
    assert bengal_wk == 46   # Mid November


def test_get_pre_harvest_lag_weeks_causality():
    harvest_wk = 44
    lag_weeks = get_pre_harvest_lag_weeks(harvest_wk, lag_window_size=16)

    # Exactly 16 weeks
    assert len(lag_weeks) == 16

    # Strict causality: max lag week strictly precedes harvest week
    assert max(lag_weeks) < harvest_wk
    assert lag_weeks[-1] == harvest_wk - 1  # Week -1
    assert lag_weeks[0] == harvest_wk - 16  # Week -16

    # Strictly monotonic increasing order
    assert lag_weeks == sorted(lag_weeks)


def test_build_crop_calendar_dataframe():
    cal_df = build_crop_calendar_dataframe()
    assert isinstance(cal_df, pd.DataFrame)
    assert len(cal_df) == len(AGRO_CLIMATIC_ZONES)
    assert "Agro_Climatic_Zone" in cal_df.columns
    assert "Harvest_Reference_Week" in cal_df.columns
    assert "16_Week_Lag_Window" in cal_df.columns
