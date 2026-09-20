"""Agro-Climatic Reference Periods and Crop Phenological Calendars for India."""
from typing import Any, Dict, Optional
import pandas as pd

# 15 Planning Commission / ICAR Agro-Climatic Planning Zones
AGRO_CLIMATIC_ZONES = {
    "Western Himalayan Region": {
        "zone_id": 1,
        "states": ["Himachal Pradesh", "Uttarakhand"],
        "kharif_sowing_weeks": (20, 24),    # May - June (early sowing due to cold onset)
        "kharif_flowering_weeks": (33, 37), # August - early September
        "kharif_harvest_week": 41,          # Mid October (Week 41)
        "kharif_harvest_month": "October",
        "description": "High altitude valleys with cooler temperatures; shorter crop duration."
    },
    "Eastern Himalayan Region": {
        "zone_id": 2,
        "states": ["Assam"],
        "kharif_sowing_weeks": (22, 26),    # June
        "kharif_flowering_weeks": (38, 41), # Late September - October
        "kharif_harvest_week": 46,          # Mid November (Week 46 - Sali rice)
        "kharif_harvest_month": "November",
        "description": "Heavy monsoon rainfall, high humidity, Sali winter rice dominant."
    },
    "Lower Gangetic Plain Region": {
        "zone_id": 3,
        "states": ["West Bengal"],
        "kharif_sowing_weeks": (24, 28),    # June - July
        "kharif_flowering_weeks": (39, 42), # Late September - October
        "kharif_harvest_week": 46,          # Mid November (Aman rice)
        "kharif_harvest_month": "November",
        "description": "Alluvial delta plains, high water availability, Aman rice dominant."
    },
    "Middle Gangetic Plain Region": {
        "zone_id": 4,
        "states": ["Bihar"],
        "kharif_sowing_weeks": (24, 28),    # June - July
        "kharif_flowering_weeks": (38, 41), # Late September - October
        "kharif_harvest_week": 45,          # Early November (Week 45)
        "kharif_harvest_month": "November",
        "description": "Fertile alluvial plains subject to flooding; medium duration rice."
    },
    "Upper Gangetic Plain Region": {
        "zone_id": 5,
        "states": ["Uttar Pradesh"],
        "kharif_sowing_weeks": (23, 27),    # June - early July
        "kharif_flowering_weeks": (37, 40), # September
        "kharif_harvest_week": 43,          # Late October (Week 43)
        "kharif_harvest_month": "October",
        "description": "Canal and tube-well irrigated sub-humid plains; wheat rotation."
    },
    "Trans-Gangetic Plain Region": {
        "zone_id": 6,
        "states": ["Punjab", "Haryana"],
        "kharif_sowing_weeks": (21, 25),    # Late May - June
        "kharif_flowering_weeks": (36, 39), # Early September
        "kharif_harvest_week": 42,          # Mid October (Week 42)
        "kharif_harvest_month": "October",
        "description": "Extensively irrigated semi-arid plains; high-yielding mechanized rice."
    },
    "Eastern Plateau and Hill Region": {
        "zone_id": 7,
        "states": ["Chhattisgarh", "Jharkhand"],
        "kharif_sowing_weeks": (24, 28),    # June - July
        "kharif_flowering_weeks": (38, 41), # Late September - October
        "kharif_harvest_week": 45,          # Early November (Week 45)
        "kharif_harvest_month": "November",
        "description": "Red and yellow rainfed soils, rolling topography; 'Rice bowl of central India'."
    },
    "Central Plateau and Hill Region": {
        "zone_id": 8,
        "states": ["Madhya Pradesh", "Rajasthan"],
        "kharif_sowing_weeks": (24, 28),    # June - July
        "kharif_flowering_weeks": (37, 40), # September
        "kharif_harvest_week": 43,          # Late October (Week 43)
        "kharif_harvest_month": "October",
        "description": "Mixed black and red soils, semi-arid plateau; rainfed with canal pockets."
    },
    "Western Plateau and Hill Region": {
        "zone_id": 9,
        "states": ["Maharashtra"],
        "kharif_sowing_weeks": (24, 28),    # June - July
        "kharif_flowering_weeks": (37, 41), # September - early October
        "kharif_harvest_week": 44,          # Late October / early November (Week 44)
        "kharif_harvest_month": "October",
        "description": "Black basaltic soils, drought-prone rainshadow zones."
    },
    "Southern Plateau and Hill Region": {
        "zone_id": 10,
        "states": ["Telangana", "Karnataka"],
        "kharif_sowing_weeks": (25, 29),    # Late June - July
        "kharif_flowering_weeks": (39, 42), # Late September - October
        "kharif_harvest_week": 46,          # Mid November (Week 46)
        "kharif_harvest_month": "November",
        "description": "Deccan plateau, tank and well irrigation; red sandy loam soils."
    },
    "East Coast Plains and Hills Region": {
        "zone_id": 11,
        "states": ["Odisha", "Andhra Pradesh", "Tamil Nadu"],
        "kharif_sowing_weeks": (26, 30),    # July
        "kharif_flowering_weeks": (40, 44), # October
        "kharif_harvest_week": 47,          # Late November (Week 47)
        "kharif_harvest_month": "November",
        "description": "Coastal alluvial deltas prone to NE monsoon and post-monsoon cyclones."
    },
    "West Coast Plains and Ghats Region": {
        "zone_id": 12,
        "states": ["Kerala"],
        "kharif_sowing_weeks": (21, 24),    # Late May - June (early SW monsoon arrival)
        "kharif_flowering_weeks": (34, 38), # August - early September
        "kharif_harvest_week": 41,          # Mid October (Virippu harvest: Week 41)
        "kharif_harvest_month": "October",
        "description": "High rainfall coastal strip and Western Ghats; Virippu (autumn) rice."
    },
    "Gujarat Plains and Hills Region": {
        "zone_id": 13,
        "states": ["Gujarat"],
        "kharif_sowing_weeks": (25, 28),    # Late June - July
        "kharif_flowering_weeks": (38, 41), # Late September - October
        "kharif_harvest_week": 44,          # Late October (Week 44)
        "kharif_harvest_month": "October",
        "description": "Arid to sub-humid plains; canal and tube-well fed rice pockets."
    },
}

# State-to-default-zone mapping for all 20 states in our dataset
STATE_TO_ZONE = {
    "Himachal Pradesh": "Western Himalayan Region",
    "Uttarakhand": "Western Himalayan Region",
    "Assam": "Eastern Himalayan Region",
    "West Bengal": "Lower Gangetic Plain Region",
    "Bihar": "Middle Gangetic Plain Region",
    "Uttar Pradesh": "Upper Gangetic Plain Region",
    "Punjab": "Trans-Gangetic Plain Region",
    "Haryana": "Trans-Gangetic Plain Region",
    "Chhattisgarh": "Eastern Plateau and Hill Region",
    "Jharkhand": "Eastern Plateau and Hill Region",
    "Madhya Pradesh": "Central Plateau and Hill Region",
    "Rajasthan": "Central Plateau and Hill Region",
    "Maharashtra": "Western Plateau and Hill Region",
    "Telangana": "Southern Plateau and Hill Region",
    "Karnataka": "Southern Plateau and Hill Region",
    "Odisha": "East Coast Plains and Hills Region",
    "Andhra Pradesh": "East Coast Plains and Hills Region",
    "Tamil Nadu": "East Coast Plains and Hills Region",
    "Kerala": "West Coast Plains and Ghats Region",
    "Gujarat": "Gujarat Plains and Hills Region",
}


def get_crop_reference_week(state: str, agro_climate: Optional[str] = None) -> int:
    """Retrieve the standard Kharif rice harvest reference week (T_ref) for a given state/zone.
    
    Parameters
    ----------
    state : str
        State name (title case or cleaned).
    agro_climate : Optional[str]
        Optional explicit agro-climatic zone name.
        
    Returns
    -------
    int
        Harvest reference calendar week number (1 to 52).
    """
    clean_state = state.strip().title()

    if agro_climate and agro_climate in AGRO_CLIMATIC_ZONES:
        return AGRO_CLIMATIC_ZONES[agro_climate]["kharif_harvest_week"]

    if clean_state in STATE_TO_ZONE:
        zone = STATE_TO_ZONE[clean_state]
        return AGRO_CLIMATIC_ZONES[zone]["kharif_harvest_week"]

    # Fallback to national median Kharif harvest week (Week 44: late October)
    return 44


def get_pre_harvest_lag_weeks(harvest_week: int, lag_window_size: int = 16) -> list[int]:
    """Compute the exact ordered sequence of calendar weeks strictly preceding harvest.
    
    w_{-16}, w_{-15}, ..., w_{-1} where w_{-1} = harvest_week - 1.
    All weeks are strictly < harvest_week to guarantee zero lookahead leakage.
    """
    # Weeks from (harvest_week - lag_window_size) up to (harvest_week - 1)
    return [harvest_week - lag for lag in range(lag_window_size, 0, -1)]


def build_crop_calendar_dataframe() -> pd.DataFrame:
    """Generate a structured tabular dataframe of India's agro-climatic rice crop calendar."""
    records = []
    for zone_name, zinfo in AGRO_CLIMATIC_ZONES.items():
        harvest_wk = zinfo["kharif_harvest_week"]
        lags = get_pre_harvest_lag_weeks(harvest_wk, lag_window_size=16)
        records.append({
            "Zone_ID": zinfo["zone_id"],
            "Agro_Climatic_Zone": zone_name,
            "Representative_States": ", ".join(zinfo["states"]),
            "Sowing_Weeks": f"Weeks {zinfo['kharif_sowing_weeks'][0]}–{zinfo['kharif_sowing_weeks'][1]}",
            "Flowering_Weeks": f"Weeks {zinfo['kharif_flowering_weeks'][0]}–{zinfo['kharif_flowering_weeks'][1]}",
            "Harvest_Reference_Week": harvest_wk,
            "Harvest_Month": zinfo["kharif_harvest_month"],
            "16_Week_Lag_Window": f"Weeks {lags[0]} to {lags[-1]}",
            "Description": zinfo["description"],
        })
    return pd.DataFrame(records)
