"""Spatial aggregation engine for mapping gridded and station precipitation to district boundaries."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DistrictSpatialAggregator:
    """Aggregates point, station, or gridded meteorological observations to district entities."""

    def __init__(self, district_id_col: str = "stable_geographic_id"):
        self.district_id_col = district_id_col

    def aggregate_daily_to_weekly(
        self,
        daily_df: pd.DataFrame,
        date_col: str = "date",
        rainfall_col: str = "rainfall_mm",
    ) -> pd.DataFrame:
        """Convert daily district-level rainfall to ISO calendar weekly cumulative precipitation.
        
        Parameters
        ----------
        daily_df : pd.DataFrame
            DataFrame containing district identifier, date, and daily rainfall in mm.
            
        Returns
        -------
        pd.DataFrame
            Weekly cumulative rainfall with columns: [district_id, year, iso_week, cumulative_rainfall_mm].
        """
        df = daily_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["year"] = df[date_col].dt.isocalendar().year.astype(int)
        df["iso_week"] = df[date_col].dt.isocalendar().week.astype(int)

        # Aggregate sum of rainfall over each 7-day ISO calendar week
        weekly_df = (
            df.groupby([self.district_id_col, "year", "iso_week"])[rainfall_col]
            .agg(cumulative_rainfall_mm="sum", n_days="count")
            .reset_index()
        )
        return weekly_df

    def aggregate_gridded_gridpoints(
        self,
        grid_df: pd.DataFrame,
        district_weights: Dict[str, Dict[Tuple[float, float], float]],
        lat_col: str = "lat",
        lon_col: str = "lon",
        rainfall_col: str = "rainfall_mm",
    ) -> pd.DataFrame:
        """Area-weighted spatial aggregation of regular lat/lon grid points into district values."""
        results = []
        for dist_id, weights in district_weights.items():
            total_weight = sum(weights.values())
            dist_records = []
            for (lat, lon), w in weights.items():
                sub = grid_df[(grid_df[lat_col] == lat) & (grid_df[lon_col] == lon)].copy()
                sub["weighted_rf"] = sub[rainfall_col] * (w / total_weight)
                dist_records.append(sub)
            if dist_records:
                combined = pd.concat(dist_records)
                agg = combined.groupby("date")["weighted_rf"].sum().reset_index()
                agg[self.district_id_col] = dist_id
                results.append(agg)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
