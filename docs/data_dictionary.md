# Phase 1 Data Dictionary: India-wide Rice Yield & Climate Dataset

This document defines every variable present in the raw source dataset (`Data.csv` / `Data(1).csv`) and processed pipeline artifacts (`data/processed/phase1_clean.csv`).

---

## 1. Raw Dataset Schema

The source dataset contains 10,279 district-year observations spanning 1990 to 2017 across 20 Indian states and 381 district names (384 distinct district codes).

| Variable Name | Raw Type | Clean Type | Unit | Role | Definition & Research Consideration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Dist.Code` | `int64` | `int64` | Dimensionless ID | Spatial Identifier | Unique district code assigned in the historical database. Globally unique per district across all states. **Excluded from numeric predictive features** to avoid pseudo-numerical spatial ranking. |
| `Sh_Dist_code` | `int64` | `int64` | Dimensionless ID | GIS Identifier | Shapefile district code. Redundant with `Dist.Code`. **Excluded from feature set**. |
| `FID_1` | `int64` | `int64` | Dimensionless ID | GIS Index | Feature identifier from GIS database export. Arbitrary integer. **Excluded from feature set**. |
| `State` | `object` | `string` | Categorical | Spatial Baseline | State name. Normalized to Title Case (`State_Clean`). One-hot encoded across 20 states to capture regional agro-climatic, soil, and agricultural policy fixed effects without target leakage. |
| `AgroClimate` | `object` | `string` | Categorical | Metadata | Broad agro-climatic zone classification. Contains 168 missing values. **Excluded from baseline predictor set** due to missingness and redundancy with state-level fixed effects. |
| `District` | `object` | `string` | Categorical | Spatial Entity | District name. Standardized to Title Case (`District_Clean`). *Critical caveat:* 3 district names (`Aurangabad`, `Bilaspur`, `Hamirpur`) exist in multiple states with different `Dist.Code`s. |
| `Year` | `int64` | `int64` | Calendar Year | Temporal Split Key | Crop observation year (1990–2017). Reserved strictly for chronological partitioning (Train: 1990–2009, Validation: 2010–2013, Test: 2014–2017). Never passed as a numerical predictor. |
| `RICE.AREA..1000.ha.` | `float64` | `float64` | Thousand Hectares ($10^3\text{ ha}$) | Descriptive Metadata | Cultivated area under rice. **Excluded from predictive features** to prevent post-planting and structural leakage into the pre-harvest forecasting baseline. |
| `RICE.PRODUCTION..1000.tons.` | `float64` | `float64` | Thousand Metric Tons ($10^3\text{ tons}$) | Descriptive Metadata | Total rice production. **Strictly excluded from predictive features** because $\text{Yield} = \frac{\text{Production}}{\text{Area}}$, which represents direct mathematical target leakage. |
| `RICE.YIELD..Kg.per.ha.` | `float64` | `float64` | Kilograms per Hectare ($\text{kg/ha}$) | **Primary Target** ($y$) | Primary crop productivity target. Range: $44.0$ to $6316.0\text{ kg/ha}$; Mean: $1936.30\text{ kg/ha}$; Median: $1874.00\text{ kg/ha}$. Zero negative or zero observations. |
| `RICE.SHARE.IN.TOTAL..Percent.` | `float64` | `float64` | Percentage ($\%$) | Descriptive Metadata | Proportion of rice in total gross cropped area. Contains 758 missing values ($7.37\%$). **Excluded from predictor set**. |
| `Crops` | `object` | `string` | Categorical | Metadata | Crop type identifier. Uniformly `"Rice"` across all 10,279 rows. Invariant metadata. |
| `IMD_RF` | `float64` | `float64` | Millimeters ($\text{mm}$) | **Numerical Predictor** ($X_1$) | District aggregate rainfall from the India Meteorological Department (IMD). Continuous positive variable. Min: $28.29\text{ mm}$, Max: $5229.08\text{ mm}$, Mean: $1026.34\text{ mm}$. |
| `IMD_Tmax` | `float64` | `float64` | Degrees Celsius ($^\circ\text{C}$) | **Numerical Predictor** ($X_2$) | Mean seasonal/annual maximum temperature from IMD. Continuous variable. Min: $24.35^\circ\text{C}$, Max: $37.28^\circ\text{C}$, Mean: $31.67^\circ\text{C}$. |
| `SD_RF_jun_nov` | `float64` | `float64` | Millimeters ($\text{mm}$) | **Numerical Predictor** ($X_3$) | Standard deviation of rainfall during the primary Kharif monsoon season (June to November). Captures intra-seasonal rainfall variability and distribution shock. Min: $0.92\text{ mm}$, Max: $51.26\text{ mm}$, Mean: $12.66\text{ mm}$. |

---

## 2. Derived Pipeline Variables

| Variable Name | Clean Type | Source / Construction | Definition & Purpose |
| :--- | :--- | :--- | :--- |
| `stable_geographic_id` | `string` | `f"{State_Clean.upper()}_{Dist.Code}"` | Unambiguous geographic identifier resolving multi-state duplicate district names. Examples: `BIHAR_920`, `MAHARASHTRA_108`, `HIMACHAL_PRADESH_18`, `CHHATTISGARH_12`. Uniqueness verified: 0 collisions with `Year`. |
| `State_Clean` | `string` | `State.str.strip().str.title()` | Standardized state name eliminating casing discrepancies (e.g. `ANDHRA PRADESH` vs `Chhattisgarh`). |
| `District_Clean` | `string` | `District.str.strip().str.title()` | Standardized district name eliminating whitespace inconsistencies. |
| `Residual` | `float64` | $\text{Actual} - \text{Predicted}$ | Signed prediction residual on the test set ($\text{kg/ha}$). Positive values indicate model underestimation; negative values indicate overestimation. |

---

## 3. Variable Exclusions and Leakage Prevention

- **Target Leakage Prohibition:** `RICE.PRODUCTION..1000.tons.`, `RICE.AREA..1000.ha.`, and `RICE.SHARE.IN.TOTAL..Percent.` are strictly excluded from all model feature matrices.
- **Geographic Numerical Fallacy:** Numeric database keys (`Dist.Code`, `Sh_Dist_code`, `FID_1`) are strictly isolated from model features.
- **Absence of Synthetic Lags:** No weekly lags (`Rain_Lag_1` ... `Rain_Lag_16`) were manufactured or imputed from aggregate variables.
