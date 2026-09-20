# Phase 1 Limitations & Scientific Constraints

This document explicitly delineates the data limitations, structural constraints, and methodological boundaries of the Phase 1 baseline study.

---

## 1. Aggregate Meteorological Limitation (No Weekly/Daily Series)

The primary constraint of the supplied dataset (`Data.csv` / `Data(1).csv`) is that rainfall (`IMD_RF`, `SD_RF_jun_nov`) and temperature (`IMD_Tmax`) are provided strictly as aggregate seasonal/annual scalars rather than continuous time series.

### Research Implication:
- **Untested 16-Week Rainfall-Lag Hypothesis:** The central hypothesis proposed in the project proposal—that representing rainfall as a 16-week pre-harvest temporal sequence outperforms aggregate seasonal totals—**cannot be tested using this dataset alone**.
- **No Synthetic Lag Generation:** In adherence to non-negotiable research rules, Phase 1 strictly forbids manufacturing synthetic weekly lags (`Rain_Lag_1` ... `Rain_Lag_16`) from aggregate totals. Such fabrication would produce artificial autocorrelation and spurious machine learning features.
- **Requirement for Next Phase:** Testing the rainfall-lag hypothesis requires an external, genuinely time-resolved meteorological source (e.g. IMD 0.25° daily gridded rainfall or ERA5 reanalysis).

---

## 2. Lack of Phenological Reference and Harvest Dates

Crop growth is governed by phenology rather than calendar years:
- Rice cultivation in India occurs across multiple seasons: **Kharif** (monsoon-sown, autumn/winter harvested), **Rabi** (winter-sown, spring harvested), and **Zaid/Summer**.
- The raw dataset reports an annual aggregate yield per district without specifying crop season, sowing date, or harvest date.
- Consequently, constructing pre-harvest lag sequences without a documented crop calendar risks misaligning meteorological observations with the true vegetative and reproductive windows.

---

## 3. District Naming, Boundary Shifts, and Key Integrity

Historical spatial analysis in India presents significant administrative boundary complications:
1. **Multi-State Homonymous Districts:** District names such as `Aurangabad` (Bihar vs. Maharashtra), `Bilaspur` (Chhattisgarh vs. Himachal Pradesh), and `Hamirpur` (Uttar Pradesh vs. Himachal Pradesh) collide when using naive `(District, Year)` keys. While resolved in this study using `State_DistCode`, external spatial joins must account for these duplicates.
2. **District Bifurcations (1990–2017):** Between 1990 and 2017, numerous Indian districts were split into multiple smaller administrative units (e.g., in newly formed states like Chhattisgarh, Jharkhand, and Telangana). In the raw dataset, older district codes aggregate certain child districts, introducing unmodeled spatial boundary shifts over the 28-year horizon.

---

## 4. Missingness and Excluded Metadata

- `AgroClimate` contains **168 missing values** ($1.63\%$). Rather than imputing arbitrary climatic zones, state-level one-hot fixed effects were utilized.
- `RICE.SHARE.IN.TOTAL..Percent.` contains **758 missing values** ($7.37\%$) and reflects post-sowing acreage allocations, necessitating exclusion from the pre-harvest predictive baseline.
- No soil characteristics, fertilizer application rates, or irrigation percentages are present in the dataset.

---

## 5. Temporal Generalization and Climate Shift

- As demonstrated by the drop in test performance ($R^2 \approx 0.12–0.13$ on 2014–2017), historical models trained on 1990–2009 struggle to generalize to recent periods without time-resolved meteorological data and non-climatic trend controls (e.g., technological yield gain, high-yielding variety adoption).
- SHAP or feature importance analysis on aggregate variables explains only aggregate statistical association; it does **not** establish causal mechanisms between weather anomalies and biological yield reductions.
