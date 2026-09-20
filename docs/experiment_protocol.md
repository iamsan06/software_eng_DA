# Phase 1 Experiment Protocol & Reproducibility Standard

**Project:** Rainfall-Lag-Aware Crop Yield Forecasting in India Using Explainable Temporal Machine Learning  
**Protocol Version:** 1.0 (Phase 1 Baseline)  
**Date:** September 2026

---

## 1. Dataset Specification

- **Primary Source File:** `data/raw/Data(1).csv` (mirrored at `data/raw/Data.csv`)
- **Encoding:** `latin1` (or `cp1252`)
- **Row Count:** 10,279 rows (header + 10,279 observations)
- **Temporal Span:** 1990 to 2017 (28 years)
- **Spatial Coverage:** 20 States, 381 Districts (384 unique `Dist.Code` values)
- **Processed Artifact:** `data/processed/phase1_clean.csv`

---

## 2. Experimental Random Seed & Determinism

- **Global Seed:** `42`
- Applied across:
  - Python internal hash / RNG
  - NumPy RNG (`np.random.seed(42)`)
  - Scikit-Learn `RandomForestRegressor(random_state=42)`
  - XGBoost `XGBRegressor(random_state=42)`

---

## 3. Chronological Partitioning Rules

Partitioning is strictly temporal, sorting records by calendar crop year:

$$\begin{aligned}
\mathcal{D}_{\text{train}} &= \{ (x_i, y_i) \mid \text{Year}_i \in [1990, 2009] \} \quad (n = 7,361; 71.61\%) \\
\mathcal{D}_{\text{val}}   &= \{ (x_i, y_i) \mid \text{Year}_i \in [2010, 2013] \} \quad (n = 1,448; 14.09\%) \\
\mathcal{D}_{\text{test}}  &= \{ (x_i, y_i) \mid \text{Year}_i \in [2014, 2017] \} \quad (n = 1,470; 14.30\%)
\end{aligned}$$

No shuffling or random splitting is permitted on the temporal axis.

---

## 4. Feature Specification & Transformations

### Numerical Features ($p = 3$):
1. `IMD_RF` (continuous aggregate rainfall, mm)
2. `IMD_Tmax` (continuous mean maximum temperature, °C)
3. `SD_RF_jun_nov` (continuous June–November rainfall std dev, mm)

### Categorical Feature ($k = 20$):
1. `State_Clean` (one-hot encoded via `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)`)

### Leakage Guard:
The `OneHotEncoder` is fitted **exclusively on $\mathcal{D}_{\text{train}}$** and subsequently transforms $\mathcal{D}_{\text{val}}$ and $\mathcal{D}_{\text{test}}$. Total input matrix width: $3 + 20 = 23$ columns.

---

## 5. Model Configurations & Validation Tuning

### 5.1 Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
- Base parameters: `n_estimators=200`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`, `random_state=42`, `n_jobs=-1`
- Tuning candidate set: `max_depth` $\in \{8, 12, 16, 20\}$
- Criterion: Minimum Validation MAE on $\mathcal{D}_{\text{val}}$
- Selected parameter: `max_depth=16`
- Artifact: `results/models/random_forest.joblib`

### 5.2 XGBoost Regressor (`xgboost.XGBRegressor`)
- Base parameters: `n_estimators=250`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, `random_state=42`, `n_jobs=-1`
- Tuning candidates:
  - `max_depth=4`, `learning_rate=0.05`
  - `max_depth=6`, `learning_rate=0.05`
  - `max_depth=6`, `learning_rate=0.03`
- Criterion: Minimum Validation MAE on $\mathcal{D}_{\text{val}}$ with early stopping monitoring on validation loss
- Selected configuration: `max_depth=4`, `learning_rate=0.05`
- Artifact: `results/models/xgboost.joblib`

---

## 6. Software Environment & Execution Dependencies

Recorded in `results/reports/environment.txt`:
- **Python:** 3.12.4 (`MSC v.1940 64 bit AMD64`) on `win32`
- **pandas:** `3.0.6`
- **numpy:** `2.5.3`
- **scikit-learn:** `1.9.1`
- **xgboost:** `3.4.1`
- **matplotlib:** `3.11.2`
- **seaborn:** `0.13.2`
- **pyyaml:** `6.0.3`
- **pytest:** `9.1.1`
- **joblib:** `1.6.0`

---

## 7. Execution Commands for Exact Reproduction

```bash
# 1. Run unit test suite
pytest -v

# 2. Run end-to-end pipeline and execute all 4 notebooks
python scripts/build_and_run_notebooks.py
```
All artifacts, figures, predictions, and metric tables are generated deterministically.
