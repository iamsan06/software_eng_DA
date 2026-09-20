# Rainfall-Lag-Aware Crop Yield Forecasting in India Using Explainable Temporal Machine Learning

A research and software engineering implementation of leakage-free machine learning pipelines for district-level crop yield forecasting across India.

---

## 1. Project Architecture Overview

```
SE DA/
├── configs/
│   ├── phase1.yaml                          # Central configuration for Phase 1 baseline
│   └── phase2.yaml                          # Central configuration for Phase 2 & 3 lag models
├── data/
│   ├── raw/
│   │   ├── Data(1).csv                      # Validated India-wide agricultural dataset (10,279 rows)
│   │   └── Data.csv
│   ├── interim/
│   └── processed/
│       └── phase1_clean.csv                 # Cleaned dataset with compound spatial keys
├── docs/
│   ├── data_dictionary.md                   # Complete variable dictionary (types, units, roles)
│   ├── phase1_methodology.md                # Phase 1 mathematical & empirical methodology
│   ├── experiment_protocol.md               # Reproducibility protocol, seeds, hyperparameters
│   ├── phase1_findings.md                   # Executed empirical findings and spatial error breakdown
│   ├── limitations.md                       # Comprehensive discussion of data limitations
│   ├── crop_calendar_reference.md           # ICAR/DES 15 Agro-Climatic Zones phenological windows
│   ├── phase2_lag_methodology.md            # 16-week pre-harvest lag sequence methodology
│   └── sequence_models_architecture.md      # PyTorch LSTM/GRU deep sequence architecture specs
├── notebooks/
│   ├── 01_data_audit.ipynb                  # Executed raw data audit and integrity tests
│   ├── 02_preprocessing_and_eda.ipynb       # Executed cleaning, feature table, and 9 EDA figures
│   ├── 03_baseline_models.ipynb             # Executed RF and XGBoost training, tuning, and evaluation
│   ├── 04_phase1_results.ipynb              # Executed presentation notebook with final results
│   └── 05_crop_calendar_and_lag_spec.ipynb  # Executed crop calendar, lag sequence & LSTM/GRU notebook
├── results/
│   ├── figures/                             # 13 high-resolution scientific plots (300 DPI)
│   ├── models/                              # Trained joblib binaries (Random Forest, XGBoost)
│   ├── predictions/                         # District-level test set prediction CSVs
│   ├── reports/                             # Pytest test results and environment specification
│   └── tables/                              # 10 comprehensive metric and audit CSV tables
├── src/
│   ├── __init__.py
│   ├── config.py                            # YAML config parser and path validator
│   ├── data_loader.py                       # Encoding-safe raw CSV loader
│   ├── validation.py                        # Strict schema, range, and zero-leakage assertions
│   ├── preprocessing.py                     # Data cleaning and stable geographic key resolution
│   ├── splitting.py                         # Strict chronological splitting (Train/Val/Test)
│   ├── models.py                            # Baseline Random Forest and XGBoost pipelines
│   ├── evaluation.py                        # MAE, RMSE, R2, and disaggregated spatial error metrics
│   ├── visualization.py                     # High-resolution scientific visualization routines
│   ├── crop_calendar.py                     # Official 15 Agro-Climatic Zone phenological calendars
│   ├── spatial_aggregation.py               # Gridded-to-district daily/weekly spatial aggregator
│   ├── lag_construction.py                  # Causal 16-week pre-harvest lag extraction (w_-16 to w_-1)
│   ├── lag_models.py                        # Lag-aware Random Forest & XGBoost tabular pipelines
│   └── sequence_models.py                   # PyTorch LSTM and GRU neural networks and trainer
├── tests/
│   ├── test_data_loader.py                  # Loader integrity tests
│   ├── test_validation.py                   # Leakage and key integrity tests
│   ├── test_preprocessing.py                # Cleaning and stable key tests
│   ├── test_splitting.py                    # Chronological causality tests
│   ├── test_evaluation.py                   # Metric computation tests
│   ├── test_crop_calendar.py                # Zonal calendar & week causality tests
│   ├── test_lag_construction.py             # 16-week lag extraction tests
│   └── test_sequence_models.py              # PyTorch LSTM/GRU tensor and forward pass tests
├── scripts/
│   ├── build_and_run_notebooks.py           # Programmatic builder for notebooks 01-04
│   └── build_and_run_phase2_nb.py           # Programmatic builder for notebook 05
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## 2. Phase 1 Executed Empirical Benchmark

Evaluated on the held-out chronological test partition (**2014–2017**, $N = 1,470$ observations) after training on **1990–2009** ($N = 7,361$) and tuning on **2010–2013** ($N = 1,448$):

| Model | Split | Observations | MAE ($\text{kg/ha}$) | RMSE ($\text{kg/ha}$) | $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Train (1990–2009) | 7,361 | 276.574 | 370.373 | 0.8351 |
| **Random Forest** | Validation (2010–2013) | 1,448 | 606.299 | 774.581 | 0.3386 |
| **Random Forest** | **Test (2014–2017)** | **1,470** | **727.194** | **914.197** | **0.1333** |
| **XGBoost** | Train (1990–2009) | 7,361 | 311.077 | 415.272 | 0.7926 |
| **XGBoost** | Validation (2010–2013) | 1,448 | 601.744 | 772.979 | 0.3414 |
| **XGBoost** | **Test (2014–2017)** | **1,470** | **732.229** | **921.292** | **0.1198** |

---

## 3. Phase 2 & Phase 3 Specifications

1. **Agro-Climatic Phenological Anchoring (`src/crop_calendar.py`):**
   - Official sowing, flowering, and harvest windows mapped across all 15 Agro-Climatic Planning Zones.
   - Deterministic harvest reference week $T_{\text{ref}}$ (e.g., Week 41 in Kerala, Week 42 in Punjab, Week 43 in UP, Week 46 in West Bengal).
2. **Causal 16-Week Pre-Harvest Lag Sequence (`src/lag_construction.py`):**
   - Extracts cumulative weekly rainfall for weeks $(T_{\text{ref}} - 16)$ through $(T_{\text{ref}} - 1)$.
   - Strict guarantee: $w_{-1} = T_{\text{ref}} - 1 < T_{\text{ref}}$ (zero lookahead leakage).
3. **Lag-Aware Tabular ML (`src/lag_models.py`):**
   - Lag-Aware Random Forest and XGBoost with empirical rainfall-response profiling.
4. **Deep Sequential Learning (`src/sequence_models.py`):**
   - PyTorch `RiceLSTM` (2-layer LSTM) and `RiceGRU` (2-layer GRU) with multi-modal tensor fusion (16-week temporal weather sequence + static spatial fixed effects).
   - `SequenceTrainer` with AdamW, gradient clipping, and validation early stopping.

---

## 4. Environment & Reproduction

### Setup Virtual Environment:
```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
```

### Run Full Test Suite (24 Tests):
```bash
.\.venv\Scripts\pytest -v
```

### Re-execute All Notebooks:
```bash
.\.venv\Scripts\python.exe scripts/build_and_run_notebooks.py
.\.venv\Scripts\python.exe scripts/build_and_run_phase2_nb.py
```
