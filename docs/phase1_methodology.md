# Phase 1 Methodology: India-wide Baseline Crop Yield Forecasting

**Project:** Rainfall-Lag-Aware Crop Yield Forecasting in India Using Explainable Temporal Machine Learning  
**Lead Research / Software Engineer Report**  
**Phase:** 1 (Baseline Empirical Implementation)

---

## 1. Research Question

Phase 1 investigates the following foundational empirical question:

> *"Can historical district-level rice yield in India be predicted from the available aggregate rainfall and temperature variables using a leakage-safe chronological modelling pipeline, and what baseline performance can be established before introducing temporally resolved rainfall lags?"*

Establishing this baseline is scientifically indispensable. In agricultural machine learning literature, sequence models or complex lag architectures are often claimed to improve yield prediction without first establishing whether an aggregate climate baseline already captures the underlying signal. Phase 1 provides this leak-free benchmark on actual historical data across India.

---

## 2. Dataset Scope and Integrity

The empirical dataset consists of 10,279 district-year observations covering 28 consecutive crop seasons from **1990 to 2017** across 20 Indian states and 381 district names (384 distinct district codes).

### 2.1 Geographic Transition
While the preliminary proposal was framed around Kerala, the actual validated multi-year dataset encompasses the entirety of India's major rice-growing states (e.g. Uttar Pradesh, West Bengal, Punjab, Bihar, Odisha, Andhra Pradesh, Tamil Nadu, Kerala, Assam, Chhattisgarh, Maharashtra, etc.). The study scope is therefore explicitly India-wide.

### 2.2 Data Integrity
- **Raw File Encoding:** The source CSV is encoded in `latin1` due to non-breaking spaces ($0\text{xA0}$) present in header strings.
- **Completeness:** The primary target (`RICE.YIELD..Kg.per.ha.`) and meteorological variables (`IMD_RF`, `IMD_Tmax`, `SD_RF_jun_nov`) contain **0 missing values** across all 10,279 observations.
- **Validity:** All yields and rainfall figures are strictly positive ($\text{Yield} \ge 44.0\text{ kg/ha}$, $\text{Rainfall} \ge 28.29\text{ mm}$).

---

## 3. Geographic Key Integrity & Disambiguation

A naive assumption that `(District, Year)` constitutes a unique primary key fails in India due to shared historical district names across state boundaries:
1. `Aurangabad`: Exists in **Bihar** (`Dist.Code = 920`) and **Maharashtra** (`Dist.Code = 108`).
2. `Bilaspur`: Exists in **Chhattisgarh** (`Dist.Code = 12`) and **Himachal Pradesh** (`Dist.Code = 18`).
3. `Hamirpur`: Exists in **Uttar Pradesh** (`Dist.Code = 138`) and **Himachal Pradesh** (`Dist.Code = 20`).

Evaluating duplicates on `(District, Year)` produces **76 collisions**. Dropping these records would discard genuine agricultural data.

### Resolution:
`Dist.Code` is globally unique per administrative district. We construct a composite stable geographic key:
$$\text{stable\_geographic\_id} = \text{State\_Clean} \mathbin{\Vert} \text{"\_"} \mathbin{\Vert} \text{Dist.Code}$$
Example: `BIHAR_920` vs `MAHARASHTRA_108`.  
Verifying uniqueness on `(stable_geographic_id, Year)` yields **0 duplicates** across all 10,279 rows.

---

## 4. Preprocessing Protocol

Preprocessing follows a strict, reproducible sequence:
1. **Raw Ingestion:** Ingest raw CSV without mutating source files.
2. **Exact Deduplication Check:** Verifies 0 duplicate rows exist.
3. **Type Casting:** Cast `Year`, `Dist.Code` to integers; yield and weather variables to IEEE 754 float64.
4. **Text Standardization:** Strip whitespace and convert `State` and `District` to Title Case (`State_Clean`, `District_Clean`).
5. **Validation Assertions:** Confirm non-negative yield and rainfall via automated assertions.
6. **Audit Accounting:** Record filter statistics in `results/tables/preprocessing_summary.csv`. All 10,279 raw rows are retained without silent drops.

---

## 5. Feature Selection and Leakage Prevention

The feature space is strictly partitioned into predictors and excluded variables:

```
Candidate Feature Pool
├── Included Numerical Predictors
│   ├── IMD_RF (Aggregate Rainfall, mm)
│   ├── IMD_Tmax (Mean Max Temperature, °C)
│   └── SD_RF_jun_nov (Monsoon Rainfall Std Dev, mm)
├── Included Categorical Predictor
│   └── State_Clean (One-Hot Encoded Spatial Fixed Effects)
└── Strictly Excluded Variables
    ├── Target: RICE.YIELD..Kg.per.ha.
    ├── Direct Target Leakage: RICE.PRODUCTION..1000.tons., RICE.AREA..1000.ha.
    ├── High Missingness: RICE.SHARE.IN.TOTAL..Percent., AgroClimate
    ├── Pseudo-Numeric IDs: FID_1, Dist.Code, Sh_Dist_code
    └── Split Key: Year
```

### Feature Inclusion Matrix:
| Feature | Unit | Role | Included? | Justification |
| :--- | :--- | :--- | :--- | :--- |
| `RICE.YIELD..Kg.per.ha.` | kg/ha | Target | **YES (Target)** | Primary forecasting variable |
| `IMD_RF` | mm | Predictor | **YES (Feature)** | Aggregate rainfall available in raw data |
| `IMD_Tmax` | °C | Predictor | **YES (Feature)** | Thermal stress indicator |
| `SD_RF_jun_nov` | mm | Predictor | **YES (Feature)** | Intra-seasonal monsoon rainfall variability |
| `State_Clean` | categorical | Predictor | **YES (Feature)** | One-hot encoded state baseline (encoder fit only on train) |
| `RICE.PRODUCTION..1000.tons.` | 1000 tons | Descriptive | **NO (Leakage)** | $\text{Yield} = \text{Prod}/\text{Area}$ (tautological leakage) |
| `RICE.AREA..1000.ha.` | 1000 ha | Descriptive | **NO (Leakage)** | Sowing / post-planting area |
| `Dist.Code`, `FID_1` | integer | Identifier | **NO (Index)** | Preventing false continuous distance assumption |
| `Rain_Lag_1` ... `16` | mm | Lag Features | **FORBIDDEN** | Not present in raw dataset; synthetic lags prohibited |

---

## 6. Chronological Temporal Split

Agricultural forecasting is inherently sequential. Random cross-validation (e.g. `KFold` or `train_test_split`) leaks future temporal weather and macroeconomic trends into past predictions, producing spuriously optimistic evaluation metrics.

We enforce a strict chronological split across distinct calendar years:
- **Training Period:** 1990–2009 (20 crop seasons, $\approx 71.6\%$ of observations, $n = 7,361$)
- **Validation Period:** 2010–2013 (4 crop seasons, $\approx 14.1\%$ of observations, $n = 1,448$)
- **Testing Period:** 2014–2017 (4 crop seasons, $\approx 14.3\%$ of observations, $n = 1,470$)

### Causality Assertions:
$$\max(\text{Train Years}) = 2009 < \min(\text{Validation Years}) = 2010$$
$$\max(\text{Validation Years}) = 2013 < \min(\text{Test Years}) = 2014$$
$$\text{Train} \cap \text{Validation} \cap \text{Test} = \emptyset$$

The test partition remained completely untouched during model architecture selection and hyperparameter tuning.

---

## 7. Baseline Machine Learning Models

Two canonical non-linear tree-based architectures were implemented and benchmarked:

### 7.1 Random Forest Regressor
- Ensemble of decorrelated bagged decision trees.
- Hyperparameter tuning over tree depth ($d \in \{8, 12, 16, 20\}$) using validation MAE.
- Selected configuration: `n_estimators=200`, `max_depth=16`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`.

### 7.2 XGBoost Regressor
- Scalable gradient-boosted decision tree algorithm minimizing squared error loss with $L_1$ and $L_2$ regularization.
- Validation tuning over depth and learning rate ($d \in \{4, 6\}$, $\eta \in \{0.03, 0.05\}$).
- Selected configuration: `n_estimators=250`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`.

---

## 8. Evaluation Metrics

Model performance is quantified using three standard regression metrics:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

Metrics are reported across Train, Validation, and Test splits, with the primary benchmark established on the held-out Test split.
