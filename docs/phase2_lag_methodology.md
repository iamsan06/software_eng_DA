# Phase 2 Methodology: 16-Week Pre-Harvest Rainfall Lag Modeling

**Project:** Rainfall-Lag-Aware Crop Yield Forecasting in India Using Explainable Temporal Machine Learning  
**Phase:** 2 (Lag-Aware Representation & Tabular Models)

---

## 1. Core Hypothesis & Experimental Objective

In Phase 1, we established that aggregate seasonal rainfall (`IMD_RF`) and temperature (`IMD_Tmax`) achieve modest test performance ($R^2 \approx 0.12–0.13, \text{MAE} \approx 727–732\text{ kg/ha}$) because aggregate scalars cannot discern the temporal distribution of rainfall across the growing season.

Phase 2 tests the central research question:
> *"Can district-level rice yield in India be predicted more accurately by representing rainfall as a temporally ordered sequence of pre-harvest observations ($w_{-16}$ to $w_{-1}$) rather than an aggregate seasonal variable, and which pre-harvest rainfall periods are most influential to the prediction?"*

---

## 2. Experimental Design: Baseline vs. Lag-Aware Comparison

To isolate the predictive value of rainfall timing without confounding it with model architecture:

$$\begin{aligned}
\text{Experiment A (Phase 1 Baseline):} \quad & \hat{y} = f(\text{IMD\_RF}, \text{IMD\_Tmax}, \text{SD\_RF}, \text{State}) \\
\text{Experiment B (Phase 2 Lag-Aware):} \quad & \hat{y} = f(w_{-16}, w_{-15}, \dots, w_{-1}, \text{IMD\_Tmax}, \text{State})
\end{aligned}$$

Both experiments use the exact same algorithm classes (`RandomForestRegressor`, `XGBRegressor`), identical chronological splits (Train: 1990–2009, Val: 2010–2013, Test: 2014–2017), and identical spatial fixed effects.

---

## 3. Pre-Harvest Lag Sequence Extraction

1. **Agro-Climatic Temporal Anchoring:**
   Each district observation is mapped to its zone-specific Kharif harvest reference week $T_{\text{ref}} \in [41, 47]$ via `src.crop_calendar`.
2. **Backward-Looking Extraction Window:**
   The 16-week pre-harvest vector $[w_{-16}, w_{-15}, \dots, w_{-1}]$ spans:
   - Early vegetative / transplanting: $w_{-16}$ to $w_{-13}$
   - Active tillering: $w_{-12}$ to $w_{-8}$
   - Panicle initiation: $w_{-7}$ to $w_{-4}$
   - Flowering / Anthesis: $w_{-3}$ to $w_{-2}$
   - Grain filling / Pre-harvest drainage: $w_{-1}$
3. **Leakage Safeguards:**
   - Week $-1$ ends strictly before harvest commencement.
   - Future meteorological observations are strictly excluded.
   - Temporal split boundaries are preserved without cross-contamination.

---

## 4. Explainability & Rainfall-Response Profiles (SHAP)

For trained tree-based models, feature importances and SHAP values are extracted across the 16 weekly lag dimensions:
$$I(w_{-k}) = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \left| \phi_k^{(i)} \right| \quad \text{for } k \in \{1, 2, \dots, 16\}$$
Plotting $I(w_{-k})$ against week index $-k$ generates the empirical **Rainfall-Response Profile**, pinpointing whether moisture anomalies during flowering ($w_{-3}, w_{-2}$) exert higher predictive impact than early vegetative rainfall ($w_{-16}, w_{-15}$).
