# Agro-Climatic Reference Periods & Rice Crop Calendar for India

**Document Version:** 1.0 (Phase 2 Specification)  
**Standard Basis:** Directorate of Economics and Statistics (DES), Ministry of Agriculture & Farmers Welfare, Government of India, and Indian Council of Agricultural Research (ICAR).

---

## 1. Context and Agronomic Rationale

Rice (*Oryza sativa*) cultivation across India is characterized by diverse microclimates, monsoon arrival timelines, and water regimes. In agricultural machine learning, arbitrary assumptions regarding sowing or harvest reference points introduce severe temporal misalignment between weather anomalies and biological crop vulnerability.

To construct defensible 16-week pre-harvest rainfall sequences ($w_{-16}$ to $w_{-1}$) withoutlookahead leakage, this document formalizes the phenological calendar across India's **15 Agro-Climatic Regional Planning Zones**.

---

## 2. Phenological Stages of Rice

A typical medium-duration rice cultivar requires approximately 120–140 days (17–20 weeks) from nursery sowing to grain maturity:

| Phase | Duration | Phenological Stage | Water Sensitivity & Critical Climate Impacts |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Weeks $-16$ to $-13$ | Sowing, Nursery & Transplanting | Crucial for standing water puddle preparation and seedling establishment; delayed monsoon stalls transplanting. |
| **Stage 2** | Weeks $-12$ to $-8$ | Tillering & Vegetative Growth | Determines productive tiller count; moderate rainfall required; drought induces root stunting. |
| **Stage 3** | Weeks $-7$ to $-4$ | Panicle Initiation & Stem Elongation | Critical phase; moisture stress severely impairs panicle size and spikelet count per panicle. |
| **Stage 4** | Weeks $-3$ to $-2$ | Flowering & Anthesis | **Peak vulnerability window**; drought or thermal extremes ($>35^\circ\text{C}$) cause spikelet sterility. Excessive downpours during flowering wash away pollen. |
| **Stage 5** | Week $-1$ | Grain Filling & Ripening | Requires gradual field drainage; unseasonal deluge causes lodging and pre-harvest grain sprouting. |
| **Harvest** | Week $T_{\text{ref}}$ | Physical Harvest & Threshing | Reference cutoff point. Meteorological observations occurring during or after $T_{\text{ref}}$ **must not enter** pre-harvest predictive feature vectors. |

---

## 3. Zonal Reference Table

| Zone ID | Agro-Climatic Zone | Primary States | Kharif Sowing Window | Kharif Flowering Window | Harvest Reference Week ($T_{\text{ref}}$) | 16-Week Pre-Harvest Lag Window ($w_{-16} \to w_{-1}$) |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| **1** | **Western Himalayan** | HP, Uttarakhand | Weeks 20–24 (May–Jun) | Weeks 33–37 (Aug–Sep) | **Week 41** (Mid Oct) | Weeks 25 to 40 |
| **2** | **Eastern Himalayan** | Assam, NE | Weeks 22–26 (Jun) | Weeks 38–41 (Sep–Oct) | **Week 46** (Mid Nov) | Weeks 30 to 45 |
| **3** | **Lower Gangetic Plain** | West Bengal | Weeks 24–28 (Jun–Jul) | Weeks 39–42 (Sep–Oct) | **Week 46** (Mid Nov) | Weeks 30 to 45 |
| **4** | **Middle Gangetic Plain** | Bihar, East UP | Weeks 24–28 (Jun–Jul) | Weeks 38–41 (Sep–Oct) | **Week 45** (Early Nov) | Weeks 29 to 44 |
| **5** | **Upper Gangetic Plain** | Central & West UP | Weeks 23–27 (Jun–Jul) | Weeks 37–40 (Sep) | **Week 43** (Late Oct) | Weeks 27 to 42 |
| **6** | **Trans-Gangetic Plain** | Punjab, Haryana | Weeks 21–25 (Late May–Jun)| Weeks 36–39 (Early Sep) | **Week 42** (Mid Oct) | Weeks 26 to 41 |
| **7** | **Eastern Plateau & Hills**| Chhattisgarh, Jharkhand| Weeks 24–28 (Jun–Jul) | Weeks 38–41 (Sep–Oct) | **Week 45** (Early Nov) | Weeks 29 to 44 |
| **8** | **Central Plateau & Hills**| MP, East Rajasthan | Weeks 24–28 (Jun–Jul) | Weeks 37–40 (Sep) | **Week 43** (Late Oct) | Weeks 27 to 42 |
| **9** | **Western Plateau & Hills**| Maharashtra | Weeks 24–28 (Jun–Jul) | Weeks 37–41 (Sep–Oct) | **Week 44** (Late Oct) | Weeks 28 to 43 |
| **10** | **Southern Plateau & Hills**| Telangana, Karnataka | Weeks 25–29 (Late Jun–Jul)| Weeks 39–42 (Sep–Oct) | **Week 46** (Mid Nov) | Weeks 30 to 45 |
| **11** | **East Coast Plains & Hills**| Odisha, AP, Tamil Nadu| Weeks 26–30 (Jul) | Weeks 40–44 (Oct) | **Week 47** (Late Nov) | Weeks 31 to 46 |
| **12** | **West Coast & Ghats** | Kerala, Konkan | Weeks 21–24 (Late May–Jun)| Weeks 34–38 (Aug–Sep) | **Week 41** (Mid Oct) | Weeks 25 to 40 |
| **13** | **Gujarat Plains & Hills**| Gujarat | Weeks 25–28 (Late Jun–Jul)| Weeks 38–41 (Sep–Oct) | **Week 44** (Late Oct) | Weeks 28 to 43 |

---

## 4. Leakage Guard Implementation

For every district observation $i$, the pre-harvest lag sequence is defined as:
$$w_{-k}^{(i)} = \sum_{t \in \text{ISO\_Week}(T_{\text{ref}}^{(i)} - k)} R_{\text{daily}}^{(i)}(t) \quad \text{for } k \in \{1, 2, \dots, 16\}$$

1. **Temporal Precedence:** Because $k \ge 1$, $\max(T_{\text{ref}}^{(i)} - k) = T_{\text{ref}}^{(i)} - 1 < T_{\text{ref}}^{(i)}$.
2. **Causal Strictness:** Zero meteorological observations during or after $T_{\text{ref}}$ are incorporated into predictors.
