# Phase 3 Sequence Deep Learning Architecture: LSTM and GRU

**Project:** Rainfall-Lag-Aware Crop Yield Forecasting in India Using Explainable Temporal Machine Learning  
**Phase:** 3 (Sequential Deep Learning)

---

## 1. Motivation for Recurrent Sequence Architectures

While tabular decision trees treat the 16 weekly rainfall lags as independent feature columns, recurrent neural networks (RNNs) natively preserve:
1. **Temporal Ordering:** Sequential flow of meteorological events ($w_{-16} \to w_{-15} \to \dots \to w_{-1}$).
2. **Hidden State Accumulation:** Internal memory states tracking cumulative soil moisture deficit or excess over consecutive weeks.
3. **Non-Linear Dynamics:** Gated memory mechanisms (forget gates, update gates) that selectively retain or extinguish past rainfall anomalies based on current weather shocks.

---

## 2. Input Tensor Formulation & Fusion

Each training observation combines two distinct data modalities:

```
[Sequential Input Tensor]
Shape: (Batch, 16 Weeks, Feature_Dim)
  w_{-16} -> w_{-15} -> ... -> w_{-1} (Weekly Cumulative Rainfall mm)
           │
           ▼
  [Recurrent Encoder: 2-Layer LSTM / GRU]
           │
     Final State h_16
           │
           ├────────────────────────────┐
           ▼                            ▼
  [Concatenation Layer]  ◄──  [Static Spatial Vector]
           │                  Shape: (Batch, 20 States One-Hot)
           ▼
  [Multi-Layer Perceptron Head]
    Linear(Hidden + 20, 64) -> ReLU -> Dropout(0.2)
    Linear(64, 32)          -> ReLU
    Linear(32, 1)
           │
           ▼
  Predicted Yield \hat{y} (kg/ha)
```

---

## 3. Mathematical Architectures

### 3.1 RiceLSTM (`RiceLSTM`)
For input vector $x_t \in \mathbb{R}^F$ at week $t \in [1, 16]$:
$$\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad &&\text{(Forget Gate)} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad &&\text{(Input Gate)} \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \quad &&\text{(Candidate Memory Cell)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad &&\text{(Updated Memory Cell)} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad &&\text{(Output Gate)} \\
h_t &= o_t \odot \tanh(c_t) \quad &&\text{(Hidden Output State)}
\end{aligned}$$

### 3.2 RiceGRU (`RiceGRU`)
$$\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \quad &&\text{(Update Gate)} \\
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \quad &&\text{(Reset Gate)} \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \quad &&\text{(Candidate Hidden State)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad &&\text{(Interpolated State)}
\end{aligned}$$

---

## 4. Optimization & Training Protocol

- **Loss Function:** Mean Squared Error ($\text{MSE}$)
  $$\mathcal{L} = \frac{1}{B} \sum_{i=1}^B (y_i - \hat{y}_i)^2$$
- **Optimizer:** AdamW with learning rate $\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$.
- **Gradient Clipping:** Maximum gradient $L_2$ norm capped at $5.0$ to prevent exploding gradients.
- **Patience & Early Stopping:** Monitor validation loss over 10 consecutive epochs; best checkpoint restored automatically.
- **Leakage Control:** All model weights, hyperparameter tuning, and early stopping criteria use strictly $\mathcal{D}_{\text{train}}$ and $\mathcal{D}_{\text{val}}$.
