# Quantitative Research Analysis: Regression Objective Failure & Reformulation Options

> **Author:** Antigravity Quant Engineering  
> **Target Reviewer:** ChatGPT Independent Auditor  
> **Status:** Research & Architectural Recommendation  
> **Document:** `training/regression_objective_analysis.md`

---

## 1. Executive Summary

Empirical audit of the baseline retraining run revealed that while the **classification head** achieved $54.88\%$ directional accuracy on validation data, the **regression head** achieved only **$50.15\%$** (identical to random guessing), with a validation Mean Squared Error ($\text{MSE}$) of **$1.0016$**.

Crucially, in live trading (`live/signal_engine.py` and `live/trader.py`), execution decisions are dictated entirely by the **regression head**:
$$\text{pred} = \frac{\text{raw\_reg}}{\text{TARGET\_MULTIPLIER}}$$
Because $\text{pred}$ is effectively random noise around zero ($\mu = 0.0295, \sigma = 0.0230$), filtering for higher confidence ($|\text{pred}| \ge 0.0450$) drops directional accuracy to **$48.51\%$** (worse than random).

This document demonstrates that this failure is **not** a training defect or convergence bug, but a **fundamental mathematical mismatch** between the chosen loss function ($\text{MSE}$) and the discrete binary nature of Triple Barrier Method ($\pm 1.0$) targets.

---

## 2. Mathematical Formulation of the Objective Failure

### 2.1 The Current Formulation (Approach A)
- **Target Variable:** $y \in \{-1.0, +1.0\}$, derived from Triple Barrier labeling ($+1$ for Take Profit hit first, $-1$ for Stop Loss hit first).
- **Target Distribution:** Approximately symmetric:
  $$P(y = +1) = p \approx 0.507, \quad P(y = -1) = 1 - p \approx 0.493$$
- **Loss Function:**
  $$\mathcal{L}_{\text{reg}}(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$

### 2.2 Proof of Output Collapse
Consider an estimator $\hat{y}(x)$ parameterized by weights $\theta$. The expected loss under the joint distribution $P(x, y)$ is:
$$\mathbb{E}_{x, y} [(\hat{y}(x) - y)^2] = \mathbb{E}_x \left[ \mathbb{E}_{y|x} [(\hat{y}(x) - y)^2 \mid x] \right]$$

For any fixed input conditioning $x$, expanding the conditional expectation:
$$\mathbb{E}_{y|x} [(\hat{y}(x) - y)^2] = \hat{y}(x)^2 - 2\hat{y}(x)\mathbb{E}[y \mid x] + \mathbb{E}[y^2 \mid x]$$

Differentiating with respect to $\hat{y}(x)$ and setting to zero yields the optimal prediction:
$$\hat{y}^*(x) = \mathbb{E}[y \mid x]$$

Under binary targets $y \in \{-1, +1\}$:
$$\mathbb{E}[y \mid x] = (+1) \cdot P(y = +1 \mid x) + (-1) \cdot P(y = -1 \mid x) = 2 P(y = +1 \mid x) - 1$$

In financial time series (specifically 1-hour ETH futures), signal-to-noise ratios are low. For realistic conditioned probabilities $P(y = +1 \mid x) \in [0.48, 0.53]$:
$$\hat{y}^*(x) \in [2(0.48) - 1, \, 2(0.53) - 1] = [-0.04, \, +0.06]$$

### 2.3 The Trivial Baseline Theorem
When the network cannot confidently condition on features $x$ (or when regularization dominates):
$$\hat{y}(x) \to \mathbb{E}[y] = 2(0.507) - 1 = +0.014 \approx 0.0$$

The MSE of a trivial constant estimator $\hat{y} = 0$ is:
$$\text{MSE}_{\text{trivial}} = \frac{1}{N} \sum_{i=1}^N (0 - y_i)^2 = \frac{1}{N} \sum_{i=1}^N (\pm 1)^2 = 1.0000$$

If the network predicts with conviction (e.g., $\hat{y} = +0.8$), but is wrong (the target is $-1.0$), the squared penalty is:
$$(-1.0 - 0.8)^2 = 3.24$$
Conversely, if it retreats to $\hat{y} = 0.0$, the penalty on an error is only:
$$(-1.0 - 0.0)^2 = 1.00$$

**Conclusion:** The MSE loss actively penalizes conviction on binary targets. It provides strong gradient pressure driving all predictions to the neighborhood of zero. The model's achieved validation MSE of $1.0015$ demonstrates that the network converged to this trivial equilibrium.

---

## 3. Empirical Evidence from Baseline Retraining

| Metric | Measured Value | Theoretical Implication |
| :--- | :--- | :--- |
| **Validation MSE** | $1.0015$ | Essentially identical to the $1.0000$ trivial zero baseline ($1.0015 \times \text{trivial}$). |
| **Validation MAE** | $1.0001$ | Exactly equal to $|0 - (\pm 1)| = 1.0$. |
| **Regression Output Mean ($\mu$)** | $+0.0295$ | Matches the slight long bias of the dataset ($50.72\%$). |
| **Regression Output Std ($\sigma$)** | $0.0230$ | Predictions are compressed in a razor-thin band around zero. |
| **Regression Output Range** | $[-0.0479, +0.0812]$ | Never reaches meaningful magnitude. |
| **Regression Directional Accuracy** | **$50.15\%$** | Coin-toss performance; no directional edge. |
| **Filtered Directional Accuracy ($|\text{pred}| \ge 0.0450$)** | **$48.51\%$** | High-magnitude predictions are outliers with negative edge. |
| **Classification Head Accuracy** | **$54.88\%$** | Modest positive predictive signal exists in latent representations. |

The empirical results rigorously confirm the mathematical proof: the regression head is structurally dysfunctional under the current objective.

---

## 4. Evaluation of Reformulation Options

To resolve this issue, three distinct approaches are evaluated:

```mermaid
flowchart TD
    subgraph Current ["Approach A: Current (Flawed)"]
        A1["TBM Labels: {-1, +1}"] --> A2["MSELoss"]
        A2 --> A3["Outputs collapse to ~0\nAccuracy: 50.15%"]
    end

    subgraph ClfPath ["Approach B: Pure Classification (Recommended Minimal)"]
        B1["TBM Labels: {0, 1}"] --> B2["BCEWithLogitsLoss / Focal Loss"]
        B2 --> B3["Calibrated Probabilities P(Win)\nAccuracy: 54.88%"]
    end

    subgraph RegPath ["Approach C: Continuous Return (Deep Quant)"]
        C1["Forward Log Return / PnL"] --> C2["Huber / Sharpe Loss"]
        C2 --> C3["Continuous Expected Value\nMagnitude & Sign"]
    end
```

### Approach A: Current Setup (Status Quo)
- **Loss:** $\text{MSELoss}$ on $y \in \{-1.0, +1.0\}$.
- **Verdict:** **UNUSABLE**. The regression head cannot learn directional conviction under quadratic penalties on binary outcomes. Live trading must not rely on this signal.

---

### Approach B: Pure Directional Classification (Recommended Minimal Intervention)
- **Formulation:** Map TBM labels to $\{0, 1\}$ (where $0 = \text{SL hit first}, 1 = \text{PT hit first}$).
- **Loss:** `BCEWithLogitsLoss(pos_weight=...)` or `BinaryCrossEntropy`.
- **Inference Signal:**
  $$p = \sigma(\text{logits}) = \frac{1}{1 + e^{-\text{logits}}}$$
  $$\text{Trade Signal} = \begin{cases} 
  \text{BUY} & \text{if } p > 0.50 + \delta \\
  \text{SELL} & \text{if } p < 0.50 - \delta \\
  \text{HOLD} & \text{otherwise}
  \end{cases}$$
- **Advantages:**
  1. The classification head is **already implemented** in `HybridModel` and currently trains via `BCEWithLogitsLoss`.
  2. Already demonstrated **$54.88\%$** validation directional accuracy in the baseline run.
  3. Well-calibrated probabilities enable dynamic position sizing (e.g., fractional Kelly criterion).
  4. Requires zero architectural changes to `HybridModel`—only routing `live/signal_engine.py` to use `clf_out` rather than `reg_out`.

---

### Approach C: Continuous Expected Return Regression
- **Formulation:** Replace the discrete $\pm 1$ TBM label with a continuous financial target:
  - Option C.1: Forward realized return over the barrier window:
    $$r_{\tau} = \frac{P_{\tau} - P_0}{P_0}$$
    where $\tau$ is the stopping time when a barrier is hit or the horizon expires.
  - Option C.2: Volatility-normalized forward return:
    $$z = \frac{r_{\tau}}{\text{ATR}_{\tau}}$$
- **Loss Function:** Huber Loss (Smooth L1) or asymmetric Sharpe-ratio loss:
  $$\mathcal{L}_{\text{Huber}}(y, \hat{y}) = \begin{cases} 
  \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\
  \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
  \end{cases}$$
- **Advantages:**
  1. Restores continuous magnitude information (large moves matter more than small moves).
  2. Directly estimates expected monetary return rather than binary hit rate.
- **Drawbacks:**
  1. Requires modifying feature engineering / target calculation in `data/indicators.py`.
  2. Higher target variance and noise level compared to barrier classification.

---

## 5. Comparative Trade-off Matrix

| Criterion | Approach A (Current) | Approach B (BCE Classification) | Approach C (Continuous Return) |
| :--- | :--- | :--- | :--- |
| **Mathematical Soundness** | ❌ Flawed (collapse to 0) | ✅ Rigorous for binary targets | ✅ Rigorous for continuous targets |
| **Validation Accuracy (Demonstrated)** | $50.15\%$ (No edge) | **$54.88\%$** (Positive edge) | Untested (Requires retraining) |
| **Required Model Architecture Changes** | None | **None** (Head already exists) | None (Same head, new target) |
| **Required Target Engineering Changes** | None | None (Binary mapping) | Requires new target calculation |
| **Live Execution Alignment** | Misaligned (uses flawed head) | Aligns with best-performing head | Aligns with expected value |
| **Implementation Risk** | High (trading noise) | **Lowest** | Moderate |

---

## 6. Recommendations for Auditor Review

1. **Short-Term Recommendation (Execution Safety):**
   Prior to any future live trading or backtesting, align `live/signal_engine.py` with the **classification head** (`clf_out`) using calibrated probability thresholds ($p > 0.52$ for BUY, $p < 0.48$ for SELL), rather than the non-predictive regression head.

2. **Retraining Recommendation:**
   When retraining is authorized:
   - Either train as a pure directional classifier (Approach B) optimizing classification cross-entropy and balanced accuracy; OR
   - Maintain the multi-task architecture but reformulate the regression head to predict continuous volatility-adjusted forward return $z = r / \text{ATR}$ with Huber loss (Approach C).

3. **Strict Compliance Note:**
   In accordance with the quantitative research protocol, **no changes to live trading logic, architecture, or configuration have been enacted in code**. This document is submitted for independent review by ChatGPT before any changes are committed or retrained.
