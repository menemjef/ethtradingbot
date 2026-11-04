# Quantitative Research Analysis: Statistical Foundations and Calibration of the Retraining Objectives

> **Author:** Antigravity Quantitative Research  
> **Target Reviewer:** Independent Quant Auditor  
> **Status:** Mathematical Correction & Objective Assessment  
> **Document:** `training/regression_objective_analysis.md`  
> **Commit Context:** Post-Audit Revision Pass 2

---

## 1. Executive Summary & Mathematical Clarification

In the previous revision, an erroneous claim was made asserting that Mean Squared Error (MSE) on binary $\{-1.0, +1.0\}$ Triple Barrier targets "fundamentally penalizes conviction and causes output collapse." 

**This previous assertion is mathematically false.**

Under standard statistical decision theory, optimizing MSE on binary targets $y \in \{-1, +1\}$ is a **strictly proper scoring rule** whose Bayes-optimal estimator is the conditional expectation:
$$\hat{y}^*(x) = \mathbb{E}[y \mid x] = (+1) \cdot P(y = +1 \mid x) + (-1) \cdot P(y = -1 \mid x) = 2P(y = +1 \mid x) - 1$$

Therefore, minimizing MSE does **not** cause an arbitrary mathematical collapse; rather, it yields an unconstrained estimator of the **signed conditional probability score** $2p(x) - 1$. 

In financial time series characterized by low signal-to-noise ratios (SNR), the true conditional probability $P(y = +1 \mid x)$ typically fluctuates in a narrow band around the unconditional prior (e.g., $p \in [0.48, 0.52]$). Consequently, the Bayes-optimal prediction $\hat{y}^*(x)$ naturally lies in $[-0.04, +0.04]$. An output close to zero is the mathematically optimal response of a consistent estimator to weak conditioning information—**not** an objective pathology.

This document rigorously delineates four distinct concepts:
1. **A Mathematically Valid Objective:** What MSE actually optimizes and estimates.
2. **An Empirically Ineffective Model:** Why the current trained model exhibits chance-level ranking.
3. **A Poorly Calibrated Prediction:** Why raw regression outputs and uncalibrated sigmoids require separate calibration.
4. **An Unsuitable Target for the Intended Economic Objective:** The distinction between statistical barrier hit probabilities and portfolio utility.

---

## 2. Theoretical Formulation: What MSE Actually Estimates

### 2.1 Bayes-Optimal Prediction under Quadratic Loss
Let $(X, Y) \sim \mathcal{D}$ where $X \in \mathbb{R}^d$ is the feature space and $Y \in \{-1, +1\}$ denotes the binary Triple Barrier outcome. We seek an estimator $f: \mathbb{R}^d \to \mathbb{R}$ minimizing the expected squared error risk:
$$\mathcal{R}(f) = \mathbb{E}_{(X,Y)} \left[ (f(X) - Y)^2 \right]$$

By the law of total expectation:
$$\mathcal{R}(f) = \mathbb{E}_X \left[ \mathbb{E}_{Y|X} \left[ (f(X) - Y)^2 \mid X = x \right] \right]$$

For any fixed $x$, the inner expectation is:
$$\mathbb{E}_{Y|X} \left[ (f(x) - Y)^2 \mid X = x \right] = f(x)^2 - 2f(x)\mathbb{E}[Y \mid X = x] + \mathbb{E}[Y^2 \mid X = x]$$
Since $Y \in \{-1, +1\}$, $Y^2 \equiv 1$ almost surely. The convex quadratic in $f(x)$ attains its unique global minimum at:
$$f^*(x) = \mathbb{E}[Y \mid X = x]$$

Expanding the conditional expectation for binary $Y$:
$$f^*(x) = (+1) \cdot P(Y = +1 \mid X = x) + (-1) \cdot P(Y = -1 \mid X = x)$$
Let $p(x) \equiv P(Y = +1 \mid X = x)$. Since $P(Y = -1 \mid X = x) = 1 - p(x)$:
$$f^*(x) = p(x) - (1 - p(x)) = 2p(x) - 1$$

There exists an exact linear isomorphism between the Bayes-optimal MSE prediction and the posterior class probability:
$$p(x) = \frac{f^*(x) + 1}{2}$$

### 2.2 Rebuttal: Why Near-Zero Outputs Are Mathematically Expected
In market regimes with low predictive predictability:
$$p(x) = 0.50 + \epsilon(x), \quad |\epsilon(x)| \le 0.02$$
The optimal regression output is:
$$f^*(x) = 2(0.50 + \epsilon(x)) - 1 = 2\epsilon(x) \in [-0.04, +0.04]$$

The theoretical MSE of this Bayes-optimal predictor is:
$$\mathbb{E}[(f^*(X) - Y)^2] = \mathbb{E}[(2p(X) - 1 - Y)^2] = 1 - \mathbb{E}[(2p(X) - 1)^2] = 1 - 4\,\text{Var}(p(X))$$
Because $\text{Var}(p(X)) \approx O(10^{-4})$, the minimal attainable MSE is naturally:
$$\text{MSE}^* \approx 1 - 4(0.0004) = 0.9984 \approx 1.0000$$

**Conclusion:** A validation MSE of $0.9997$ or $1.0015$ and outputs clustered near zero do not indicate a mathematical defect of MSE. Rather, they confirm that the estimator has correctly converged near the unconditional expectation $\mathbb{E}[Y] \approx 0$ under near-symmetric noise. **MSE should not be abandoned simply because outputs are close to zero.**

---

## 3. Four-Way Methodological Breakdown

```mermaid
flowchart TD
    subgraph A ["1. Mathematical Validity"]
        A1["min E[(f(x) - y)^2]"] --> A2["f*(x) = 2P(y=+1|x) - 1\nStrictly proper scoring rule"]
    end
    subgraph B ["2. Empirical Efficacy"]
        B1["Model Rank Correlation"] --> B2["Validation Directional Acc: 49.24% - 50.15%\nNo statistical ranking separation"]
    end
    subgraph C ["3. Probability Calibration"]
        C1["Raw Sigmoid / MSE Score"] --> C2["Brier: 0.2514 | ECE: 0.0392\nRequires separate calibration split"]
    end
    subgraph D ["4. Economic Utility"]
        D1["Binary TBM Barrier Hit"] --> D2["Ignores move magnitude, slippage & fee drag\nMisaligned with Sharpe / utility"]
    end
```

### 3.1 Mathematically Valid Objective vs. Empirically Ineffective Model
- **Mathematical Validity:** The MSE objective is statistically consistent for conditional probabilities.
- **Empirical Inefficacy:** The issue with the current baseline model is not that $\hat{y}$ is small, but that $\text{sign}(\hat{y})$ does **not** correlate with $\text{sign}(y)$ on out-of-sample data:
  - Validation Directional Accuracy: **$49.24\% - 50.15\%$** (indistinguishable from a fair coin toss).
  - Top Confidence Decile Accuracy ($|\hat{y}| \ge 0.0450$): **$49.81\%$** (zero statistical edge even at the extremes).
- The network's learned representations have failed to extract conditional features that separate $p(x) > 0.50$ from $p(x) < 0.50$. The model is empirically ineffective, not mathematically invalid.

---

### 3.2 Poorly Calibrated Prediction vs. Calibration Correction
Even if a model exhibits modest directional accuracy, uncalibrated scores cannot be trusted for sizing or thresholding:
- **Raw Regression Output:** While $f^*(x) = 2p(x) - 1$ holds asymptotically, empirical neural network predictions $\hat{y}(x)$ are unconstrained and distorted by regularizers ($L_2$ weight decay, dropout).
- **Classification Sigmoid Output:** In `model/trainer.py`, the classification head was trained with hardcoded `pos_weight=1.17` and `label_smoothing=0.25`. This induced:
  1. **Bullish Prior Bias:** In a balanced dataset ($50.7\%$ positive), setting `pos_weight=1.17` multiplies positive log-odds by $1.17$, artificially driving predicted probabilities above their empirical frequencies.
  2. **Probability Compression:** Label smoothing of $0.25$ forces targets from $\{0, 1\}$ into $[0.125, 0.875]$, artificially compressing sigmoid outputs into a narrow range ($\mu \approx 0.5298, \sigma \approx 0.0123$).
- **Calibration Metrics (Measured on Validation Data):**
  - **Brier Score:** $0.2514$ (baseline uninformative prior Brier score: $0.2500$).
  - **Log Loss:** $0.6959$ (random guessing log-loss: $\ln(2) \approx 0.6931$).
  - **Expected Calibration Error (ECE):** $0.0392$.
- **Correction:** Sigmoid outputs must **never** be assumed to be calibrated probabilities. A dedicated **calibration partition** (Platt scaling / temperature scaling or isotonic regression) must be evaluated post-training before using probabilities in downstream risk management.

---

### 3.3 Unsuitable Target for the Intended Economic Objective
A trading bot's ultimate objective is expected utility or risk-adjusted return (Sharpe, Calmar, Profit Factor), not binary barrier hit rates:
- **Target Limitation:** A binary $+1$ label treats a trade that barely touches $+2.0 \times \text{ATR}$ identical to a massive $+10 \times \text{ATR}$ trend. Conversely, a $-1$ label treats an immediate stop identical to a trade that oscillated for 11 hours before ticking down.
- **Economic Asymmetry:** Trading incurs friction:
  $$\text{Realized PnL} = \Delta P - \text{Commissions} - \text{Slippage} - \text{Funding}$$
  Predicting barrier hit probability without incorporating trade duration or path-dependent cost drag creates an economic mismatch between the loss function and the execution engine.

---

## 4. Evaluation of Loss Formulations & Checkpoint Selection

| Objective Formulation | Loss Function | What It Estimates | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Signed Probability (Regression)** | $\text{MSELoss}$ | $2P(y=+1 \mid x) - 1$ | Mathematically valid; proper scoring rule; symmetric penalty. | Near-zero outputs require delicate thresholding; unconstrained range. |
| **Calibrated Classification** | $\text{BCEWithLogitsLoss}$ (Data-driven pos_weight) | $\ln\left(\frac{p(x)}{1 - p(x)}\right)$ | Outputs bounded in $(0, 1)$; naturally maps to Kelly betting & dynamic sizing. | Sigmoid is uncalibrated without post-hoc scaling; sensitive to class imbalance. |
| **Continuous Expected Return** | Huber Loss / Smooth L1 | $\mathbb{E}[r_{\tau} / \text{ATR} \mid x]$ | Directly estimates expected return magnitude; penalizes small vs large moves appropriately. | Higher target variance; noise in volatile regimes. |

### Checkpoint Selection Criteria
Currently, `best_model.pt` is selected solely by minimizing validation regression MSE:
- **Consequence:** Minimizing MSE strictly rewards models whose outputs stay closest to the sample mean ($\approx 0$). A model predicting constant $\hat{y} = 0.02$ achieves $\text{MSE} \approx 0.999$, easily beating a model with directional conviction that makes occasional false discoveries.
- **Recommendation:** Decouple checkpoint selection criteria based on the research objective:
  - For classification: Select by minimum validation **Log Loss** or maximum **Balanced Accuracy**.
  - For regression: Select by **Directional Accuracy** or **Information Coefficient (Rank IC)** alongside MSE.

---

## 5. Summary of Audit Recommendations

1. **Retain MSE as a Valid Statistical Estimator:** Do not reject MSE due to near-zero magnitude; small magnitude is the Bayes-optimal response to weak signal.
2. **Remove Hardcoded Weights & Smoothing:** Use data-driven `pos_weight = N_neg / N_pos` calculated strictly from the training split; set label smoothing to $0.0$ during baseline calibration.
3. **Enforce Chronological Partitioning:** Divide data into Train (60%), Validation/Calibration (20%), and Untouched Test (20%) with purged horizons.
4. **Calibrate Probabilities:** Apply temperature scaling or isotonic regression on the calibration split before evaluating probability thresholds.
5. **Acknowledge Live Invariant:** Maintain live execution untouched pending complete audit alignment.
