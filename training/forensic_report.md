# Quantitative Research Audit: Retraining Pipeline Forensic Report (Pass 2)

> **Branch:** `chatgpt-audit-review`  
> **Repository:** `menemjef/ethtradingbot`  
> **Date:** 2026-11-04  
> **Status:** Second Correction Pass Complete (Awaiting Retraining Authorization)

---

## Executive Summary

Following a second-pass quantitative research audit of commit `3957d5ffd5ee9a952e6e7f17dddee576d0ba75c1`, a comprehensive set of corrections was implemented across the retraining pipeline, research documents, test fixtures, and statistical evaluation modules.

All unit and integration tests (23 pipeline and quantitative tests, 8 audit bridge tests) pass with 100% success. A dedicated reproducibility script (`training/reproduce_baseline_audit.py`) was created and verified, establishing full cryptographic provenance.

**Strict Governance Constraints Maintained:**
- No live trading was enabled, started, or executed.
- No Binance private API endpoints were queried.
- No AWS EC2 deployments were performed.
- Trading strategy parameters, position sizing, leverage, and entry/exit rules remain completely untouched.
- Pre-retraining model and scaler backups remain untouched at `C:\Python Projects\trading_bot_backups\baseline_pre_retrain\`.
- **No retraining run was started.** Execution remains paused for independent auditor review.

---

## 1. Mathematical Correction: Regression Objective Analysis

The previous audit document contained an erroneous mathematical claim asserting that Mean Squared Error (MSE) on binary $\{-1.0, +1.0\}$ Triple Barrier targets "fundamentally penalizes conviction and causes output collapse."

### Mathematical Reality
For binary $y \in \{-1, +1\}$, the Bayes-optimal prediction under quadratic loss is:
$$\hat{y}^*(x) = \mathbb{E}[y \mid x] = (+1) \cdot P(y = +1 \mid x) + (-1) \cdot P(y = -1 \mid x) = 2P(y = +1 \mid x) - 1$$

MSE is a strictly proper scoring rule that estimates the **signed conditional probability score**. In low signal-to-noise ratio financial regimes where $P(y = +1 \mid x) \in [0.48, 0.52]$, the Bayes-optimal conditional expectation naturally lies in $[-0.04, +0.04]$. 

We distinguish four separate aspects:
1. **Mathematically Valid Objective:** Minimizing MSE is statistically consistent for $2P(y = +1 \mid x) - 1$.
2. **Empirically Ineffective Model:** The baseline model's predictions do not exhibit out-of-sample directional ranking power ($\approx 49.24\% - 50.15\%$ directional accuracy).
3. **Poorly Calibrated Prediction:** Raw neural network regression outputs and sigmoids require post-hoc calibration (e.g., Platt scaling on a separate partition) before probability interpretation.
4. **Unsuitable Target for Economic Utility:** Binary barrier outcomes do not capture continuous return magnitude or execution friction.

MSE is **not** abandoned solely because outputs are close to zero; rather, its statistical meaning is clarified and evaluated alongside classification objectives.

---

## 2. Classification Loss Weighting and Calibration Investigation

### 2.1 Investigation of `pos_weight = 1.17` and Label Smoothing
In `model/trainer.py`:
- Previous configuration hardcoded `pos_weight = 1.17` and `label_smoothing = 0.25`.
- In a balanced training partition ($50.7\%$ positive, $49.3\%$ negative), the empirical class ratio is:
  $$\frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{49.3}{50.7} \approx 0.9724$$
- Setting `pos_weight = 1.17` multiplied positive class log-odds by $1.17$, inducing an artificial **$+20.3\%$ positive (bullish) bias** into the loss function.
- Applying label smoothing of $0.25$ compressed the target probabilities into $[0.125, 0.875]$, artificially narrowing the sigmoid output distribution ($\mu \approx 0.5298, \sigma \approx 0.0123$).

### 2.2 Calibration Metrics on Empirical Validation Data
Using the baseline retrained checkpoint (`ac224bb7...`):
- **Brier Score:** $0.2514$ (baseline uninformative prior Brier score: $0.2500$)
- **Log Loss:** $0.6959$ (random guessing log-loss: $\ln(2) \approx 0.6931$)
- **Expected Calibration Error (ECE):** $0.0392$ across 10 probability bins

### 2.3 Implementation Updates
1. `model/trainer.py` now supports `pos_weight_mode`:
   - `"balanced"` (default): Calculates $\text{pos\_weight} = N_{\text{neg}} / N_{\text{pos}}$ strictly from the training partition.
   - `"unweighted"`: Sets $\text{pos\_weight} = 1.0$.
   - `"legacy_1.17"`: Retained for backward reproducibility.
2. Label smoothing defaults to $0.0$ for clean, uncompromised probability calibration.
3. Live inference continues to use the existing execution engine untouched.

---

## 3. Model Selection and Chronological Research Design

### 3.1 Checkpoint Selection Criteria
- Previously, `best_model.pt` was selected solely by minimizing validation regression MSE.
- Minimizing MSE favors predictions closest to the unconditional mean ($\approx 0$), penalizing directional risk-taking.
- `model/trainer.py` now supports configurable selection criteria (`checkpoint_metric`):
  - `"regression_mse"`: Selects checkpoint minimizing validation MSE.
  - `"classification_loss"`: Selects checkpoint minimizing validation BCE log-loss.
  - `"directional_accuracy"`: Selects checkpoint maximizing directional classification accuracy.

### 3.2 Chronological 3-Partition Split
Implemented `split_train_val_test_purged()` in `data/preprocessor.py`:
- **Training Partition (60%):** Model parameter optimization.
- **Validation/Calibration Partition (20%):** Early stopping, hyperparameter comparison, probability calibration.
- **Untouched Final Test Partition (20%):** Evaluated strictly out-of-sample once. **No threshold tuning or hyperparameter optimization is permitted on this partition.**
- Purge gaps of $H = 12$ hours are enforced at both boundaries.

### 3.3 Walk-Forward Evaluation and Uncertainty Estimation
- Implemented `generate_walk_forward_splits()` supporting rolling and expanding window walk-forward validation with purged test partitions.
- Implemented `block_bootstrap_confidence_interval()` using moving blocks of size $H = 12$ to calculate valid confidence intervals and standard errors on overlapping time-series targets without artificially deflating variance.

---

## 4. TBM Ambiguous Candle Resolution: Default Exclusion

### 4.1 Directional Label Bias in Stop-First
In `data/indicators.py`, assigning $-1$ to every candle where both PT and SL are breached in the same bar induced systematic bearish label bias for directional training.

### 4.2 Engineering Resolution
`add_tbm_target()` updated with:
- `ambiguous_resolution = "exclude"` (default): Ambiguous bars receive `np.nan` and are discarded from training, guaranteeing unbiased directional labels.
- `ambiguous_resolution = "stop_first"`: Retained strictly for conservative trade execution / backtest simulation.
- `df_ltf`: When lower-timeframe OHLC data (e.g. 1m or 5m) is available, sequential intra-bar order is evaluated chronologically to identify whether PT or SL occurred first.

### 4.3 Empirical Impact (5,000 Candles Scanned)
- Total bars: 4,988
- Ambiguous bars: 8 ($0.16\%$)
- Legacy overshoot resolution: 7 wins / 1 loss
- Discarded under `"exclude"`: Exactly 8 samples ($0.16\%$)
- Impact on class distribution: negligible ($0.14\%$ shift), eliminating artificial bias.

---

## 5. Test Suite Strengthening

The test suite in `tests/test_retraining_corrections.py` was expanded to 13 comprehensive tests:
1. `test_multipage_pagination_over_2000_candles`: Requests 2,500 candles across 1,000-candle page boundaries; verifies cursor progression and strictly monotonic timestamps.
2. `test_safe_termination_on_repeated_non_progressing_page`: Simulates stagnant API returns; verifies safe termination without infinite loops.
3. `test_shortfall_reporting_when_exchange_has_fewer_candles`: Verifies explicit shortfall logging.
4. `test_duplicate_and_missing_interval_detection`: Tests deduplication and explicit `raise_on_gap` data-integrity exceptions.
5. `test_warmup_bars_fetching_guarantee`: Tests pre-roll candle fetching.
6. `test_ambiguous_candle_resolution_modes`: Verifies `"exclude"`, `"stop_first"`, `"overshoot"`, and lower-timeframe sequential ordering.
7. `test_head_specific_metrics_and_calibration`: Verifies Brier score, log loss, ECE, and reliability curve generation.
8. `test_trivial_zero_regression_baseline`: Verifies trivial zero MSE = 1.0.
9. `test_three_way_purged_partitioning`: Proves isolation of Train, Val, and Untouched Test partitions.
10. `test_walk_forward_splits_generator`: Verifies expanding train partitions and purged test partitions.
11. `test_block_bootstrap_uncertainty_estimation`: Tests moving block bootstrap confidence interval calculation.
12. `test_end_to_end_scaler_fitting_and_purged_split_integration`: End-to-end integration test proving scalers fit strictly on purged training data without validation contamination.
13. `test_tbm_parameter_propagation`: Verifies `config.py` parameters propagate to labeling.

---

## 6. Reproducibility & Cryptographic Provenance

A dedicated script (`training/reproduce_baseline_audit.py`) was executed to verify all empirical figures. Outputs are recorded in `training/reproducibility_audit_report.json`.

| Artifact / Metric | Provenance Value | Verification Status |
| :--- | :--- | :--- |
| **Active Checkpoint SHA-256** | `ac224bb7731f26b46d3f1cdc8cf39457742f7afde81799b31d67f9e02e3505bb` | Verified (`checkpoints/best_model.pt`) |
| **Pre-Retrain Backup SHA-256** | `8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95` | Verified (`baseline_pre_retrain/best_model.pt`) |
| **Active Scaler Candle SHA-256** | `b8e8da465b61cc47dd43ecbef3fd47df037362f4b17e6a54e1f3620b29dab5e4` | Verified (`checkpoints/scaler_candle.pkl`) |
| **Active Scaler Market SHA-256** | `9d62bd29c91cfa45762c1ed23f8eb21e3eddc65be2d613a63ebfd2af4cced281` | Verified (`checkpoints/scaler_market.pkl`) |
| **Reproduction Script** | `python training/reproduce_baseline_audit.py` | Executed, Exit Code 0 |
| **Reproduction Report JSON** | `training/reproducibility_audit_report.json` | Generated from live runtime evaluation |
| **Regression Accuracy (5,000 sample)** | $49.24\%$ | Reproduced |
| **Classification Accuracy (5,000 sample)** | $49.04\%$ | Reproduced |
| **Regression MSE vs. Trivial** | $0.9997$ vs $1.0000$ | Reproduced |
| **Brier Score / Log Loss / ECE** | $0.2514$ / $0.6959$ / $0.0392$ | Calculated on validation partition |
| **Ambiguous Bars** | 8 bars out of 4,988 ($0.16\%$) | Reproduced (7 wins / 1 loss under legacy overshoot) |

---

## 7. Cumulative Test Verification Summary

```text
======================================================================
1. Retraining & Methodology Test Suite (tests/test_retraining_corrections.py)
----------------------------------------------------------------------
test_duplicate_and_missing_interval_detection ......... ok
test_multipage_pagination_over_2000_candles ........... ok
test_safe_termination_on_repeated_non_progressing_page . ok
test_shortfall_reporting_when_exchange_has_fewer_candles ok
test_warmup_bars_fetching_guarantee ................... ok
test_head_specific_metrics_and_calibration ............ ok
test_trivial_zero_regression_baseline ................. ok
test_ambiguous_candle_resolution_modes ................ ok
test_tbm_parameter_propagation ....................... ok
test_block_bootstrap_uncertainty_estimation ........... ok
test_end_to_end_scaler_fitting_and_purged_split ....... ok
test_three_way_purged_partitioning .................... ok
test_walk_forward_splits_generator .................... ok

2. System Integration Suite (tests/test_full_pipeline.py)
----------------------------------------------------------------------
test_circuit_breaker_on_repeated_exchange_errors ...... ok
test_cooldown_blocks_reentry .......................... ok
test_long_signal_pipeline ............................. ok
test_minimum_notional_compliance ...................... ok
test_short_signal_pipeline ............................ ok
test_weak_signal_waiting .............................. ok

3. Risk & Signal Engines (tests/test_risk_manager.py, test_signal_engine.py)
----------------------------------------------------------------------
test_daily_reset ...................................... ok
test_drawdown_limit_long .............................. ok
test_leverage_cap_scaling ............................. ok
test_inference_pipeline ............................... ok

Ran 23 tests in 1.661s — OK (0 failures, 0 errors)
======================================================================
Audit Bridge V2 Suite (tests/test_audit_bridge.py)
----------------------------------------------------------------------
Results: 8 passed, 0 failed, 8 total — OK
============================================================
```

---

## 8. Unresolved Limitations & Open Research Decisions

Prior to initiating any full retraining run, the following architectural and experimental decisions remain open for auditor determination:

1. **Target Formulation Choice for Retraining:**
   - **Option A (Signed Probability):** Maintain binary TBM labels with $\text{MSELoss}$, interpreting outputs as $2p(x) - 1$, but evaluating checkpoint selection via Directional Accuracy rather than raw MSE.
   - **Option B (Calibrated Classification):** Train pure classification head using $\text{BCEWithLogitsLoss}$ with data-driven `pos_weight`, followed by temperature scaling on the calibration partition.
   - **Option C (Continuous Forward Return):** Replace discrete TBM targets with forward volatility-adjusted returns ($z = r / \text{ATR}$) trained with Huber loss.
2. **Threshold Optimization Governance:**
   - Thresholds must be calibrated on the Validation/Calibration partition only; the final Test partition must remain strictly out-of-sample and evaluated once.
3. **Execution Routing Alignment:**
   - Currently, `live/signal_engine.py` routes to the regression head. If Option B is selected, signal engine routing must be aligned with calibrated probability thresholds prior to live trading.
