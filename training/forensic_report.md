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


---

## Pass 3: Training Pipeline Integration (2026-11-04)

> **Commit:** `(pending — changes staged for export)`  
> **Status:** All integrations complete, 25 unittest + 8 audit bridge tests passing. Awaiting retraining authorization.

### 9. Three-Way Chronological Split Integration

**Requirement:** The existing `split_train_val_test_purged()` helper in `data/preprocessor.py` was not connected to the actual training pipeline.

**Changes Applied:**

| Component | Before | After |
|---|---|---|
| `model/trainer.py` | 80/20 train/val via `split_train_val_purged()` | 60/20/20 train/val/test via `split_train_val_test_purged()` with `use_three_way_split=True` |
| Purge gaps | Single H=12 gap at train-val boundary | Dual H=12 purge gaps at both train→val and val→test boundaries |
| Scaler fitting | Fitted on purged training data | Unchanged — fitted on purged training data only |
| Checkpoint selection | Validation partition | Validation partition only (test partition never seen during training) |
| Final test evaluation | Not implemented | Frozen model evaluated on untouched test partition after all choices are finalized |

**train.py Changes:**
- Default `--warmup-bars` changed from 50 to **720** (30-day Fibonacci retracement lookback)
- Added `--allow-gaps` flag (default: False, raises `ValueError` on detected gaps during training)
- Added `--train-ratio` (default 0.60) and `--val-ratio` (default 0.20) arguments
- Added `--walk-forward` flag and `--wf-folds` (default 4)
- Explicit partition boundary logging with bar counts at each stage
- Comprehensive results logging: validation metrics, calibration temperature, test partition accuracy, Brier score, and bootstrap CIs

### 10. Temperature Scaling Calibration Integration

**Pipeline Flow:**
1. Train model, select best checkpoint via validation partition
2. After training: fit temperature parameter $T > 0$ on validation/calibration partition logits via L-BFGS
3. Apply frozen $T$ to test partition for calibrated probability evaluation
4. Report Brier score, Log Loss, and ECE against constant-prior baselines on both partitions

### 11. Block Bootstrap Uncertainty Estimation

After evaluating the frozen model on the untouched test partition, block bootstrap confidence intervals are computed for:
- Directional accuracy
- Regression MSE
- Calibrated Brier score

Block sizes: [12, 24, 48, 168] hours. Each with 500 bootstrap replications and 95% percentile CIs.

### 12. Walk-Forward Cross-Validation Pipeline

`model/walk_forward.py` is now fully integrated and callable via `train.py --walk-forward`:
- Independent training on each chronological fold
- Scalers fitted only on each fold's training window
- Purge gaps applied between each fold's training and test windows
- Trade simulation with maker/taker fees and slippage
- Fold-by-fold metrics: directional accuracy, regression MSE, Brier score, log loss, win rate, profit factor, net PnL, max drawdown
- Aggregate summary across all folds

### 13. Data Fetcher Gap Pagination Fix

The fetcher's pagination no longer breaks prematurely on intermediate exchange outage gaps when `target_end_ms` is set. Instead:
- Tracks consecutive empty pages (max 3 retries) with cursor forward-leap
- Continues toward `target_end_ms` across gaps
- Preserves cursor progression safety (`next_since <= since` break)
- `raise_on_gap=True` enforces gap-free data for training

### 14. Test Suite (25 Tests)

| Suite | Count | Status |
|---|---|---|
| `TestDataFetcherCorrections` | 5 | ✅ All pass |
| `TestTBMCorrections` | 2 | ✅ All pass |
| `TestEvaluationAndCalibration` | 2 | ✅ All pass |
| `TestTemporalLeakageAndPartitions` | 6 | ✅ All pass (including new gap traversal, walk-forward execution) |
| `TestFullPipeline` | 6 | ✅ All pass |
| `TestRiskManager` | 3 | ✅ All pass |
| `TestSignalEngine` | 1 | ✅ All pass |
| Audit Bridge | 8 | ✅ All pass |
| **Total** | **33** | **✅ 100% pass rate** |

### 15. Files Modified in Pass 3

| File | Change Type |
|---|---|
| `train.py` | Modified — 3-way split wiring, warmup default, walk-forward, comprehensive logging |
| `model/trainer.py` | Modified — 3-way split partitioning, temperature scaling, block bootstrap, results manifest |
| `model/calibration.py` | New — Temperature scaling, Brier/LogLoss/ECE evaluation |
| `model/walk_forward.py` | New — Full walk-forward engine with trade simulation |
| `data/fetcher.py` | Modified — Gap pagination fix |
| `data/preprocessor.py` | Modified — 3-way split, walk-forward splits, block bootstrap helpers |
| `data/indicators.py` | Modified — Ambiguous resolution default, logger |
| `tests/test_retraining_corrections.py` | Modified — 15 tests (added gap traversal, walk-forward execution) |
| `training/forensic_report.md` | Updated — This section |


---

## Pass 4: Execution Blockers and Backtest Integrity (2026-11-04)

> **Commit:** `(pending staged export)`  
> **Status:** All execution blockers resolved, OHLC-aware simulation implemented, 33 discover tests + 8 audit bridge tests passing.

### 16. Resolution of Training Entry-Point Blockers

Two runtime crashes in `train.py` were identified and resolved:
1. **Argument Mismatch:** `train_model()` previously accepted `purge_horizon`, while `train.py` passed `purge_bars`. Resolved by accepting both `purge_horizon` and `purge_bars` (alias) in `train_model()`, harmonizing `train.py` to `purge_horizon`.
2. **Metric Schema Inconsistency:** `compute_comprehensive_metrics()` returned `"sample_count": N`, while `train.py` queried `test_metrics["validation_samples"]`. Resolved by providing both `"sample_count"` and `"validation_samples"` in the returned dictionary, standardizing all callers on `"sample_count"`.
3. **Variable Scope Resolution:** Eliminated shadowing and late import statements in `train.py` (`TARGET_HORIZON` and `numpy as np`) to ensure clean module-level accessibility.
4. **Integration Test:** Added `test_train_py_full_orchestration_mocked` running `train.py.main()` through all 5 pipeline stages down to final result reporting.

### 17. Pre-Roll Lookback and Retained Dataset Isolation

- **Feature Warmup on Full History:** Indicator computation (`build_features()`) operates on the complete fetched candle history ($N_{\text{requested}} + N_{\text{pre-roll}}$).
- **Trimming Post-Computation:** The dataset is sliced to the requested start timestamp (`df_feat = df_feat_full.loc[requested_start_ts:].copy()`) **only after** all rolling indicators have completed warm-up.
- **Fibonacci Validation:** Verified that the 720-period Fibonacci feature (`fib_618_dist_180d`) is valid and finite at index 0 of the retained dataset.
- **Pre-Roll Exclusion:** Zero pre-roll observations are included as labeled training samples.
- **Cache Data Verification:** Local cache loading verifies monotonic UTC datetime index, deduplicates timestamps, and enforces contiguous 1h intervals when `--allow-gaps` is false.

### 18. Continuous Hourly Windows (Zero Gap Compression)

- **Elimination of Pre-Window Dropna:** Ceased calling `df_feat.dropna(subset=["target"])` prior to sliding window creation. The hourly feature timeline remains continuous and unbroken.
- **Contiguous Lookback Construction:** Every lookback window $[i - \text{LOOKBACK} : i]$ spans exactly contiguous 1-hour steps. No window skips an hour internally.
- **Post-Construction Filtering:** Windows with NaN targets (e.g. ambiguous barrier breaches or endpoint bounds) are filtered out after construction.
- **Timestamp Tracking:** Each window carries its decision timestamp, decision index, and target resolution timestamp (`t_{\text{dec}} + H \times \Delta t`).
- **Timestamp-Based Purging:** `split_train_val_test_purged()` purges training samples where $t_{\text{resolution}} > T_{\text{val\_start}}$, and validation samples where $t_{\text{resolution}} > T_{\text{test\_start}}$, mathematically verifying zero target leakage across partition boundaries.

### 19. OHLC-Aware Barrier Trade Simulator (`simulate_trades_ohlc`)

Rebuilt `model/walk_forward.py` trade simulator with realistic microstructure modeling:
- **Next-Bar Execution:** Entry orders execute at candle $i+1$ **OPEN** (after signal is confirmed at candle $i$ close), with modeled slippage (`open * (1 + slippage)` for long).
- **Intrabar Barrier Checks:** Take-profit and stop-loss evaluate against subsequent candle **High and Low** prices.
- **Intrabar Collision Resolution:** If both TP and SL are touched in the same bar, conservatively resolves **STOP-FIRST**.
- **Holding Horizon Exit:** Position exits at candle close of bar $(t_{\text{entry}} + H)$ with slippage.
- **Forced End-of-Fold Close:** Open positions at the final bar of a test fold are marked-to-market and closed at final candle CLOSE (eliminating silent trade omissions).
- **Explicit Fee Breakdown:** Separates entry fee (taker) and exit fee (maker for limit TP, taker for stop/horizon/collision).

### 20. Walk-Forward Methodology and Result Aggregation

- **Clear Boundary Enforcement:** For each fold, $T_{\text{train\_end}} + H \le T_{\text{test\_start}}$, guaranteeing that all training labels have resolved before out-of-sample predictions begin.
- **Historical Context:** Preceding historical candles provide lookback context for the first test prediction without lookahead.
- **Inner Chronological Validation:** Divides fold training history into inner train (75%) and inner val (25% with purge) with early stopping, preventing overtraining.
- **Pooled Profit Factor:** Aggregates all trade ledgers across folds to compute true pooled profit factor ($\sum \text{wins} / \sum |\text{losses}|$), replacing unweighted arithmetic averaging.
- **Sequential Compounded Return:** Computes compounded net return $\prod (1 + R_k) - 1$ across chronological folds.
- **Reporting Metrics:** Added `mean_win_rate_pct`, `pooled_win_rate_pct`, `worst_max_drawdown_pct`, and `compounded_net_return_pct`.

### 21. Calibration and Test Isolation

- **Prior Baseline from Training/Calibration:** `evaluate_calibration()` now accepts `fixed_prior`. When evaluating the untouched test partition, calibration metrics (Brier score and Log Loss) are measured against the prior positive rate fixed on the training/calibration set, not the test set.
- **Untouched Final Test Partition:** Completely isolated from model selection and temperature fitting.


---

## Pass 5: Artifact Integrity and Walk-Forward Corrections (2026-11-04)

> **Commit:** `(staged on chatgpt-audit-review)`  
> **Status:** Priority 0 artifact forensic audit completed, non-destructive test harness enforced, epoch propagation fixed, pre-`fillna` 720-bar Fibonacci validation implemented, walk-forward signal alignment & inner-val scaler isolation verified, and realistic OHLC stop-gap & itemized accounting integrated. All 38 `unittest discover` tests + 8 `test_audit_bridge.py` tests pass with zero artifact mutation.

### 22. Priority 0 — Active Model and Scaler Integrity Audit

#### A. Forensic Hash Comparison of Active Artifacts (`checkpoints/`)

| Artifact | Previously Recorded Active Hash | Current SHA-256 in `checkpoints/` | Status & Provenance |
|---|---|---|---|
| `checkpoints/best_model.pt` | `ac224bb7731f26b46d3f1cdc8cf39457742f7afde81799b31d67f9e02e3505bb` | `0dfa18e77e5afaafa94f12ad239ba80379a418f77569ccd50aa261b1328789f2` | **Overwritten during Pass 4 test run** by `test_train_py_full_orchestration_mocked()` (which mocked `fetch_ohlcv` and `sys.argv` without redirecting `config.MODEL_PATH`). |
| `checkpoints/scaler_candle.pkl` | `b8e8da465b61cc47dd43ecbef3fd47df037362f4b17e6a54e1f3620b29dab5e4` | `7104dc5965e75a6cee2224ff4a97cd00fe9c17c5d847ca5646d32e0ac8ad401d` | **Overwritten during Pass 4 test run** by `test_train_py_full_orchestration_mocked()` (unpatched `config.SCALER_CANDLE_PATH`). |
| `checkpoints/scaler_market.pkl` | `9d62bd29c91cfa45762c1ed23f8eb21e3eddc65be2d613a63ebfd2af4cced281` | `b94a1c728b0a090c3ff7215ecc03a149ee19c9c5cfa996bc98e6d2391716723c` | **Overwritten during Pass 4 test run** by `test_train_py_full_orchestration_mocked()` (unpatched `config.SCALER_MARKET_PATH`). |

#### B. Preserved Baseline Backups (`C:\Python Projects\trading_bot_backups\baseline_pre_retrain\`)

| Backup Artifact | Expected Baseline Hash | Verified Current SHA-256 | Integrity Status |
|---|---|---|---|
| `best_model.pt` | `8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95` | `8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95` | **100% INTACT & UNTOUCHED** |
| `scaler_candle.pkl` | `d5e5e30533e3ad133dd9385a5d84f3c0793c96ec91d1b7db12e97722a19bb19c` | `d5e5e30533e3ad133dd9385a5d84f3c0793c96ec91d1b7db12e97722a19bb19c` | **100% INTACT & UNTOUCHED** |
| `scaler_market.pkl` | `d7bc7ebdc32810f1c65eb6ad51273cccb380cf20882e0513fa009e0387390e2d` | `d7bc7ebdc32810f1c65eb6ad51273cccb380cf20882e0513fa009e0387390e2d` | **100% INTACT & UNTOUCHED** |

#### C. Additional Untouched Multi-Seed Checkpoints in `checkpoints/`

A full filesystem search across `C:\Python Projects` and user directories confirmed no secondary copy of `ac224bb7...` was cached outside `checkpoints/`. However, the suffixed multi-seed checkpoints and scalers in `checkpoints/` were never touched by `test_train_py_full_orchestration_mocked()`:

| File in `checkpoints/` | Verified SHA-256 | Last Modified |
|---|---|---|
| `best_model_42.pt` | `8c352dfe8f637af3ca3cb05e192e44fcec47848e5f7b067f88481ee9939bdcdd` | `2026-03-27 11:39:11` |
| `best_model_1337.pt` | `66aa63f4d334b4970fd87ea576A58e9504d50cc960f4ebb8ba4944f98cb62864` | `2026-03-27 12:12:41` |
| `best_model_7.pt` | `a0136e12ad106c914f898b989d4172233dd348c4367710a0401892089d5b10d6` | `2026-03-27 12:49:44` |
| `scaler_candle_42.pkl` / `_1337.pkl` / `_7.pkl` | `c82037685e2aef7c3f8136af6f4321c89a3d3c49bf775916d1a33080320eaf33` | `2026-03-27` |
| `scaler_market_42.pkl` / `_1337.pkl` / `_7.pkl` | `a6dfa45c8991c96efec1525063b231684d77c72da7e86427657d60e4aff70f03` | `2026-03-27` |

Per Priority 0 rules 6 and 7, no older backup was blindly restored over `checkpoints/best_model.pt`, and no files were deleted or overwritten.

---

### 23. Non-Destructive Integration Testing (`test_train_py_full_orchestration_mocked`)

- `fit_and_save_scalers()` in `data/preprocessor.py` now accepts optional `scaler_candle_path` and `scaler_market_path`.
- `train_model()` in `model/trainer.py` now accepts optional `model_path`.
- `train.py` resolves `model_path`, `scaler_candle_path`, and `scaler_market_path` dynamically from `config` at runtime and passes them explicitly to `fit_and_save_scalers()` and `train_model()`.
- `test_train_py_full_orchestration_mocked()` now creates a `tempfile.TemporaryDirectory()`, patches `config.MODEL_PATH`, `config.SCALER_CANDLE_PATH`, and `config.SCALER_MARKET_PATH` to paths inside `tmpdir`, verifies temporary artifacts are created in `tmpdir`, and asserts that the SHA-256 hashes of all three production files in `checkpoints/` are identical before and after the test.

---

### 24. Fast and Test Mode Epoch Propagation

- Previously, `model/trainer.py` bound `MAX_EPOCHS` and `PATIENCE` via `from config import MAX_EPOCHS, PATIENCE` at module import time, so modifying `config.MAX_EPOCHS` inside `train.py` had no effect on `train_model()`.
- `train_model()` now accepts explicit `max_epochs: Optional[int] = None` and `patience: Optional[int] = None` parameters, defaulting to `config.MAX_EPOCHS` and `config.PATIENCE` when `None`, and records `epochs_trained`, `max_epochs_configured`, and `patience_configured` in `validation_metrics`.
- `train.py` computes `max_epochs = 5` (`patience = 3`) for `--fast` and `max_epochs = 30` for `--test`, passing both explicitly into `train_model()` and `run_walk_forward_evaluation()`.
- Verified by `test_fast_and_test_mode_epoch_propagation`.

---

### 25. Pre-`fillna(0.0)` 720-Bar Fibonacci Warm-Up Validation

- `build_features()` in `data/preprocessor.py` replaces remaining auxiliary `NaN` values with `0.0` at the end of feature assembly, which previously masked whether `fib_618_dist_180d` (720-hour rolling Fibonacci retracement) was `NaN` at `requested_start_ts`.
- Added `validate_fibonacci_warmup()` in `data/preprocessor.py` and integrated it into `build_features()` **before** `df = df.fillna(0.0)`:
  1. Verifies that at least `min_history_bars=720` valid OHLC observations exist in `df_raw` strictly before `requested_start_ts`.
  2. Verifies that raw `fib_618_dist_180d` at `requested_start_ts` (and across the entire retained slice `df.loc[requested_start_ts:]`) is non-`NaN` **before** `fillna(0.0)` is applied.
  3. Stores `df.attrs["raw_fib_valid_mask"]` and `df.attrs["fib_warmup_validation"]` on the returned DataFrame.
- `train.py` passes `requested_start_ts=requested_start_ts, min_warmup_bars=720` to `build_features()` for **both** live fetches and cached CSV datasets (`--use_cache`).
- Verified by `test_fibonacci_warmup_validation_fails_on_insufficient_history_and_cache`.

---

### 26. Walk-Forward Signal Timestamp Alignment

- In `model/walk_forward.py`:
  - For a test fold covering indices $[k_{\text{start}}, k_{\text{end}})$, prediction $j \in [0, N_{\text{test}}-1]$ is evaluated at decision candle $k_{\text{start}} + j - 1$ **CLOSE** using the contiguous lookback window $[k_{\text{start}} + j - \text{LOOKBACK}, k_{\text{start}} + j - 1]$.
  - Specifically, the first test prediction ($j = 0$) uses decision timestamp $T_{k_{\text{start}}-1}$ (the candle immediately before the test period) and executes at the first test candle's **OPEN** ($T_{k_{\text{start}}}$).
  - The final test candle ($j = N_{\text{test}} - 1$) cannot open a new position outside the fold (`i < N - 1` guard on entry; any open trade is marked to market and closed at the final test candle's close with `exit_reason = "end_of_fold"`).
  - Every trade record includes explicit `decision_time` and `entry_time` timestamps with `decision_time + 1h == entry_time`.
- Verified by `test_walk_forward_signal_timestamp_alignment`.

---

### 27. Scaler Isolation in Walk-Forward Inner Validation

- In `run_walk_forward_evaluation()`, each fold's outer training history $[0, \text{tr\_end\_idx})$ is partitioned into:
  - Inner training slice: `df_inner_train = df_feat.iloc[tr_start_idx:inner_train_end_idx]` (purged by `purge_horizon` before inner validation start).
  - Inner validation slice: `df_inner_val = df_feat.iloc[inner_val_start_idx:tr_end_idx]`.
- `scaler_c` and `scaler_m` are fitted **exclusively** on `df_inner_train`.
- Verified by `test_walk_forward_inner_validation_scaler_isolation`, which injects $10^9$ outliers into the inner validation slice and proves `scaler_c.mean_`, `scaler_c.var_`, `scaler_m.mean_`, and `scaler_m.var_` are bit-for-bit identical (`rtol=0, atol=0`).

---

### 28. Realistic OHLC Execution Accounting

- **Stop-Loss & Take-Profit Opening Gaps:** In `simulate_trades_ohlc()`, if a candle opens beyond the stop trigger (`curr_open <= sl_price` for LONG, `curr_open >= sl_price` for SHORT), the trade exits at the worse opening gap price `curr_open` with slippage and `exit_reason = "stop_loss_gap"`.
- **Itemized PnL Decomposition:** Every trade record reports:
  - `gross_price_return`: unadjusted price return between `raw_entry_price` and `raw_exit_price`
  - `entry_slippage_cost` and `exit_slippage_cost` (`total_slippage_cost`)
  - `entry_fee_cost` and `exit_fee_cost`
  - `funding_cost`: `0.0` (with `funding_modeling_note: "not_modeled_in_ohlc_simulation"`)
  - `net_return`: net realized return after slippage and fees
- **Compounded Equity Curve & Intratrade Drawdown:**
  - Tracks hourly mark-to-market equity using intratrade adverse excursion (`curr_low` for LONG, `curr_high` for SHORT) to compute true maximum drawdown (`max_drawdown_pct`).
  - Computes compounded net return (`compounded_net_return_pct`) from the multiplicative equity curve $\prod (1 + r_m) - 1$, reported separately from the uncompounded arithmetic sum (`sum_net_pnl_pct`).
- Verified by `test_ohlc_stop_loss_gap_and_itemized_accounting`.

---

### 29. Pre-Test and Post-Test Artifact Hash Verification

| Artifact Path | Pre-Test SHA-256 | Post-Test SHA-256 | Changed? |
|---|---|---|---|
| `checkpoints/best_model.pt` | `0dfa18e77e5afaafa94f12ad239ba80379a418f77569ccd50aa261b1328789f2` | `0dfa18e77e5afaafa94f12ad239ba80379a418f77569ccd50aa261b1328789f2` | **No (0 diff)** |
| `checkpoints/scaler_candle.pkl` | `7104dc5965e75a6cee2224ff4a97cd00fe9c17c5d847ca5646d32e0ac8ad401d` | `7104dc5965e75a6cee2224ff4a97cd00fe9c17c5d847ca5646d32e0ac8ad401d` | **No (0 diff)** |
| `checkpoints/scaler_market.pkl` | `b94a1c728b0a090c3ff7215ecc03a149ee19c9c5cfa996bc98e6d2391716723c` | `b94a1c728b0a090c3ff7215ecc03a149ee19c9c5cfa996bc98e6d2391716723c` | **No (0 diff)** |
| `baseline_pre_retrain/best_model.pt` | `8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95` | `8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95` | **No (0 diff)** |
| `baseline_pre_retrain/scaler_candle.pkl` | `d5e5e30533e3ad133dd9385a5d84f3c0793c96ec91d1b7db12e97722a19bb19c` | `d5e5e30533e3ad133dd9385a5d84f3c0793c96ec91d1b7db12e97722a19bb19c` | **No (0 diff)** |
| `baseline_pre_retrain/scaler_market.pkl` | `d7bc7ebdc32810f1c65eb6ad51273cccb380cf20882e0513fa009e0387390e2d` | `d7bc7ebdc32810f1c65eb6ad51273cccb380cf20882e0513fa009e0387390e2d` | **No (0 diff)** |

---

### 30. Remaining Limitations

1. **Overwritten Unweighted Active Artifact (`ac224bb7...`):** Because Pass 4's `test_train_py_full_orchestration_mocked()` ran before path isolation was added, the single unsuffixed `checkpoints/best_model.pt` (`ac224bb7...`) was overwritten by a 5-epoch synthetic smoke-test checkpoint (`0dfa18e7...`). All pre-retrain baseline artifacts (`8b7b979e...`) and suffixed 20,000-candle multi-seed checkpoints (`best_model_42.pt`, `best_model_1337.pt`, `best_model_7.pt`) remain intact. A fresh controlled production training run (once approved) or explicit promotion of a verified checkpoint will be needed before live deployment.
2. **Historical Perpetual Funding Cost in OHLC Walk-Forward Simulator:** While `simulate_trades_ohlc()` now itemizes `funding_cost: 0.0` with explicit disclosure (`"not_modeled_in_ohlc_simulation"`), integrating historical 8-hour perpetual funding rate schedules into trade-level PnL requires passing aligned historical funding rate series into `simulate_trades_ohlc()`.
3. **1-Hour OHLC Granularity for Intrabar Collisions:** When both TP and SL are touched within the same 1-hour candle during walk-forward simulation, the simulator conservatively resolves `stop_loss_collision` (stop-first) unless lower-timeframe (e.g., 1-minute or 15-minute) sub-bars are supplied.


---

## Final Targeted Correction: Walk-Forward Pre-Roll Exclusion & Warm-Up Enforcement (2026-11-04)

> **Baseline Commit Inspected:** `dcf18cf215cc4aa7a358b65ee9e541dd4577d6fe`  
> **Status:** Walk-forward pre-roll exclusion and 720-bar Fibonacci warm-up validation enforced; 3 new targeted regression tests added; all 41 `unittest discover` tests + 8 `test_audit_bridge.py` tests pass with zero artifact mutation.

### 31. Defect Confirmation & Resolution

#### A. Defect Confirmed in Commit `dcf18cf`
- In `train.py`, `requested_start_ts` was computed after data fetching (`df_raw.index[-limit]`), but the `if args.walk_forward:` branch invoked `run_walk_forward_evaluation(df_raw, ...)` without passing `requested_start_ts` or `min_warmup_bars`.
- Inside `model/walk_forward.py`, `run_walk_forward_evaluation()` called `build_features(df_raw, add_targets=False)` without `requested_start_ts`, skipped the 720-bar pre-`fillna(0.0)` Fibonacci warm-up validation, and calculated `N_total = len(df_feat)` across the full dataset including the 720 pre-roll candles.

#### B. Old vs. New Walk-Forward Preprocessing Order

| Step | Old Order (`dcf18cf`) | New Order (Corrected) |
|---|---|---|
| 1 | Receive `df_raw` (including pre-roll) | Receive `df_raw` (including pre-roll), `requested_start_ts`, and `min_warmup_bars=720` |
| 2 | Call `build_features(df_raw, add_targets=False)` without `requested_start_ts` | Compute indicators on complete `df_raw` and validate $\ge 720$ valid prior OHLC bars + non-`NaN` raw `fib_618_dist_180d` at `requested_start_ts` **before** `fillna(0.0)` |
| 3 | Add TBM labels on unsliced `df_feat` (including pre-roll) | Slice retained dataset: `df_feat = df_feat_full.loc[requested_start_ts:].copy()` |
| 4 | Compute `N_total = len(df_feat)` (including pre-roll) | Generate TBM labels (`add_tbm_target`) strictly on the sliced retained `df_feat` |
| 5 | Construct fold boundaries over pre-roll + requested bars | Compute `N_total = len(df_feat)` and construct all outer-train, inner-train, inner-val, and outer-test fold boundaries strictly on the retained dataset |

#### C. Walk-Forward Dataset Metadata (`dataset_metadata`)
`run_walk_forward_evaluation()` now returns `dataset_metadata` containing:
- `requested_start_ts`: Timestamp where requested dataset begins
- `retained_start_ts`: First timestamp of retained `df_feat` (`== requested_start_ts`)
- `retained_end_ts`: Final timestamp of retained `df_feat`
- `raw_rows`: Total rows in `df_raw` (pre-roll + requested)
- `pre_roll_rows_excluded`: Exact count of excluded pre-roll rows (`720`)
- `retained_rows`: Exact count of retained rows (`N_total`)
- `min_warmup_bars`: Enforced minimum pre-roll warm-up bars (`720`)
- `fib_warmup_validation`: Output dictionary from `validate_fibonacci_warmup()`

#### D. Targeted Regression Tests (`TestWalkForwardPreRollExclusion`)
1. **`test_walk_forward_preroll_exclusion` (Test A):** Feeds 720 pre-roll + 360 requested candles (`1,080` total) with `requested_start_ts = df_raw.index[720]`. Verifies `pre_roll_rows_excluded == 720`, `retained_rows == 360`, `retained_start_ts == requested_start_ts`, and asserts that all fold `train_timestamps`, `inner_val_timestamps`, `outer_train_timestamps`, `test_timestamps`, `first_decision_ts`, and trade `decision_time` / `entry_time` / `exit_time` are $\ge \text{requested\_start\_ts}$.
2. **`test_walk_forward_rejects_insufficient_preroll` (Test B):** Feeds 719 pre-roll + 360 requested candles with `requested_start_ts = df_raw.index[719]`. Verifies `run_walk_forward_evaluation()` raises `ValueError` (`Fibonacci warm-up validation failed: only 719 valid historical observations...`) before `HybridModel` is ever instantiated.
3. **`test_train_entrypoint_passes_requested_start_ts_to_walk_forward` (Test C):** Verifies `train.py --walk-forward` passes its computed `requested_start_ts` and `min_warmup_bars=720` directly to `run_walk_forward_evaluation()` without touching any checkpoint or scaler files.
