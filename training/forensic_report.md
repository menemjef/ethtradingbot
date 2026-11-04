# Quantitative Research Audit: Retraining Pipeline Forensic Report

> **Branch:** `chatgpt-audit-review`  
> **Repository:** `menemjef/ethtradingbot`  
> **Date:** 2026-11-04  
> **Status:** Forensic Audit & Pipeline Corrections Complete (Pre-Retraining Review)

---

## Executive Summary

Following an independent review of the baseline retraining run and published codebase, an in-depth forensic investigation was conducted across the data ingestion, labeling, preprocessing, model objective, and evaluation components. 

Six critical issues were identified and systematically resolved. In addition, an execution stall during test execution was isolated, diagnosed, and resolved. All unit and integration test suites (19 pipeline tests, 8 audit bridge tests) now pass with 100% success.

No live trading, private Binance API connections, EC2 deployments, or unauthorized strategy/architecture changes were performed. All baseline model checkpoints and scalers remain securely preserved.

---

## 1. Issue 1: Historical Data Fetching Bug (Stale Dataset)

### 1.1 Root Cause Analysis
The previous `fetch_ohlcv()` implementation in `data/fetcher.py` attempted to calculate an initial start timestamp via:
```python
since = exchange.milliseconds() - ((limit + buffer) * tf_ms)
```
and then paginated forward. However, due to Binance USDM pagination constraints, API limits, and historical data gaps, the forward-fetching loop terminated prematurely at:
- **Last Fetched Timestamp:** `2026-08-13 04:00:00 UTC`
- **Execution Date:** `2026-11-04 17:40:00 UTC`
- **Temporal Gap:** Approximately 83 days (~2,000 hours / 2.7 months) of stale, missing data.

Consequently, the baseline model was trained on historical data ending nearly three months in the past, completely omitting recent market regimes.

### 1.2 Engineering Corrections
`data/fetcher.py` was refactored with the following guarantees:
1. **Anchor to Latest Completed Candle:** Calculates the most recent closed candle open time:
   $$\text{latest\_completed\_open} = \left(\lfloor \text{now\_ms} / \text{tf\_ms} \rfloor \times \text{tf\_ms}\right) - \text{tf\_ms}$$
2. **Deterministic Backward Range Target:** Sets `target_end_ms = latest_completed_open` and `target_start_ms = latest_completed_open - (total_needed - 1) * tf_ms`.
3. **Strict Exclusion of Unclosed Candles:** By default (`exclude_unclosed=True`), drops any candle whose open timestamp corresponds to the currently forming bar ($\text{timestamp} > \text{now\_ms} - \text{tf\_ms}$).
4. **Data Quality Verification:**
   - Detects and logs duplicate timestamps (`df.drop_duplicates(subset="open_time", keep="last")`).
   - Verifies continuous chronological intervals, logging warnings if intervals are missing.
   - Detects and explicitly warns on exchange data shortfall (`len(df) < limit`).
5. **Indicator Warm-up Pre-roll:** Added `warmup_bars` parameter to ensure indicators have sufficient historical lookback without truncating the requested dataset size.

---

## 2. Issue 2: TBM Configuration Parameter Discrepancy

### 2.1 Investigation & Reconciliation
In the initial retraining report, it was claimed that the Triple Barrier Method (TBM) used:
- Profit Target: $3.0 \times \text{ATR}$
- Stop Loss: $1.5 \times \text{ATR}$

However, inspection of `config.py` revealed:
```python
TBM_PT_MULT = 2.0  # Take-profit ATR multiplier
TBM_SL_MULT = 2.0  # Stop-loss ATR multiplier
TARGET_HORIZON = 12  # Maximum holding bars
```

### 2.2 Forensic Origin
The $3.0 / 1.5$ figures were traced to `live/trader.py` within `_calculate_position_size()`:
```python
sl_dist = atr * 1.5
tp_dist = atr * 3.0
```
These parameters govern live order risk/reward execution brackets, **not** dataset labeling. The previous report author conflated live execution brackets with data labeling hyperparameters. 

The baseline training run actually executed with symmetric $2.0 / 2.0$ multipliers, which produced a balanced dataset ($50.72\%$ Long, $49.28\%$ Short). Tests have been added to guarantee that `config.py` parameters (`TBM_PT_MULT`, `TBM_SL_MULT`, `TARGET_HORIZON`) strictly propagate into `add_tbm_target()`.

---

## 3. Issue 3: Evaluation Metric Mismatch (Classification vs. Regression)

### 3.1 Forensic Finding
In the baseline retraining report, a validation accuracy of **$54.88\%$** was highlighted, suggesting a modest predictive edge. 

Forensic auditing of `model/trainer.py` revealed that this metric was calculated exclusively from the **classification head** logits:
```python
# Legacy metric calculation:
pred_label = (torch.sigmoid(clf_out) > 0.5).float()
```
However, in production execution (`live/signal_engine.py` and `live/trader.py`), trade decisions are driven exclusively by the **regression head**:
```python
pred = float(reg_out.squeeze().cpu().numpy()) / TARGET_MULTIPLIER
if pred > SIGNAL_BUY_THRESHOLD:
    signal = "BUY"
elif pred < SIGNAL_SELL_THRESHOLD:
    signal = "SELL"
```

### 3.2 Empirical Verification
Evaluating both heads independently on the validation set revealed:
- **Classification Head Directional Accuracy:** $54.88\%$
- **Regression Head Directional Accuracy:** **$50.15\%$** (indistinguishable from a random coin toss)
- **Threshold-Filtered Regression Accuracy ($|\text{pred}| \ge 0.0450$):** **$48.51\%$** (worse than random guessing)

The reported $54.88\%$ accuracy created a false impression of model competency. The model that actually drives orders possessed zero statistical edge.

### 3.3 Engineering Corrections
`model/trainer.py` now implements `compute_comprehensive_metrics()`, which independently reports:
- `trading_signal_directional_accuracy` (regression head / `TARGET_MULTIPLIER` — matching live trading)
- `classification_directional_accuracy` (classification logits)
- `regression_mse`, `regression_mae`, `trivial_zero_mse`, `mse_vs_trivial_ratio`
- `classification_balanced_accuracy`, sensitivity/specificity, and confusion matrices
- Threshold-filtered accuracy with sample counts for both heads

---

## 4. Issue 4: Regression Objective Failure (MSE on Binary $\pm 1$ Labels)

### 4.1 Theoretical & Mathematical Flaw
The model architecture employs a multi-task head:
1. `clf_out`: Binary Cross Entropy on labels $\in \{0, 1\}$
2. `reg_out`: Mean Squared Error on labels $\in \{-1.0, +1.0\}$

For a binary target $y \in \{-1, +1\}$ with balanced distribution $P(y = +1) \approx P(y = -1) \approx 0.5$, the expected squared error for a constant prediction $\hat{y} = c$ is:
$$\mathbb{E}[(c - y)^2] = c^2 - 2c\mathbb{E}[y] + \mathbb{E}[y^2] = c^2 - 2c(0) + 1 = c^2 + 1$$
This is strictly minimized when $c = 0$, resulting in $\text{MSE} = 1.0000$.

### 4.2 Empirical Confirmation
During training:
- **Baseline Model Best Validation MSE:** **$1.0015$**
- **Trivial Zero Baseline MSE:** **$1.0000$**
- **Model Output Distribution:** $\text{mean} = 0.0295$, $\text{std} = 0.0230$, $\text{min} = -0.0479$, $\text{max} = 0.0812$.

The regression head simply learned to output values near zero, collapsing toward the unconditional expectation. Because $\pm 1$ TBM labels lack continuous magnitude information, optimizing with MSE penalizes directional conviction and encourages output collapse. (See accompanying document `training/regression_objective_analysis.md` for full research analysis and recommendations).

---

## 5. Issue 5: Temporal Leakage at Train/Validation Boundary

### 5.1 Leakage Mechanism
The Triple Barrier Method looks forward up to `TARGET_HORIZON = 12` hours to determine whether price hits PT or SL first. 

In the original pipeline:
```python
train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]
```
The targets for training observations between `split_idx - 12` and `split_idx - 1` evaluated future price movements extending into the validation timeline. Furthermore, fitting the feature scalers (`StandardScaler`) on the unpurged training split allowed validation-period price dynamics to inform training scaling parameters.

### 5.2 Implementation of Purged Splits
We introduced `split_train_val_purged()` in `data/preprocessor.py`:
- Calculates `purge_size = max(0, purge_horizon)` (where `purge_horizon = TARGET_HORIZON = 12`).
- Restricts training samples to indices $[0, \text{split\_idx} - \text{purge\_size})$.
- Discards the boundary samples $[ \text{split\_idx} - \text{purge\_size}, \text{split\_idx} )$.
- Begins validation evaluation strictly at `split_idx`.
- Restricts `StandardScaler.fit()` strictly to the purged training partition.

Mathematical proof of isolation is verified in `test_purge_prevents_target_overlap_with_validation`.

---

## 6. Issue 6: Ambiguous Candle Resolution in TBM

### 6.1 Problem Definition
In high-volatility 1-hour candles, both the Take Profit barrier ($+2.0 \times \text{ATR}$) and Stop Loss barrier ($-2.0 \times \text{ATR}$) can be breached within the same bar. Because standard OHLC data lacks intra-bar tick sequencing, it is impossible to determine whether PT or SL was hit first.

The legacy implementation used an "overshoot" heuristic:
$$\text{overshoot}_{\text{PT}} = \text{High} - \text{PT}, \quad \text{overshoot}_{\text{SL}} = \text{SL} - \text{Low}$$
If $\text{overshoot}_{\text{PT}} > \text{overshoot}_{\text{SL}}$, the bar was labeled as a winning trade (+1).

### 6.2 Quantitative Impact & Conservative Standard
An empirical scan across 5,000 hourly ETH candles revealed:
- Total bars: 4,988
- Ambiguous bars: 8 ($0.16\%$)
- Legacy resolution: 7 wins / 1 loss (a bullish bias of $0.14\%$)

While the quantitative impact is minor, the overshoot heuristic violates sound risk management. In `data/indicators.py`, `add_tbm_target()` was updated with an `ambiguous_resolution` parameter defaulting to `"stop_first"`, which conservatively assumes the stop-loss was triggered first (-1.0). Options for `"overshoot"` and `"exclude"` (label as NaN) are also supported.

---

## 7. Diagnostic & Resolution of Stalled Test Task-5423

### 7.1 Diagnostic Record
During initial verification of the test suite, execution stalled indefinitely on `task-5423`.
- **Log Path:** `.../.system_generated/tasks/task-5423.log`
- **Stalled Test:** `TestDataFetcherCorrections.test_duplicate_and_missing_interval_detection`
- **Last Terminal Output:**
  ```text
  test_duplicate_and_missing_interval_detection (__main__.TestDataFetcherCorrections)
  Verify that duplicate timestamps are deduplicated and missing intervals logged. ...
  ```
- **Process State:** Two Python child processes (PIDs 5416, 18520) were active, consuming $100\%$ of a CPU core (>200 CPU seconds accumulated).

### 7.2 Root Cause
The test fixture configured the exchange mock with a static return value:
```python
mock_exchange.fetch_ohlcv.return_value = mock_candles  # Fixed 5 candles (02:00 to 06:00 UTC)
```
In `data/fetcher.py`, the pagination loop:
1. Called `exchange.fetch_ohlcv(..., since=since, limit=fetch_n)`
2. Checked if `raw[-1][0] >= target_end_ms` ($06:00 < 11:00 \implies \text{False}$)
3. Checked if `len(raw) < fetch_n` ($5 < 5 \implies \text{False}$)
4. Advanced `since = raw[-1][0] + tf_ms`

Because `return_value` returned the exact same 5 candles on every call regardless of `since`, `raw[-1][0]` never reached `target_end_ms`, and `len(raw)` was never less than `fetch_n`. The loop ran forever.

### 7.3 Resolution & Verification
Both `test_duplicate_and_missing_interval_detection` and `test_shortfall_reporting_when_exchange_has_fewer_candles` were refactored to use dynamic `side_effect` functions:
```python
def mock_fetch(symbol, timeframe, since, limit):
    return [c for c in mock_candles if c[0] >= since][:limit]
mock_exchange.fetch_ohlcv.side_effect = mock_fetch
```
- Re-run with timeout: **PASSED in 0.050 seconds**.
- Full test suite execution: **19/19 tests PASSED in 1.332 seconds**.
- Audit bridge suite: **8/8 tests PASSED**.

---

## 8. Summary of Test Verification Results

```text
======================================================================
1. Retraining Pipeline Verification Suite (tests/test_retraining_corrections.py)
----------------------------------------------------------------------
test_duplicate_and_missing_interval_detection ... ok
test_pagination_and_latest_completed_candle ... ok
test_shortfall_reporting_when_exchange_has_fewer_candles ... ok
test_head_specific_metrics_separation ... ok
test_trivial_zero_regression_baseline ... ok
test_ambiguous_candle_stop_first_vs_overshoot ... ok
test_tbm_parameter_propagation ... ok
test_purge_prevents_target_overlap_with_validation ... ok
test_scaler_fitting_restricted_to_purged_training ... ok

2. Full System Integration Suite (tests/test_full_pipeline.py)
----------------------------------------------------------------------
test_circuit_breaker_on_repeated_exchange_errors ... ok
test_cooldown_blocks_reentry ... ok
test_long_signal_pipeline ... ok
test_minimum_notional_compliance ... ok
test_short_signal_pipeline ... ok
test_weak_signal_waiting ... ok

3. Risk & Signal Management Suites (tests/test_risk_manager.py, test_signal_engine.py)
----------------------------------------------------------------------
test_daily_reset ... ok
test_drawdown_limit_long ... ok
test_leverage_cap_scaling ... ok
test_inference_pipeline ... ok

Ran 19 tests in 1.332s — OK (0 failures, 0 errors)
======================================================================
Audit Bridge V2 Suite (tests/test_audit_bridge.py)
----------------------------------------------------------------------
Results: 8 passed, 0 failed, 8 total
======================================================================
```

---

## 9. Next Steps
1. Review accompanying mathematical document: `training/regression_objective_analysis.md`.
2. Await independent ChatGPT audit approval before initiating any retraining runs.
3. Upon approval, execute baseline retraining with clean data ingestion, purged validation boundaries, and decoupled metric evaluation.
