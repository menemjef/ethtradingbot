# Resolution of Outstanding Audit Findings (Pass 5 Review)

## 1. Issue 1: Prediction/Target One-Candle Alignment
- **Original Defect:** The model generated predictions for an entry at candle $j$ open using information available up to candle $j-1$ close. However, predictive metric evaluation compared `test_signals[j]` against `df_test_window["target"].iloc[j]`, creating a 1-candle forward lookahead/misalignment.
- **Root Cause:** In the target generator (`add_tbm_target`), row $k$ target is calculated from `close[k]` predicting price action starting at bar $k+1$. Therefore, the target associated with decision bar $j-1$ is strictly `df_feat["target"].iloc[j-1]`.
- **Exact Code Fix:** In `model/walk_forward.py` (lines 675-715) and research runner:
  - Predictions made at candle $j-1$ close are evaluated against `df_feat["target"].iloc[te_start_idx + j - 1]`.
  - The final bar $j = N_{	ext{test}} - 1$ is strictly excluded from predictive evaluation because it cannot execute an entry in that fold.
  - Separate timestamps recorded: $t_{	ext{decision}}$, $t_{	ext{target}}$, $t_{	ext{execution}}$.
  - Asserted $t_{	ext{decision}} < t_{	ext{execution}}$ and $t_{	ext{target}} == t_{	ext{decision}}$.
- **Regression Test:** `tests/test_audit_findings_fixes.py::test_01_prediction_target_one_candle_alignment` and `test_02_chronological_timestamp_ordering_and_final_bar_exclusion`. Both pass (100% directional accuracy on synthetic inverted target dataset vs 0% on flawed alignment).

## 2. Issue 2: MajorityBaseline Aggregation & Distinction
- **Original Defect:** `research/backtest_metrics.json` reported 380 trades at fold level but 0.0% win rate and zero gross profit/loss at the aggregate level, creating an internal contradiction with the compounded loss (-36.51%).
- **Root Cause:** In `simulate_trades_ohlc`, `sim_maj` trades were generated but never appended to `all_oos_trades`. When aggregation filtered `all_oos_trades` by model name, `MajorityBaseline` was empty and took the fallback zeros branch while compounding fold returns directly.
- **Exact Code Fix:**
  - Added full MajorityBaseline trade logging to `all_oos_trades` and `research/trade_log_oos.csv`.
  - Unified metrics computation in `utils/quant_evaluation.py::compute_reconciled_trade_metrics`.
  - Clearly separated:
    - `MajorityClassPredictiveBaseline`: Statistical predictor outputting training set majority class (+1 or -1).
    - `MajorityBaseline` (Always Long): Executable trading baseline entering continuous long positions.
- **Regression Test:** `tests/test_audit_findings_fixes.py::test_03_majority_baseline_trade_logging_and_reconciliation`.

## 3. Issue 3: HybridModel Edge & Directional Asymmetry Analysis
- **Original Defect:** HybridModel generated 59 trades across 3,500 bars, all LONG, win rate 47.46%, profit factor 0.629, expectancy -25.33 bps, -14.27% unlevered return.
- **Investigation & Findings:**
  1. Prediction Distribution: Raw regression output has a positive mean bias ($+0.0335$, median $+0.0313$).
  2. With symmetric threshold $0.045$, the upper bound is at the 65th percentile (1,201 trigger bars), while the lower bound $-0.045$ is at the 99.9th negative percentile (only 3 trigger bars across 3,500 candles).
  3. The model suffers from severe directional asymmetry and does not possess an edge over simple baselines after 5 bps taker fees and 3 bps slippage.
- **Recommendation:** Do not promote current HybridModel weights to production.

## 4. Issue 4: Dependence-Aware Statistical Uncertainty
- **Original Defect:** Previous confidence intervals used naive IID standard errors ($p \pm 1.96 \sqrt{p(1-p)/n}$), ignoring temporal dependence and volatility clustering in financial time series.
- **Exact Code Fix:** Implemented in `utils/quant_evaluation.py`:
  - Moving Block Bootstrap (Kunsch 1989) with 2,000 resamples and 24-hour block size for both predictive metrics and trade expectancy.
  - Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014) penalizing strategy selection bias across multiple trials $K$.
- **Regression Test:** `tests/test_audit_findings_fixes.py::test_04_moving_block_bootstrap_preserves_temporal_dependence` and `test_05_deflated_sharpe_ratio_penalizes_multiple_trials`.

## 5. Issue 5: Reproducibility Manifest & Checkpoint Provenance
- **Original Defect:** Lack of a deterministic manifest clarifying whether baseline checkpoints or retrained models generated reported numbers, and missing artifact SHA-256 hashes.
- **Exact Code Fix:** Created `research/research_reproducibility_manifest.json` cataloging:
  - Commit SHAs, environment versions, random seeds.
  - Parquet dataset hashes for raw candles and funding rates.
  - Frozen baseline checkpoint (`8b7b979e...`) and scaler hashes vs synthetic Pass 4 dummy checkpoint (`0dfa18e7...`).
  - Reproduction command: `python scratch/run_strategy_research.py`.
