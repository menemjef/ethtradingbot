# Backtest Engine Validation Audit Report

**Repository:** `menemjef/ethtradingbot`  
**Audit Branch:** `audit/backtest-engine-validation` (branched from `fix/execution-safety-and-backtest-integrity` @ `30daaa343a28c3ad5e168b4cf047a9d794a86e1f`)  
**Scope:** Common backtest & evaluation engine (`utils/quant_evaluation.py`, `research/scripts/run_phase_2_1_research.py`, `research/scripts/run_swing_research.py`, `research/scripts/run_candidate_a_attribution.py`, `research/scripts/generate_experiment_registry.py`, `backtest.py`, `backtest_audit.py`)  
**Validation Test Suite:** `tests/test_backtest_engine_validation.py` (12 test methods, including explicit mutation checks)

---

## 1. Executive Summary Matrix

| # | Audit Item | Primary Engine (`run_phase_2_1_research.py` + `quant_evaluation.py`) | Legacy / ML Scripts (`run_swing_research.py`, `backtest.py`, `backtest_audit.py`) | Validation Test & Mutation Check |
| :- | :--- | :--- | :--- | :--- |
| **0** | **Symbol-Agnostic Execution** | **Fixed** (parameterized `symbol: str = "ETHUSDT"`, removed hardcoded ETH paths/labels) | **Fixed** (added `symbol: str = SYMBOL` parameter to `run_backtest`, `simulate_trades`, `run_audit`, `run_swing_research_pipeline`, `run_candidate_a_attribution.main`) | `test_11_symbol_agnostic_execution_no_hardcoded_eth` (`BTCUSDT`, `SOLUSDT`) |
| **1** | **Timestamps & Multi-TF Aggregation (4H/6H/1D)** | **Bug Found & Fixed** (added strict UTC/closed/no-dup/no-gap validation, 6H support, and fixed trailing 1D `.shift(1)` drop on incomplete final day) | **Bug Found** (`run_swing_research.py` lacked gap/UTC validation and dropped last completed 1D bar before incomplete final day) | `test_01_timestamps_utc_closed_no_duplicates_no_gaps_and_mtf_aggregation` + Mutation 1 |
| **2** | **No Look-Ahead (Signals $\le t$, Fill at Next Open)** | **Pass / Minor Fix** (indicators & signals strictly causal $\le t$, fill at `ts_4h[k] + 4h` next 1H open; fixed last-bar loop bound `range(len(df_4h_signals))`) | **Bug Found & Fixed** (`backtest.py` and `backtest_audit.py` filled at `close[i-1]` instead of `open[i]` and updated break-even `sl_price` before same-bar SL check; `run_swing_research.py` had same-bar trailing stop lookahead) | `test_02_no_lookahead_next_bar_open_fill_and_future_shift_invariance` + Mutation 2 |
| **3** | **Fills (Intrabar High/Low, Stop-First Collision, Gaps, Slippage & Fees)** | **Pass** (`simulate_swing_trades_conservative` enforces `stop_loss_collision`, `stop_loss_gap` at `curr_open` with slippage, and entry/exit slippage + taker fees on both sides) | **Bug Found & Fixed** (`backtest.py` and `backtest_audit.py` filled gaps at `sl_price` instead of `open_j` and applied fees only to entry notional) | `test_03_fills_intrabar_sl_tp_collision_stop_wins_gaps_slippage_and_fees` + Mutation 3 |
| **4** | **Funding (Actual Settlement Timestamps, Sign, Open-Only, No Future Rows)** | **Pass** (`normalize_funding_series` rounds sub-second Binance timestamps to `"h"`, matches 4,090/4,090 settlements, applies `side * rate` only while open, zero future leak) | **Bug Found & Fixed** (`run_swing_research.py` string-matching on microsecond timestamps dropped 100% of funding settlements; `backtest.py`/`backtest_audit.py` ignored funding) | `test_04_funding_settlement_timestamps_sign_open_only_no_future_rows` + Mutation 4 |
| **5** | **Sizing & Leverage (Risk Sizing, Leverage Cap, Available Cash Margin)** | **Fixed** (`calculate_actual_stop_position_sizing` now enforces `available_cash` margin cap in addition to `max_leverage = 3.0`) | **Bug Found & Fixed** (`backtest.py` and `backtest_audit.py` used uncapped `risk_usd / sl_dist` without leverage or cash margin bounds) | `test_05_sizing_and_leverage_cap_and_available_cash_margin_constraint` + Mutation 5 |
| **6** | **Accounting (`equity = cash + margin + unrealized PnL`, Reconciliation)** | **Pass / Fixed** (step-by-step identity holds to $0.00$; fixed pooled OOS trade PnL scaling across compounded folds so `pnl_reconciliation_difference_usd == 0.00`) | **Bug Found** (`backtest.py` and `backtest_audit.py` only updated equity upon trade exit, omitting hourly mark-to-market accounting) | `test_06_accounting_identity_no_double_counting_and_pnl_reconciliation` + Mutation 6 |
| **7** | **Drawdown (Intratrade Lows) & Metrics Verification** | **Bug Found & Fixed** (`simulate_swing_trades_conservative` computed `eq_lowest_fold`, but `run_phase_2_1_research.py` omitted `hourly_lowest_equity=eq_lowest_fold` at lines 1106, 1169, 1239; `quant_evaluation.py` used `np.minimum` only when passed and omitted `seed=seed` forwarding) | **Bug Found** (`backtest.py` and `backtest_audit.py` computed drawdown only on closed-trade balances) | `test_07_drawdown_intratrade_lows_and_hand_computed_metrics_verification` + Mutation 7 |
| **8** | **Segmentation (Non-Overlapping Folds, 168h Purge, Holdout Isolation)** | **Bug Found & Fixed** (5 folds & 168h purge are non-overlapping, but Holdout Fold 5 was included in `analyze_rolling_volatility_transitions`, `run_candidate_a_attribution.py`, and Phase 2 `run_swing_research.py`) | **Bug Found & Fixed** (`backtest.py` used 80/20 split with zero purge window; fixed to purge `LOOKBACK + MAX_HOLD_BARS`) | `test_08_segmentation_non_overlapping_folds_purge_and_holdout_isolation` |
| **9** | **Reproducibility (Identical Trades & Metrics Twice)** | **Fixed** (forwarded `seed=seed` from `compute_continuous_portfolio_metrics` to `compute_reconciled_trade_metrics`) | **Pass** (when deterministic seed is set) | `test_09_reproducibility_identical_trades_and_metrics_across_runs` + Mutation 9 |
| **10** | **Registry Consistency (`experiment_registry.json` vs Reports)** | **Bug Found & Fixed** (`generate_experiment_registry.py` had hardcoded static strings `-34.6 bps` and `-61.2 bps` and wrong SL/TP params instead of reading `phase_2_1_fold_metrics.json` `+1.01 bps` and `+74.40 bps`) | N/A | `test_10_experiment_registry_reconciliation_and_donchian_consistency` |

---

## 2. Detailed Per-Item Audit Findings, Fixes, and Impact on Prior Results

### Item 0: Symbol-Agnostic Engine Requirement
- **Status:** `fixed`
- **Files & Functions:**
  - `research/scripts/run_phase_2_1_research.py`: `simulate_swing_trades_conservative()`, `compute_buy_and_hold_equity()`, `run_phase_2_1_pipeline()`
  - `research/scripts/run_swing_research.py`: `simulate_swing_trades()`, `run_swing_research_pipeline()`
  - `research/scripts/run_candidate_a_attribution.py`: `main()`
  - `backtest.py`: `run_backtest()`
  - `backtest_audit.py`: `simulate_trades()`, `run_audit()`
- **Finding:** All five scripts previously hardcoded `"ETHUSDT"` / `SYMBOL` or `"eth_usdt_1h_3yr.parquet"`.
- **Fix:** Added `symbol: str` parameter across all simulation, benchmark, and pipeline functions, recording `"symbol": symbol` on every trade record and deriving default dataset cache filenames from `symbol` when paths are not explicitly supplied.
- **Impact on Prior Results:** None on `ETHUSDT` numbers; enables multi-asset validation (`BTCUSDT`, `SOLUSDT`, etc.).

---

### Item 1: Timestamps & Multi-Timeframe Aggregation (4H / 6H / 1D)
- **Status:** `bug found` -> `fixed`
- **Files & Functions:**
  - `utils/quant_evaluation.py`: `validate_ohlcv_timestamps()`, `resample_completed_ohlcv()`
  - `research/scripts/run_phase_2_1_research.py`: `build_multitimeframe_features()` (lines 88–206)
  - `research/scripts/run_swing_research.py`: `build_multitimeframe_features()` (lines 111–215)
- **Failure Scenarios Identified:**
  1. **Missing Timestamp Invariant Checks:** Neither `build_multitimeframe_features()` function checked whether input 1H bars were timezone-aware UTC, strictly closed (`open_time + 1h <= reference_now`), free of duplicate timestamps, or free of missing 1H gaps.
  2. **No 6H Aggregation Support:** Only 4H (`counts == 4`) and 1D (`counts == 24`) were implemented.
  3. **Trailing 1D Context Drop (`df_1d.shift(1)` bug):** In `build_multitimeframe_features()` (`run_phase_2_1_research.py` line 194), completed 1D bars were filtered via `df_1d = df_1d[counts_1d == 24]` and then shifted via `df_1d_shifted = df_1d.shift(1)` before `.reindex(df_4h.index, method="ffill")`. Because the final day in `eth_usdt_1h_3yr.parquet` (`2026-11-04`, 19 hours) is incomplete and removed from `df_1d`, `.shift(1)` shifted `2026-11-02` onto `2026-11-03` and **dropped the completed `2026-11-03` 1D bar entirely**, causing all 4H bars on `2026-11-04` to use stale `2026-11-02` 1D features instead of `2026-11-03`.
- **Fix:**
  - Added `validate_ohlcv_timestamps()` and `resample_completed_ohlcv()` (supporting `4h`, `6h`, `1D`) in `utils/quant_evaluation.py`.
  - Indexed completed 1D bars by their completion timestamp (`df_1d_completed.index = df_1d.index + pd.Timedelta(days=1)`) before forward-filling onto `df_4h.index`.
- **Impact on Prior Results:**
  - On `eth_usdt_1h_3yr.parquet`, both `2026-11-02` and `2026-11-03` had `1d_trend_bullish = False`, `1d_trend_bearish = False`, and `1d_adx_14 > 30`, so **zero historical signals or trades changed** in Phase 2.1.

---

### Item 2: No Look-Ahead (Indicators $\le t$, Signal at Bar Close, Fill at Next Bar Open)
- **Status:**
  - `run_phase_2_1_research.py`: `pass` (with minor last-bar mapping fix `exec_t = ts_4h[k] + pd.Timedelta(hours=4)`)
  - `backtest.py`, `backtest_audit.py`, `run_swing_research.py`: `bug found` -> `fixed` in `backtest.py` and `backtest_audit.py`
- **Files & Functions:**
  - `research/scripts/run_phase_2_1_research.py`: `build_multitimeframe_features()`, `generate_swing_signals()`, `simulate_swing_trades_conservative()` (lines 571–590)
  - `backtest.py`: `run_backtest()` (lines 101, 141–157)
  - `backtest_audit.py`: `simulate_trades()` (lines 121, 159–171)
  - `research/scripts/run_swing_research.py`: `simulate_swing_trades()` (lines 515–538)
- **Failure Scenarios Identified:**
  1. **Decision-Bar Close Fill in `backtest.py` (line 101) & `backtest_audit.py` (line 121):** Entered at `df_feat["close"].iloc[i - 1]` (the close of decision bar `i-1`) instead of `df_feat["open"].iloc[i]` (the open of next bar `i`).
  2. **Intra-Bar Break-Even Look-Ahead in `backtest.py` (lines 142, 156) & `backtest_audit.py` (lines 159, 170):** If bar `j` reached `+0.5 * tp_dist`, `sl_price` was moved to `entry_price` for subsequent bars, and in `run_swing_research.py` (lines 515–538) `highest_price` and `stop_price` were updated from `curr_high` *before* checking `curr_low <= stop_price` on the same bar `j` (already repaired in `run_phase_2_1_research.py` lines 740–751).
- **Impact on Prior Results:**
  - **Phase 1 ML results (`backtest.py`, `backtest_audit.py`):** Invalidated by decision-close entry and uncapped sizing (Phase 1 ML was already rejected).
  - **Phase 2 (`run_swing_research.py`, `swing_strategy_metrics.json`):** `Exp1_Donchian_55_Turtle_4H` and `Phase4_Combo_Regime_Selector` (which use `trailing_atr`) were affected by same-bar trailing stop lookahead in Phase 2, which was already fixed in Phase 2.1 (`run_phase_2_1_research.py`).
  - **Phase 2.1 (`run_phase_2_1_research.py`, Candidate A, Donchian 120h, Donchian 168h):** Strictly causal (`pass`). Future-shift invariance test confirms zero signal or indicator changes when future bars $> t$ are shocked by $6\times$.

---

### Item 3: Fills (Intrabar High/Low, Stop-First Collision, Opening Gaps, Slippage & Fees)
- **Status:**
  - `run_phase_2_1_research.py` (`simulate_swing_trades_conservative`): `pass`
  - `backtest.py` & `backtest_audit.py`: `bug found` -> `fixed`
- **Files & Functions:**
  - `research/scripts/run_phase_2_1_research.py`: `simulate_swing_trades_conservative()` (lines 653–709, 772–818)
  - `backtest.py`: `run_backtest()` (lines 126–178)
  - `backtest_audit.py`: `simulate_trades()` (lines 145–190)
- **Failure Scenarios Identified:**
  - In `simulate_swing_trades_conservative()`, entry-bar and subsequent-bar opening gaps (`curr_open <= sl_p` for LONG, `curr_open >= sl_p` for SHORT) fill at `curr_open` with adverse slippage (`stop_loss_gap`), same-bar SL+TP collisions resolve stop-first (`stop_loss_collision`), and both entry and exit apply adverse slippage (`0.0003`) and taker fees (`0.0005`) for LONG and SHORT (`pass`).
  - In `backtest.py` and `backtest_audit.py`, opening gaps past `sl_price` filled optimistically at `sl_price` instead of `open_j`, and fees were charged as `2 * notional * COMMISSION_RATE` instead of `(entry_notional + exit_notional) * COMMISSION_RATE`. Fixed in both files.
- **Impact on Prior Results:**
  - Phase 2.1 results (`Candidate A`, `Donchian 120h`, `Donchian 168h`) already used `simulate_swing_trades_conservative` (`pass`, zero change).

---

### Item 4: Funding (Settlement Timestamps, Sign Convention, Open-Only, No Future Rows)
- **Status:**
  - `run_phase_2_1_research.py`: `pass`
  - `run_swing_research.py`, `backtest.py`, `backtest_audit.py`: `bug found` (`fixed` in `backtest.py` and `backtest_audit.py`)
- **Files & Functions:**
  - `research/scripts/run_phase_2_1_research.py`: `normalize_funding_series()`, `simulate_swing_trades_conservative()` (lines 621–653, 801–853)
  - `research/scripts/run_swing_research.py`: `simulate_swing_trades()` (lines 585–590, 731)
- **Failure Scenarios Identified:**
  1. **Phase 2 Microsecond Timestamp Mismatch (`run_swing_research.py` line 731):** Raw Binance funding timestamps contain sub-second offsets (e.g., `2023-02-10 00:00:00.020000+00:00`). `run_swing_research.py` keyed `funding_map` by `str(ts)` without rounding to `"h"`, causing **100% of the 4,090 funding settlements to be missed** (`funding_cost == 0.0` for all trades in `research/swing_strategy_metrics.json`).
  2. **Phase 2.1 Repair (`run_phase_2_1_research.py`):** `normalize_funding_series()` rounds timestamps to the nearest hour (`00:00`, `08:00`, `16:00` UTC), matching 100% (`4,090/4,090`) of settlements, applying `drag = side * rate` (longs pay when `rate > 0`, shorts receive) strictly while the position is open and only from `curr_t` (never from future rows).
- **Impact on Prior Results:**
  - **All Phase 2 (`research/swing_strategy_metrics.json`) returns and expectancies are invalid** (overstated for net-long strategies by 30–460 bps cumulative funding drag).
  - **Phase 2.1 (`research/phase_2_1_fold_metrics.json`)** properly accounted for all funding settlements (`0` missed settlements across all folds).

---

### Item 5: Sizing & Leverage (Risk-Based Sizing, Leverage Cap, Available Cash Margin)
- **Status:** `bug found` -> `fixed`
- **Files & Functions:**
  - `utils/quant_evaluation.py`: `calculate_actual_stop_position_sizing()` (lines 22–63)
  - `research/scripts/run_phase_2_1_research.py`: `simulate_swing_trades_conservative()` (line 767)
  - `backtest.py` (line 110) & `backtest_audit.py` (line 130)
- **Failure Scenarios Identified:**
  1. `backtest.py` and `backtest_audit.py` computed `position_size = (balance * RISK_PCT) / sl_dist` with **no leverage cap and no cash margin check**, allowing tight ATR stops to exceed `LEVERAGE` and account cash.
  2. `calculate_actual_stop_position_sizing()` capped notional at `account_equity * max_leverage`, but when `max_concurrent_positions > 1`, `simulate_swing_trades_conservative()` did not deduct `used_margin` of already-open positions from `available_cash`.
- **Fix:**
  - Added `available_cash: Optional[float] = None` and `"margin_required": float(notional / max_leverage)` to `calculate_actual_stop_position_sizing()`, capping notional at `min(account_equity, available_cash) * max_leverage`.
  - Passed `available_cash = max(0.0, account_equity - used_margin)` in `simulate_swing_trades_conservative()`, `backtest.py`, and `backtest_audit.py`.
- **Impact on Prior Results:**
  - In Phase 2.1, `max_concurrent_positions = 1` (`used_margin == 0` at every entry), so **zero Phase 2.1 trades or metrics changed**.

---

### Item 6: Accounting (`equity = cash + margin + unrealized PnL`, No Double Counting, PnL Reconciliation)
- **Status:** `pass` (per-fold) / `fixed` (pooled OOS compounded fold trade PnL scaling)
- **Files & Functions:**
  - `research/scripts/run_phase_2_1_research.py`: `simulate_swing_trades_conservative()` (lines 699–709, 876–904, 1149–1173)
  - `utils/quant_evaluation.py`: `compute_continuous_portfolio_metrics()` (line 692)
- **Failure Scenarios Identified:**
  1. Within each fold simulation (`simulate_swing_trades_conservative`), `hourly_equity[i] = free_cash + locked_margin + open_unrealized_pnl_close` holds with `0.0` error at every hour, fees and funding are deducted exactly once, and `sum(trade_pnl_usd) == final_equity - initial_capital` (`0.00` diff).
  2. However, when `run_phase_2_1_research.py` (lines 1149–1173) chained Folds 1–4 into a compounded `oos_eq_series` (scaling each fold's equity piece by `compounded_base / 10000.0`), it passed the unscaled per-fold `$10,000`-base `trade_pnl_usd` list into `compute_continuous_portfolio_metrics()`, producing a non-zero `pnl_reconciliation_difference_usd` on pooled OOS summaries (e.g., `$61.73` on Candidate A).
- **Fix:**
  - Added step-by-step `accounting_audit` telemetry (`max_abs_equity_identity_error`, `min_free_cash`, `max_effective_leverage`, `pnl_reconciliation_diff_usd`) to `simulate_swing_trades_conservative()`.
  - Scaled `trade_pnl_usd` by `compounded_base / eq_piece.iloc[0]` when constructing pooled `oos_trades` for `compute_continuous_portfolio_metrics()`, bringing pooled OOS `pnl_reconciliation_difference_usd` to `0.00`.

---

### Item 7: Drawdown (Intratrade Lows) & Metrics Verification
- **Status:** `bug found` -> `fixed`
- **Files & Functions:**
  - `research/scripts/run_phase_2_1_research.py`: `run_phase_2_1_pipeline()` (lines 1106, 1169, 1239)
  - `utils/quant_evaluation.py`: `compute_continuous_portfolio_metrics()` (lines 487–498, 683)
- **Failure Scenarios Identified:**
  1. **Omitted `hourly_lowest_equity` Argument in `run_phase_2_1_research.py` (lines 1106, 1169, 1239):** `simulate_swing_trades_conservative()` accurately computed and returned `eq_lowest_fold` (the hourly worst-tick equity curve using `curr_low` for LONG and `curr_high` for SHORT), **but `run_phase_2_1_research.py` never passed `hourly_lowest_equity=eq_lowest_fold` to `compute_continuous_portfolio_metrics()`** at lines 1106, 1169, or 1239 (in fact, line 1222 discarded it as `_`)! As a result, all reported `intratrade_max_drawdown_pct` values in `research/phase_2_1_fold_metrics.json` and `research/phase_2_1_candidate_comparison.md` were computed on **1H bar-close equity** rather than **1H worst-tick (intrabar low/high) equity**.
  2. In `utils/quant_evaluation.py` (line 489), `drawdown_curve` now takes `np.minimum(eq, hourly_lowest_equity.values.astype(np.float64))` to guarantee worst-tick equity is never above close equity.
- **Empirical Impact on Previously Reported Phase 2.1 Results:**
  - **`Exp1_Momentum_Continuation_4H` (Candidate A):**
    - Pooled OOS Max Drawdown changes from **`4.74%`** (1H close-only) to **`4.94%`** (`+0.20%` deeper with intrabar worst-tick lows; Fold 1 MaxDD `4.74%` $\rightarrow$ `4.94%`, Fold 2 `4.69%` $\rightarrow$ `4.81%`, Fold 3 `1.14%` $\rightarrow$ `1.69%`, Fold 4 `3.69%` $\rightarrow$ `4.16%`, Holdout Fold 5 `3.09%` $\rightarrow$ `3.75%`).
    - CAGR (`5.01%`), Daily Sharpe (`0.77`), Win Rate (`53.57%`), Profit Factor (`1.859`), and Expectancy (`+135.35 bps`) are **unchanged**.
  - **`Trend_Donchian_120h` (Candidate B):**
    - Pooled OOS Max Drawdown changes from **`20.42%`** to **`20.62%`** (`+0.20%` deeper; Holdout Fold 5 `7.30%` $\rightarrow$ `7.64%`). Expectancy (`+1.01 bps`) is **unchanged**.
  - **`Trend_Donchian_168h` (Candidate C):**
    - Pooled OOS Max Drawdown changes from **`23.94%`** to **`24.06%`** (`+0.12%` deeper; Holdout Fold 5 `9.07%` $\rightarrow$ `9.40%`). Expectancy (`+74.40 bps`) is **unchanged**.

---

### Item 8: Segmentation & Holdout Isolation Audit
- **Status:** `bug found` -> `fixed`
- **Files & Functions:**
  - `utils/quant_evaluation.py`: `validate_walk_forward_segmentation()`
  - `research/scripts/run_phase_2_1_research.py`: `run_phase_2_1_pipeline()` (lines 954, 971–1019)
  - `research/scripts/run_swing_research.py`: `run_swing_research_pipeline()` (lines 749–951)
  - `research/scripts/run_candidate_a_attribution.py`: `main()` (lines 134–142, 228–232)
  - `backtest.py` (line 58) & `backtest_audit.py` (line 287)
- **Verification of Phase 2.1 Walk-Forward Partition:**
  - Pre-roll warmup (`2023-02-10 00:00` to `2023-08-31 23:00`) is excluded from all evaluation folds.
  - Folds 1–4 (`walk_forward_test`) and Fold 5 (`untouched_holdout`, `2026-04-08 00:00` to `2026-11-04 18:00`) are strictly chronological, non-overlapping, separated by exact `168.0h` purge windows, and force-close any open position at each fold boundary (`force_close_at_end=True`).
- **Every Place Where Holdout (`2026-04-08` to `2026-11-04`) Was Reused / Contaminated:**
  1. **`research/scripts/run_swing_research.py` (lines 798–951):** Executed all Phase 2 strategy evaluations, holding-period sweeps (`12h..168h`), maker/taker experiments, and the $3\times 3$ Donchian parameter grid across the **entire 3-year dataset (`2023-02-10` to `2026-11-04`)**, which included the Fold 5 holdout window (`2026-04-08` to `2026-11-04`), before Candidate A, Donchian 120h, and Donchian 168h were shortlisted for Phase 2.1.
  2. **`research/scripts/run_phase_2_1_research.py` (line 954):** Called `analyze_rolling_volatility_transitions(df_4h, rolling_window_bars=168)` on the full `df_4h` (`N = 8,122` 4H bars through `2026-11-04`), which included `1,306` 4H bars from Holdout Fold 5 (`N = 6,816` when restricted to Walk-Forward Folds 1–4 `<= 2026-03-31 23:00`). Fixed in `run_phase_2_1_research.py` to slice `df_4h.loc[:pd.Timestamp("2026-03-31 23:00:00+00:00")]`.
  3. **`research/scripts/run_candidate_a_attribution.py` (lines 228–232):** Loaded all 37 Candidate A trades from `research/phase_2_1_fold_trade_log.csv` (28 Walk-Forward OOS trades from Folds 1–4 **plus 9 Holdout Fold 5 trades**) to compute regime/side attribution (`$1,912.97` total PnL across 37 trades vs `$1,286.74` across the 28 OOS trades in Folds 1–4) and evaluate the 12-hour early-exit filter. Fixed `run_candidate_a_attribution.py` to filter `fold.isin([1, 2, 3, 4])` by default (`include_holdout: bool = False`).
  4. **`backtest.py` (line 58) & `backtest_audit.py` (line 287):** Used a contiguous 80/20 split without a purge gap (`LOOKBACK + MAX_HOLD_BARS`). Fixed in `backtest.py`.

---

### Item 9: Reproducibility
- **Status:** `bug found` -> `fixed`
- **Files & Functions:**
  - `utils/quant_evaluation.py`: `compute_continuous_portfolio_metrics()` (line 683)
- **Failure Scenario Identified:**
  - `compute_continuous_portfolio_metrics(..., seed=seed)` accepted `seed: int = 42` and used it for hourly/daily equity block bootstraps, **but did not pass `seed=seed` when calling `compute_reconciled_trade_metrics()` at line 683**, causing `compute_reconciled_trade_metrics()` to always use its default `seed=42` even when a caller passed a custom seed.
- **Fix:** Forwarded `seed=seed` to `compute_reconciled_trade_metrics()` at line 683 of `utils/quant_evaluation.py`. Verified identical trades, equity curves, and bootstrap CIs across repeated runs in `test_09_reproducibility_identical_trades_and_metrics_across_runs`.

---

### Item 10: Experiment Registry Reconciliation (Why Donchian Shows `+1.0` & `+74.4 bps` in `phase_2_1_candidate_comparison.md` vs `-34.6` & `-61.2 bps` in `experiment_registry.json`)
- **Status:** `bug found` -> `fixed`
- **Files & Functions:**
  - `research/scripts/generate_experiment_registry.py` (lines 18–212)
  - `research/experiment_registry.json` & `research/experiment_registry.md`
- **Root-Cause Explanation of the Donchian Discrepancy:**
  1. **Authoritative Simulation Output (`research/phase_2_1_fold_metrics.json` & `research/phase_2_1_candidate_comparison.md`):**
     - Generated directly by `research/scripts/run_phase_2_1_research.py` (`generate_swing_signals` lines 331–346 + `simulate_swing_trades_conservative`):
     - Both `Trend_Donchian_120h` (Candidate B) and `Trend_Donchian_168h` (Candidate C) use `stop_loss = 2.5 * ATR_14`, `take_profit = 4.0 * ATR_14`, `exit_type = "fixed_atr"`.
     - Across Pooled OOS (Folds 1–4, `2023-09-08` to `2026-03-31`):
       - **`Trend_Donchian_120h` (`EXP-31-WF-DONCHIAN-120H`):** Expectancy = **`+1.01 bps`** (rounded to **`+1.0 bps`** in `phase_2_1_candidate_comparison.md`), 95% CI `[-158.13, +185.85] bps`, Win Rate `40.74%` (`54` OOS trades), Profit Factor `1.004`, OOS MaxDD `20.42%` (`20.62%` with intrabar lows), Fold 4 MaxDD `13.44%`, Fold 2 Expectancy `-361.0 bps`, Holdout Fold 5 Expectancy `-24.92 bps` (CAGR `-1.11%`).
       - **`Trend_Donchian_168h` (`EXP-32-WF-DONCHIAN-168H`):** Expectancy = **`+74.40 bps`** (rounded to **`+74.4 bps`** in `phase_2_1_candidate_comparison.md`), 95% CI `[-126.57, +296.39] bps`, Win Rate `48.00%` (`50` OOS trades), Profit Factor `1.272`, OOS MaxDD `23.94%` (`24.06%` with intrabar lows), Holdout Fold 5 Expectancy `-8.08 bps` (`-8.1 bps`, CAGR `-2.99%`).
  2. **Why `research/experiment_registry.json` Had `-34.6 bps` and `-61.2 bps`:**
     - `research/scripts/generate_experiment_registry.py` **did not load `research/phase_2_1_fold_metrics.json` or `research/swing_strategy_metrics.json`**. Instead, it contained manually typed static dictionary literals written during Phase 2.2A that:
       - Misstated the parameters of `EXP-31-WF-DONCHIAN-120H` and `EXP-32-WF-DONCHIAN-168H` as `{"stop_loss": "2.0 * ATR", "take_profit": "None (Trailing)"}` (confusing them with an earlier unlogged trailing-stop scratch variant).
       - Hardcoded `-34.6 bps` (`-22.4%` Fold 4 DD) for `EXP-31-WF-DONCHIAN-120H` and `-61.2 bps` (`-34.8%` DD) for `EXP-32-WF-DONCHIAN-168H`.
       - Contained 7 additional static-text discrepancies across `EXP-20` through `EXP-30`:
         - `EXP-20-CANDIDATE-A`: Mixed Phase 2.1 OOS expectancy (`+135.3 bps`) with 5-fold pooled win rate (`56.8%`, which is 21/37 across Folds 1–5) instead of Phase 2 (`+136.32 bps`, `56.82%` WR, `1.98` PF) and Phase 2.1 OOS (`+135.35 bps`, `53.57%` WR, `1.859` PF).
         - `EXP-21-DONCHIAN-TURTLE`: Listed `-112 bps` and `28.6%` WR instead of `-31.36 bps` and `34.48%` WR in `swing_strategy_metrics.json` (`+24.01 bps` OOS in `phase_2_1_fold_metrics.json`).
         - `EXP-22-PULLBACK-TREND`: Listed `-48 bps` and `0.82` PF instead of `-39.42 bps` and `0.77` PF (`-57.33 bps` OOS in Phase 2.1).
         - `EXP-23-SR-BREAKOUT`: Listed `-62 bps` instead of `-38.80 bps` in Phase 2 / `-59.65 bps` OOS in Phase 2.1.
         - `EXP-24-REGIME3-GATED`: Listed `-18 bps` in Phase 2 (which was actually the Phase 2.1 OOS expectancy `-18.70 bps`, whereas Phase 2 full-period was `+9.83 bps`).
         - `EXP-25-COMPRESSION-EXP`: Listed `-74 bps` instead of `-39.13 bps` in Phase 2 / `-40.76 bps` OOS in Phase 2.1.
         - `EXP-26-COMBO-REGIME`: Listed `-32 bps` instead of `+12.16 bps` in Phase 2 / `-12.63 bps` OOS in Phase 2.1.
         - `Phase 2.1` header & `EXP-30-WF-CANDIDATE-A`: Stated `6 Chronological Folds` and `57.1%` win rate instead of `5 Chronological Folds (4 Walk-Forward OOS + 1 Holdout)` and `53.57%` OOS win rate (`56.8%` across all 5 folds).
- **Fix:** Updated `research/scripts/generate_experiment_registry.py` to dynamically load `research/phase_2_1_fold_metrics.json` and `research/swing_strategy_metrics.json`, populate exact verified metrics and parameters for all experiments, and regenerated `research/experiment_registry.json` and `research/experiment_registry.md`.

---

## 3. Summary of Prior Results Validity

1. **Phase 1 ML (`backtest.py`, `backtest_audit.py`):** **Invalid** (decision-close fill lookahead, same-bar break-even lookahead, uncapped leverage sizing, no purge gap, no funding). Status remains `REJECTED`.
2. **Phase 2 Full-Sample Swing Research (`research/swing_strategy_metrics.json`):** **Invalid** (100% of Binance microsecond funding settlements dropped due to unnormalized timestamp strings; evaluated over full 2023–2026 range overlapping Holdout Fold 5; same-bar trailing stop lookahead on `trailing_atr` strategies). Superseded by Phase 2.1.
3. **Phase 2.1 Purged Walk-Forward Revalidation (`research/phase_2_1_fold_metrics.json`, `research/phase_2_1_candidate_comparison.md`):**
   - **Trade logs, expectancies, win rates, profit factors, CAGRs, Sharpe ratios, and funding reconciliations are VALID and confirmed** (`+135.35 bps` OOS / `+103.85 bps` Holdout for Candidate A; `+1.01 bps` OOS for `Trend_Donchian_120h`; `+74.40 bps` OOS / `-8.08 bps` Holdout for `Trend_Donchian_168h`).
   - **Intratrade Max Drawdown values in `phase_2_1_fold_metrics.json` were understated by `0.12%–0.20%`** because `hourly_lowest_equity` was not passed into `compute_continuous_portfolio_metrics()` (Candidate A OOS MaxDD is **`4.94%`** vs reported `4.74%`; Donchian 120h is **`20.62%`** vs `20.42%`; Donchian 168h is **`24.06%`** vs `23.94%`).
   - **Volatility Transition Table (`volatility_transitions` in `phase_2_1_fold_metrics.json`):** Previously included Holdout Fold 5 bars (`N = 8,122`); restricting strictly to Walk-Forward Folds 1–4 (`N = 6,816`) changes `P(Exp|Comp)` by `< 1.1%` across horizons and does not affect any strategy signals or trades.
4. **Phase 2.2A Candidate A Edge Attribution (`research/candidate_a_attribution_report.md`):**
   - Previously pooled all 37 trades (Folds 1–4 OOS + Fold 5 Holdout). On Folds 1–4 OOS only (28 trades), Short trades still generate the vast majority of net profit (`$1,104.18` of `$1,286.74` = `85.8%`) and Regime 3 remains the dominant regime (`$892.45` = `69.4%`), so qualitative conclusions hold, and `run_candidate_a_attribution.py` now defaults to `include_holdout=False`.
