# AI Review Context — ETH/USDT Futures Trading Bot

> **Provenance Note:** `AI_REVIEW_CONTEXT.md` was not present in the repository at commit `60cbf445a854158705814708e19b99130a5b5aa1` and was created during the `chore/ai-research-council-setup` initialization. All information below is strictly derived from verified source files and the Git commit history on `chatgpt-audit-review`. No historical events or experimental results have been invented or reconstructed.

---

## 1. Project Overview & Current Baseline

- **Repository:** `menemjef/ethtradingbot`
- **Instrument & Timeframe:** `ETH/USDT:USDT` perpetual futures (Binance USD-M), 1-hour primary timeframe (`TIMEFRAME = "1h"` in `config.py`), with optional multi-timeframe (15m, 4h) and external funding/open-interest feature hooks.
- **Canonical Audit Branch:** `chatgpt-audit-review`
- **Current Verified Engineering Baseline Commit:** `60cbf445a854158705814708e19b99130a5b5aa1`
- **Governance & Review Protocol:** See [`research/AI_REVIEW_PROTOCOL.md`](research/AI_REVIEW_PROTOCOL.md).

---

## 2. Verified Git Commit History on `chatgpt-audit-review`

| Commit SHA | Summary |
|---|---|
| `f26020a6507d853a461886dc07f268535d825150` | Initial sanitized repository export and Audit Bridge v2 publication infrastructure |
| `3957d5ffd5ee9a952e6e7f17dddee576d0ba75c1` | Pass 1 retraining pipeline audit fixes: multi-page pagination, purged split, TBM ambiguous candle handling, decoupled evaluation metrics |
| `3e3462e1a30f3ba420ecf63d7a2de30c4c727b72` | Pass 2 correction: isolated stalled test, mathematical MSE analysis, class-weight & calibration diagnostics, 3-way split & block bootstrap utilities |
| `03e4117545107639a0af8d243e733aeca5ec21d2` | Pass 3: integrated 3-way split, walk-forward pipeline, post-hoc temperature scaling (`model/calibration.py`), and fetcher gap pagination |
| `1f4ca4a4c1fa1921663ee96762a30c162acd340e` | Pass 4: resolved training entry-point crashes, pre-roll exclusion in `train.py`, continuous hourly lookback windows, and OHLC walk-forward simulator |
| `dcf18cf215cc4aa7a358b65ee9e541dd4577d6fe` | Pass 5: non-destructive orchestration tests, explicit epoch propagation (`max_epochs`/`patience`), pre-`fillna` 720-bar Fibonacci validation, walk-forward signal timestamp alignment, inner-validation scaler isolation, and OHLC stop-gap/itemized accounting |
| `60cbf445a854158705814708e19b99130a5b5aa1` | Final targeted Pass 5 correction: enforced 720-bar pre-roll exclusion and Fibonacci warm-up validation inside `run_walk_forward_evaluation()` |

---

## 3. Repository Architecture & Key Modules

| File Path | Role in Pipeline |
|---|---|
| `config.py` | Central configuration: `SYMBOL = "ETH/USDT:USDT"`, `TIMEFRAME = "1h"`, `FETCH_LIMIT = 2500`, `LOOKBACK = 72`, `TARGET_HORIZON = 12`, `TBM_PT_MULT = 2.0`, `TBM_SL_MULT = 2.0`, `SIGNAL_THRESHOLD = 0.004`, `CANDLE_FEATURES` (8), `MARKET_FEATURES` (44). |
| `data/fetcher.py` | Public CCXT OHLCV pagination (`fetch_ohlcv`) with `warmup_bars` pre-roll support, incomplete-candle exclusion, cursor-stall protection, deduplication, and `raise_on_gap` interval verification. |
| `data/indicators.py` | Technical & quantitative feature computation (`compute_indicators`, including 720-bar `_fib_retracement` -> `fib_618_dist_180d`) and Triple Barrier Method labeling (`add_tbm_target` with `ambiguous_resolution="exclude"` default for training). |
| `data/preprocessor.py` | Pre-`fillna(0.0)` 720-bar Fibonacci validation (`validate_fibonacci_warmup`), feature assembly (`build_features`), scaler persistence (`fit_and_save_scalers`), continuous hourly window construction (`build_windows`), 3-way timestamp-purged split (`split_train_val_test_purged`), and moving block bootstrap (`block_bootstrap_confidence_interval`). |
| `model/architecture.py` | Dual-branch `HybridModel`: Branch 1 (1D CNN over 8 candle features) + Branch 2 (stacked LSTM + multi-head self-attention over 44 market features) -> shared trunk -> dual heads (`reg_head` with `Tanh` and `clf_head` outputting raw logits). |
| `model/trainer.py` | Training orchestration (`train_model`): 60/20/20 purged train/val/test split, data-driven `pos_weight`, checkpoint selection on validation partition, temperature scaling calibration, and block bootstrap CIs on the frozen final test partition. |
| `model/calibration.py` | Post-hoc logit temperature scaling (`TemperatureScaler`), Brier score, log loss, expected calibration error (ECE), and 10-bin reliability curves against a fixed training/calibration prior. |
| `model/walk_forward.py` | Out-of-sample walk-forward engine (`run_walk_forward_evaluation`) and OHLC barrier trade simulator (`simulate_trades_ohlc`): pre-roll exclusion, inner-train scaler isolation, next-bar open execution (`decision_time` at bar $k-1$ close -> `entry_time` at bar $k$ open), stop-loss opening gap fills (`stop_loss_gap`), stop-first intrabar collision handling, intratrade hourly drawdown, itemized fees/slippage, and compounded equity curve aggregation. |
| `train.py` | CLI entry point supporting standard 3-way purged training and `--walk-forward` evaluation, `--fast` (5 epochs), `--test` (30 epochs), `--use_cache`, and `--warmup-bars 720`. |
| `live/signal_engine.py`, `live/trader.py`, `live/execution_engine.py` | Live inference and paper/live execution state machine (currently untouched during research audit passes). |
| `training/forensic_report.md` | Detailed audit log and mathematical/forensic record across Passes 1 through 5. |
| `tests/test_retraining_corrections.py` | 31-test verification suite covering fetcher pagination, TBM labeling, calibration, 3-way purged splits, non-destructive `train.py` orchestration, Fibonacci warm-up validation, inner-val scaler isolation, OHLC stop-gap accounting, and walk-forward pre-roll exclusion. |

---

## 4. Verified Artifact Integrity State (at Baseline `60cbf44`)

1. **Preserved Pre-Retrain Baseline Backups (`baseline_pre_retrain/`):**
   - `best_model.pt`: `8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95` (Intact)
   - `scaler_candle.pkl`: `d5e5e30533e3ad133dd9385a5d84f3c0793c96ec91d1b7db12e97722a19bb19c` (Intact)
   - `scaler_market.pkl`: `d7bc7ebdc32810f1c65eb6ad51273cccb380cf20882e0513fa009e0387390e2d` (Intact)
2. **Untouched Multi-Seed Checkpoints (`checkpoints/`):**
   - `best_model_42.pt`: `8c352dfe8f637af3ca3cb05e192e44fcec47848e5f7b067f88481ee9939bdcdd`
   - `best_model_1337.pt`: `66aa63f4d334b4970fd87ea576a58e9504d50cc960f4ebb8ba4944f98cb62864`
   - `best_model_7.pt`: `a0136e12ad106c914f898b989d4172233dd348c4367710a0401892089d5b10d6`
   - `scaler_candle_42.pkl` (`_1337`, `_7`): `c82037685e2aef7c3f8136af6f4321c89a3d3c49bf775916d1a33080320eaf33`
   - `scaler_market_42.pkl` (`_1337`, `_7`): `a6dfa45c8991c96efec1525063b231684d77c72da7e86427657d60e4aff70f03`
3. **Unsuffixed Active Checkpoint (`checkpoints/best_model.pt`):**
   - Currently `0dfa18e77e5afaafa94f12ad239ba80379a418f77569ccd50aa261b1328789f2` (`scaler_candle.pkl`: `7104dc59...`, `scaler_market.pkl`: `b94a1c72...`), which was overwritten during Pass 4 by a 5-epoch synthetic test run prior to Pass 5's `tempfile.TemporaryDirectory()` test isolation fix. It must **not** be used for live trading.

---

## 5. Core Research Objective for Round 01

Establish whether the current `ETH/USDT:USDT` 1-hour hybrid CNN+LSTM+Attention strategy—evaluated under the corrected Pass 5 pipeline (`60cbf445a854158705814708e19b99130a5b5aa1`) with 720-bar pre-roll exclusion, timestamp-purged 3-way and walk-forward splits, inner-train scaler isolation, and OHLC next-bar open execution with stop-loss gap accounting—has a statistically defensible and economically viable predictive edge after realistic transaction costs.
