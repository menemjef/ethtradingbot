# Baseline Model Retraining Report (ChatGPT Audit Review)

> **IMPORTANT DISCLAIMER**  
> This document is a technical training-performance report documenting model convergence, validation directional accuracy, loss trajectories, and data consistency under historical market conditions. **It is NOT proof of profitability, financial advice, or a guarantee of live trading success.**

---

## 1. Executive Summary

A clean baseline model retraining run was executed using the existing training pipeline without any modifications to model architecture, trading targets, feature definitions, threshold parameters, risk limits, or configuration.

- **Execution Command:** `python train.py --limit 20000 --seed 42`
- **Execution Date:** 2026-11-04 17:39:36 to 17:42:33 (Local UTC+5:30)
- **Target Symbol / Timeframe:** `ETH/USDT:USDT` (Binance USDM Futures, 1h candles)
- **Compute Device:** NVIDIA GeForce RTX 4050 Laptop GPU (`cuda`)
- **Pre-Retraining Backups:** Backed up to `C:\Python Projects\trading_bot_backups\baseline_pre_retrain` with verified cryptographic hashes.
- **Retraining Outcome:** Completed successfully. Early stopping triggered at epoch 32 (best epoch: 2). Artifacts saved and validated.

---

## 2. Dataset Manifest & Data Quality

| Metric | Measured Value |
| :--- | :--- |
| **Candles Requested** | 20,000 |
| **Candles Fetched** | 20,000 |
| **Unique Timestamps** | 20,000 |
| **Duplicate Timestamps** | 0 |
| **Missing Intervals (1h)** | 0 (continuous hourly time series) |
| **Time Range** | `2024-05-01 21:00:00+00:00` &rarr; `2026-08-13 04:00:00+00:00` |
| **Dataset Canonical SHA-256** | `80f52a5bfe4f6c037876c4df916e933e7e9f694920a409111d028052276fac58` |
| **Feature Engineered Rows** | 19,985 (after indicator warm-up and TBM lookahead drop) |
| **Total Features** | 53 (including target column) |
| **Target Methodology** | Triple Barrier Method (TBM): PT=3.0 &times; ATR, SL=1.5 &times; ATR, Horizon=12 |
| **Target Distribution (Long: +1.0)** | 10,136 samples (50.718%) |
| **Target Distribution (Short: -1.0)** | 9,849 samples (49.282%) |

---

## 3. Training & Validation Setup

- **Model Architecture:** `HybridModel` (Parallel 1D CNN feature extractors + Bi-directional LSTM sequence model + multi-task regression & classification heads)
- **Feature Partitioning:**
  - `X_candle`: Shape `(19961, 24, 8)` (8 candle-level OHLCV & micro features, 24-step lookback)
  - `X_market`: Shape `(19961, 24, 44)` (44 technical & regime indicators, 24-step lookback)
  - Total input dimension: 52 features
- **Split Ratio:** 80% Train (`15,968` samples), 20% Validation (`3,993` samples)
- **Batch Size:** 128
- **Optimizer:** Adam (`lr=0.0005`, `weight_decay=0.001`) with 5-epoch linear warmup and cosine annealing to `1e-6`
- **Loss Functions:**
  - Classification: `BCEWithLogitsLoss` (`pos_weight=1.17`, `label_smoothing=0.25`), weight = 1.0
  - Regression: `MSELoss`, weight = 1.0
- **Random Seed:** 42 (enforced across Python `random`, `numpy`, and `torch`)
- **Patience Limit:** 30 epochs (based on validation loss)

---

## 4. Retraining Metrics & Convergence

| Epoch Metric | Value |
| :--- | :--- |
| **Total Epochs Trained** | 32 |
| **Best Validation Epoch** | **Epoch 2** |
| **Early Stopping Triggered** | **Epoch 32** (patience = 30 exhausted after no loss improvement since epoch 2) |
| **Best Validation Loss (MSE)** | **1.0015** |
| **Train Total Loss at Best Epoch** | 1.7853 |
| **Lowest Train Total Loss** | 1.2980 (at epoch 31 & 32) |
| **Top-20% Confidence Accuracy (Best Epoch)** | 47.74% |
| **Final Validation Directional Accuracy (All Samples)** | **51.29%** |
| **Threshold-Filtered Validation Accuracy (|reg| &ge; 0.0450)** | **53.59%** (on 948 confident validation samples) |

---

## 5. Artifact Verification & Cryptographic Provenance

### 5.1 Pre-Retraining Model Checkpoints (Backed Up)
Stored externally at `C:\Python Projects\trading_bot_backups\baseline_pre_retrain`:
- `best_model.pt`: `8B7B979EC0E586FB02B1A98852329DB0D0A38B1BAACE91C96E28B3646132BD95`
- `scaler_candle.pkl`: `D5E5E30533E3AD133DD9385A5D84F3C0793C96EC91D1B7DB12E97722A19BB19C`
- `scaler_market.pkl`: `D7BC7EBDC32810F1C65EB6AD51273CCCB380CF20882E0513FA009E0387390E2D`

### 5.2 Newly Trained Model Checkpoints & Scalers
Saved in `checkpoints/`:
- `best_model.pt`:
  - Size: 321,634 bytes
  - SHA-256: `ac224bb7731f26b46d3f1cdc8cf39457742f7afde81799b31d67f9e02e3505bb`
- `scaler_candle.pkl`:
  - Size: 807 bytes
  - SHA-256: `b8e8da465b61cc47dd43ecbef3fd47df037362f4b17e6a54e1f3620b29dab5e4`
- `scaler_market.pkl`:
  - Size: 1671 bytes
  - SHA-256: `9d62bd29c91cfa45762c1ed23f8eb21e3eddc65be2d613a63ebfd2af4cced281`

### 5.3 Model Health & Inference Verification
- **State Dict Loading:** Loaded into `HybridModel` on `cuda` with 0 missing or unexpected keys.
- **Output Tensors:**
  - `reg_out`: Shape `torch.Size([16, 1])`, `dtype=torch.float32`
  - `clf_out`: Shape `torch.Size([16, 1])`, `dtype=torch.float32`
- **Finite Check:** Verified 0 NaN values and 0 Infinite values on representative evaluation batches.

---

## 6. Test Suite Results

All tests executed without connecting to Binance private endpoints or placing trades:

1. **Unit & Integration Test Suite (`python -m unittest discover tests -v`):**
   - `test_circuit_breaker_on_repeated_exchange_errors`: **PASSED**
   - `test_cooldown_blocks_reentry`: **PASSED**
   - `test_long_signal_pipeline`: **PASSED**
   - `test_minimum_notional_compliance`: **PASSED**
   - `test_short_signal_pipeline`: **PASSED**
   - `test_weak_signal_waiting`: **PASSED**
   - `test_daily_reset`: **PASSED**
   - `test_drawdown_limit_long`: **PASSED**
   - `test_leverage_cap_scaling`: **PASSED**
   - `test_inference_pipeline`: **PASSED**
   - **Result: 10 passed, 0 failed, 0 skipped.**

2. **Audit Bridge V2 Suite (`python tests/test_audit_bridge.py`):**
   - Secret redaction across all categories (Bearer, JWT, API keys, Binance, Telegram, DB URIs): **PASSED**
   - Nested JSON and log stream sanitization: **PASSED**
   - Publication safety gate enforcement: **PASSED**
   - Backtest metrics validation & consistency checks: **PASSED**
   - 99-trade audit reconciliation: **PASSED**
   - Runner execution capture: **PASSED**
   - Native hook handler simulation: **PASSED**
   - MCP server protocol compliance and path restriction: **PASSED**
   - **Result: 8 passed, 0 failed, 0 skipped.**

**Total: 18 tests executed, 18 passed, 0 failed, 0 skipped.**

---

## 7. Diagnostics and Observations

1. **Convergence Pattern:** Training loss continued to decrease steadily from 1.8729 down to 1.2980 across epochs 1 through 32, while validation MSE reached its minimum early at epoch 2 (1.0015) and subsequently rose to 1.62. This pattern is consistent with neural network capacity memorizing training noise, correctly triggering early stopping.
2. **Threshold Signal Filtering:** Filtering predictions by confidence magnitude (|regression prediction| &ge; 0.0450) improved directional accuracy from 51.29% across all samples to 53.59% on the 948 high-confidence validation samples.
3. **Regime Extraction:** Standard `hmmlearn` non-convergence warnings occurred during regime detection on specific rolling windows; these are handled safely by the feature pipeline with fallback defaults.
