# ETH/USDT Futures Backtest Metrics: Out-of-Sample Walk-Forward Evaluation

## 1. Executive Summary & Audit Scope

- **Asset & Market:** `ETH/USDT:USDT` (Binance USD-M Perpetual Futures)
- **Timeframe:** `1h`
- **Historical Range:** `2026-06-11 23:00:00+00:00` to `2026-11-04 18:00:00+00:00`
- **Total Raw Candles Fetched:** `4220`
- **Fibonacci Pre-Roll Warm-up Bars Excluded from Labeling:** `720`
- **Retained Evaluation Window:** `3500` candles
- **Walk-Forward Folds:** `4` temporal non-overlapping folds
- **Barrier Multipliers:** Profit Target = `2.0x ATR`, Stop Loss = `2.0x ATR`, Max Horizon = `12` bars
- **Assumed Leverage Cap:** `3x`
- **Risk Budget per Trade:** `2.5%` of equity
- **Active Model Checkpoint Hash:** `8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95` (Baseline Model)
- **Report Generated At:** `2026-11-04T19:05:18.634202+00:00`

> [!IMPORTANT]
> **Methodological Realism & Lookahead Elimination:**
> - Signals are calculated at decision bar `i-1` close and executed strictly at bar `i` open with 3 bps slippage.
> - Barrier ATR is calculated at decision bar `i-1` close (zero lookahead into bar `i` ATR).
> - Take-profit, stop-loss, and horizon exits execute with market taker fees (5 bps).
> - Historical 8-hour Binance funding rate settlements (00:00, 08:00, 16:00 UTC) are deducted based on holding duration.
> - Conservative STOP-FIRST rule applies to intra-bar high/low barrier collisions.

## 2. Chronological Walk-Forward Fold Partitioning

| Fold | Train Window | Test Window | Test Candles |
| :--- | :--- | :--- | :--- |
| Fold 1 | 2026-06-11 23:00 to 2026-07-10 22:00 | 2026-07-11 11:00 to 2026-08-09 06:00 | 692 |
| Fold 2 | 2026-06-11 23:00 to 2026-08-08 18:00 | 2026-08-09 07:00 to 2026-09-07 02:00 | 692 |
| Fold 3 | 2026-06-11 23:00 to 2026-09-06 14:00 | 2026-09-07 03:00 to 2026-10-05 22:00 | 692 |
| Fold 4 | 2026-06-11 23:00 to 2026-10-05 10:00 | 2026-10-05 23:00 to 2026-11-04 18:00 | 716 |

## 3. Aggregate Out-of-Sample Performance Comparison

### 3.1 Trading Performance Across Models

| Metric | Hybrid Model (Baseline) | HistGradientBoosting | Logistic Regression | Majority Baseline | No-Trade (Cash) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Out-of-Sample Trades** | 59 | 376 | 338 | 380 | 0 |
| **Long / Short Trades** | 59 / 0 | 173 / 203 | 71 / 267 | - | - |
| **Win Rate** | 47.46% | 43.62% | 39.64% | - | - |
| **Profit Factor** | 0.629 | 0.693 | 0.605 | - | - |
| **Average Trade Return** | -0.2533% | -0.1919% | -0.2691% | - | - |
| **Median Trade Return** | -0.2258% | -0.2304% | -0.4208% | - | - |
| **Expectancy** | -25.33 bps | -19.19 bps | -26.91 bps | - | - |
| **Compounded Net Return (Unlevered)** | **-14.27%** | **-52.73%** | **-60.77%** | 0.0% | 0.0% |
| **Account Levered Return (3x Cap, 2.5% Risk)** | **-23.05%** | **-72.19%** | **-79.58%** | 0.0% | 0.0% |
| **Max Drawdown (Unlevered)** | 10.75% | 25.35% | 25.95% | 0.0% | 0.0% |
| **Avg Holding Duration** | 6.9 hrs | 6.4 hrs | 6.3 hrs | - | - |
| **Fee Costs / Slippage Costs / Funding Costs** | 5.9% / 3.54% / 0.1295% | 37.6% / 22.56% / -0.0981% | 33.8% / 20.28% / -0.9302% | - | - |
| **Stop-Loss / Take-Profit Exits** | 26 / 18 | 144 / 123 | 138 / 100 | - | - |
| **Intrabar Collisions (Stop-First)** | 0 | 1 | 1 | - | - |
| **Opening Gap Exits** | 0 | 0 | 0 | - | - |
| **Unclosed / Invalid Trades** | 0 | 0 | 0 | 0 | 0 |

### 3.2 Predictive Statistical Metrics Across Models

| Metric | Hybrid Model | HistGradientBoosting | Logistic Regression | Majority Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Directional Accuracy** | 53.3% | 50.32% | 44.27% | 54.01% |
| **Regression MSE** | 0.989142 | 1.341213 | - | - |

## 4. Execution-Cost & Funding Stress Sensitivity Matrix

| Scenario | Taker Fee | Slippage | Funding Rate Mode | Total Trades | Compounded Net Return |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Case** | 5 bps | 3 bps | historical | 59 | **-14.28%** |
| **Low Cost (Maker/Zero Slip)** | 3 bps | 1 bps | historical | 59 | **-10.92%** |
| **High Cost (Adverse Slippage)** | 7 bps | 5 bps | historical | 59 | **-17.5%** |
| **Funding Stress (+10.95% APR)** | 5 bps | 3 bps | stress_10bps | 59 | **-14.66%** |
| **Funding Excluded** | 5 bps | 3 bps | none | 59 | **-14.17%** |

## 5. Walk-Forward Fold Breakdown (Hybrid Model)

| Fold | Test Range | Trades | Win Rate | Compounded Net Return | Max Drawdown | Directional Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fold 1 | 2026-07-11 to 2026-08-09 | 5 | 60.0% | 1.74% | 1.78% | 49.28% |
| Fold 2 | 2026-08-09 to 2026-09-07 | 6 | 33.33% | -4.57% | 6.11% | 61.42% |
| Fold 3 | 2026-09-07 to 2026-10-05 | 26 | 50.0% | -5.08% | 8.96% | 56.36% |
| Fold 4 | 2026-10-05 to 2026-11-04 | 22 | 45.45% | -6.98% | 10.75% | 46.37% |
