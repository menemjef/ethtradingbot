# ETH/USDT Futures Backtest Metrics: Out-of-Sample Walk-Forward Evaluation (Reconciled)

## 1. Audit Scope & Reconciliation Invariants

- **Prediction/Target Alignment (Issue 1 Resolved):** Evaluated strictly against `target[j-1]` with separate decision, target, and execution timestamps.
- **MajorityBaseline Reconciliation (Issue 2 Resolved):** Full trade logging with complete trade records and zero discrepancy between JSON and Markdown.
- **Retained Evaluation Window:** 3,500 continuous 1h candles (2026-06-11 23:00:00+00:00 to 2026-11-04 18:00:00+00:00).

## 2. Reconciled Trading Performance Comparison

| Metric | Hybrid Model (Baseline) | HistGradientBoosting | Majority Baseline (Always Long) | No-Trade (Cash) |
| :--- | :--- | :--- | :--- | :--- |
| **Total OOS Trades** | 59 | 376 | 380 | 0 |
| **Long / Short Trades** | 59 / 0 | 173 / 203 | 380 / 0 | - |
| **Win Rate** | 47.46% | 43.62% | 48.95% | - |
| **Profit Factor** | 0.631 | 0.693 | 0.81 | - |
| **Expectancy** | -25.19 bps | -19.18 bps | -11.03 bps | 0.0 bps |
| **95% Bootstrap CI (Expectancy)** | [-61.44, -0.1] bps | [-30.16, -8.41] bps | [-24.15, 0.37] bps | - |
| **Compounded Net Return (Unlevered)** | **-14.21%** | **-52.72%** | **-36.05%** | 0.0% |
| **Account Levered Return (3x Cap, 2.5% Risk)** | **-22.94%** | **-72.19%** | **-53.99%** | 0.0% |
| **Max Drawdown (Unlevered)** | 19.0% | 55.38% | 41.35% | 0.0% |
| **Deflated Sharpe Ratio (DSR)** | 0.0 | 0.0 | 0.0 | - |
| **Fee / Slippage / Funding Drag** | 5.9% / 3.54% / 0.0499% | 37.6% / 22.56% / -0.1242% | 38.0% / 22.8% / 0.6868% | - |

## 3. Reconciled Predictive Metrics (Aligned at Decision Candle j-1)

| Metric | Hybrid Model (Baseline) | HistGradientBoosting | Majority Class Baseline |
| :--- | :--- | :--- | :--- |
| **Directional Accuracy** | 53.34% | 49.68% | 54.06% |
| **Regression MSE** | 0.989594 | 1.383778 | 0.985941 |
