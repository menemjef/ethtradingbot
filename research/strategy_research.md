# Systematic Strategy Research: Multi-Year Quantitative Evaluation

## 1. Executive Summary & Research Mandate

- **Asset & Market:** `ETH/USDT:USDT` (Binance USD-M Perpetual Futures)
- **Timeframe:** `1h`
- **Historical Range:** `2023-02-18 05:00:00+00:00` to `2026-11-04 18:00:00+00:00` (32534 hourly bars / 3.7 years)
- **Walk-Forward Folds:** `6` non-overlapping chronological test folds
- **Transaction Costs:** 5 bps taker fee entry, 5 bps taker fee exit, 3 bps slippage per side, 8-hour historical funding rate settlements
- **Report Generated At:** `2026-11-04T19:41:24.808339+00:00`

## 2. Multi-Year Performance Comparison Across All Strategies

| Strategy / Baseline | Trades | Win Rate | Profit Factor | Expectancy (bps) | 95% Bootstrap CI (bps) | Compounded Return | Levered Return (3x) | Max Drawdown | Sharpe | DSR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy_A_Trend_Donchian** | 378 | 51.06% | 1.039 | 3.68 bps | [-21.33, 30.11] | **4.7%** | **-2.67%** | 41.28% | 0.643 | 0.0 |
| **Strategy_A_Donchian_Only** | 1548 | 44.7% | 0.816 | -15.82 bps | [-25.3, -7.13] | **-93.29%** | **-99.17%** | 93.67% | -3.272 | 0.0 |
| **Strategy_B_Volatility_Expansion** | 573 | 43.8% | 0.702 | -19.37 bps | [-29.84, -9.83] | **-68.45%** | **-86.08%** | 70.1% | -6.157 | 0.0 |
| **Strategy_C_Mean_Reversion** | 110 | 49.09% | 0.643 | -28.87 bps | [-51.87, -5.35] | **-28.17%** | **-43.23%** | 34.02% | -6.307 | 0.0 |
| **Strategy_D_Funding_Crowding** | 977 | 47.39% | 0.841 | -13.33 bps | [-25.45, -1.93] | **-77.08%** | **-92.91%** | 83.89% | -2.507 | 0.0 |
| **Strategy_E_Pullback_Trend** | 1233 | 45.09% | 0.739 | -20.2 bps | [-28.55, -12.35] | **-92.97%** | **-99.0%** | 93.16% | -4.317 | 0.0 |
| **Strategy_Combo_Rule_Selector** | 698 | 47.85% | 0.898 | -8.56 bps | [-23.29, 7.13] | **-51.43%** | **-73.87%** | 53.93% | -1.717 | 0.0 |
| **Baseline_Always_Long** | 4188 | 47.28% | 0.788 | -17.08 bps | [-22.52, -11.87] | **-99.96%** | **-100.0%** | 99.96% | -3.636 | 0.0 |
| **Baseline_Always_Short** | 4190 | 46.32% | 0.802 | -15.92 bps | [-21.03, -10.76] | **-99.93%** | **-100.0%** | 99.93% | -3.373 | 0.0 |
| **Baseline_NoTrade_Cash** | 0 | 0.0% | 0.0 | 0.0 bps | [0.0, 0.0] | **0.0%** | **0.0%** | 0.0% | 0.0 | 0.0 |
| **Strategy_Meta_Labeling** | 103 | 43.69% | 0.75 | -28.25 bps | [-73.65, 8.52] | **-27.3%** | **-43.03%** | 36.02% | -3.93 | 0.0 |

## 3. Transaction Cost & Execution Drag Breakdown

| Strategy | Fee Drag | Slippage Drag | Funding Drag | Net Compounded Return |
| :--- | :--- | :--- | :--- | :--- |
| **Strategy_A_Trend_Donchian** | 37.8% | 22.68% | 1.9665% | **4.7%** |
| **Strategy_A_Donchian_Only** | 154.8% | 92.88% | 2.6478% | **-93.29%** |
| **Strategy_B_Volatility_Expansion** | 57.3% | 34.38% | 0.2373% | **-68.45%** |
| **Strategy_C_Mean_Reversion** | 11.0% | 6.6% | 0.0154% | **-28.17%** |
| **Strategy_D_Funding_Crowding** | 97.7% | 58.62% | -3.4883% | **-77.08%** |
| **Strategy_E_Pullback_Trend** | 123.3% | 73.98% | 2.7592% | **-92.97%** |
| **Strategy_Combo_Rule_Selector** | 69.8% | 41.88% | 1.8093% | **-51.43%** |
| **Baseline_Always_Long** | 418.8% | 251.28% | 19.6265% | **-99.96%** |
| **Baseline_Always_Short** | 419.0% | 251.4% | -19.6265% | **-99.93%** |
| **Baseline_NoTrade_Cash** | 0.0% | 0.0% | 0.0% | **0.0%** |
| **Strategy_Meta_Labeling** | 10.3% | 6.18% | 0.1436% | **-27.3%** |

## 4. Cost Sensitivity Scenarios

| Scenario | Strategy A (Trend Donchian) | Strategy B (Vol Expansion) | Strategy Combo (Selector) | Baseline Always Long |
| :--- | :--- | :--- | :--- | :--- |
| **Base Case** | 2.57% (Exp: 3.14 bps) | -68.49% (Exp: -19.39 bps) | -52.42% (Exp: -8.84 bps) | -99.96% (Exp: -17.1 bps) |
| **Low Cost (Maker/Zero Slip)** | 43.22% (Exp: 11.9 bps) | -48.36% (Exp: -10.68 bps) | -13.66% (Exp: -0.32 bps) | -98.72% (Exp: -8.99 bps) |
| **Adverse Cost (High Slip)** | -28.12% (Exp: -6.28 bps) | -79.89% (Exp: -27.36 bps) | -74.11% (Exp: -17.57 bps) | -100.0% (Exp: -25.38 bps) |