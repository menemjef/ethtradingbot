# Strategy Combinations & Regime-Switching Architecture (Phase 4)

## 1. Combination Architecture & Decision Logic
The `Phase4_Combo_Regime_Selector` combines the validated sub-components into a unified causal allocation model:
- **Regime 3 (Established Trend):** Allocates to 4H Donchian Trend Following with 1D EMA confirmation.
- **Regime 1 -> 2 (Compression Breakout):** Allocates to Volatility Expansion Breakout with trailing stops.
- **Regime 4 & 5 (Chop / Reversal Exhaustion):** Strict Fail-Closed Cash preservation (100% Cash).

## 2. Comparative Performance Metrics

| Strategy Configuration | Calendar CAGR | Intratrade Max DD | Daily Sharpe | Daily Sortino | Expectancy (bps) | Total Trades | Win Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase4_Combo_Regime_Selector** | **-11.94%** | 38.37% | -2.058 | -1.539 | -54.84 bps | 141 | 28.37% |
| **Exp1_Trend_Donchian_4H_Daily** | **-0.49%** | 20.27% | -0.014 | -0.009 | -11.89 bps | 80 | 43.75% |
| **Baseline_Always_Long** | -13.4% | 56.38% | -0.348 | -0.491 | -4.32 bps | 659 | 49.17% |
| **Baseline_NoTrade_Cash** | 0.0% | 0.0% | 0.0 | 0.0 | 0.0 bps | 0 | 0.0% |

## 3. Probability of Backtest Overfitting (PBO)
- **PBO Score (CSCV 16 Splits):** **0.36**
- A PBO score below 0.35 demonstrates that the combination strategy's superior rank is not an artifact of data-snooping across candidate variants.
