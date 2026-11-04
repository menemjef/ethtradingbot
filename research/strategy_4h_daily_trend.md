# Multi-Timeframe 4H and Daily Trend Following Research (Experiment 1)

## 1. Executive Summary & Design Principles
- **Core Hypothesis:** Transitioning from hourly micro-movements to multi-timeframe swing trading (4H entries filtered by completed 1D trend structure) significantly improves risk-adjusted returns by expanding profit per trade relative to transaction costs.
- **Causality Enforcement:**
  - 1D trend features (`close > ema_50 > ema_200`, `adx_14 > 20`) are strictly shifted by 1 calendar day.
  - 4H Donchian channel breakouts (20, 55 periods) are lagged by 1 bar; decisions made at 4H bar close execute at the next 1H candle open.
  - Zero lookahead bias: intra-bar excursion and mark-to-market drawdown are evaluated bar-by-bar on completed 1H intervals.

## 2. Multi-Timeframe Strategy Performance Comparison (Base Case: 5 bps fee, 3 bps slip)

| Strategy Name | Trades | Win Rate | Profit Factor | Expectancy (bps) | 95% Bootstrap CI | Calendar CAGR | Intratrade Max DD | Daily Sharpe | Daily Sortino |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp1_Trend_Donchian_4H_Daily** | 80 | 43.75% | 0.947 | -11.89 bps | [-117.51, 106.81] | **-0.49%** | 20.27% | -0.014 | -0.009 |
| **Exp1_Donchian_55_Turtle_4H** | 208 | 37.02% | 0.701 | -31.36 bps | - | **-8.23%** | 32.45% | -0.955 | -0.896 |
| **Exp1_Pullback_Trend_4H** | 154 | 36.36% | 0.741 | -40.49 bps | - | **-17.64%** | 62.3% | -0.811 | -0.793 |
| **Exp1_SR_Breakout_Retest_4H** | 250 | 40.4% | 0.831 | -34.93 bps | - | **-12.24%** | 47.16% | -0.517 | -0.618 |
| **Exp1_Momentum_Continuation_4H** | 44 | 52.27% | 1.318 | 53.08 bps | [-56.42, 169.93] | **2.9%** | 9.24% | 0.443 | 0.251 |

## 3. Exit Behavior Comparison
- **Fixed ATR Exits (3x ATR stop, 3x ATR target):** Provides symmetric payoff structure.
- **Trailing ATR Stops (2.5x Chandelier Exit):** Captured longer trending tails during strong momentum phases (e.g. Q4 2023 and Q1 2024), but surrendered unrealized profit during choppy consolidation regimes.
- **Intratrade Drawdown Observation:** Drawdown measured with bar-by-bar hourly mark-to-market fidelity increased by ~5-8% compared to closed-trade equity jumps, confirming the Phase 0 audit finding that intratrade excursion must be incorporated into risk budgets.
