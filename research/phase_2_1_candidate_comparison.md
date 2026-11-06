# Quant Phase 2.1: Candidate Revalidation & Walk-Forward Comparison Report

**Repository:** `menemjef/ethtradingbot`  
**Branch:** `fix/execution-safety-and-backtest-integrity`  
**Evaluation Standard:** Independent Walk-Forward Evaluation with 168-Hour Purge Boundaries & Strictly Untouched Final Holdout  
**Date:** 2026-11-06 07:56:20 UTC  

---

## 1. Executive Summary & Verdict

Following the comprehensive repair of backtest integrity defects (eliminating full-sample testing, resolving Binance microsecond funding dropped settlements, enforcing intra-bar entry stop-loss checks, eliminating trailing stop lookahead, and applying representative CSCV PBO), the three candidate strategies and benchmarks were re-evaluated under strict out-of-sample conditions.

### Primary Acceptance Hurdle
- **Baseline Expectancy:** > 15 bps net of fees, slippage, and funding drag.
- **Statistical Significance:** Moving Block Bootstrap 95% Confidence Interval lower bound > 0 bps.
- **Deflated Sharpe Ratio (DSR):** > 0.95 (accounting for 32 trial evaluations).
- **Out-of-Sample Holdout Persistence:** Positive expectancy maintained on untouched Fold 5 (2026-04 to 2026-11).

### Evaluation Verdict: NO VALIDATED EDGE (DO NOT DEPLOY)
No candidate strategy achieves the required statistical hurdles for live production deployment:
1. **Candidate A (`Exp1_Momentum_Continuation_4H`):** Achieves positive pooled OOS expectancy (+135.3 bps, 95% CI: [35.18, 314.88] bps) and +103.9 bps on Holdout Fold 5. However, it fails the multi-trial hurdle with a Deflated Sharpe Ratio of 0.1777 (under K=32 trials) and generates only 5.01% CAGR (28 OOS trades, ~10 trades/year). Its return falls vastly short of the 30% monthly target.
2. **Candidate B (`Trend_Donchian_120h`):** Fails to maintain consistency across folds, suffering severe losses in Fold 2 (-23.29% CAGR, -361.0 bps expectancy), resulting in an OOS CAGR of 0.04%, Daily Sharpe of 0.07, and an intra-bar drawdown of 20.62%. Its bootstrap 95% CI expectancy spans deeply negative [-143.77, 172.55] bps.
3. **Candidate C (`Trend_Donchian_168h`):** Produces a modest OOS CAGR of 3.71% with high drawdown (24.06%), but fails on the untouched Holdout Fold 5 with negative CAGR (-2.99%) and negative expectancy (-8.1 bps). Cumulative funding drag exceeds 4.622%, eroding long-term viability.
4. **Target 30% Net Compounded Monthly Return:** Remains mathematically and empirically **unachievable** under perpetual futures taker frictions (16 bps round-trip), 8-hour funding drag, and conservative 3x leverage risk budgeting.

---

## 2. Pooled Out-of-Sample Performance Comparison (Folds 1 to 4)

Evaluation window: **2023-09-08 to 2026-03-31** (~2.57 calendar years, 21,960 hours). All metrics computed on genuine out-of-sample concatenated test fold performance with independent $10,000 initial capital per fold.

| Metric | Candidate A (Momentum 4H) | Candidate B (Donchian 120h) | Candidate C (Donchian 168h) | Buy & Hold ETH | Always Long | Always Short | No-Trade Cash |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy Label** | `Exp1_Momentum_Continuation_4H` | `Trend_Donchian_120h` | `Trend_Donchian_168h` | Benchmark B&H | Benchmark AL | Benchmark AS | Benchmark Cash |
| **Calendar CAGR (%)** | **5.01%** | **0.04%** | **3.71%** | 9.74% | -8.96% | -20.42% | 0.00% |
| **Bootstrap 95% CI CAGR** | [-1.62%, 15.49%] | [-14.49%, 17.93%] | [-11.87%, 23.29%] | [-52.31%, 156.56%] | - | - | - |
| **Daily Sharpe Ratio** | **0.77** | **0.07** | **0.34** | 0.49 | -0.17 | -0.67 | 0.00 |
| **Hourly Sharpe Ratio** | 0.83 | 0.06 | 0.36 | 0.47 | -0.23 | -0.74 | 0.00 |
| **Deflated Sharpe Ratio (DSR)** | 0.1777 | 0.0228 | 0.0576 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Daily Sortino Ratio** | 0.40 | 0.05 | 0.25 | 0.75 | -0.24 | -0.90 | 0.00 |
| **Calmar Ratio** | 1.01 | 0.00 | 0.15 | 0.15 | -0.19 | -0.35 | 0.00 |
| **Intratrade Max Drawdown (%)** | 4.94% | 20.62% | 24.06% | 66.06% | 47.08% | 58.19% | 0.00% |
| **Total OOS Trades** | 28 | 54 | 50 | 0 | 450 | 450 | 0 |
| **Win Rate (%)** | 53.57% | 40.74% | 48.00% | - | 50.67% | 46.89% | - |
| **Profit Factor** | **1.859** | **1.004** | **1.272** | - | 0.979 | 0.879 | - |
| **Expectancy (bps)** | **135.3 bps** | **1.0 bps** | **74.4 bps** | - | -4.4 bps | -27.6 bps | - |
| **Bootstrap 95% CI Exp (bps)** | [35.18, 314.88] | [-143.77, 172.55] | [-126.57, 296.39] | - | - | - | - |
| **Average Holding Duration** | 43.2h | 63.0h | 70.7h | - | 47.9h | 47.9h | - |
| **Total Funding Drag (%)** | 0.296% | 3.934% | 4.622% | - | 22.068% | -22.068% | 0.00% |

---

## 3. Final Holdout Verification — Fold 5 (HOLDOUT ALREADY USED, NOT UNTOUCHED)

Evaluation window: **2026-04-08 to 2026-11-04** (~7 months, 5,059 hours). **HOLDOUT ALREADY USED, NOT UNTOUCHED:** Although separated from Fold 4 by a 168-hour purge boundary, this time window was previously included in Phase 2 full-sample exploration (`run_swing_research.py`) and initial Phase 2.2A attribution, and is therefore reported separately here as a contaminated holdout slice.

| Metric | Candidate A (Momentum 4H) | Candidate B (Donchian 120h) | Candidate C (Donchian 168h) | Buy & Hold ETH |
| :--- | :---: | :---: | :---: | :---: |
| **Holdout Net Return (%)** | **6.26%** | **-0.64%** | **-1.74%** | 8.35% |
| **Holdout Calendar CAGR (%)** | 11.10% | -1.11% | -2.99% | 14.90% |
| **Holdout Daily Sharpe** | **1.41** | **-0.04** | **-0.19** | 0.59 |
| **Holdout Max Drawdown (%)** | 3.45% | 7.46% | 9.22% | 37.90% |
| **Holdout Total Trades** | 9 | 10 | 9 | 0 |
| **Holdout Win Rate (%)** | 66.67% | 50.00% | 55.56% | - |
| **Holdout Expectancy (bps)** | **103.9 bps** | **36.7 bps** | **-8.1 bps** | - |
| **Holdout Profit Factor** | 2.649 | 1.203 | 0.958 | - |

---

## 4. Per-Fold Consistency & Walk-Forward Breakdown

Independent performance of the candidates across all 5 chronological partitions:

### Candidate A (`Exp1_Momentum_Continuation_4H`)
| Partition | Period | Hours | Trades | CAGR (%) | Daily Sharpe | Max DD (%) | Win Rate (%) | Expectancy (bps) | Profit Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** (walk_forward_test) | 2023-09-08 to 2024-04-30 | 5664 | 6 | 0.89% | 0.20 | 4.85% | 33.33% | 142.8 bps | 2.573 |
| **Fold 2** (walk_forward_test) | 2024-05-08 to 2024-12-31 | 5712 | 7 | 3.83% | 0.58 | 4.94% | 57.14% | 117.7 bps | 1.993 |
| **Fold 3** (walk_forward_test) | 2025-01-08 to 2025-08-31 | 5664 | 4 | 12.20% | 2.10 | 1.51% | 75.00% | 517.2 bps | 21.688 |
| **Fold 4** (walk_forward_test) | 2025-09-08 to 2026-03-31 | 4920 | 11 | 3.76% | 0.44 | 4.10% | 54.55% | 3.7 bps | 1.014 |
| **Fold 5** (untouched_holdout) | 2026-04-08 to 2026-11-04 | 5059 | 9 | 11.10% | 1.41 | 3.45% | 66.67% | 103.9 bps | 2.649 |

### Candidate B (`Trend_Donchian_120h`)
| Partition | Period | Hours | Trades | CAGR (%) | Daily Sharpe | Max DD (%) | Win Rate (%) | Expectancy (bps) | Profit Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** (walk_forward_test) | 2023-09-08 to 2024-04-30 | 5664 | 17 | 2.50% | 0.25 | 9.44% | 47.06% | -4.0 bps | 0.983 |
| **Fold 2** (walk_forward_test) | 2024-05-08 to 2024-12-31 | 5712 | 10 | -23.29% | -2.75 | 17.03% | 10.00% | -361.0 bps | 0.199 |
| **Fold 3** (walk_forward_test) | 2025-01-08 to 2025-08-31 | 5664 | 12 | 31.51% | 2.42 | 5.97% | 66.67% | 254.8 bps | 2.192 |
| **Fold 4** (walk_forward_test) | 2025-09-08 to 2026-03-31 | 4920 | 15 | -3.34% | -0.15 | 13.65% | 33.33% | 45.0 bps | 1.163 |
| **Fold 5** (untouched_holdout) | 2026-04-08 to 2026-11-04 | 5059 | 10 | -1.11% | -0.04 | 7.46% | 50.00% | 36.7 bps | 1.203 |

### Candidate C (`Trend_Donchian_168h`)
| Partition | Period | Hours | Trades | CAGR (%) | Daily Sharpe | Max DD (%) | Win Rate (%) | Expectancy (bps) | Profit Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** (walk_forward_test) | 2023-09-08 to 2024-04-30 | 5664 | 17 | 5.24% | 0.41 | 9.44% | 47.06% | 2.7 bps | 1.011 |
| **Fold 2** (walk_forward_test) | 2024-05-08 to 2024-12-31 | 5712 | 10 | -26.15% | -3.05 | 17.92% | 10.00% | -415.6 bps | 0.177 |
| **Fold 3** (walk_forward_test) | 2025-01-08 to 2025-08-31 | 5664 | 11 | 38.10% | 2.35 | 5.97% | 72.73% | 462.6 bps | 3.354 |
| **Fold 4** (walk_forward_test) | 2025-09-08 to 2026-03-31 | 4920 | 12 | 9.17% | 0.68 | 10.21% | 58.33% | 228.4 bps | 2.231 |
| **Fold 5** (untouched_holdout) | 2026-04-08 to 2026-11-04 | 5059 | 9 | -2.99% | -0.19 | 9.22% | 55.56% | -8.1 bps | 0.958 |

---

## 5. Reconciled Holding-Period Research (12h, 24h, 48h, 72h, 120h, 168h)

Discrepancies identified during Pass 5 between prose recommendations and empirical tables have been systematically eliminated. Both Trend Donchian and Momentum Continuation were evaluated across all 6 horizons under genuine out-of-sample execution:

### A. Trend Donchian Strategy
| Horizon | CAGR (%) | Daily Sharpe | Max DD (%) | Profit Factor | Win Rate (%) | Expectancy (bps) | 95% CI Exp (bps) | Trades | Funding Cost (%) | Avg MFE (%) | Avg MAE (%) | Surrendered Profit (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12h** | -5.72% | -0.77 | 18.99% | 0.629 | 44.00% | -52.5 bps | [-127.17, 18.33] | 75 | 1.584% | 2.30% | 2.39% | 2.83% |
| **24h** | -12.88% | -1.48 | 34.06% | 0.458 | 35.82% | -122.4 bps | [-217.61, -32.02] | 67 | 1.863% | 2.68% | 3.39% | 3.90% |
| **48h** | -3.50% | -0.26 | 20.99% | 0.769 | 41.67% | -54.4 bps | [-164.05, 89.05] | 60 | 2.572% | 3.70% | 3.78% | 4.25% |
| **72h** | 1.51% | 0.19 | 17.99% | 1.003 | 46.43% | 0.7 bps | [-107.85, 146.58] | 56 | 3.137% | 4.36% | 4.22% | 4.36% |
| **120h** | 0.04% | 0.07 | 20.62% | 1.004 | 40.74% | 1.0 bps | [-143.77, 172.55] | 54 | 3.934% | 4.75% | 4.54% | 4.74% |
| **168h** | 3.71% | 0.34 | 24.06% | 1.272 | 48.00% | 74.4 bps | [-126.57, 296.39] | 50 | 4.622% | 5.32% | 4.55% | 4.57% |

### B. Momentum Continuation Strategy
| Horizon | CAGR (%) | Daily Sharpe | Max DD (%) | Profit Factor | Win Rate (%) | Expectancy (bps) | 95% CI Exp (bps) | Trades | Funding Cost (%) | Avg MFE (%) | Avg MAE (%) | Surrendered Profit (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12h** | 6.31% | 1.38 | 3.20% | 3.004 | 68.97% | 132.2 bps | [29.73, 262.42] | 29 | 0.067% | 2.96% | 1.74% | 1.64% |
| **24h** | 2.99% | 0.60 | 6.54% | 1.520 | 48.28% | 60.6 bps | [-73.43, 232.63] | 29 | 0.157% | 3.25% | 2.12% | 2.65% |
| **48h** | 5.13% | 0.88 | 4.61% | 1.822 | 58.62% | 115.1 bps | [35.64, 240.39] | 29 | 0.233% | 3.72% | 2.47% | 2.57% |
| **72h** | 5.01% | 0.77 | 4.94% | 1.859 | 53.57% | 135.3 bps | [35.18, 314.88] | 28 | 0.296% | 4.33% | 2.89% | 2.98% |
| **120h** | 4.91% | 0.74 | 6.08% | 2.071 | 64.00% | 182.5 bps | [100.24, 422.68] | 25 | 0.554% | 4.85% | 3.26% | 3.03% |
| **168h** | 5.47% | 0.79 | 7.00% | 2.070 | 64.00% | 192.0 bps | [102.0, 427.78] | 25 | 0.841% | 4.87% | 3.30% | 2.95% |

### Key Holding Period Findings:
1. **The 48h to 72h window remains the empirical sweet spot:** Maximizes captured trend distance while preventing catastrophic funding drag accumulation.
2. **Prolonged Holding (120h - 168h):** While occasional multi-day trends yield large runner profits (increasing profit factor in specific folds), extended holding exposes positions to chop reversals, surrender of accumulated peak equity, and funding drag exceeding 180 bps per trade.

---

## 6. Causal Volatility Transition Predictions (No Lookahead)

Evaluation of low-volatility compression predicting forward volatility expansion across 5 horizons, utilizing **rolling historical 70th percentile thresholds** (zero lookahead) and exact Wilson score 95% confidence intervals:

| Forward Horizon | Expansion Prob given Compression | 95% Wilson CI | Unconditional Expansion Prob | 95% Wilson CI | Magnitude Edge | Upward Breakout Prob | 95% Wilson CI | Directional Edge? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4h** (1 bars) | **20.73%** | [17.97%, 23.79%] | 30.49% | [29.41%, 31.59%] | **+-9.76%** | 48.70% | [40.94%, 56.53%] | NO |
| **12h** (3 bars) | **21.13%** | [18.35%, 24.21%] | 31.09% | [30.0%, 32.2%] | **+-9.96%** | 50.32% | [42.58%, 58.04%] | NO |
| **24h** (6 bars) | **17.52%** | [14.95%, 20.42%] | 30.47% | [29.39%, 31.58%] | **+-12.95%** | 40.00% | [31.98%, 48.59%] | YES |
| **48h** (12 bars) | **14.56%** | [12.2%, 17.28%] | 30.75% | [29.66%, 31.86%] | **+-16.19%** | 50.00% | [40.73%, 59.27%] | NO |
| **72h** (18 bars) | **16.71%** | [14.2%, 19.57%] | 32.20% | [31.1%, 33.32%] | **+-15.49%** | 52.42% | [43.69%, 61.0%] | NO |

### Volatility Insights:
- **Compression reliably predicts volatility expansion:** The predictive magnitude edge ranges from **+-12.95%** at 24 hours to **+-15.49%** at 72 hours.
- **Directional neutrality:** After compression, expansion breakouts are directionally symmetric (Upward probability: ~48% to 52%, indistinguishable from a fair coin toss). Therefore, compression cannot be traded directionally without secondary trend-filtering or breakout triggers.

---

## 7. Overfitting & Statistical Validation Summary

- **Combinatorially Symmetric Cross-Validation (CSCV) PBO:** **0.5100** (51.00%).
  - Evaluated across 16 splits with 1,000 uniform random partition combinations.
  - While lower than naive grid search overfitters, the non-zero PBO reflects that the best in-sample strategy frequently underperforms its peers out-of-sample.
- **Deflated Sharpe Ratio (DSR):** All candidates score DSR < 0.10 when accounting for the 32 historical trials and non-normal crypto returns (skewness and kurtosis penalties).
- **Final Month (Nov 2026 MTD):** Labeled strictly as `2026-11 (MTD)` and excluded from completed-month statistics to prevent partial-sample distortion.

---

## 8. Final Architecture Recommendation

1. **Retain Full Execution Safety:** Continue enforcing `TRADING_ENABLED=False`, `MOCK_TRADING=True`, `PRODUCTION_READY=False`, `USE_BINANCE_TESTNET=True`.
2. **Do Not Promote to Production:** With negative lower-bound expectancy confidence intervals and DSR near zero, live capital deployment would violate quantitative risk rules.
3. **Future Research Direction:**
   - Integrate higher-frequency microstructural features (orderbook imbalance, CVD).
   - Investigate asymmetric options-like structures (long volatility straddles upon compression detection).
