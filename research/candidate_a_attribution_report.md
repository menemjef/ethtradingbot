# Candidate A Edge Attribution Report (`Exp1_Momentum_Continuation_4H`)

**Repository**: `menemjef/ethtradingbot`  
**Working Branch**: `audit/backtest-engine-validation`  
**Strategy**: `Exp1_Momentum_Continuation_4H` (Candidate A)  
**Primary OOS Scope (`include_holdout=False`)**: Walk-Forward Folds 1–4 (`2023-09-08` to `2026-03-31`, 28 completed OOS trades)  
**Separate Holdout Scope**: Fold 5 (`2026-04-08` to `2026-11-04`, 9 completed trades — **HOLDOUT ALREADY USED, NOT UNTOUCHED**)  
**Strategy Rules**: Frozen baseline (3-candle pullback + engulfing volume breakout in 4H trend, 1.5x ATR SL, 3.0x ATR TP, 72h max holding)  

---

## 1. Executive Summary (Walk-Forward OOS Folds 1–4, `include_holdout=False`)

A causal empirical edge attribution analysis was performed on Candidate A (`Exp1_Momentum_Continuation_4H`) strictly across the **28 completed Walk-Forward Out-of-Sample trades in Folds 1–4** (`include_holdout=False`), excluding Holdout Fold 5 from the primary attribution statistics.

### Key OOS Discoveries (Folds 1–4, $N = 28$):
1. **Short-Side Superiority**: Short trades contributed **78.3% of all OOS strategy profits** (+$1,007.30 of +$1,286.76 uncompounded fold sum PnL; +$1,104.18 of +$1,286.74 compounded OOS PnL). Short trades ($N=19$) delivered a 57.89% win rate, 2.06 Profit Factor, and +169.5 bps expectancy, compared to Long trades ($N=9$) which generated a 44.44% win rate, 1.42 Profit Factor, and +63.3 bps expectancy.
2. **First 12-Hour Decisive Trajectory**: The first 12 hours after entry strongly bifurcate trade outcomes. OOS trades that were positive at $t=12\text{h}$ ($N=17$) achieved a **76.47% win rate**, a Profit Factor of **13.66**, and an expectancy of **+387.9 bps** (+$2,444.44 PnL). Conversely, OOS trades underwater at $t=12\text{h}$ ($N=11$) had only an **18.18% win rate**, a Profit Factor of **0.28**, and an expectancy of **-255.0 bps** (-$1,157.68 PnL).
3. **Regime 3 (Strong Trend) Concentration**: Strong trend regimes (4H ADX $\ge 25$, $N=14$) produced **93.9% of OOS net profit** (+$1,207.86) with +239.7 bps expectancy and 2.75 Profit Factor. Compression regimes ($N=9$) lost -$221.23 (-20.2 bps expectancy, 0.84 PF), whereas expansion regimes ($N=4$) generated +31.1 bps (+$50.95).
4. **Volume Sweet-Spot**: OOS entries with 1.8x to 2.5x volume expansion ($N=16$) produced +202.3 bps expectancy and 2.45 PF (+$944.81), and moderate volume 1.4x to 1.8x ($N=5$) produced +168.6 bps expectancy and 3.00 PF (+$440.47). Extreme volume spikes (> 2.5x, $N=4$) suffered from exhaustion (+12.8 bps expectancy, MAE 3.63%), while baseline volume (< 1.4x, $N=3$) had negative expectancy (-113.9 bps, 0.55 PF).
5. **Surrendered Profit and Time Horizon Decay**: 8 OOS trades (28.6% of OOS trades) expired at the 72-hour horizon, generating a negative expectancy of -180.7 bps and -$433.86 net loss after achieving an average MFE of 2.54%.

---

## 2. Walk-Forward OOS Statistical Attribution Breakdown (Folds 1–4, `include_holdout=False`)

### 2.1 Side Asymmetry (Long vs. Short — OOS Folds 1–4)

| Metric | LONG (OOS) | SHORT (OOS) | Overall OOS (Folds 1–4) |
| :--- | :--- | :--- | :--- |
| **Trade Count ($N$)** | 9 (32.1%) | 19 (67.9%) | 28 (100.0%) |
| **Win Rate** | 44.44% | **57.89%** | 53.57% |
| **95% CI Win Rate (Wilson)** | [18.88%, 73.33%] | **[36.28%, 76.86%]** | [35.81%, 70.47%] |
| **Profit Factor** | 1.42 | **2.06** | 1.86 |
| **Expectancy (bps)** | +63.3 bps | **+169.5 bps** | +135.4 bps |
| **95% CI Expectancy (Trade Bootstrap)**| [-239.6, +340.3] bps | **[-69.6, +408.2] bps** | [-50.8, +319.1] bps |
| **95% CI Expectancy (Moving Block)** | — | — | **[+35.18, +314.88] bps** |
| **Net PnL (USD, Uncompounded Fold Sum)** | +$279.46 | **+$1,007.30** | +$1,286.76 |
| **Average MFE** | 3.58% | **4.68%** | 4.33% |
| **Average MAE** | 3.61% | **2.56%** | 2.90% |
| **MFE / MAE Ratio** | 0.99 | **1.83** | 1.49 |
| **Average Duration (hours)** | 49.2h | 40.4h | 43.2h |

---

### 2.2 Market Regime at Entry (OOS Folds 1–4)

| Market Regime | Trades ($N$) | Win Rate (95% CI) | Profit Factor | Expectancy (95% CI) | Net PnL (USD) | MFE / MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Regime 3: Strong Trend (ADX $\ge 25$)** | 14 | 57.14% [32.6%, 78.6%] | **2.75** | **+239.7 bps** [-27.0, +501.0] | **+$1,207.86** | **1.89** |
| **Regime 1: Compression (BB Bandwidth Low)** | 9 | 44.44% [18.9%, 73.3%] | 0.84 | -20.2 bps [-212.3, +178.1] | -$221.23 | 0.85 |
| **Regime 2: Breakout Expansion** | 4 | 50.00% [15.0%, 85.0%] | 1.09 | +31.1 bps [-661.8, +724.0] | +$50.95 | 1.39 |
| **Regime 5: Sideways / Chop (ADX $\le 20$)** | 1 | 100.0% [20.7%, 100.0%] | 99.0 | +492.0 bps [+492.0, +492.0] | +$249.18 | 2.84 |

---

### 2.3 Relative Volume Conditions at Entry (OOS Folds 1–4)

| Relative Volume Tier | Trades ($N$) | Win Rate (95% CI) | Profit Factor | Expectancy (95% CI) | Net PnL (USD) | Avg MFE / MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Moderate Volume (1.4x – 1.8x)** | 5 | **80.00%** [37.6%, 96.4%] | **3.00** | +168.6 bps [-149.6, +484.5] | +$440.47 | **2.14** (3.59% / 1.68%) |
| **High Volume (1.8x – 2.5x)** | 16 | 50.00% [28.0%, 72.0%] | **2.45** | **+202.3 bps** [-57.4, +455.0] | **+$944.81** | **1.73** (5.00% / 2.90%) |
| **Extreme Volume (> 2.5x)** | 4 | 50.00% [15.0%, 85.0%] | 1.05 | +12.8 bps [-568.6, +527.1] | +$168.96 | 1.01 (3.65% / 3.63%) |
| **Baseline Volume (< 1.4x)** | 3 | 33.33% [6.2%, 79.2%] | 0.55 | -113.9 bps [-457.2, +412.6] | -$267.48 | 0.73 (2.88% / 3.95%) |

---

### 2.4 Bitcoin Trend Alignment and ETH Relative Strength (OOS Folds 1–4)

| Context Factor | Category | Trades ($N$) | Win Rate | Expectancy | Net PnL (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BTC Trend Alignment** | Aligned with BTC (Same Dir) | 26 | 53.85% | +143.2 bps | +$1,294.00 |
| | Counter to BTC Trend | 2 | 50.00% | +33.3 bps | -$7.24 |
| **ETH 24H Relative Strength**| ETH Underperforming BTC | 18 | 55.56% | +164.5 bps | +$978.77 |
| | ETH Outperforming BTC | 10 | 50.00% | +82.8 bps | +$307.99 |

---

### 2.5 First 12-Hour Trajectory Analysis (OOS Folds 1–4)

| 12-Hour Status | Trades ($N$) | Win Rate (95% CI) | Profit Factor | Expectancy (95% CI) | Net PnL (USD) | MFE / MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive at 12 Hours ($PnL_{12h} > 0$)** | 17 (60.7%) | **76.47%** [52.7%, 90.4%] | **13.66** | **+387.9 bps** [+191.4, +569.7] | **+$2,444.44** | **3.61** |
| **Negative at 12 Hours ($PnL_{12h} \le 0$)** | 11 (39.3%) | **18.18%** [5.1%, 47.7%] | **0.28** | **-255.0 bps** [-491.6, +9.9] | **-$1,157.68** | **0.41** |

---

### 2.6 Exit Mechanism Breakdown & MFE/MAE Profile (OOS Folds 1–4)

| Exit Reason | Trades ($N$) | Win Rate | Expectancy | Net PnL (USD) | Avg MFE | Avg MAE | Avg Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Take Profit (3.0x ATR)** | 13 | 100.0% | +621.1 bps | +$3,202.80 | 7.41% | 1.32% | 34.0 hours |
| **Stop Loss (1.5x ATR)** | 5 | 0.0% | -491.2 bps | -$1,300.37 | 0.66% | 5.97% | 32.6 hours |
| **Horizon Expiry (72h)** | 8 | 12.50% | -180.7 bps | -$433.86 | 2.54% | 3.59% | 72.0 hours |
| **Fold Boundary Close** | 2 | 50.0% | -191.3 bps | -$181.81 | 0.67% | 2.70% | 15.0 hours |

---

## 3. Holdout Fold 5 Results — HOLDOUT ALREADY USED, NOT UNTOUCHED

> **IMPORTANT — HOLDOUT ALREADY USED, NOT UNTOUCHED**: Fold 5 (`2026-04-08` to `2026-11-04`, 9 trades) was previously included in the Phase 2 full-sample exploration (`run_swing_research.py`) and in the initial Phase 2.2A 37-trade attribution analysis. It is isolated in this section for transparency and must not be treated as an untouched holdout.

### 3.1 Holdout Fold 5 Summary & Side Split ($N = 9$)

| Slice | Trades ($N$) | Win Rate (95% CI) | Profit Factor | Expectancy (95% CI) | Net PnL (USD) | Avg MFE / MAE | Avg Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall Fold 5** | 9 | 66.67% [35.4%, 87.9%] | 2.65 | +103.9 bps [-56.5, +264.6] | +$626.21 | 1.56 (3.36% / 2.15%) | 58.3h |
| **LONG (Fold 5)** | 5 | 40.00% [11.8%, 76.9%] | 0.84 | -18.1 bps [-203.6, +199.6] | +$14.71 | 1.70 (3.43% / 2.02%) | 59.0h |
| **SHORT (Fold 5)** | 4 | 100.00% [51.0%, 100.0%] | 99.0 | +256.4 bps [+126.7, +386.0] | +$611.50 | 1.41 (3.27% / 2.32%) | 57.5h |

### 3.2 Holdout Fold 5 Regime, Trajectory, and Exit Breakdown ($N = 9$)

| Dimension | Category | Trades ($N$) | Win Rate | Profit Factor | Expectancy | Net PnL (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Market Regime** | Regime 1: Compression | 4 | 75.00% | 2.89 | +140.2 bps | +$458.52 |
| | Regime 3: Strong Trend | 4 | 50.00% | 1.86 | +58.2 bps | +$110.15 |
| | Regime 5: Sideways / Chop | 1 | 100.00% | 99.0 | +141.7 bps | +$57.54 |
| **First 12h Trajectory** | Positive at 12h ($PnL_{12h} > 0$) | 4 | 100.00% | 99.0 | +312.2 bps | +$764.18 |
| | Negative at 12h ($PnL_{12h} \le 0$) | 5 | 40.00% | 0.45 | -62.7 bps | -$137.97 |
| **Exit Mechanism** | Take Profit (3.0x ATR) | 3 | 100.00% | 99.0 | +393.8 bps | +$708.57 |
| | Horizon Expiry (72h) | 6 | 50.00% | 0.57 | -41.0 bps | -$82.36 |
| | Stop Loss (1.5x ATR) | 0 | — | — | — | $0.00 |

---

## 4. Conclusions and Strategy Implications

1. **Candidate A Edge Persistence in OOS Folds 1–4**:
   Across the 28 strictly out-of-sample walk-forward trades in Folds 1–4, Candidate A achieves an expectancy of **+135.35 bps** (`+135.4 bps`, Moving Block Bootstrap 95% CI: `[+35.18, +314.88] bps`) and a Profit Factor of **1.859** (`1.86`). However, with `DSR = 0.1777` under $K=32$ trials and only ~11 trades/year (`5.01%` CAGR), it does not pass multi-trial deployment hurdles.
2. **Structural Bottlenecks**:
   - **Low Trade Frequency**: 28 trades over 2.57 OOS years (~10.9 trades/year).
   - **Horizon Expiry Drag**: 8 of 28 OOS trades (28.6%) expire at the 72-hour horizon for a net loss of -$433.86 (-180.7 bps expectancy).
   - **Directional Asymmetry**: Short trades generate 78.3% of OOS dollar profits (+169.5 bps expectancy vs +63.3 bps on Longs), and Regime 3 (Strong Trend) generates 93.9% of OOS dollar profits (+239.7 bps vs -20.2 bps in Regime 1 Compression).
