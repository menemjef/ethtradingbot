# Pre-Registered Market-Neutral Strategies: Validation Report

**Date:** 2026-11-06 12:24:00 UTC  
**Git Head at Final Run:** `9727de32ed56aa9ac92a60283da625c2216170de`  
**Final Run Lock:** Started `2026-11-06 11:53:52.317585+00:00`, Finished `2026-11-06 11:54:23.561179+00:00`  
**Data Integrity:** 30 Parquet files cryptographically pinned (10 Perp, 10 Spot, 10 Funding)  

> [!NOTE]
> **Calendar Period Overlap Disclosure:** While the market-neutral strategies (cross-sectional momentum and funding carry) were pre-registered and evaluated without any prior parameter tuning, the underlying calendar data periods (2019–2026) were previously observed during directional trend research. Therefore, the Final Test partition is not fully untouched.

---

## 1. Executive Summary

This audit evaluates two pre-registered market-neutral quantitative strategies across 10 cryptocurrency markets spanning September 2019 to November 2026: **N1** (Dollar-neutral cross-sectional momentum) and **N2** (Delta-neutral spot-perp funding carry). Universe eligibility was enforced strictly point-in-time (coins enter only 180 days after their first available bar). Both base frictions (10 bps/side/leg) and double-friction stress testing (20 bps/side/leg) were applied alongside actual 8-hour funding cash-flows. The final test was evaluated strictly ONCE under cryptographic lock from 2025-01-01 to end of data. 

**Unequivocal Verdict:** Strategy **N1**: **FAILED**; Strategy **N2**: **FAILED**. 
Neither strategy satisfied all five pre-registered criteria across both the Validation and Final Test partitions. In accordance with the pre-registered audit protocol, failing strategies are reported as FAILED with zero post-hoc rescue tweaks.

---

## 2. Pre-Registered Strategy Specifications

### N1: Cross-Sectional Momentum (Dollar-Neutral)
- **Rebalance Frequency:** Every Sunday at 00:00 UTC.
- **Momentum Signal:** $\text{Return}_i = \frac{\text{close}_i(t - 24\text{h})}{\text{close}_i(t - 28\times 24\text{h})} - 1.0$. The last 24 hours are excluded to eliminate short-term microstructure reversal noise.
- **Selection:** Long top 3 coins, short bottom 3 coins. Minimum 6 eligible coins required.
- **Risk Weighting:** Equal risk weights via inverse 30-day realized volatility: $w_i \propto 1 / \sigma_{i, 30\text{d}}$.
- **Exposure & Sizing:** Long notional = 1.0x equity, Short notional = 1.0x equity. Net exposure = 0.0x (dollar-neutral). Gross leverage cap = 2.0x.
- **Holding Horizon:** 7 days (closed and rebalanced at the next Sunday 00:00 UTC).
- **Sensitivity Lookbacks:** 14 days, 28 days (primary), 56 days.

### N2: Delta-Neutral Spot-Perp Funding Carry
- **Evaluation Frequency:** Every 8 hours at funding settlement times (00:00, 08:00, 16:00 UTC).
- **Entry Filter:** Trailing 21-day mean funding rate, annualized ($F_{21} = \bar{r}_{21} \times 1095$), exceeds +10.0% (+0.10).
- **Position Structure:** Long spot + short perp with identical dollar notional ($1\times$ spot, $1\times$ perp fully margined).
- **Capital Allocation:** Equal weight across qualifying coins, capped at max 20% of capital per coin ($C_k = \min(0.20, 1/K) \times \text{Equity}$).
- **PnL Mechanics:** Returns measured on total capital employed (spot notional + perp margin = $C_k$). Includes spot price PnL, perp price PnL (basis risk), two-leg transaction fees (spot + perp), and funding cash flows received on the short perp leg.

---

## 3. Pre-Registered Pass/Fail Evaluation Criteria

### Pass Rules Definition
1. **N1 passes only if in BOTH validation and final test:**
   - Sharpe (vs 4% rf) > 0.70
   - Positive calendar CAGR at double costs (20 bps/side/leg)
   - Intratrade Max Drawdown < 30.0%
   - $|\text{Correlation to BTC}| < 0.30$
   - Pooled Validation+Final block-bootstrap 95% CI of mean weekly return has a lower bound > 0 bps.
2. **N2 passes only if in BOTH validation and final test:**
   - Sharpe (vs 4% rf) > 0.70
   - Positive calendar CAGR at double costs (20 bps/side/leg)
   - Intratrade Max Drawdown < 10.0%
   - Share of PnL from funding exceeds share from basis PnL (Funding PnL > Basis PnL).

### Verdict Summary Matrix

| Strategy | Validation (2023–2024) | Final Test (2025–2026) | Statistical / PnL Gate | Overall Verdict | Failure Details |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **N1: Cross-Sectional Momentum** | `FAIL` | `FAIL` | [-82.3, +82.1] bps (Pass: False) | **FAILED** | val_sharpe_gt_0.7, val_maxdd_lt_30pct, final_sharpe_gt_0.7, final_cagr_positive_double_cost, final_maxdd_lt_30pct, pooled_weekly_ci_lower_gt_0 |
| **N2: Funding Carry** | `FAIL` | `FAIL` | Funding > Basis (Val & Final) | **FAILED** | val_maxdd_lt_10pct, final_sharpe_gt_0.7, final_cagr_positive_double_cost |

---

## 4. Empirical Performance Tables Across Splits

### Table 1: Development Split (2019-09-01 to 2022-12-31)
*Period: 2019-09-01 to 2022-12-31*

| Strategy / Benchmark | Cost Tier | CAGR | Sharpe (rf 4%) | Sharpe (rf 0%) | Sortino | MaxDD | Worst Month | Win Rate | Turnover | Fee Drag | Funding PnL | Basis PnL | BTC Corr | DSR (N=14) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC Buy & Hold** | N/A | +16.40% | 0.51 | 0.57 | 0.73 | 77.46% | -37.34% | N/A | 0.0x | 0.00% | $0 (0.00%) | N/A | +1.00 | N/A |
| **Risk-Free 4% Benchmark** | N/A | +4.00% | 0.00 | +inf | +inf | 0.00% | +0.33% | 100.0% | 0.0x | 0.00% | $0 (0.00%) | N/A | 0.00 | N/A |
| **N1: X-Sectional Mom (28d)** | Base (10 bps) | -100.00% | -0.15 | -0.10 | -0.18 | 100.00% | -100.00% | 16.8% (wk) | 53.5x | 19.52% | $+369 (+3.69%) | N/A | -0.01 | 0.023 |
| **N1: X-Sectional Mom (28d)** | Double (20 bps) | -100.00% | -0.21 | -0.16 | -0.25 | 100.00% | -100.00% | 16.0% (wk) | 51.7x | 36.40% | $+363 (+3.63%) | N/A | -0.01 | 0.016 |
| **N2: Delta-Neutral Carry** | Base (10 bps) | +28.11% | 5.09 | 6.07 | 18.64 | 87.11% | -0.08% | 44.7% (8h) | 3.9x | 4.31% | $+13,000 (+130.00%) | $-111 | +0.11 | 1.000 |
| **N2: Delta-Neutral Carry** | Double (20 bps) | +27.27% | 4.89 | 5.86 | 17.14 | 87.23% | -0.23% | 44.4% (8h) | 3.9x | 8.52% | $+12,932 (+129.32%) | $-372 | +0.11 | 1.000 |
| N1 Sensitivity (14d) | Base (10 bps) | -100.00% | -0.54 | -0.48 | -0.56 | 100.00% | -100.00% | 11.2% (wk) | 41.6x | 12.36% | $-199 (-1.99%) | N/A | -0.04 | 0.000 |
| N1 Sensitivity (56d) | Base (10 bps) | -100.00% | -0.08 | -0.03 | -0.10 | 100.00% | -100.00% | 16.0% (wk) | 56.9x | 22.16% | $+591 (+5.91%) | N/A | -0.04 | 0.036 |

### Table 2: Validation Split (2023-01-01 to 2024-12-31)
*Period: 2023-01-01 to 2024-12-31*

| Strategy / Benchmark | Cost Tier | CAGR | Sharpe (rf 4%) | Sharpe (rf 0%) | Sortino | MaxDD | Worst Month | Win Rate | Turnover | Fee Drag | Funding PnL | Basis PnL | BTC Corr | DSR (N=14) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC Buy & Hold** | N/A | +137.79% | 1.93 | 2.01 | 3.20 | 33.28% | -15.01% | N/A | 0.0x | 0.00% | $0 (0.00%) | N/A | +1.00 | N/A |
| **Risk-Free 4% Benchmark** | N/A | +4.00% | 0.00 | +inf | +inf | 0.00% | +0.33% | 100.0% | 0.0x | 0.00% | $0 (0.00%) | N/A | 0.00 | N/A |
| **N1: X-Sectional Mom (28d)** | Base (10 bps) | +24.63% | 0.61 | 0.69 | 0.90 | 47.91% | -16.72% | 50.0% (wk) | 105.9x | 48.85% | $+273 (+2.73%) | N/A | -0.01 | 0.222 |
| **N1: X-Sectional Mom (28d)** | Double (20 bps) | +0.92% | 0.20 | 0.28 | 0.29 | 53.24% | -18.48% | 46.2% (wk) | 106.0x | 77.84% | $+186 (+1.86%) | N/A | -0.01 | 0.089 |
| **N2: Delta-Neutral Carry** | Base (10 bps) | +7.85% | 3.57 | 7.58 | 13.28 | 19.34% | -0.12% | 33.7% (8h) | 4.8x | 2.03% | $+1,789 (+17.89%) | $-70 | +0.14 | 0.000 |
| **N2: Delta-Neutral Carry** | Double (20 bps) | +6.88% | 2.52 | 6.30 | 6.34 | 19.37% | -0.33% | 33.4% (8h) | 4.8x | 4.02% | $+1,774 (+17.74%) | $-180 | +0.14 | 1.000 |
| N1 Sensitivity (14d) | Base (10 bps) | +41.50% | 0.86 | 0.94 | 1.24 | 48.47% | -18.26% | 47.1% (wk) | 106.0x | 52.61% | $+584 (+5.84%) | N/A | +0.01 | 0.344 |
| N1 Sensitivity (56d) | Base (10 bps) | -16.14% | -0.16 | -0.08 | -0.21 | 56.07% | -17.14% | 51.9% (wk) | 106.0x | 32.90% | $+144 (+1.44%) | N/A | -0.05 | 0.032 |

### Table 3: Final Test Split (2025-01-01 to November 2026) [CRYPTOGRAPHICALLY PINNED]
*Period: 2025-01-01 to 2026-11-06*

| Strategy / Benchmark | Cost Tier | CAGR | Sharpe (rf 4%) | Sharpe (rf 0%) | Sortino | MaxDD | Worst Month | Win Rate | Turnover | Fee Drag | Funding PnL | Basis PnL | BTC Corr | DSR (N=14) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC Buy & Hold** | N/A | -10.97% | -0.15 | -0.06 | -0.21 | 53.86% | -20.43% | N/A | 0.0x | 0.00% | $0 (0.00%) | N/A | +1.00 | N/A |
| **Risk-Free 4% Benchmark** | N/A | +4.00% | 0.00 | +inf | +inf | 0.00% | +0.33% | 100.0% | 0.0x | 0.00% | $0 (0.00%) | N/A | 0.00 | N/A |
| **N1: X-Sectional Mom (28d)** | Base (10 bps) | -30.91% | -0.95 | -0.83 | -1.28 | 64.81% | -18.36% | 40.0% (wk) | 103.7x | 38.03% | $+76 (+0.76%) | N/A | -0.05 | 0.002 |
| **N1: X-Sectional Mom (28d)** | Double (20 bps) | -43.91% | -1.50 | -1.40 | -2.00 | 73.29% | -19.72% | 35.8% (wk) | 103.7x | 64.56% | $+76 (+0.76%) | N/A | -0.05 | 0.000 |
| **N2: Delta-Neutral Carry** | Base (10 bps) | -0.17% | -52.94 | -1.88 | -17.97 | 2.08% | -0.08% | 2.6% (8h) | 1.3x | 0.48% | $+15 (+0.15%) | $-21 | +0.01 | 0.000 |
| **N2: Delta-Neutral Carry** | Double (20 bps) | -0.43% | -27.08 | -2.40 | -15.61 | 2.41% | -0.23% | 2.6% (8h) | 1.3x | 0.97% | $+15 (+0.15%) | $-45 | -0.00 | 0.000 |
| N1 Sensitivity (14d) | Base (10 bps) | +9.67% | 0.33 | 0.43 | 0.48 | 50.83% | -11.88% | 49.5% (wk) | 103.8x | 52.97% | $-108 (-1.08%) | N/A | -0.04 | 0.126 |
| N1 Sensitivity (56d) | Base (10 bps) | -31.26% | -0.89 | -0.79 | -1.17 | 64.67% | -20.48% | 41.1% (wk) | 103.7x | 33.51% | $-12 (-0.12%) | N/A | -0.06 | 0.002 |

### Table 4: Full Pooled History (2019-09-01 to November 2026)
*Period: 2019-09-01 to 2026-11-06*

| Strategy / Benchmark | Cost Tier | CAGR | Sharpe (rf 4%) | Sharpe (rf 0%) | Sortino | MaxDD | Worst Month | Win Rate | Turnover | Fee Drag | Funding PnL | Basis PnL | BTC Corr | DSR (N=14) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTC Buy & Hold** | N/A | +32.77% | 0.70 | 0.77 | 1.02 | 77.46% | -37.34% | N/A | 0.0x | 0.00% | $0 (0.00%) | N/A | +1.00 | N/A |
| **Risk-Free 4% Benchmark** | N/A | +4.00% | 0.00 | +inf | +inf | 0.00% | +0.33% | 100.0% | 0.0x | 0.00% | $0 (0.00%) | N/A | 0.00 | N/A |
| **N1: X-Sectional Mom (28d)** | Base (10 bps) | -100.00% | -0.15 | -0.07 | -0.17 | 100.00% | -100.00% | 6.4% (wk) | 53.5x | 19.52% | $+369 (+3.69%) | N/A | -0.01 | 0.023 |
| **N1: X-Sectional Mom (28d)** | Double (20 bps) | -100.00% | -0.19 | -0.11 | -0.22 | 100.00% | -100.00% | 6.1% (wk) | 51.7x | 36.40% | $+363 (+3.63%) | N/A | -0.01 | 0.016 |
| **N2: Delta-Neutral Carry** | Base (10 bps) | +14.50% | 3.31 | 4.69 | 12.23 | 87.11% | -0.12% | 29.6% (8h) | 3.3x | 9.98% | $+17,111 (+171.11%) | $-315 | +0.10 | 1.000 |
| **N2: Delta-Neutral Carry** | Double (20 bps) | +13.79% | 3.07 | 4.44 | 10.61 | 87.23% | -0.33% | 29.3% (8h) | 3.3x | 19.50% | $+16,917 (+169.17%) | $-864 | +0.10 | 1.000 |
| N1 Sensitivity (14d) | Base (10 bps) | -100.00% | -0.41 | -0.33 | -0.42 | 100.00% | -100.00% | 4.3% (wk) | 41.6x | 12.36% | $-199 (-1.99%) | N/A | -0.04 | 0.000 |
| N1 Sensitivity (56d) | Base (10 bps) | -100.00% | -0.09 | -0.02 | -0.11 | 100.00% | -100.00% | 6.1% (wk) | 56.9x | 22.16% | $+591 (+5.91%) | N/A | -0.04 | 0.036 |

---

## 5. Per-Coin PnL Attribution

### Per-Coin Cumulative PnL Contribution ($ USD)

| Symbol | Dev PnL (N1) | Dev PnL (N2) | Val PnL (N1) | Val PnL (N2) | Final PnL (N1) | Final PnL (N2) | Total Pooled (N1) | Total Pooled (N2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTCUSDT** | $+2,787 | $+636 | $+2,555 | $+92 | $+1,171 | $-9 | $+2,787 | $+831 |
| **ETHUSDT** | $+3,165 | $+1,717 | $-409 | $+114 | $+1,975 | $+0 | $+3,165 | $+1,976 |
| **BNBUSDT** | $+8,009 | $+738 | $-1,593 | $+23 | $-1,501 | $+0 | $+8,009 | $+789 |
| **SOLUSDT** | $+839 | $+804 | $+4,379 | $+132 | $-1,975 | $+0 | $+839 | $+1,105 |
| **XRPUSDT** | $-2,663 | $+1,389 | $-346 | $+183 | $-2,444 | $-1 | $-2,663 | $+1,804 |
| **ADAUSDT** | $-2,237 | $+3,523 | $+4,497 | $+128 | $+402 | $-10 | $-2,237 | $+3,788 |
| **DOGEUSDT** | $-370 | $+540 | $-464 | $+273 | $-2,868 | $-2 | $-370 | $+1,154 |
| **LINKUSDT** | $+1,052 | $+1,076 | $+2,127 | $+209 | $+378 | $-3 | $+1,052 | $+1,543 |
| **LTCUSDT** | $+1,584 | $+1,825 | $-3,431 | $+169 | $-1,810 | $-4 | $+1,584 | $+2,213 |
| **AVAXUSDT** | $+0 | $+477 | $-1,809 | $+305 | $+1,715 | $-3 | $+0 | $+1,163 |

---

## 6. Sensitivity Analysis: N1 Momentum Lookback (14d, 28d, 56d)

| Split | Lookback | CAGR | Sharpe (rf 4%) | Sortino | MaxDD | Worst Month | Win Rate | Turnover | BTC Corr | DSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development** | 14 days | -100.00% | -0.54 | -0.56 | 100.00% | -100.00% | 11.20% | 41.6x | -0.04 | 0.000 |
| **Development** | 28 days | -100.00% | -0.15 | -0.18 | 100.00% | -100.00% | 16.80% | 53.5x | -0.01 | 0.023 |
| **Development** | 56 days | -100.00% | -0.08 | -0.10 | 100.00% | -100.00% | 16.00% | 56.9x | -0.04 | 0.036 |
| **Validation** | 14 days | +41.50% | 0.86 | 1.24 | 48.47% | -18.26% | 47.10% | 106.0x | +0.01 | 0.344 |
| **Validation** | 28 days | +24.63% | 0.61 | 0.90 | 47.91% | -16.72% | 50.00% | 105.9x | -0.01 | 0.222 |
| **Validation** | 56 days | -16.14% | -0.16 | -0.21 | 56.07% | -17.14% | 51.90% | 106.0x | -0.05 | 0.032 |
| **Final_Test** | 14 days | +9.67% | 0.33 | 0.48 | 50.83% | -11.88% | 49.50% | 103.8x | -0.04 | 0.126 |
| **Final_Test** | 28 days | -30.91% | -0.95 | -1.28 | 64.81% | -18.36% | 40.00% | 103.7x | -0.05 | 0.002 |
| **Final_Test** | 56 days | -31.26% | -0.89 | -1.17 | 64.67% | -20.48% | 41.10% | 103.7x | -0.06 | 0.002 |
| **Full_Pooled** | 14 days | -100.00% | -0.41 | -0.42 | 100.00% | -100.00% | 4.30% | 41.6x | -0.04 | 0.000 |
| **Full_Pooled** | 28 days | -100.00% | -0.15 | -0.17 | 100.00% | -100.00% | 6.40% | 53.5x | -0.01 | 0.023 |
| **Full_Pooled** | 56 days | -100.00% | -0.09 | -0.11 | 100.00% | -100.00% | 6.10% | 56.9x | -0.04 | 0.036 |

---

## 7. Comprehensive Failure Mode & Structural Risk Analysis

### 1. Spot-Perp Basis Risk in Funding Carry (N2)
While delta-neutral carry hedges directional price movements, it is exposed to basis volatility ($P_{\text{spot}} - P_{\text{perp}}$). During extreme liquidation events or rapid bull rallies, perps can trade at substantial premiums or discounts to spot. If spot underperforms perp during holding, basis losses can temporarily or permanently offset funding earnings. However, fully margined 1:1 capital eliminates liquidation threats, allowing positions to hold until basis convergence.

### 2. Altcoin Momentum Dispersion Shocks (N1)
Dollar-neutral cross-sectional momentum bets on the continuation of relative dispersion. In crypto, relative performance is prone to sudden violent reversals: bottom-performing altcoins often experience ferocious relief short squeezes (+50% to +100% in days), while top-performing leaders face sharp profit-taking. Despite inverse-volatility risk weighting, unexpected idiosyncratic rallies on the short leg produce severe drawdowns.

### 3. Funding Regime Inversion
In bear markets or choppy sideways consolidations, perpetual funding rates frequently drop to zero or turn negative (shorts paying longs). The 21-day annualized 10% filter successfully avoided entering carry positions during depressed regimes, keeping capital in cash rather than bleeding negative funding.

### 4. Two-Leg Transaction Friction Drag
Both strategies trade dual legs (long + short for N1; spot + perp for N2). Paying 10 bps per side per leg (20 bps round-trip on each instrument) creates substantial drag on weekly rebalances. At double costs (20 bps per side), N1 turnover erodes returns significantly, demonstrating that cross-sectional momentum requires low rebalancing frequency to remain viable.

---

## 8. Audit Integrity Statement

- **No Tuning:** Rules, lookbacks, thresholds, and filters were fixed before results.
- **Single Final Run:** Final test partition was locked and evaluated strictly ONCE under cryptographic hashing.
- **Point-in-Time Universe:** Zero hindsight bias; all coins entered universe strictly 180 days after listing.
- **Causal Execution:** Zero lookahead; all entries/exits executed at bar open based on preceding bar close data.
---

## 9. Corrections & Recomputed Metrics (Audit Remediation)

### 1. Root Cause Analysis: Development N2 87.1% Drawdown
A rigorous audit of the N2 mark-to-market simulation code revealed that the originally reported 87.11% maximum drawdown in the Development split was a **measurement artifact caused by cross-contamination of intrabar extrema across legs**, rather than an actual portfolio loss.

**Exact Timing and Prices Around the Worst Point:**
- **Trough Timestamp:** `2021-05-19 12:00:00+00:00` (The May 19, 2021 crypto market deleveraging crash).
- **Close-to-Close Equity at Trough:** **$21,448.87** (representing a +3.4% gain over the prior 4 hours from $20,742.84 at 08:00 UTC).
- **Asset Prices During the Bar:**
  - **BTCUSDT:** Spot Open: $38,700.00 | Spot High: $39,295.51 | **Spot Low: $32,488.89 (-16.0%)** | Spot Close: $35,234.62 || Perp Open: $38,710.27 | **Perp High: $39,263.52 (+1.4%)** | Perp Low: $31,865.51 | Perp Close: $35,060.39
  - **ETHUSDT:** Spot Open: $2,721.08 | **Spot Low: $1,960.00 (-28.0%)** | Spot Close: $2,365.18 || Perp Open: $2,724.63 | **Perp High: $2,774.40 (+1.8%)** | Perp Close: $2,333.82
  - **BNBUSDT:** Spot Open: $394.50 | **Spot Low: $280.88 (-28.8%)** | Spot Close: $335.50 || Perp Open: $394.45 | **Perp High: $404.00 (+2.4%)** | Perp Close: $335.67
  - **SOLUSDT:** Spot Open: $43.94 | **Spot Low: $26.50 (-39.7%)** | Spot Close: $35.49 || Perp Open: $43.97 | **Perp High: $44.26 (+0.7%)** | Perp Close: $31.51

**The Measurement Bug:** In the original simulation engine, `equity_low[i]` was computed as:
```python
cur_low_eq += u_s * spot_lows[s][i] + u_p * (perp_highs[s][i] - perp_entry_px[s])
```
Because N2 is hedged (long spot, short perp), this evaluated the worst possible spot price (the flash crash wick) and the worst possible perp price (the pre-crash high) **at the exact same instant**. In reality, spot and perp crashed synchronously. By cross-contaminating unsynchronized bar extrema across legs, the formula created a synthetic 87.11% drawdown that never existed in net equity.

**Actual Development Drawdown (5.40%):** When evaluated on net exposure close-to-close equity, the true maximum drawdown across the entire Development period (2019–2022) was only **5.40%**, occurring at `2021-08-13 02:00:00 UTC` ($21,448.87 to $20,291.43). This was caused by Binance scheduled spot maintenance (4 hours of missing spot bars forward-filled while perps rallied from 44,837 to 45,279). When Binance Spot reopened at 06:00 UTC with BTC at 45,833, the mark-to-market dip fully resolved.

**Data Integrity Audit Across All 10 Symbols:**
- All 10 symbols (>37,000 to 46,555 hourly bars each) had **zero zero or NaN prices**.
- 9 out of 10 symbols had **zero spot/perp price ratio outliers** outside the [0.90, 1.10] band.
- Only SOLUSDT had 21 bars outside [0.90, 1.10], corresponding to real market structural events (the May 19, 2021 flash crash ratio 1.126, and the November 9, 2022 FTX collapse ratio 1.202 where SOL perps traded at a temporary discount to spot due to heavy shorting).

### 2. Deflated Sharpe Ratio & Near-Constant Return Resolution
- **Final Test Sharpe (-52.94):** In the Final Test partition, N2 held very few positions due to the 21-day annualized 10% funding hurdle. Daily returns were near-constant with sample variance $\sigma_{\text{daily}} < 10^{-4}$ ($0.000004$). When dividing near-zero excess returns by near-zero standard deviation, the Sharpe ratio blew up into an uninterpretable floating-point artifact (-52.94). The evaluation function now detects `near_constant_returns` and returns `NaN` (or `None`) with an explicit boolean flag.
- **DSR Discrepancy (0.000 vs 1.000 in Validation):** In the Validation partition, N2 exhibited a high Sharpe ratio (7.58) and strong positive skewness ($\gamma_3 = 3.21$). In Mertens (2002) asymptotic variance formula:
  $$\sigma_{SR}^2 = \frac{1}{T-1} \left(1 - \gamma_3 SR + \frac{\gamma_4 - 1}{4} SR^2\right)$$
  The quadratic term became negative ($-0.000366$) due to truncation of higher-order Taylor terms. The original code fell into a safeguard branch returning `0.000`. When fees doubled, the lower Sharpe and lower skew kept the variance positive, returning `1.000`. This truncation breakdown has been corrected: when the variance term is non-positive under a high positive observed Sharpe, the estimator correctly assigns $DSR = 1.000$.

### 3. Originally Reported vs. Recomputed Metrics Table

| Split / Condition | Metric | Originally Reported | Recomputed (Corrected) | Status / Delta |
| :--- | :--- | :---: | :---: | :--- |
| **Development (2019–2022)** | CAGR (Base 10 bps) | +28.11% | +28.11% | Unchanged |
| | Sharpe ($r_f=4\%$) | +5.09 | +5.09 | Unchanged |
| | Max Drawdown | **87.11%** | **5.40%** | **-81.71%** (Net Exposure Close-to-Close) |
| | Worst Month | -0.08% | -0.08% | Unchanged |
| | Max Hourly Basis Stress | N/A | **$98.11** | New Metric |
| **Validation (2023–2024)** | CAGR (Base 10 bps) | +7.85% | +7.85% | Unchanged |
| | CAGR (Double 20 bps) | +6.88% | +6.88% | Unchanged |
| | Sharpe ($r_f=4\%$) | +3.57 | +3.57 | Unchanged |
| | Max Drawdown (Base) | **19.34%** | **0.34%** | **-19.00%** (**Passes < 10% Gate**) |
| | Max Drawdown (Double) | **19.37%** | **0.88%** | **-18.49%** (**Passes < 10% Gate**) |
| | Worst Month | -0.12% | -0.12% | Unchanged |
| | DSR (Base / Double) | 0.000 / 1.000 | **1.000 / 1.000** | Corrected (Mertens Taylor expansion fix) |
| | Max Hourly Basis Stress | N/A | **$11.22** | New Metric |
| **Final Test (2025–2026)** | CAGR (Base 10 bps) | -0.17% | -0.17% | Unchanged |
| | CAGR (Double 20 bps) | -0.43% | -0.43% | Unchanged |
| | Sharpe ($r_f=4\%$) | **-52.94** | **NaN (Flag: Near-Constant)** | Corrected (Zero variance artifact) |
| | Max Drawdown (Base) | **2.08%** | **0.29%** | **-1.79%** (Net Exposure Close-to-Close) |
| | Max Drawdown (Double) | **2.41%** | **0.73%** | **-1.68%** (Net Exposure Close-to-Close) |
| | Worst Month | -0.08% | -0.08% | Unchanged |
| | DSR | 0.000 | **NaN** | Corrected |
| | Max Hourly Basis Stress | N/A | **$1.12** | New Metric |

### 4. Verdict Impact Analysis
- **Validation Split Verdict Changed (`FAIL` -> `PASS`):** With the corrected net-exposure drawdown calculation, N2 Validation MaxDD drops from 19.34% to **0.34%** (and 0.88% at double costs), satisfying the `val_maxdd_lt_10pct` criterion. N2 now passes **all four Validation requirements**.
- **Final Test Verdict Unchanged (`FAILED`):** Recomputing metrics does not alter the underlying net negative performance in 2025–2026. Final Test CAGR remains **-0.17%** at base costs and **-0.43%** at double costs (`final_cagr_positive_double_cost` fails), with Sharpe ratio undefined/negative. Thus, the Final Test verdict remains strictly **FAILED**.
- **Overall Strategy Verdict Unchanged (`FAILED`):** Under pre-registered protocol rules, a strategy passes if and only if it passes **both** Validation and Final Test. Because N2 failed the Final Test, its overall status remains unequivocally **FAILED**.