# Pre-Registered Volatility-Targeted BTC/ETH vs. Buy-and-Hold: Research Report

**Date:** 2026-11-07 09:59:38 UTC  
**Git Commit:** `0c7650ae65219dbed5dc9166351e9924c1c26a40`  
**Final Run Log:** Read logged to `data/run_log.jsonl` at `2026-11-07T09:58:50.013499+00:00`  
**Evaluated Strategies:** V1 (Vol-Targeted HAR-RV), V2 (200-day SMA Trend Filter), Benchmark (Buy-and-Hold)  
**Universe:** BTCUSDT and ETHUSDT (evaluated separately)  

> [!NOTE]
> **Calendar Period Overlap Disclosure:** The underlying calendar periods (Development: 2020-09 to 2024-12, Final: 2025-01 to 2026-11) were observed during earlier directional trend, funding carry, and market-neutral research. Therefore, while strategy rules, parameters, and pass criteria were pre-registered without tuning, the Final Test partition is not fully untouched.

---

## Standing Backtest Engine Control Suite

Per repository protocol, every research report must include the standing engine control suite results verified on the validated backtest engine:

| # | Control Name | Target / Asset | Observed Value | Threshold / Target | Status | Diagnostics |
| :-: | :--- | :--- | :--- | :--- | :-: | :--- |
| 1 | **Random-Signal Control** | BTCUSDT | Mean: $-49.46, 5-95%: [$-1379.89, $1292.65] | <= $0.00 and close to $-116.55 | `PASS` | Mean Net PnL: $-49.46 (Target -Cost: $-116.55, Diff: $67.09, SEM: $36.90) | 5th-95th percentile: [$-1379.89, $1292.65] across 500 repeats (58 trades/rep) |
| 1 | **Random-Signal Control** | ETHUSDT | Mean: $-114.78, 5-95%: [$-1870.56, $1736.91] | <= $0.00 and close to $-124.00 | `PASS` | Mean Net PnL: $-114.78 (Target -Cost: $-124.00, Diff: $9.22, SEM: $48.80) | 5th-95th percentile: [$-1870.56, $1736.91] across 500 repeats (64 trades/rep) |
| 2 | **Perfect-Foresight Control** | BTCUSDT | +143800900.4% Gain, 100.0% Win | Gain > +100%, Win >= 95% | `PASS` | Initial: $10000.00 -> Final: $14380100041.19 (+143800900.4%), Trades: 2840, Win Rate: 100.0% |
| 2 | **Perfect-Foresight Control** | ETHUSDT | +195722641333.4% Gain, 100.0% Win | Gain > +100%, Win >= 95% | `PASS` | Initial: $10000.00 -> Final: $19572264143342.81 (+195722641333.4%), Trades: 3469, Win Rate: 100.0% |
| 3 | **Shifted-Date Control** | BTCUSDT G0 | Real SR: 1.06, Med Shifted: 0.78, >=Real: 16.0%, PnL 5/50/95: [$802, $2950, $6188] | Med SR < 50% Real & >=Real < 10% | `NO TIMING EVIDENCE` | Real SR: 1.06 (PnL $4521.28) -> Shifted Median SR: 0.78, >=Real: 16.0%, PnL 5/50/95: [$802, $2950, $6188] across 50 shifts (30-365d) |
| 3 | **Shifted-Date Control** | BTCUSDT G1 | Real SR: 1.01, Med Shifted: 0.63, >=Real: 22.0%, PnL 5/50/95: [$-495, $1894, $4386] | Med SR < 50% Real & >=Real < 10% | `NO TIMING EVIDENCE` | Real SR: 1.01 (PnL $3469.91) -> Shifted Median SR: 0.63, >=Real: 22.0%, PnL 5/50/95: [$-495, $1894, $4386] across 50 shifts (30-365d) |
| 3 | **Shifted-Date Control** | BTCUSDT G2 | Real SR: 1.07, Med Shifted: 0.79, >=Real: 16.0%, PnL 5/50/95: [$762, $3014, $5080] | Med SR < 50% Real & >=Real < 10% | `NO TIMING EVIDENCE` | Real SR: 1.07 (PnL $4646.84) -> Shifted Median SR: 0.79, >=Real: 16.0%, PnL 5/50/95: [$762, $3014, $5080] across 50 shifts (30-365d) |
| 3 | **Shifted-Date Control** | BTCUSDT F0 | Real SR: 0.65, Med Shifted: 0.15, >=Real: 8.0%, PnL 5/50/95: [$-2307, $578, $8951] | Med SR < 50% Real & >=Real < 10% | `PASS` | Real SR: 0.65 (PnL $7699.75) -> Shifted Median SR: 0.15, >=Real: 8.0%, PnL 5/50/95: [$-2307, $578, $8951] across 50 shifts (30-365d) |
| 3 | **Shifted-Date Control** | ETHUSDT G0 | Real SR: 0.71, Med Shifted: 0.54, >=Real: 32.0%, PnL 5/50/95: [$-282, $1850, $4222] | Med SR < 50% Real & >=Real < 10% | `NO TIMING EVIDENCE` | Real SR: 0.71 (PnL $2682.32) -> Shifted Median SR: 0.54, >=Real: 32.0%, PnL 5/50/95: [$-282, $1850, $4222] across 50 shifts (30-365d) |
| 3 | **Shifted-Date Control** | ETHUSDT G1 | Real SR: 0.75, Med Shifted: 0.47, >=Real: 18.0%, PnL 5/50/95: [$-1107, $1235, $2514] | Med SR < 50% Real & >=Real < 10% | `NO TIMING EVIDENCE` | Real SR: 0.75 (PnL $2230.48) -> Shifted Median SR: 0.47, >=Real: 18.0%, PnL 5/50/95: [$-1107, $1235, $2514] across 50 shifts (30-365d) |
| 3 | **Shifted-Date Control** | ETHUSDT G2 | Real SR: 0.89, Med Shifted: 0.53, >=Real: 14.0%, PnL 5/50/95: [$-253, $1849, $4623] | Med SR < 50% Real & >=Real < 10% | `NO TIMING EVIDENCE` | Real SR: 0.89 (PnL $3767.26) -> Shifted Median SR: 0.53, >=Real: 14.0%, PnL 5/50/95: [$-253, $1849, $4623] across 50 shifts (30-365d) |
| 3 | **Shifted-Date Control** | ETHUSDT F0 | Real SR: 0.85, Med Shifted: 0.25, >=Real: 4.0%, PnL 5/50/95: [$-2716, $1764, $11206] | Med SR < 50% Real & >=Real < 10% | `PASS` | Real SR: 0.85 (PnL $11934.62) -> Shifted Median SR: 0.25, >=Real: 4.0%, PnL 5/50/95: [$-2716, $1764, $11206] across 50 shifts (30-365d) |
| 3 | **Shifted-Date Control (Synthetic)** | SYNTH_SKILL | Real SR: 64.21, Med Shifted: -0.32, >=Real: 0.0%, PnL 5/50/95: [$-1010, $-137, $1170] | Med SR < 50% Real & >=Real < 10% | `PASS` | Synthetic (Timing Skill): Real SR 64.21 -> Shifted Median SR -0.32, >=Real: 0.0%, PnL 5/50/95: [$-1010, $-137, $1170] |
| 3 | **Shifted-Date Control (Synthetic)** | SYNTH_NOSKILL | Real SR: 0.04, Med Shifted: 0.04, >=Real: 100.0%, PnL 5/50/95: [$16, $16, $16] | Med SR < 50% Real & >=Real < 10% | `NO TIMING EVIDENCE` | Synthetic (No Skill): Real SR 0.04 -> Shifted Median SR 0.04, >=Real: 100.0%, PnL 5/50/95: [$16, $16, $16] |
| 4 | **Buy-and-Hold Replication** | BTCUSDT | Rel Diff: 0.000000% | Rel Diff <= 0.10% | `PASS` | Analytical Eq: $11489.5868 | Simulated Engine Eq: $11489.5868 | Rel Diff: 0.000000% ($0.0000) [Funding: $1605.60, Entry Fee: $10.00, Exit Fee: $13.12] |
| 4 | **Buy-and-Hold Replication** | ETHUSDT | Rel Diff: 0.000000% | Rel Diff <= 0.10% | `PASS` | Analytical Eq: $73749.5529 | Simulated Engine Eq: $73749.5529 | Rel Diff: 0.000000% ($0.0000) [Funding: $10586.84, Entry Fee: $10.00, Exit Fee: $84.43] |
| 5 | **Cost Monotonicity** | BTCUSDT | PnL: [$605.26, $531.87, $458.95] | Strictly decreasing (0 > 10 > 20 bps) | `PASS` | Net PnL @ 0 bps: $605.26 > @ 10 bps: $531.87 > @ 20 bps: $458.95 (Fee drops: -$73.40, -$72.92) |
| 5 | **Cost Monotonicity** | ETHUSDT | PnL: [$1062.74, $990.76, $919.22] | Strictly decreasing (0 > 10 > 20 bps) | `PASS` | Net PnL @ 0 bps: $1062.74 > @ 10 bps: $990.76 > @ 20 bps: $919.22 (Fee drops: -$71.99, -$71.54) |
| 6 | **Timing Control** | BTCUSDT | PnL Diff: $0.2293 | Diff > $0.00 (Fills use Next Open) | `PASS` | Standard (Next Open) PnL: $368.0596 vs Modified (Bar Close) PnL: $368.2889 | Diff: $0.2293 (Entry prices altered: True) |
| 6 | **Timing Control** | ETHUSDT | PnL Diff: $2.8057 | Diff > $0.00 (Fills use Next Open) | `PASS` | Standard (Next Open) PnL: $837.1801 vs Modified (Bar Close) PnL: $839.9859 | Diff: $2.8057 (Entry prices altered: True) |
| 7 | **Accounting Control** | BTCUSDT | Diff: $0.000000 | Discrepancy < $0.01 | `PASS` | Delta Equity: $770.2772 | Sum of 39 Trade PnLs: $770.2772 | Discrepancy: $0.000000 (< $0.01) |
| 7 | **Accounting Control** | ETHUSDT | Diff: $0.000000 | Discrepancy < $0.01 | `PASS` | Delta Equity: $1179.7762 | Sum of 45 Trade PnLs: $1179.7762 | Discrepancy: $0.000000 (< $0.01) |
| 8 | **Report-vs-Code Control** | All Reports (4 Files) | 72 Verified, 0 Discrepancies | 0 Mismatches > Rounding | `PASS` | Verified 72 reported metrics across 4 research reports against saved JSON results. Mismatches exceeding rounding tolerance: 0 |


---

## Pre-Registered Decision & Pass Rules

A strategy passes for a coin **only if in BOTH Development and Final splits**:
1. **Sharpe Ratio:** $\text{Sharpe}_{\text{strat}} \ge \text{Sharpe}_{\text{benchmark}}$
2. **Max Drawdown:** $|\text{MaxDD}_{\text{strat}}| \le 0.60 \times |\text{MaxDD}_{\text{benchmark}}|$
3. **Double Costs Resilience:** $\text{CAGR}_{\text{20bps}} > 0$ whenever Benchmark CAGR (at 10 bps) is positive.

> [!IMPORTANT]
> **Risk-Adjusted vs. Absolute Performance:** These rules evaluate whether volatility targeting or trend filtering delivers structural risk-adjusted improvement and drawdown containment relative to buy-and-hold. In a secular bear market or drawdown, a passing strategy may still experience negative CAGR, provided it preserves capital and curtails downside risk below 60% of buy-and-hold.

---

## Executive Summary of Strategy Verdicts

| Symbol | Strategy | Status | Sharpe (Dev / Final) | MaxDD (Dev / Final) | Double Costs CAGR (Dev / Final) | Verdict Details |
| :-: | :--- | :-: | :---: | :---: | :---: | :--- |
| **BTCUSDT** | **V1** | `FAIL` | 0.88 (BH: 0.84) / -0.07 (BH: -0.11) | 68.7% (BH: 80.7%) / 48.7% (BH: 54.2%) | +32.1% / -9.2% | MaxDD > 60% of B&H (Dev: 68.7% vs 48.4%, Final: 48.7% vs 32.5%) |
| **BTCUSDT** | **V2** | `FAIL` | 1.13 (BH: 0.84) / -0.37 (BH: -0.11) | 50.0% (BH: 80.7%) / 32.1% (BH: 54.2%) | +45.0% / -12.4% | Sharpe lower than B&H (Dev: 1.13 vs 0.84, Final: -0.37 vs -0.11); MaxDD > 60% of B&H (Dev: 50.0% vs 48.4%, Final: 32.1% vs 32.5%) |
| **ETHUSDT** | **V1** | `FAIL` | 0.72 (BH: 0.81) / 0.09 (BH: -0.01) | 61.9% (BH: 80.0%) / 57.9% (BH: 68.8%) | +24.3% / -7.2% | Sharpe lower than B&H (Dev: 0.72 vs 0.81, Final: 0.09 vs -0.01); MaxDD > 60% of B&H (Dev: 61.9% vs 48.0%, Final: 57.9% vs 41.3%) |
| **ETHUSDT** | **V2** | `FAIL` | 0.98 (BH: 0.81) / 0.09 (BH: -0.01) | 60.3% (BH: 80.0%) / 38.8% (BH: 68.8%) | +51.4% / -3.8% | MaxDD > 60% of B&H (Dev: 60.3% vs 48.0%, Final: 38.8% vs 41.3%) |

---

## Performance Metrics Table

### Development Split (2020-09-01 to 2024-12-31)

| Coin | Strategy | Friction | CAGR | Sharpe | Sortino | Max Drawdown | Worst Month | Avg Exposure | Ann. Turnover | Cost Drag |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| BTCUSDT | Buy-and-Hold | 10bps | +39.0% | 0.84 | 0.89 | 80.7% | -37.4% | 1.00 | 0.23x | 39.14% |
| BTCUSDT | V1 (Vol-Target) | 10bps | +33.6% | 0.88 | 0.92 | 68.7% | -25.5% | 0.78 | 11.29x | 22.79% |
| BTCUSDT | V2 (200d SMA) | 10bps | +45.5% | 1.13 | 0.77 | 50.0% | -43.0% | 0.40 | 3.46x | 35.36% |
| BTCUSDT | Buy-and-Hold | 20bps | +39.0% | 0.84 | 0.88 | 80.7% | -37.4% | 1.00 | 0.23x | 39.12% |
| BTCUSDT | V1 (Vol-Target) | 20bps | +32.1% | 0.85 | 0.90 | 69.6% | -25.6% | 0.78 | 11.29x | 24.29% |
| BTCUSDT | V2 (200d SMA) | 20bps | +45.0% | 1.12 | 0.77 | 50.2% | -43.2% | 0.40 | 3.46x | 36.34% |
| ETHUSDT | Buy-and-Hold | 10bps | +38.5% | 0.81 | 0.83 | 80.0% | -44.9% | 1.00 | 0.23x | 54.18% |
| ETHUSDT | V1 (Vol-Target) | 10bps | +25.4% | 0.72 | 0.74 | 61.9% | -22.7% | 0.66 | 8.44x | 20.58% |
| ETHUSDT | V2 (200d SMA) | 10bps | +52.0% | 0.98 | 0.72 | 60.3% | -27.4% | 0.51 | 4.38x | 54.61% |
| ETHUSDT | Buy-and-Hold | 20bps | +38.5% | 0.81 | 0.83 | 80.0% | -44.9% | 1.00 | 0.23x | 54.16% |
| ETHUSDT | V1 (Vol-Target) | 20bps | +24.3% | 0.71 | 0.72 | 62.1% | -22.8% | 0.66 | 8.44x | 21.76% |
| ETHUSDT | V2 (200d SMA) | 20bps | +51.4% | 0.97 | 0.72 | 60.5% | -27.7% | 0.51 | 4.38x | 55.94% |

### Final Split (2025-01-01 to 2026-11-05)

| Coin | Strategy | Friction | CAGR | Sharpe | Sortino | Max Drawdown | Worst Month | Avg Exposure | Ann. Turnover | Cost Drag |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| BTCUSDT | Buy-and-Hold | 10bps | -13.1% | -0.11 | -0.11 | 54.2% | -21.5% | 1.00 | 0.54x | 4.13% |
| BTCUSDT | V1 (Vol-Target) | 10bps | -8.7% | -0.07 | -0.08 | 48.7% | -19.1% | 0.87 | 4.66x | 4.28% |
| BTCUSDT | V2 (200d SMA) | 10bps | -11.6% | -0.37 | -0.24 | 32.1% | -18.0% | 0.40 | 9.75x | 2.83% |
| BTCUSDT | Buy-and-Hold | 20bps | -13.1% | -0.11 | -0.11 | 54.2% | -21.5% | 1.00 | 0.54x | 4.18% |
| BTCUSDT | V1 (Vol-Target) | 20bps | -9.2% | -0.09 | -0.09 | 48.9% | -19.1% | 0.87 | 4.66x | 4.69% |
| BTCUSDT | V2 (200d SMA) | 20bps | -12.4% | -0.41 | -0.26 | 32.8% | -18.0% | 0.40 | 9.75x | 3.61% |
| ETHUSDT | Buy-and-Hold | 10bps | -21.7% | -0.01 | -0.02 | 68.8% | -32.4% | 1.00 | 0.54x | 3.21% |
| ETHUSDT | V1 (Vol-Target) | 10bps | -6.7% | 0.09 | 0.10 | 57.9% | -19.8% | 0.72 | 5.09x | 3.28% |
| ETHUSDT | V2 (200d SMA) | 10bps | -3.0% | 0.09 | 0.05 | 38.8% | -22.6% | 0.24 | 7.59x | 2.50% |
| ETHUSDT | Buy-and-Hold | 20bps | -21.7% | -0.02 | -0.02 | 68.8% | -32.4% | 1.00 | 0.54x | 3.26% |
| ETHUSDT | V1 (Vol-Target) | 20bps | -7.2% | 0.08 | 0.09 | 58.1% | -19.8% | 0.72 | 5.09x | 3.75% |
| ETHUSDT | V2 (200d SMA) | 20bps | -3.8% | 0.07 | 0.03 | 39.2% | -23.1% | 0.24 | 7.59x | 3.19% |

---

## Moving-Block Bootstrap Analysis (10-Day Blocks, 2000 Resamples)

Pairwise paired bootstrap of daily excess returns testing whether strategy Sharpe ratio exceeds buy-and-hold: $\Delta \text{Sharpe} = \text{Sharpe}_{\text{strat}} - \text{Sharpe}_{\text{BH}}$.

| Split | Coin | Strategy | Realized $\Delta$ Sharpe | 95% Bootstrap CI | Bootstrap p-val | Significant ($lpha=0.05$)? |
| :-: | :-: | :--- | :-: | :-: | :-: | :-: |
| Development | BTCUSDT | V1 (Vol-Target) | +0.034 | [-0.195, +0.312] | 0.7020 | `NO` |
| Development | BTCUSDT | V2 (200d SMA) | +0.286 | [-0.482, +1.100] | 0.4230 | `NO` |
| Development | ETHUSDT | V1 (Vol-Target) | -0.083 | [-0.415, +0.224] | 0.5880 | `NO` |
| Development | ETHUSDT | V2 (200d SMA) | +0.171 | [-0.431, +0.828] | 0.5920 | `NO` |
| Final | BTCUSDT | V1 (Vol-Target) | +0.035 | [-0.149, +0.201] | 0.7750 | `NO` |
| Final | BTCUSDT | V2 (200d SMA) | -0.264 | [-1.704, +1.072] | 0.6100 | `NO` |
| Final | ETHUSDT | V1 (Vol-Target) | +0.107 | [-0.200, +0.388] | 0.4970 | `NO` |
| Final | ETHUSDT | V2 (200d SMA) | +0.102 | [-1.423, +1.434] | 0.9360 | `NO` |

---

## Random-Exposure Control (500 Repeats)

Verifies whether strategy performance can be explained by uninformative random exposure with identical average exposure and turnover.

| Split | Coin | Strategy | Real Sharpe | Random 5th-95th Sharpe | Random Median Sharpe | % Random >= Real | Finding |
| :-: | :-: | :--- | :-: | :-: | :-: | :-: | :--- |
| Development | BTCUSDT | V1 (Vol-Target) | 0.88 | [0.56, 1.05] | 0.80 | 30.6% | `EXPLAINED BY RANDOM` |
| Development | BTCUSDT | V2 (200d SMA) | 1.13 | [0.23, 1.18] | 0.73 | 8.8% | `SUPERIOR TO RANDOM` |
| Development | ETHUSDT | V1 (Vol-Target) | 0.72 | [0.49, 1.01] | 0.77 | 64.2% | `EXPLAINED BY RANDOM` |
| Development | ETHUSDT | V2 (200d SMA) | 0.98 | [0.42, 1.07] | 0.74 | 11.8% | `EXPLAINED BY RANDOM` |
| Final | BTCUSDT | V1 (Vol-Target) | -0.07 | [-0.37, 0.13] | -0.14 | 31.0% | `EXPLAINED BY RANDOM` |
| Final | BTCUSDT | V2 (200d SMA) | -0.37 | [-0.72, 0.50] | -0.15 | 72.6% | `EXPLAINED BY RANDOM` |
| Final | ETHUSDT | V1 (Vol-Target) | 0.09 | [-0.39, 0.41] | 0.02 | 38.4% | `EXPLAINED BY RANDOM` |
| Final | ETHUSDT | V2 (200d SMA) | 0.09 | [-0.83, 0.75] | -0.02 | 38.2% | `EXPLAINED BY RANDOM` |

---

## Detailed Analysis & Structural Findings

### BTCUSDT Performance Analysis
#### **V1 (Vol-Targeted HAR-RV)**:
- **Drawdown Containment:** In Development, max drawdown was 68.7% vs. Buy-and-Hold's 80.7% (target <= 60% of B&H = 48.4%). In Final, max drawdown was 48.7% vs. B&H's 54.2% (target <= 32.5%).
- **Sharpe Comparison:** Development Sharpe was 0.88 (B&H: 0.84); Final Sharpe was -0.07 (B&H: -0.11).
- **Cost Sensitivity:** At double friction (20 bps per side), CAGR was +32.1% in Dev and -9.2% in Final.
- **Pre-Registration Status:** **`FAIL`** (MaxDD > 60% of B&H (Dev: 68.7% vs 48.4%, Final: 48.7% vs 32.5%)).

#### **V2 (200-day SMA Trend)**:
- **Drawdown Containment:** In Development, max drawdown was 50.0% vs. Buy-and-Hold's 80.7% (target <= 60% of B&H = 48.4%). In Final, max drawdown was 32.1% vs. B&H's 54.2% (target <= 32.5%).
- **Sharpe Comparison:** Development Sharpe was 1.13 (B&H: 0.84); Final Sharpe was -0.37 (B&H: -0.11).
- **Cost Sensitivity:** At double friction (20 bps per side), CAGR was +45.0% in Dev and -12.4% in Final.
- **Pre-Registration Status:** **`FAIL`** (Sharpe lower than B&H (Dev: 1.13 vs 0.84, Final: -0.37 vs -0.11); MaxDD > 60% of B&H (Dev: 50.0% vs 48.4%, Final: 32.1% vs 32.5%)).

### ETHUSDT Performance Analysis
#### **V1 (Vol-Targeted HAR-RV)**:
- **Drawdown Containment:** In Development, max drawdown was 61.9% vs. Buy-and-Hold's 80.0% (target <= 60% of B&H = 48.0%). In Final, max drawdown was 57.9% vs. B&H's 68.8% (target <= 41.3%).
- **Sharpe Comparison:** Development Sharpe was 0.72 (B&H: 0.81); Final Sharpe was 0.09 (B&H: -0.01).
- **Cost Sensitivity:** At double friction (20 bps per side), CAGR was +24.3% in Dev and -7.2% in Final.
- **Pre-Registration Status:** **`FAIL`** (Sharpe lower than B&H (Dev: 0.72 vs 0.81, Final: 0.09 vs -0.01); MaxDD > 60% of B&H (Dev: 61.9% vs 48.0%, Final: 57.9% vs 41.3%)).

#### **V2 (200-day SMA Trend)**:
- **Drawdown Containment:** In Development, max drawdown was 60.3% vs. Buy-and-Hold's 80.0% (target <= 60% of B&H = 48.0%). In Final, max drawdown was 38.8% vs. B&H's 68.8% (target <= 41.3%).
- **Sharpe Comparison:** Development Sharpe was 0.98 (B&H: 0.81); Final Sharpe was 0.09 (B&H: -0.01).
- **Cost Sensitivity:** At double friction (20 bps per side), CAGR was +51.4% in Dev and -3.8% in Final.
- **Pre-Registration Status:** **`FAIL`** (MaxDD > 60% of B&H (Dev: 60.3% vs 48.0%, Final: 38.8% vs 41.3%)).

---

## Pre-Registration & Methodological Disclosures

- **Zero Parameter Tuning:** The 40% annualized volatility target, 25% rebalance band, 200-day SMA period, and all pass rules were fixed prior to running any simulations.
- **Strict Causal Stamping:** All signals at day D strictly ingested market data <= 23:55 UTC of day D-1, with fills executed at day D open.
- **Walk-Forward Integrity:** Ridge regression models were refit monthly on past data only, with standard scaler parameters fit strictly on training windows.
- **Single Final Run Logging:** The Final Test split (2025-01-01 to end) was logged to `data/run_log.jsonl` and accessed strictly once.
- **Calendar Overlap:** The calendar period overlap with prior experiments is disclosed in full.