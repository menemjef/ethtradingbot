# Pre-Registered Test: S&R Breakout + Triple Barrier + Heuristic Score

> **EXECUTIVE SUMMARY & FORMAL VERDICT**  
> **OVERALL VERDICT:** `FAILED`  
> **Execution Date:** 2026-11-09T06:52:50.020200+00:00  
> **Commit Hash:** `393bfa371acba9f5710316e32310a11e80fa4834`  
> **Universe:** BTCUSDT, ETHUSDT, and Pooled 10-Coin Set (180-day Point-in-Time rule)  
> **Splits:** Development (Sep 2019 – Dec 2024), Final (Jan 2025 – Nov 2026, Single-Read Holdout)  
> **Note on Final Split:** Calendar periods in the Final split were observed by earlier trend/predictivity tests.  

---

## 1. Strategy Specification & Parameters (Fixed Pre-Registered)
- **Bar Frequency:** 1-Hour OHLCV.
- **Levels (Swing Pivots):** High/low is extremum of 5 bars before and 5 bars after; confirmed strictly 5 bars later (causal). Active for 14 days (336 hours) or until broken.
- **Touches:** Later bars (spaced $\ge 6$ bars apart) whose high/low comes within $0.25 \times \text{ATR14}$ of the level before the breakout.
- **Breakout Entry:** Bar closes beyond level by $> 0.25 \times \text{ATR14}$ with $\ge 2$ touches. Fill at NEXT bar open. One trade per level.
- **Barriers:** Stop = $2.0 \times \text{ATR14}$, Take-Profit = $3.0 \times \text{ATR14}$, Vertical Barrier = 24 bars. Stop wins on same-bar touch; opening gaps fill at open with slippage.
- **Heuristic Score (0-3):** Volume (signal volume $\ge 1.5\times$ 20-bar rolling average), Maturity ($\ge 3$ touches), Trend (completed daily EMA50 > EMA200 for long, < for short).
- **Sizing & Risk Gate:** Trade only if score $\ge 2$. Score 3 risks 1.0% equity, score 2 risks 0.5%. Leverage cap 2.0x. No pyramiding (at most 1 open position per coin).
- **Costs:** 10 bps per side base fee, 20 bps double costs, actual funding cash flows.

---

## 2. Standing Engine Controls Suite

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

## 3. Strategy Performance Metrics Table

### Performance Across Universes, Splits, and Fee Levels
| Split | Universe | Fee Level | Trades | Win Rate | Expectancy (bps) [95% CI] | Profit Factor | Sharpe | CAGR | Max DD | Fees ($) | Funding ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Development | `BTCUSDT` | 10 bps | 503 | 43.74% | -8.02 [-27.3, 11.1] | 0.90 | -0.42 | -3.38% | 23.71% | $4397.60 | $-156.04 |
| Development | `BTCUSDT` | 20 bps (2x) | 503 | 42.74% | -28.02 [-47.3, -8.9] | 0.70 | -1.60 | -11.65% | 50.52% | $7055.99 | $-128.27 |
| Development | `ETHUSDT` | 10 bps | 422 | 44.08% | -1.59 [-26.2, 23.1] | 0.95 | -0.21 | -1.72% | 29.08% | $3265.20 | $-181.88 |
| Development | `ETHUSDT` | 20 bps (2x) | 422 | 42.89% | -21.58 [-46.2, 3.1] | 0.78 | -1.10 | -7.74% | 45.70% | $5617.73 | $-165.44 |
| Development | `POOLED_10COIN` | 10 bps | 3613 | 43.07% | -7.46 [-20.8, 7.3] | 0.95 | -0.41 | -14.52% | 68.13% | $19228.84 | $-994.33 |
| Development | `POOLED_10COIN` | 20 bps (2x) | 3613 | 41.93% | -27.46 [-40.8, -12.7] | 0.83 | -1.92 | -44.36% | 95.89% | $16934.58 | $-687.32 |
| Final | `BTCUSDT` | 10 bps | 155 | 45.16% | -9.50 [-30.1, 9.4] | 0.92 | -0.33 | -2.57% | 10.81% | $2146.36 | $-7.32 |
| Final | `BTCUSDT` | 20 bps (2x) | 155 | 42.58% | -29.50 [-50.1, -10.6] | 0.61 | -1.99 | -13.47% | 26.25% | $3859.54 | $-7.89 |
| Final | `ETHUSDT` | 10 bps | 150 | 45.33% | -8.62 [-42.5, 25.8] | 0.94 | -0.20 | -1.65% | 10.66% | $1173.93 | $-10.91 |
| Final | `ETHUSDT` | 20 bps (2x) | 150 | 44.00% | -28.65 [-62.5, 5.8] | 0.76 | -1.07 | -7.63% | 17.76% | $2218.59 | $-10.39 |
| Final | `POOLED_10COIN` | 10 bps | 1386 | 43.36% | -13.01 [-30.7, 4.3] | 0.92 | -0.62 | -19.98% | 45.73% | $9330.65 | $-157.78 |
| Final | `POOLED_10COIN` | 20 bps (2x) | 1386 | 41.63% | -33.00 [-50.7, -15.7] | 0.76 | -2.49 | -54.14% | 79.89% | $12015.22 | $-110.61 |

### Share of Trades by Exit Type
| Split | Universe | Stop Loss (%) | Take Profit (%) | Time-Out (24h) (%) |
| :--- | :--- | :--- | :--- | :--- |
| Development | `BTCUSDT` | 46.32% | 35.79% | 17.89% |
| Development | `ETHUSDT` | 47.16% | 35.78% | 17.06% |
| Development | `POOLED_10COIN` | 47.83% | 35.04% | 17.13% |
| Final | `BTCUSDT` | 45.16% | 36.77% | 18.06% |
| Final | `ETHUSDT` | 45.33% | 34.00% | 20.67% |
| Final | `POOLED_10COIN` | 48.63% | 34.85% | 16.52% |

### Expectancy Breakdown by Heuristic Score (Development Split)
| Symbol | Score 0 (Exp bps / N) | Score 1 (Exp bps / N) | Score 2 (Exp bps / N) | Score 3 (Exp bps / N) |
| :--- | :--- | :--- | :--- | :--- |
| `BTCUSDT` | -61.5 bps (81) | 4.0 bps (311) | -13.3 bps (377) | 7.6 bps (126) |
| `ETHUSDT` | -58.6 bps (77) | -23.4 bps (283) | 4.1 bps (306) | -16.5 bps (116) |

### Expectancy Breakdown by Heuristic Score (Final Split)
| Symbol | Score 0 (Exp bps / N) | Score 1 (Exp bps / N) | Score 2 (Exp bps / N) | Score 3 (Exp bps / N) |
| :--- | :--- | :--- | :--- | :--- |
| `BTCUSDT` | -38.5 bps (39) | -31.8 bps (111) | -19.3 bps (111) | 15.2 bps (44) |
| `ETHUSDT` | -16.7 bps (22) | 32.6 bps (107) | -7.5 bps (105) | -11.2 bps (45) |

### Paired Bootstrap Test: Score >= 2 vs Score <= 1 (Final Split)
- **Mean Expectancy Difference (Score $\ge 2$ minus Score $\le 1$):** `3.39 bps`
- **95% Bootstrap Confidence Interval:** `[-13.73, 21.06] bps`
- **Empirical p-value:** `0.3515`
- **Statistically Significant (CI excludes zero):** `False`

### Buy-and-Hold Benchmark Comparisons
| Split | Symbol | Strategy CAGR | B&H CAGR | Strategy Sharpe | B&H Sharpe | Strategy Max DD | B&H Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Development | `BTCUSDT` | -3.38% | 27.83% | -0.42 | 0.70 | 23.71% | 89.51% |
| Development | `ETHUSDT` | -1.72% | 44.78% | -0.21 | 0.93 | 29.08% | 98.49% |
| Final | `BTCUSDT` | -2.57% | -7.67% | -0.33 | -0.28 | 10.81% | 32.16% |
| Final | `ETHUSDT` | -1.65% | -11.32% | -0.20 | -0.24 | 10.66% | 42.56% |

---

## 4. Experimental Controls Evaluation

| Control Test | Target / Universe | Metric / Observed | Threshold / Requirement | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Shifted-Date Control** | BTCUSDT | Real SR: -0.42, Med Shifted: -1.51, >=Real: 0.0% | Med SR < 50% Real & >=Real < 10% | `NO TIMING EVIDENCE` |
| **Shifted-Date Control** | ETHUSDT | Real SR: -0.21, Med Shifted: -1.23, >=Real: 0.0% | Med SR < 50% Real & >=Real < 10% | `NO TIMING EVIDENCE` |
| **Random-Filter Control** | BTCUSDT | p-value = 0.3900 (Random Exp: -10.6 bps) | p < 0.05 | `FAIL` |
| **Random-Entry Control** | BTCUSDT | p-value = 0.0920 (Random Exp: -21.8 bps) | p < 0.05 | `FAIL` |
| **Perfect-Foresight Check** | BTCUSDT | Win Rate: 100.00%, Gain: $4450.94 | Win Rate >= 95%, Gain > 0 | `PASS` |
| **No-Lookahead Mutation Check** | BTCUSDT | Zero Signal Mutation on Future Alteration | Bit-for-bit identity | `PASS` |

---

## 5. Pre-Registered Pass Rules Verification
1. **Rule (a) - Positive Expectancy in Both Splits & Final Criteria:** `FAIL`  
   - Requirement: Expectancy > 0 in Development and Final splits. In Final, 95% CI lower bound > 0 or pooled trade count >= 150 with positive expectancy.
2. **Rule (b) - Final Double Costs Sharpe > 0.7 and CAGR > 0:** `FAIL`  
   - Requirement: Strategy maintains Sharpe > 0.7 and positive CAGR at 20 bps fee friction in Final split.
3. **Rule (c) - Score Monotonicity (Score >= 2 beats Score <= 1):** `FAIL`  
   - Requirement: Paired bootstrap 95% CI of expectancy difference strictly excludes zero.
4. **Rule (d) - Beats Random-Filter & Random-Entry Controls:** `FAIL`  
   - Requirement: Both controls show p < 0.05.
5. **Rule (e) - Shifted-Date Timing Evidence:** `FAIL`  
   - Requirement: Strict shifted-date control rule satisfied.

### Standing Verdict
**`FAILED`**

---

## 6. Limitations & Research Discussion
1. **Breakout Degradation in Modern Regimes:** Classical horizontal support and resistance breakout models suffer substantial friction in modern electronic cryptocurrency markets. The prevalence of liquidity sweeps and false breakouts results in high stop-out rates.
2. **Microstructure Frictions:** With standard 10 bps and stress-tested 20 bps execution friction, edge in 24-hour holding trades is compressed, requiring high win rates or skewed reward-to-risk ratios to survive.
3. **Strict Research Discipline:** In accordance with pre-registered protocol, no parameter tuning, threshold shifting, or rescue tweaks are applied. The standing verdict is reported objectively from saved results.
