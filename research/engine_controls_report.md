# Standing Engine Control Suite: Audit Report

**Date:** 2026-11-09 06:10:17 UTC  
**Overall Status:** **`16 PASSED, 7 NO TIMING EVIDENCE, 0 FAILED`** (16 Passed, 7 No Timing Evidence, 0 Failed)  
**Evaluated Systems:** Validated Backtest Engine, Trend Strategies (G0/G1/G2/F0), Market-Neutral (N1/N2), OI & Volatility Predictivity Code.

---

## 1. Standing Engine Control Pass/Fail Matrix

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

## 2. Control Methodology & Findings

1. **Random-Signal Control (500 repeats):** Confirms that an uninformative entry signal has zero gross alpha and incurs expected negative friction close to minus total transaction costs (fees, slippage, and funding). A strictly non-positive mean verifies absence of upward survival bias or phantom drift.
2. **Perfect-Foresight Control:** Asserts that an engine given 1-step lookahead information captures large profits, verifying direction sign, execution, and sizing.
3. **Shifted-Date Control (50 random shifts between 30 and 365 days):** Applied to G0, G1, G2, and F0 on BTCUSDT and ETHUSDT, alongside synthetic positive and negative timing benchmarks. A strategy passes only if its median shifted Sharpe is below 50% of the real Sharpe AND fewer than 10% of shifts match or exceed the real Sharpe. Otherwise, status is `NO TIMING EVIDENCE` (an empirical finding).
4. **Buy-and-Hold Replication:** Proves mathematical identity between engine continuous holding and the analytical price return plus funding minus entry and exit costs within 0.1%.
5. **Cost Monotonicity:** Verifies strictly monotonic performance degradation from 0 bps to 10 bps to 20 bps transaction friction.
6. **Timing Control:** Verifies that moving execution fills from next bar open to current bar close alters entry prices and trade returns, proving no instantaneous close lookahead.
7. **Accounting Control:** Proves double-entry accounting reconciliation: the sum of trade-level net PnLs exactly equals final minus initial portfolio equity to the cent ($0.01).
8. **Report-vs-Code Control:** Recomputes metrics from published research reports against saved JSON result artifacts, verifying zero unverified text deviations or copy-paste errors.

---
