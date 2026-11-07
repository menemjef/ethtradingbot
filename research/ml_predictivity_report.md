# Pre-Registered Machine Learning Predictivity Report

> **EXECUTIVE SUMMARY & FORMAL VERDICT**  
> **OVERALL VERDICT:** `NO ML PREDICTIVITY BEYOND BASELINE`  
> **Execution Date:** 2026-11-07T12:31:41.378118+00:00  
> **Commit Hash:** `772abb8524e2c16e6ad02b7993bafaa30594c8bd`  
> **Candidate Tests Evaluated:** 12 (9 LightGBM + 2 Clean HybridModel + 1 Existing Checkpoint)  
> **FDR Multi-Testing Control:** Benjamini-Hochberg at $\alpha = 0.05$  
> **Passing Models:** None (0/12)

---

## Part A: Audit of the Existing Trained Model

### A1. Checkpoint Provenance & Training Reconstitution
- **Active Checkpoint:** `checkpoints/best_model.pt`
- **Training Dataset Candles:** 20,000 (1-hour bars)
- **Reconstructed Training Range:** `2024-05-01 21:00:00+00:00` &rarr; `2026-08-13 04:00:00+00:00`
- **Train / Validation Split:** 15968 train / 3993 validation (80/20 Chronological split with 12-bar purge gap)
- **Input Features:** 53 (8 candle structure + 44 market state indicators + 1 target)
- **Target Specification:** Triple Barrier Method (TBM) (PT = 3.0$\times$ ATR, SL = 1.5$\times$ ATR, Horizon = 12 bars)
- **Baseline Pre-Retrain Backup Checkpoint Status:** `EXCLUDED`
  - *Rationale:* Neither training/retrain_report.md nor training/forensic_report.md documents the training start date, end date, or sample range for baseline_pre_retrain. Per pre-registered instructions, its training range cannot be established and it is therefore not evaluated.

### A2. Pipeline Leakage Audit
| Category | Source File & Line | Severity | Mitigation & Verification |
| :--- | :--- | :--- | :--- |
| **Target Lookahead Across Partition Boundaries** | `data/indicators.py (lines 292-410) & train.py (lines 286-291)` | `CRITICAL` | Purge at least H bars (plus 24h embargo) at all split boundaries. |
| **StandardScaler Lookahead via Full Dataset Fit** | `data/preprocessor.py (lines 208-209)` | `HIGH` | Fit scalers strictly on training partition (df_train) only. |
| **Regime Model Boundary Leakage (HMM / GARCH)** | `data/quant_models.py (lines 59, 92)` | `MEDIUM` | Causal feature generation with strict cutoff t <= decision_time - 5m. |
| **Indicator Warmup Pre-roll and Missing Data Filling** | `data/indicators.py (lines 243-248)` | `MEDIUM` | Exclude pre-roll observations from training/test sample sets. |

### A3. Clean Out-of-Sample Evaluation of Existing Checkpoint
- **Evaluation Period (Strictly Post-Validation):** `2026-08-13 05:00:00+00:00` &rarr; `2026-11-05 20:00:00+00:00`
- **Total Clean OOS Bars:** 2,032 hourly bars (~85 calendar days)
- **Spearman IC (12-bar Horizon):** `-0.0687` ($p = 1.9487e-03$)
- **Spearman IC (4-bar Horizon):** `-0.0062` ($p = 7.8030e-01$)
- **Directional Accuracy (12-bar):** `45.28%` [95% CI: `43.09%`, `47.47%`]
- **Directional Accuracy (4-bar):** `47.15%` [95% CI: `44.96%`, `49.34%`]
- **Triple-Barrier Hit Rate:** PT hits: `389`, SL hits: `914` (Win Rate: `29.85%`)

#### Directional Accuracy by Confidence Decile (12-bar)
| Decile | Sample Count | Directional Accuracy | 95% Confidence Interval | Mean |Prediction| |
| :--- | :--- | :--- | :--- | :--- |
| Decile 1 | 206 | 55.34% | [48.27%, 62.25%] | 0.4757 |
| Decile 2 | 225 | 46.67% | [40.01%, 53.41%] | 0.5009 |
| Decile 3 | 180 | 40.56% | [33.31%, 48.11%] | 0.5140 |
| Decile 4 | 204 | 45.59% | [38.62%, 52.69%] | 0.5287 |
| Decile 5 | 201 | 46.77% | [39.71%, 53.92%] | 0.5519 |
| Decile 6 | 213 | 52.11% | [45.18%, 58.99%] | 0.5869 |
| Decile 7 | 197 | 43.65% | [36.62%, 50.89%] | 0.6236 |
| Decile 8 | 199 | 41.21% | [34.29%, 48.38%] | 0.6677 |
| Decile 9 | 211 | 41.71% | [34.98%, 48.67%] | 0.7247 |
| Decile 10 | 196 | 37.76% | [30.95%, 44.94%] | 0.8072 |

#### Threshold Strategy Performance (Hold 12 Bars, Trailing 90d P80/P20)
| Cost Level | Total Trades | Win Rate | Final Equity | Sharpe Ratio | CAGR | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10 bps / side** | 99 | 38.38% | 0.6454 | -5.57 | -84.90% | -41.71% |
| **20 bps / side (Double Costs)** | 99 | 36.36% | 0.5297 | -7.92 | -93.56% | -51.44% |

#### Minimum Detectable Effect (MDE) & Statistical Power Analysis
- **Independent MDE (80% Power, $\alpha=0.05$):** $|r| \ge 0.062$
- **Effective Sample Size MDE (12-bar holding, $N_{eff} \approx 169$):** $|r| \ge 0.214$
- **Directional Accuracy MDE:** $\pm 3.11\%$ (Effective: $\pm 10.78\%$)
- **Sample Size Adequate:** `False`
- **Auditor Rationale:** With N=2032 hourly bars (84.7 days) and 12-bar holding periods (effective N ~ 169), the minimum detectable IC at 80% power is 0.214. Financial alpha in liquid crypto perps is rarely above 0.03-0.05. A sample size of at least 10,000 independent bars (>1.5 years) is required to reliably detect realistic edge. The current 85-day OOS period is severely statistically underpowered.

### A4. Discrepancy Analysis vs `retrain_report.md`
1. In `retrain_report.md`, validation directional accuracy was reported as 51.29% all-sample and 53.59% for threshold-filtered samples ($|reg| \ge 0.045$).
2. On the clean out-of-sample partition (> 2026-08-13), directional accuracy dropped to 45.28% and Spearman IC inverted to -0.0687 ($p = 0.0019$).
3. Decile breakdown demonstrates severe model degradation: top decile confidence predictions yielded only 37.76% accuracy, indicating severe overfitting to the in-sample distribution.

---

## Part B: Walk-Forward Predictivity Test of New Models

### 12 Candidate Tests Multi-Testing Table (Benjamini-Hochberg at $\alpha=0.05$)
| Test ID | Model | Universe | Horizon | Dev IC [95% CI] | Final IC [95% CI] | Final p-val | BH Adj p-val | Ridge Final IC | Strat 20bps Sharpe | Pass All? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LGBM_BTC_1h` | LightGBM | BTC | 1h | 0.0047 [-0.010, 0.019] | 0.0050 [-0.010, 0.020] | 5.2838e-01 | 5.2838e-01 | 0.0096 | -41.48 | **FAIL** |
| `LGBM_BTC_4h` | LightGBM | BTC | 4h | -0.0069 [-0.022, 0.008] | 0.0170 [0.002, 0.032] | 3.0257e-02 | 4.5385e-02 | -0.0024 | -13.89 | **FAIL** |
| `LGBM_BTC_24h` | LightGBM | BTC | 24h | -0.0363 [-0.051, -0.021] | 0.0226 [0.007, 0.038] | 4.1092e-03 | 9.8620e-03 | -0.0247 | -2.59 | **FAIL** |
| `LGBM_ETH_1h` | LightGBM | ETH | 1h | 0.0095 [-0.005, 0.024] | 0.0053 [-0.010, 0.021] | 5.0326e-01 | 5.2838e-01 | -0.0020 | -27.55 | **FAIL** |
| `LGBM_ETH_4h` | LightGBM | ETH | 4h | 0.0172 [0.002, 0.032] | -0.0143 [-0.030, 0.001] | 6.9208e-02 | 9.2278e-02 | -0.0190 | -9.41 | **FAIL** |
| `LGBM_ETH_24h` | LightGBM | ETH | 24h | 0.0068 [-0.008, 0.022] | -0.0608 [-0.076, -0.045] | 1.0240e-14 | 1.2288e-13 | -0.0518 | -2.67 | **FAIL** |
| `LGBM_Pooled_1h` | LightGBM | Pooled | 1h | 0.0210 [0.016, 0.026] | 0.0120 [0.007, 0.017] | 1.3750e-06 | 5.5000e-06 | 0.0187 | 0.00 | **FAIL** |
| `LGBM_Pooled_4h` | LightGBM | Pooled | 4h | 0.0254 [0.021, 0.030] | 0.0063 [0.001, 0.011] | 1.0808e-02 | 2.1616e-02 | 0.0137 | 0.00 | **FAIL** |
| `LGBM_Pooled_24h` | LightGBM | Pooled | 24h | 0.0007 [-0.004, 0.005] | 0.0151 [0.010, 0.020] | 1.2492e-09 | 7.4950e-09 | 0.0084 | 0.00 | **FAIL** |
| `HybridModel_BTC_4h` | HybridModel | BTC | 4h | -0.0063 [-0.021, 0.009] | -0.0121 [-0.028, 0.003] | 1.2251e-01 | 1.4701e-01 | -0.0024 | -13.09 | **FAIL** |
| `HybridModel_ETH_4h` | HybridModel | ETH | 4h | 0.0010 [-0.014, 0.016] | -0.0195 [-0.035, -0.004] | 1.3278e-02 | 2.2763e-02 | -0.0190 | -9.20 | **FAIL** |
| `Existing_Checkpoint_ETH_12h` | Checkpoint_A3 | ETH | 12h | 0.0000 [0.000, 0.000] | -0.0687 [-0.112, -0.025] | 1.9487e-03 | 5.8461e-03 | -0.0518 | -7.92 | **FAIL** |

### Feature Importances (Model B: LightGBM BTC 4H)
| Feature Rank | Feature Name | Description / Category | Split Count (Importance) |
| :-: | :--- | :--- | :-: |
| 1 | `rv_30d` | 30-day realized volatility from 5m mark price returns | 52 |
| 2 | `rv_7d` | 7-day realized volatility from 5m mark price returns | 36 |
| 3 | `z_global_ls` | Global long/short account ratio z-score | 34 |
| 4 | `ret_7d` | 7-day cumulative log return | 28 |
| 5 | `ret_24h` | 24-hour cumulative log return | 25 |
| 6 | `funding_mean_7d` | 7-day trailing mean funding rate | 24 |
| 7 | `z_toptrader_ls` | Top-trader account long/short ratio z-score | 23 |
| 8 | `oi_change_24h` | 24-hour percentage change in open interest | 22 |
| 9 | `rv_1d` | 24-hour realized volatility | 21 |
| 10 | `z_premium` | Spot-perp basis / premium index z-score | 15 |
| 11 | `day_of_week` | Day of week integer [0..6] | 15 |
| 12 | `hour_of_day` | Hour of day integer [0..23] | 11 |
| 13 | `ret_4h` | 4-hour cumulative log return | 10 |
| 14 | `oi_change_4h` | 4-hour percentage change in open interest | 10 |
| 15 | `ret_1h` | 1-hour causal log return | 8 |
| 16 | `oi_change_1h` | 1-hour percentage change in open interest | 6 |
| 17 | `vol_ratio_7d` | Volume relative to 7-day rolling mean | 5 |
| 18 | `vol_ratio_24h` | Volume relative to 24-hour rolling mean | 3 |
| 19 | `z_taker_ratio` | Taker buy/sell volume ratio z-score | 2 |

### Calibration Analysis (Out-of-Fold Deciles on Development Split)
| Decile | Mean Predicted Target | Mean Realized Target | Bar Count | Calibration Slope |
| :-: | :--- | :--- | :--- | :--- |
| Decile 0 | -0.0190 | +0.0010 | 880 | Inverted |
| Decile 1 | -0.0067 | +0.0059 | 885 | Inverted |
| Decile 2 | -0.0028 | +0.0014 | 879 | Inverted |
| Decile 3 | -0.0005 | -0.0010 | 871 | Calibrated |
| Decile 4 | +0.0011 | +0.0015 | 878 | Calibrated |
| Decile 5 | +0.0028 | +0.0022 | 879 | Calibrated |
| Decile 6 | +0.0049 | +0.0025 | 877 | Calibrated |
| Decile 7 | +0.0083 | +0.0039 | 878 | Calibrated |
| Decile 8 | +0.0151 | +0.0069 | 878 | Calibrated |
| Decile 9 | +0.0339 | +0.0035 | 879 | Flattened / Overconfident |

*Calibration Diagnostic:* Predicted returns display positive monotonic rank correlation across Deciles 3-8, but compress heavily in the tails. Extreme negative predictions (Deciles 0-2) inverted, generating positive actual forward returns, while extreme positive predictions (Decile 9) flattened, failing to deliver proportional alpha.

### Experimental Controls
- **Shuffled-Target Control (LightGBM):** OOS Dev IC = `-0.00500` (Pass: `True`)
- **Shuffled-Target Control (HybridModel):** OOS Dev IC = `0.02171` (Pass: `False`)
- **Lag-Shift Control (BTC 4H LightGBM):** Unlagged IC = `-0.00688` vs 1-bar Extra Lag IC = `-0.01001` (Pass: `True`)

#### Diagnostic Rationale for Shuffled-Target Control Failure in HybridModel
While tree-based LightGBM cleanly passed the shuffled-target control ($|IC| = 0.0050 < 0.01$), the deep sequence model (Clean HybridModel with 24-step CNN-BiLSTM) exhibited an OOS IC of `+0.02171`, violating the pre-registered threshold ($|IC| < 0.01$).
This divergence highlights a critical machine learning pathology in quantitative finance: deep recurrent architectures trained on sequential rolling windows with early stopping can memorize autocorrelated feature trajectories even when target labels are randomly permuted. When validation loss fluctuates stochastically on noise, early stopping selects weights that fit spurious temporal regimes. This confirms the necessity of simple baselines (Ridge / GBDT) and formal control suites to expose pseudo-signal.

---

## Standing Engine Controls Suite

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

### Limitations & Research Findings
1. **Signal-to-Noise Ratio at 1H/4H Horizons:** In highly liquid crypto perpetual futures (BTC and ETH), price returns at 1-hour and 4-hour frequencies behave nearly as martingales. Linear and non-linear predictive correlations ($IC$) remain below $\pm 0.02$, which is indistinguishable from zero under rigorous multiple testing controls.
2. **Microstructure Friction Dominance:** With standard trading friction of 10 bps per side (and 20 bps stress tests), any model with directional accuracy near 50-52% suffers catastrophic negative Sharpe ratios ($-9.0$ to $-40.0$) due to bid-ask spread and taker fees.
3. **In-Sample Leakage vs Reality:** The existing production model (`checkpoints/best_model.pt`) reported 53.6% accuracy in training documentation but completely collapsed to 45.28% directional accuracy and inverted IC ($-0.0687$) on clean out-of-sample data, confirming that reported edge was an artifact of pipeline boundary leakage and in-sample overfitting.
4. **Strict Research Discipline:** In accordance with pre-registered protocol, zero post-hoc rescue models, alternative architectures, or hyperparameter sweeps are permitted.

### Pre-Registered Pass Rules Verification
1. **OOS IC > 0.03 with 95% CI excluding zero in BOTH Development and Final splits:** Failed across all candidate models.
2. **Benjamini-Hochberg adjusted p < 0.05 on Final split:** Evaluated across the 12 candidate tests.
3. **Beats Ridge baseline IC in both splits:** Benchmarked against linear Ridge regression.
4. **Threshold strategy Sharpe > 0.7 and CAGR > 0 at double costs (20 bps):** Failed across models.
5. **Passes shuffled-target and lag-shift controls:** Controls verified (HybridModel failed shuffled-target control).

### Standing Verdict
**`NO ML PREDICTIVITY BEYOND BASELINE`**
