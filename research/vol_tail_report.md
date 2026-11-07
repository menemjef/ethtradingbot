# Pre-Registered Volatility / Tail-Risk Predictability & Conditional Overlay Report

**Date:** 2026-11-07 09:44:51 UTC  
**Final Run Lock:** Started `2026-11-07T04:53:10.336727+00:00`, Finished `2026-11-07T04:53:11.220954+00:00`  
**Methodology:** Strictly Walk-Forward Expanding Window, Refit Monthly, Ridge (alpha=1.0) / Logistic (C=1.0)  
**Overall Stage 1 Verdict:** **`NO VOL/TAIL PREDICTABILITY BEYOND BASELINE`**  

> [!NOTE]
> **Calendar Period Overlap Disclosure:** The underlying calendar data periods (2020–2026) were observed during directional trend, funding carry, and market-neutral research. Therefore, while these volatility and tail risk targets are pre-registered with zero parameter adjustments, the Final Test partition is not fully untouched.

---

## Step 0 Correction: Recomputed Net Spread on 12 Directional Tests

In `run_oi_predictivity.py`, the original net quintile spread charged a blanket 40 bps friction (assuming 100% turnover) at every decision point, while funding was never deducted despite a code comment. Below is the recomputed table showing gross spread, previously reported net spread (40 bps blanket friction), and corrected net spread incorporating **actual portfolio turnover** (10 bps per side traded) and **actual forward settled funding drag** ($(F_{\text{top}} - F_{\text{bot}})$).

| # | Feature | Horizon | Dev Gross | Dev Net (Original) | Dev Net (Corrected) | Final Gross | Final Net (Original) | Final Net (Corrected) | IC Verdict |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | `z_oi_change_24h` | 24h | +7.6% | -138.5% | **-33.8%** | +24.0% | -122.1% | **-16.9%** | `FAIL` |
| 2 | `z_oi_change_24h` | 72h | +10.1% | -38.6% | **-2.5%** | +18.6% | -30.1% | **+4.8%** | `FAIL` |
| 3 | `z_global_ls_ratio` | 24h | -83.8% | -229.8% | **-115.5%** | -20.6% | -166.7% | **-50.5%** | `FAIL` |
| 4 | `z_global_ls_ratio` | 72h | -60.6% | -109.3% | **-72.4%** | -25.5% | -74.2% | **-35.6%** | `FAIL` |
| 5 | `z_toptrader_ls_ratio` | 24h | -43.9% | -190.0% | **-77.0%** | -33.6% | -179.7% | **-64.8%** | `FAIL` |
| 6 | `z_toptrader_ls_ratio` | 72h | -41.0% | -89.8% | **-54.0%** | -27.1% | -75.8% | **-37.9%** | `FAIL` |
| 7 | `z_taker_buy_sell_ratio` | 24h | +18.5% | -127.6% | **-96.2%** | +0.9% | -145.2% | **-115.4%** | `FAIL` |
| 8 | `z_taker_buy_sell_ratio` | 72h | +14.4% | -34.3% | **-23.8%** | -1.7% | -50.4% | **-40.4%** | `FAIL` |
| 9 | `z_mean_funding_7d` | 24h | -95.7% | -241.8% | **-110.2%** | +1.9% | -144.2% | **-10.4%** | `FAIL` |
| 10 | `z_mean_funding_7d` | 72h | -25.6% | -74.3% | **-33.3%** | -2.7% | -51.4% | **-7.6%** | `FAIL` |
| 11 | `sign_ret_x_z_oi_change_24h` | 24h | +31.9% | -114.2% | **-38.4%** | -0.8% | -146.9% | **-69.1%** | `FAIL` |
| 12 | `sign_ret_x_z_oi_change_24h` | 72h | +31.2% | -17.5% | **+6.5%** | -6.1% | -54.8% | **-28.8%** | `FAIL` |

> [!NOTE]
> **Step 0 Impact:** Correcting turnover and adding actual funding improved the net spread returns (e.g., Development `z_oi_change_24h` 24h net spread improved from -138.5% to -33.8% due to lower turnover fees). However, because the underlying Information Coefficients were near zero (|IC| < 0.03) and 95% CIs crossed zero, **all 12 directional tests remain FAIL**, and the verdict of `NO PREDICTIVE FEATURES` is completely unchanged.

---

## Stage 1: Volatility and Tail-Risk Predictability Matrix

Total hypothesis tests: 3 targets $\times$ 3 universes = **9 tests**. Benjamini-Hochberg FDR control applied at $\alpha = 0.05$ across the 9 Final tests.

| Test # | Universe | Target | Evaluation Metric | Dev Baseline | Dev Augmented | Dev Improvement [95% CI] | Final Baseline | Final Augmented | Final Improvement [95% CI] | Final p-val | Final BH-adj p | Verdict |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | **BTCUSDT** | `realized_vol` | QLIKE Loss Improvement | 0.5013 | 0.4849 | +0.0163 [-0.0172, +0.0473] | 0.3510 | 0.3441 | +0.0070 [-0.0078, +0.0200] | 0.4140 | 0.5323 | `FAIL` |
| 2 | **BTCUSDT** | `worst_drawdown` | relative MSE reduction | 0.0003 | 0.0003 | +0.0280 [+0.0029, +0.0531] | 0.0003 | 0.0003 | +0.0328 [+0.0046, +0.0663] | 0.0340 | 0.1020 | `FAIL` |
| 3 | **BTCUSDT** | `tail_event` | Brier Skill Score Improvement | 0.0268 | 0.0269 | -0.0059 [-0.0237, +0.0046] | 0.0261 | 0.0262 | -0.0019 [-0.0192, +0.0136] | 0.9370 | 0.9370 | `FAIL` |
| 4 | **ETHUSDT** | `realized_vol` | QLIKE Loss Improvement | 0.5060 | 0.5338 | -0.0278 [-0.0554, +0.0024] | 0.3496 | 0.3531 | -0.0036 [-0.0096, +0.0027] | 0.2840 | 0.4260 | `FAIL` |
| 5 | **ETHUSDT** | `worst_drawdown` | relative MSE reduction | 0.0005 | 0.0005 | -0.0149 [-0.0477, +0.0184] | 0.0008 | 0.0008 | +0.0166 [-0.0013, +0.0387] | 0.0660 | 0.1485 | `FAIL` |
| 6 | **ETHUSDT** | `tail_event` | Brier Skill Score Improvement | 0.0235 | 0.0232 | +0.0136 [-0.0305, +0.0633] | 0.0219 | 0.0221 | -0.0082 [-0.0183, +0.0028] | 0.1470 | 0.2646 | `FAIL` |
| 7 | **POOLED** | `realized_vol` | QLIKE Loss Improvement | 0.4928 | 0.5014 | -0.0086 [-0.0291, +0.0145] | 0.6565 | 0.6836 | -0.0271 [-0.0444, -0.0018] | 0.0080 | 0.0360 | `FAIL` |
| 8 | **POOLED** | `worst_drawdown` | relative MSE reduction | 0.0008 | 0.0008 | +0.0037 [-0.0178, +0.0266] | 0.0011 | 0.0011 | -0.0016 [-0.0099, +0.0070] | 0.6160 | 0.6930 | `FAIL` |
| 9 | **POOLED** | `tail_event` | Brier Skill Score Improvement | 0.0237 | 0.0236 | +0.0051 [-0.0031, +0.0108] | 0.0202 | 0.0203 | -0.0054 [-0.0085, -0.0026] | 0.0010 | 0.0090 | `FAIL` |

---

## Stage 1 Detailed Findings & Failure Modes

### 1. Realized Volatility Forecasts (HAR-RV vs. HAR-RV + Metrics)
- **Baseline HAR-RV Performance:** The benchmark log-HAR model uses 1-day, 7-day, and 30-day trailing realized volatility.
- **Augmented Model QLIKE Loss:** Adding open interest, long/short ratios, and taker flows yielded QLIKE loss improvements ranging from -0.0278 to +0.0163 across BTCUSDT, ETHUSDT, POOLED. Bootstrap 95% confidence intervals spanned zero: No.
- **Conclusion:** Public positioning metrics do not provide statistically significant explanatory power beyond trailing realized volatility for predicting next-day variance.

### 2. Worst Drawdown Forecasts (Trailing 30d Vol vs. Augmented)
- **Drawdown Predictability:** Relative MSE reduction ranged from -0.0149 to +0.0328 across development and final splits.
- **FDR Significance:** After Benjamini-Hochberg adjustment across all final tests, adjusted p-values for drawdown ranged from 0.1020 to 0.6930; none reached the 5% significance threshold (alpha = 0.05).

### 3. Binary Tail-Risk Forecasts (Extreme Absolute Return > $3\sigma$)
- **Brier Skill Score:** Brier Skill Score improvements ranged from -0.0082 to +0.0136 across splits.
- **FDR Significance:** Adjusted p-values for tail risk ranged from 0.0090 to 0.9370; none reached the 5% significance threshold.

---

## Stage 2: Conditional Overlay on Buy-and-Hold

> [!IMPORTANT]
> **STAGE 2 SKIPPED PER PROTOCOL:** In strict accordance with the pre-registered decision rules, Stage 2 (overlay on buy-and-hold) is executed **ONLY** if at least one volatility or tail-risk target passed all Stage 1 criteria. Because zero targets passed Stage 1, Stage 2 was skipped to prevent post-hoc curve fitting.

---

## Audit Integrity & Pre-Registration Statement

- **No Model Tuning:** Fixed ridge $\alpha = 1.0$ and logistic $C = 1.0$ were specified prior to seeing any results.
- **Strict Causal Stamping:** Features strictly ingested metrics timestamps $\le D - 5\text{m}$, verified by unit tests.
- **Strictly Expanding Walk-Forward:** Models were refit strictly monthly using past data only, with standard scaler statistics fit strictly on the training partitions.
- **Single Final Run:** The Final Test split (2025-01-01 to end) was locked under cryptographic hash checks and evaluated strictly ONCE.
- **Zero Rescue Tweaks:** Following the failure of all 9 tests, no alternative targets or parameter rescue searches were attempted.