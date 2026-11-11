# Pre-Registered Chart-Image CNN Predictivity Report

- **Generated Timestamp (UTC):** `2026-11-11 13:57:16 UTC`
- **Overall Verdict:** **NO IMAGE-MODEL PREDICTIVITY BEYOND NUMERIC BASELINE**
- **Test Verdict Breakdown (6 Tests):** `0 PASS`, `2 FAIL`, `4 INCONCLUSIVE`
- **Prior Final Split Exposure Disclosure:** The Final split (`2025-01-01` to `2026-10-31`) was previously seen by earlier pre-registered experiments in this repository (`vol_target_beta`, `ml_predictivity`, `sr_breakout`) as recorded in `data/run_log.jsonl`. For `chart_cnn`, it was read once through `run_logger.py` with zero post-evaluation modifications.

## 1. Stage 0: Pre-Flight Safety Checks

| Check ID | Check Name | Key Metric / Result | Threshold / Requirement | Status |
| :--- | :--- | :--- | :--- | :--- |
| **0.1** | Data Audit & Gap Filter | 10 coins audited; `1512` windows spanning gaps > 2 bars dropped; 0 forward-filled bars | Drop all windows spanning >2-bar gaps; never ffill | **PASS** |
| **0.2(a)** | Causal Bar Isolation | Max pixel diff when altering bars after $t$: `0.0e+00` | Bit-identical (`0.0`) | **PASS** |
| **0.2(b)** | Affine Scale/Shift Invariance | Max pixel diff under $7P + C$: `0.0e+00` | Identical (`0.0`) | **PASS** |
| **0.2(c)** | Zero Metadata/Price Leakage | Identical relative paths across coins/dates/prices max diff: `0.0e+00` | No price/date/coin recoverable | **PASS** |
| **0.2(d)** | Canonical Synthetic Shapes | Uptrend top-right (`2`), Flat midline (`26`), Crash bottom-right (`49`) | Expected pixel layout verified | **PASS** |
| **0.2(e)** | Blank/Constant Image Rate | Blank/constant rate across all rendered windows: `0.0000%` | $\le 1.0\%$ | **PASS** |
| **0.2(f)** | Sample Image Export | `12` PNG images written to `data/chart_cnn/sample_images/` | 12 sample images saved & listed | **PASS** |
| **0.3** | Next-Open Target & Purge/Embargo | Synthetic return diff `2.2e-16`; min purge gap `48h` ($H+24\text{h}$) | Target starts at $t+1$ open; purge $\ge H+24\text{h}$ | **PASS** |
| **0.4** | Planted-Signal Learnability | Out-of-sample Spearman IC = `0.7606` (best epoch `17`) | OOS IC $> 0.20$ | **PASS** |
| **0.5** | Model Capacity Overfitting (512 imgs) | Initial MSE `1.5878` $\rightarrow$ Final MSE `0.000007` | Overfit 512 random labels ($< 0.02$) | **PASS** |
| **0.6** | Seed Determinism | Same fold trained twice (seed 42) max prediction diff = `0.00e+00` | Max diff $\le 10^{-5}$ | **PASS** |
| **0.7** | Power & Cost Realism | `4/6` tests flagged `UNDERPOWERED` (Final MDE IC $> 0.08$) | Flag Final MDE IC $> 0.08$ up front | **PASS** |
| **0.8** | Compute Budget & Checkpointing | 1-fold benchmark `1.58s`; est. total `5.8m`; actual `76.0m` | Per-fold `.pt` & per-bar `.parquet` saved | **PASS** |

### Stage 0.1: Per-Coin 1H Data Audit & Dropped Windows

| Symbol | First Timestamp | 180d Entry Timestamp | Raw 1H Bars | Zero-Vol Removed | Gaps (>1H) | Gaps (>2 Bars) | Candidate 6H Windows | Dropped (Gap > 2 Bars) | Retained 6H Windows |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BTCUSDT` | `2019-09-08 17:00` | `2020-03-06 17:00` | 62777 | 3 | 9 | 7 | 10298 | 189 | 10109 |
| `ETHUSDT` | `2019-11-27 07:00` | `2020-05-25 07:00` | 60867 | 1 | 5 | 4 | 10004 | 108 | 9896 |
| `BNBUSDT` | `2020-02-10 08:00` | `2020-08-08 08:00` | 59066 | 0 | 7 | 7 | 9680 | 189 | 9491 |
| `SOLUSDT` | `2020-09-14 07:00` | `2021-03-13 07:00` | 53859 | 0 | 6 | 6 | 8816 | 162 | 8654 |
| `XRPUSDT` | `2020-01-06 08:00` | `2020-07-04 08:00` | 59906 | 0 | 6 | 6 | 9824 | 162 | 9662 |
| `ADAUSDT` | `2020-01-31 08:00` | `2020-07-29 08:00` | 59306 | 0 | 4 | 4 | 9744 | 108 | 9636 |
| `DOGEUSDT` | `2020-07-10 09:00` | `2021-01-06 09:00` | 55441 | 0 | 4 | 4 | 9100 | 108 | 8992 |
| `LINKUSDT` | `2020-01-17 08:00` | `2020-07-15 08:00` | 59642 | 0 | 7 | 7 | 9776 | 189 | 9587 |
| `LTCUSDT` | `2020-01-09 08:00` | `2020-07-07 08:00` | 59834 | 0 | 4 | 4 | 9832 | 108 | 9724 |
| `AVAXUSDT` | `2020-09-23 07:00` | `2021-03-22 07:00` | 53643 | 0 | 7 | 7 | 8776 | 189 | 8587 |

#### Detailed 1H Gap Windows Across All 10 Coins

- **`BTCUSDT`** (9 gaps): `2019-09-08 17:00 -> 2019-09-08 19:00` (1 missing bars), `2019-09-08 23:00 -> 2019-09-09 02:00` (2 missing bars), `2021-06-30 23:00 -> 2021-07-02 00:00` (24 missing bars), `2021-07-23 23:00 -> 2021-07-28 00:00` (96 missing bars), `2022-07-30 23:00 -> 2022-08-01 00:00` (24 missing bars), `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)
- **`ETHUSDT`** (5 gaps): `2019-11-27 08:00 -> 2019-11-27 10:00` (1 missing bars), `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)
- **`BNBUSDT`** (7 gaps): `2021-06-30 23:00 -> 2021-07-02 00:00` (24 missing bars), `2021-07-23 23:00 -> 2021-07-28 00:00` (96 missing bars), `2022-07-30 23:00 -> 2022-08-01 00:00` (24 missing bars), `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)
- **`SOLUSDT`** (6 gaps): `2021-06-30 23:00 -> 2021-07-02 00:00` (24 missing bars), `2021-07-23 23:00 -> 2021-07-28 00:00` (96 missing bars), `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)
- **`XRPUSDT`** (6 gaps): `2021-06-30 23:00 -> 2021-07-02 00:00` (24 missing bars), `2021-07-23 23:00 -> 2021-07-28 00:00` (96 missing bars), `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)
- **`ADAUSDT`** (4 gaps): `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)
- **`DOGEUSDT`** (4 gaps): `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)
- **`LINKUSDT`** (7 gaps): `2021-06-30 23:00 -> 2021-07-02 00:00` (24 missing bars), `2021-07-23 23:00 -> 2021-07-28 00:00` (96 missing bars), `2022-07-30 23:00 -> 2022-08-01 00:00` (24 missing bars), `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)
- **`LTCUSDT`** (4 gaps): `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)
- **`AVAXUSDT`** (7 gaps): `2021-06-30 23:00 -> 2021-07-02 00:00` (24 missing bars), `2021-07-23 23:00 -> 2021-07-28 00:00` (96 missing bars), `2022-07-30 23:00 -> 2022-08-01 00:00` (24 missing bars), `2022-10-01 23:00 -> 2022-10-03 00:00` (24 missing bars), `2023-02-23 23:00 -> 2023-02-25 00:00` (24 missing bars), `2026-06-28 23:00 -> 2026-06-30 00:00` (24 missing bars), `2026-10-27 23:00 -> 2026-10-30 00:00` (48 missing bars)

### Stage 0.2(f): 12 Sample Chart Images (Gitignored)

1. `data/chart_cnn/sample_images/sample_01_BTCUSDT_20191009_0000.png`
2. `data/chart_cnn/sample_images/sample_02_BTCUSDT_20200211_0000.png`
3. `data/chart_cnn/sample_images/sample_03_BTCUSDT_20201018_0000.png`
4. `data/chart_cnn/sample_images/sample_04_BTCUSDT_20211115_1200.png`
5. `data/chart_cnn/sample_images/sample_05_ETHUSDT_20191227_1200.png`
6. `data/chart_cnn/sample_images/sample_06_ETHUSDT_20200430_1200.png`
7. `data/chart_cnn/sample_images/sample_07_ETHUSDT_20210105_1200.png`
8. `data/chart_cnn/sample_images/sample_08_ETHUSDT_20220115_1200.png`
9. `data/chart_cnn/sample_images/sample_09_SOLUSDT_20201203_1200.png`
10. `data/chart_cnn/sample_images/sample_10_BNBUSDT_20200927_1200.png`
11. `data/chart_cnn/sample_images/sample_11_XRPUSDT_20201201_1200.png`
12. `data/chart_cnn/sample_images/sample_12_DOGEUSDT_20210913_1200.png`

## 2. Stage 0.7: Statistical Power & Cost Realism Table

- Empirical cross-coin $M_{\text{eff}}$ (market-neutral target): **24H** = `8.38` (raw return $M_{\text{eff}} = 1.91$); **72H** = `8.30` (raw return $M_{\text{eff}} = 1.96$).

| Test ID | Horizon | Overlap ($H/6$) | $M_{\text{eff}}$ | Dev $T$ | Dev $N_{\text{eff}}$ | Dev MDE IC (80%) | Dev Med $|r|$ (bps) | Final $T$ | Final $N_{\text{eff}}$ | Final MDE IC (80%) | Final Med $|r|$ (bps) | RT Cost (bps) | Power Status |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`BTCUSDT_24h`** | 24H | 4x | 1.00 | 4276 | 1069.0 | `0.0856` | 126.3 | 2620 | 655.0 | `0.1093` | 115.1 | 20.0 | **UNDERPOWERED** |
| **`ETHUSDT_24h`** | 24H | 4x | 1.00 | 4312 | 1078.0 | `0.0852` | 162.1 | 2620 | 655.0 | `0.1093` | 170.9 | 20.0 | **UNDERPOWERED** |
| **`POOLED_24h`** | 24H | 4x | 8.38 | 4312 | 9031.6 | `0.0295` | 198.4 | 2620 | 5487.6 | `0.0378` | 183.6 | 20.0 | **ADEQUATELY_POWERED** |
| **`BTCUSDT_72h`** | 72H | 12x | 1.00 | 4252 | 354.3 | `0.1484` | 244.1 | 2604 | 217.0 | `0.1892` | 206.3 | 20.0 | **UNDERPOWERED** |
| **`ETHUSDT_72h`** | 72H | 12x | 1.00 | 4296 | 358.0 | `0.1476` | 309.9 | 2604 | 217.0 | `0.1892` | 321.7 | 20.0 | **UNDERPOWERED** |
| **`POOLED_72h`** | 72H | 12x | 8.30 | 4296 | 2970.0 | `0.0514` | 359.2 | 2604 | 1800.3 | `0.0660` | 327.0 | 20.0 | **ADEQUATELY_POWERED** |

## 3. Walk-Forward Out-of-Sample Spearman IC Results (10-Day Moving-Block Bootstrap, 2,000 Resamples)

| Test ID | Split | $N_{\text{eff}}$ | CNN IC [95% Block CI] | CNN Raw $p$ (BH $p$) | Ridge Twin IC | LightGBM Twin IC | Momentum Ridge IC | Predict-Zero IC | Paired $\Delta$ (CNN - Best Numeric) [95% CI] | Paired $\Delta$ (CNN - Momentum) [95% CI] |
| :--- | :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| **`BTCUSDT_24h`** | Development | 1069.0 | `+0.0042` [`-0.0422`, `+0.0514`] | `0.8460` (N/A) | `-0.0236` | `-0.0225` | `-0.0129` | `+0.0000` | `+0.0267` [`-0.0286`, `+0.0779`] | `+0.0171` [`-0.0559`, `+0.0796`] |
| **`BTCUSDT_24h`** | Final | 655.0 | `-0.0020` [`-0.0682`, `+0.0480`] | `0.7770` (`0.8580`) | `+0.0042` | `+0.0349` | `-0.0160` | `+0.0000` | `-0.0369` [`-0.1023`, `+0.0177`] | `+0.0140` [`-0.0580`, `+0.0802`] |
| **`BTCUSDT_72h`** | Development | 354.3 | `+0.0455` [`-0.0270`, `+0.1125`] | `0.2350` (N/A) | `-0.0610` | `+0.0191` | `-0.0542` | `+0.0000` | `+0.0264` [`-0.0403`, `+0.0803`] | `+0.0997` [`+0.0024`, `+0.1907`] |
| **`BTCUSDT_72h`** | Final | 217.0 | `-0.0321` [`-0.1172`, `+0.0405`] | `0.3250` (`0.7720`) | `-0.0088` | `+0.0667` | `-0.0232` | `+0.0000` | `-0.0988` [`-0.1625`, `-0.0400`] | `-0.0089` [`-0.0969`, `+0.0772`] |
| **`ETHUSDT_24h`** | Development | 1078.0 | `+0.0094` [`-0.0346`, `+0.0583`] | `0.6360` (N/A) | `+0.0309` | `-0.0208` | `-0.0038` | `+0.0000` | `-0.0215` [`-0.0768`, `+0.0344`] | `+0.0132` [`-0.0597`, `+0.0786`] |
| **`ETHUSDT_24h`** | Final | 655.0 | `-0.0043` [`-0.0753`, `+0.0489`] | `0.6200` (`0.8580`) | `+0.0137` | `+0.0005` | `-0.0487` | `+0.0000` | `-0.0180` [`-0.0955`, `+0.0376`] | `+0.0444` [`-0.0269`, `+0.1071`] |
| **`ETHUSDT_72h`** | Development | 358.0 | `+0.0069` [`-0.0517`, `+0.0683`] | `0.8500` (N/A) | `+0.0066` | `-0.0006` | `-0.0810` | `+0.0000` | `+0.0003` [`-0.0749`, `+0.0727`] | `+0.0879` [`-0.0083`, `+0.1783`] |
| **`ETHUSDT_72h`** | Final | 217.0 | `-0.0279` [`-0.1224`, `+0.0504`] | `0.3860` (`0.7720`) | `-0.0033` | `-0.0201` | `-0.0851` | `+0.0000` | `-0.0246` [`-0.1086`, `+0.0455`] | `+0.0572` [`-0.0290`, `+0.1414`] |
| **`POOLED_24h`** | Development | 9031.6 | `+0.0105` [`-0.0038`, `+0.0251`] | `0.1500` (N/A) | `-0.0209` | `-0.0139` | `-0.0322` | `+0.0000` | `+0.0245` [`+0.0095`, `+0.0416`] | `+0.0428` [`+0.0267`, `+0.0611`] |
| **`POOLED_24h`** | Final | 5487.6 | `+0.0012` [`-0.0146`, `+0.0177`] | `0.8580` (`0.8580`) | `-0.0189` | `-0.0038` | `-0.0474` | `+0.0000` | `+0.0051` [`-0.0112`, `+0.0238`] | `+0.0487` [`+0.0293`, `+0.0689`] |
| **`POOLED_72h`** | Development | 2970.0 | `-0.0132` [`-0.0298`, `+0.0045`] | `0.1390` (N/A) | `-0.0258` | `-0.0177` | `-0.0515` | `+0.0000` | `+0.0045` [`-0.0142`, `+0.0251`] | `+0.0383` [`+0.0180`, `+0.0603`] |
| **`POOLED_72h`** | Final | 1800.3 | `+0.0276` [`+0.0074`, `+0.0534`] | `0.0100` (`0.0600`) | `-0.0217` | `+0.0138` | `-0.0412` | `+0.0000` | `+0.0138` [`-0.0071`, `+0.0385`] | `+0.0688` [`+0.0432`, `+0.0992`] |

## 4. Strategy Walk-Forward Backtest Metrics (10 bps/side + Funding vs. Double Costs 20 bps/side + 2x Funding)

| Test ID | Split | Cost Regime | CAGR | Sharpe | Max Drawdown | Turnover | Trades | Mean Daily Return (bps) [95% Block CI] |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`BTCUSDT_24h`** | Development | 1x (10 bps/side + fund) | `-35.33%` | `-1.73` | `-77.59%` | 888.5x | 1777 | `-11.16` [`-20.74`, `-2.23`] |
| **`BTCUSDT_24h`** | Development | 2x (20 bps/side + 2x fund) | `-52.31%` | `-3.00` | `-90.13%` | 888.5x | 1777 | `-19.47` [`-29.01`, `-10.45`] |
| **`BTCUSDT_24h`** | Final | 1x (10 bps/side + fund) | `-34.20%` | `-2.14` | `-55.25%` | 546.5x | 1093 | `-10.97` [`-18.97`, `-4.09`] |
| **`BTCUSDT_24h`** | Final | 2x (20 bps/side + 2x fund) | `-51.57%` | `-3.77` | `-73.55%` | 546.5x | 1093 | `-19.35` [`-27.36`, `-12.62`] |
| **`BTCUSDT_72h`** | Development | 1x (10 bps/side + fund) | `+3.36%` | `+0.29` | `-19.11%` | 298.3x | 1790 | `+1.24` [`-5.13`, `+7.63`] |
| **`BTCUSDT_72h`** | Development | 2x (20 bps/side + 2x fund) | `-7.01%` | `-0.39` | `-28.76%` | 298.3x | 1790 | `-1.66` [`-7.93`, `+4.63`] |
| **`BTCUSDT_72h`** | Final | 1x (10 bps/side + fund) | `-16.38%` | `-1.53` | `-30.35%` | 186.5x | 1119 | `-4.72` [`-10.20`, `+0.18`] |
| **`BTCUSDT_72h`** | Final | 2x (20 bps/side + 2x fund) | `-24.71%` | `-2.46` | `-41.95%` | 186.5x | 1119 | `-7.59` [`-12.94`, `-2.74`] |
| **`ETHUSDT_24h`** | Development | 1x (10 bps/side + fund) | `-31.56%` | `-1.07` | `-73.93%` | 914.5x | 1829 | `-9.07` [`-19.58`, `+2.12`] |
| **`ETHUSDT_24h`** | Development | 2x (20 bps/side + 2x fund) | `-49.94%` | `-2.07` | `-88.32%` | 914.5x | 1829 | `-17.61` [`-28.18`, `-6.53`] |
| **`ETHUSDT_24h`** | Final | 1x (10 bps/side + fund) | `-33.40%` | `-1.36` | `-62.41%` | 544.5x | 1089 | `-10.11` [`-24.81`, `+1.91`] |
| **`ETHUSDT_24h`** | Final | 2x (20 bps/side + 2x fund) | `-51.09%` | `-2.49` | `-77.65%` | 544.5x | 1089 | `-18.55` [`-33.05`, `-6.46`] |
| **`ETHUSDT_72h`** | Development | 1x (10 bps/side + fund) | `-14.68%` | `-0.77` | `-41.03%` | 296.2x | 1777 | `-3.88` [`-10.66`, `+3.08`] |
| **`ETHUSDT_72h`** | Development | 2x (20 bps/side + 2x fund) | `-22.83%` | `-1.31` | `-55.29%` | 296.2x | 1777 | `-6.63` [`-13.41`, `+0.42`] |
| **`ETHUSDT_72h`** | Final | 1x (10 bps/side + fund) | `-16.20%` | `-0.82` | `-34.43%` | 174.7x | 1048 | `-4.32` [`-14.50`, `+5.44`] |
| **`ETHUSDT_72h`** | Final | 2x (20 bps/side + 2x fund) | `-24.19%` | `-1.34` | `-42.79%` | 174.7x | 1048 | `-7.06` [`-17.13`, `+2.58`] |
| **`POOLED_24h`** | Development | 1x (10 bps/side + fund) | `-33.67%` | `-2.12` | `-74.62%` | 2156.0x | 17176 | `-10.76` [`-17.47`, `-3.90`] |
| **`POOLED_24h`** | Development | 2x (20 bps/side + 2x fund) | `-67.64%` | `-5.83` | `-96.48%` | 2156.0x | 17176 | `-30.35` [`-37.38`, `-23.47`] |
| **`POOLED_24h`** | Final | 1x (10 bps/side + fund) | `-53.35%` | `-6.37` | `-75.35%` | 1310.0x | 10480 | `-20.66` [`-25.65`, `-15.78`] |
| **`POOLED_24h`** | Final | 2x (20 bps/side + 2x fund) | `-77.60%` | `-12.57` | `-93.27%` | 1310.0x | 10480 | `-40.69` [`-45.70`, `-35.81`] |
| **`POOLED_72h`** | Development | 1x (10 bps/side + fund) | `-20.58%` | `-1.81` | `-51.06%` | 716.0x | 17096 | `-6.10` [`-10.80`, `-0.35`] |
| **`POOLED_72h`** | Development | 2x (20 bps/side + 2x fund) | `-36.94%` | `-3.66` | `-74.22%` | 716.0x | 17096 | `-12.41` [`-17.16`, `-6.72`] |
| **`POOLED_72h`** | Final | 1x (10 bps/side + fund) | `-14.13%` | `-1.94` | `-26.59%` | 434.0x | 10416 | `-4.09` [`-8.21`, `+0.88`] |
| **`POOLED_72h`** | Final | 2x (20 bps/side + 2x fund) | `-32.65%` | `-5.10` | `-50.78%` | 434.0x | 10416 | `-10.74` [`-14.85`, `-5.76`] |

## 5. Mandatory Controls Verification

| Control ID | Control Description | Measured Result Across 6 Tests | Pass Criterion | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Control 1** | Shuffled-Target Control | Max OOS $|\text{IC}|$ = `0.0015` | OOS $|\text{IC}| < 0.01$ | **PASS** |
| **Control 2** | 1-Bar Extra Lag-Shift Control | Max $(\text{IC}_{\text{lag1}} - \max(\text{IC}_{\text{cnn}}, 0))$ = `0.0000` | Shifting 1 extra bar does not improve positive IC | **PASS** |
| **Control 3** | Random-Image Noise Control | Max OOS $|\text{IC}|$ = `0.0017` | OOS $|\text{IC}| < 0.01$ | **PASS** |
| **Control 4** | Beats Momentum-Only Baseline | `2/6` tests beat Momentum in both splits with CI $> 0$ | Required for any test PASS | **ENFORCED** |
| **Control 5** | Standing Engine Controls (`run_engine_controls.py`) | `16` PASSED, `0` FAILED | 0 engine control failures | **PASS** |

### Per-Test Control Metrics (Full OOS)

| Test ID | Unshifted CNN OOS IC | Control 1: Shuffled-Target IC | Control 2: Lag-Shift (+1 Bar) IC | Control 3: Random-Image IC |
| :--- | ---: | ---: | ---: | ---: |
| **`BTCUSDT_24h`** | `+0.0012` | `+0.0010` | `+0.0009` | `-0.0017` |
| **`BTCUSDT_72h`** | `+0.0185` | `-0.0015` | `+0.0213` | `-0.0004` |
| **`ETHUSDT_24h`** | `+0.0038` | `+0.0000` | `-0.0020` | `-0.0013` |
| **`ETHUSDT_72h`** | `-0.0022` | `+0.0010` | `-0.0061` | `-0.0005` |
| **`POOLED_24h`** | `+0.0071` | `-0.0001` | `+0.0067` | `+0.0002` |
| **`POOLED_72h`** | `+0.0006` | `-0.0002` | `-0.0012` | `+0.0003` |

## 6. Pre-Registered Test Verdicts

| Test ID | Power Status (0.7) | (a) IC > 0.03 Both & Final CI > 0 | (b) BH Final $p < 0.05$ | (c) Beats Best Numeric & Mom (Paired CI > 0) | (d) 2x-Cost Sharpe > 0.7, CAGR > 0, Final Ret CI > 0 | (e) Controls Passed | Final Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`BTCUSDT_24h`** | `UNDERPOWERED` | FAIL | FAIL | FAIL | FAIL | PASS | **INCONCLUSIVE** |
| **`BTCUSDT_72h`** | `UNDERPOWERED` | FAIL | FAIL | FAIL | FAIL | PASS | **INCONCLUSIVE** |
| **`ETHUSDT_24h`** | `UNDERPOWERED` | FAIL | FAIL | FAIL | FAIL | PASS | **INCONCLUSIVE** |
| **`ETHUSDT_72h`** | `UNDERPOWERED` | FAIL | FAIL | FAIL | FAIL | PASS | **INCONCLUSIVE** |
| **`POOLED_24h`** | `ADEQUATELY_POWERED` | FAIL | FAIL | FAIL | FAIL | PASS | **FAIL** |
| **`POOLED_72h`** | `ADEQUATELY_POWERED` | FAIL | FAIL | FAIL | FAIL | PASS | **FAIL** |

### Overall Experiment Verdict: **NO IMAGE-MODEL PREDICTIVITY BEYOND NUMERIC BASELINE**

- **PASS:** `0`
- **FAIL (Adequately Powered):** `2` (POOLED_24h, POOLED_72h)
- **INCONCLUSIVE (Underpowered in Final Split, MDE IC > 0.08):** `4` (BTCUSDT_24h, BTCUSDT_72h, ETHUSDT_24h, ETHUSDT_72h)

## 7. Standing Engine Control Suite (`run_engine_controls.py`)

# Standing Engine Control Suite: Audit Report

**Date:** 2026-11-11 09:10:26 UTC  
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
