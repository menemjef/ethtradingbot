# Phase 2.1 & Attribution Artifact Regeneration Diff Report

**Repository:** `menemjef/ethtradingbot`  
**Branch:** `audit/backtest-engine-validation`  
**Baseline ("Before"):** Stored artifacts from commit `537c51c` / `6ecee0a` prior to re-running `run_phase_2_1_research.py` and `run_candidate_a_attribution.py`  
**Regenerated ("After"):** Stored artifacts regenerated with the validated engine (`seed=42`, identical dataset `eth_usdt_1h_3yr.parquet` SHA-256 `ffdbd16d...` and `eth_funding_3yr.parquet` SHA-256 `d741ec49...`, identical 5-fold definitions with 168h purge)

---

## 1. Pooled Walk-Forward OOS (Folds 1–4, `2023-09-08` to `2026-03-31`) — Before vs. After

| Candidate | Metric | Before | After | Status | Delta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Candidate A (`Exp1_Momentum_Continuation_4H`)** | Trade Count | `28` | `28` | `unchanged` | `0` |
| | Calendar CAGR (%) | `5.01%` | `5.01%` | `unchanged` | `0.00%` |
| | Daily / Hourly Sharpe | `0.769` / `0.835` (`0.77` / `0.83`) | `0.769` / `0.835` (`0.77` / `0.83`) | `unchanged` | `0.000` |
| | Intratrade Max Drawdown (%) | `4.74%` | `4.94%` | **`changed`** | `+0.20%` |
| | Profit Factor | `1.859` | `1.859` | `unchanged` | `0.000` |
| | Win Rate (%) | `53.57%` | `53.57%` | `unchanged` | `0.00%` |
| | Expectancy (bps) | `135.35 bps` (`135.3 bps`) | `135.35 bps` (`135.3 bps`) | `unchanged` | `0.00 bps` |
| | Expectancy 95% Bootstrap CI (bps) | `[29.88, 324.71]` | `[35.18, 314.88]` | **`changed`** | `[+5.30, -9.83] bps` |
| | Deflated Sharpe Ratio (DSR) | `0.0000` | `0.1777` | **`changed`** | `+0.1777` |
| **Candidate B (`Trend_Donchian_120h`)** | Trade Count | `54` | `54` | `unchanged` | `0` |
| | Calendar CAGR (%) | `0.04%` | `0.04%` | `unchanged` | `0.00%` |
| | Daily / Hourly Sharpe | `0.067` / `0.063` (`0.07` / `0.06`) | `0.067` / `0.063` (`0.07` / `0.06`) | `unchanged` | `0.000` |
| | Intratrade Max Drawdown (%) | `20.42%` | `20.62%` | **`changed`** | `+0.20%` |
| | Profit Factor | `1.004` | `1.004` | `unchanged` | `0.000` |
| | Win Rate (%) | `40.74%` | `40.74%` | `unchanged` | `0.00%` |
| | Expectancy (bps) | `1.01 bps` (`1.0 bps`) | `1.01 bps` (`1.0 bps`) | `unchanged` | `0.00 bps` |
| | Expectancy 95% Bootstrap CI (bps) | `[-158.13, 185.85]` | `[-143.77, 172.55]` | **`changed`** | `[+14.36, -13.30] bps` |
| | Deflated Sharpe Ratio (DSR) | `0.0000` | `0.0228` | **`changed`** | `+0.0228` |
| **Candidate C (`Trend_Donchian_168h`)** | Trade Count | `50` | `50` | `unchanged` | `0` |
| | Calendar CAGR (%) | `3.71%` | `3.71%` | `unchanged` | `0.00%` |
| | Daily / Hourly Sharpe | `0.342` / `0.365` (`0.34` / `0.36`) | `0.342` / `0.365` (`0.34` / `0.36`) | `unchanged` | `0.000` |
| | Intratrade Max Drawdown (%) | `23.94%` | `24.06%` | **`changed`** | `+0.12%` |
| | Profit Factor | `1.272` | `1.272` | `unchanged` | `0.000` |
| | Win Rate (%) | `48.00%` | `48.00%` | `unchanged` | `0.00%` |
| | Expectancy (bps) | `74.40 bps` (`74.4 bps`) | `74.40 bps` (`74.4 bps`) | `unchanged` | `0.00 bps` |
| | Expectancy 95% Bootstrap CI (bps) | `[-126.57, 296.39]` | `[-126.57, 296.39]` | `unchanged` | `[0.00, 0.00] bps` |
| | Deflated Sharpe Ratio (DSR) | `0.0000` | `0.0576` | **`changed`** | `+0.0576` |
| **Buy & Hold ETH (`Benchmark_BuyAndHold_ETH`)** | Trade Count | `0` | `0` | `unchanged` | `0` |
| | Calendar CAGR (%) | `9.74%` | `9.74%` | `unchanged` | `0.00%` |
| | Daily / Hourly Sharpe | `0.486` / `0.475` (`0.49` / `0.47`) | `0.486` / `0.475` (`0.49` / `0.47`) | `unchanged` | `0.000` |
| | Intratrade Max Drawdown (%) | `66.06%` | `66.06%` | `unchanged` | `0.00%` |
| | Profit Factor | `0.0` (`-`) | `0.0` (`-`) | `unchanged` | `0.000` |
| | Win Rate (%) | `0.0%` (`-`) | `0.0%` (`-`) | `unchanged` | `0.00%` |
| | Expectancy (bps) | `0.0 bps` (`-`) | `0.0 bps` (`-`) | `unchanged` | `0.00 bps` |
| | Expectancy 95% Bootstrap CI (bps) | `[0.0, 0.0]` (`-`) | `[0.0, 0.0]` (`-`) | `unchanged` | `[0.00, 0.00] bps` |
| | Deflated Sharpe Ratio (DSR) | `0.0000` | `0.0899` | **`changed`** | `+0.0899` |

---

## 2. Holdout Fold 5 (`2026-04-08` to `2026-11-04` — HOLDOUT ALREADY USED, NOT UNTOUCHED) — Before vs. After

| Candidate | Metric | Before | After | Status | Delta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Candidate A (`Exp1_Momentum_Continuation_4H`)** | Trade Count | `9` | `9` | `unchanged` | `0` |
| | Calendar CAGR (%) | `11.10%` | `11.10%` | `unchanged` | `0.00%` |
| | Daily / Hourly Sharpe | `1.407` / `1.177` (`1.41` / `1.18`) | `1.407` / `1.177` (`1.41` / `1.18`) | `unchanged` | `0.000` |
| | Intratrade Max Drawdown (%) | `3.09%` | `3.45%` | **`changed`** | `+0.36%` |
| | Profit Factor | `2.649` | `2.649` | `unchanged` | `0.000` |
| | Win Rate (%) | `66.67%` | `66.67%` | `unchanged` | `0.00%` |
| | Expectancy (bps) | `103.91 bps` (`103.9 bps`) | `103.91 bps` (`103.9 bps`) | `unchanged` | `0.00 bps` |
| | Expectancy 95% Bootstrap CI (bps) | `[-18.56, 176.74]` | `[-33.04, 150.89]` | **`changed`** | `[-14.48, -25.85] bps` |
| | Deflated Sharpe Ratio (DSR) | `0.0000` | `0.1111` | **`changed`** | `+0.1111` |
| **Candidate B (`Trend_Donchian_120h`)** | Trade Count | `10` | `10` | `unchanged` | `0` |
| | Calendar CAGR (%) | `-1.11%` | `-1.11%` | `unchanged` | `0.00%` |
| | Daily / Hourly Sharpe | `-0.041` / `-0.038` (`-0.04` / `-0.04`) | `-0.041` / `-0.038` (`-0.04` / `-0.04`) | `unchanged` | `0.000` |
| | Intratrade Max Drawdown (%) | `7.30%` | `7.46%` | **`changed`** | `+0.16%` |
| | Profit Factor | `1.203` | `1.203` | `unchanged` | `0.000` |
| | Win Rate (%) | `50.00%` | `50.00%` | `unchanged` | `0.00%` |
| | Expectancy (bps) | `36.73 bps` (`36.7 bps`) | `36.73 bps` (`36.7 bps`) | `unchanged` | `0.00 bps` |
| | Expectancy 95% Bootstrap CI (bps) | `[-79.20, 307.39]` | `[-79.20, 307.39]` | `unchanged` | `[0.00, 0.00] bps` |
| | Deflated Sharpe Ratio (DSR) | `0.0000` | `0.0168` | **`changed`** | `+0.0168` |
| **Candidate C (`Trend_Donchian_168h`)** | Trade Count | `9` | `9` | `unchanged` | `0` |
| | Calendar CAGR (%) | `-2.99%` | `-2.99%` | `unchanged` | `0.00%` |
| | Daily / Hourly Sharpe | `-0.191` / `-0.185` (`-0.19` / `-0.18`) | `-0.191` / `-0.185` (`-0.19` / `-0.18`) | `unchanged` | `0.000` |
| | Intratrade Max Drawdown (%) | `9.07%` | `9.22%` | **`changed`** | `+0.15%` |
| | Profit Factor | `0.958` | `0.958` | `unchanged` | `0.000` |
| | Win Rate (%) | `55.56%` | `55.56%` | `unchanged` | `0.00%` |
| | Expectancy (bps) | `-8.11 bps` (`-8.1 bps`) | `-8.11 bps` (`-8.1 bps`) | `unchanged` | `0.00 bps` |
| | Expectancy 95% Bootstrap CI (bps) | `[-167.62, 192.87]` | `[-203.16, 231.97]` | **`changed`** | `[-35.54, +39.10] bps` |
| | Deflated Sharpe Ratio (DSR) | `0.0000` | `0.0133` | **`changed`** | `+0.0133` |
| **Buy & Hold ETH (`Benchmark_BuyAndHold_ETH`)** | Trade Count | `0` | `0` | `unchanged` | `0` |
| | Calendar CAGR (%) | `14.90%` | `14.90%` | `unchanged` | `0.00%` |
| | Daily / Hourly Sharpe | `0.594` / `0.527` (`0.59` / `0.53`) | `0.594` / `0.527` (`0.59` / `0.53`) | `unchanged` | `0.000` |
| | Intratrade Max Drawdown (%) | `37.90%` | `37.90%` | `unchanged` | `0.00%` |
| | Profit Factor | `0.0` (`-`) | `0.0` (`-`) | `unchanged` | `0.000` |
| | Win Rate (%) | `0.0%` (`-`) | `0.0%` (`-`) | `unchanged` | `0.00%` |
| | Expectancy (bps) | `0.0 bps` (`-`) | `0.0 bps` (`-`) | `unchanged` | `0.00 bps` |
| | Expectancy 95% Bootstrap CI (bps) | `[0.0, 0.0]` (`-`) | `[0.0, 0.0]` (`-`) | `unchanged` | `[0.00, 0.00] bps` |
| | Deflated Sharpe Ratio (DSR) | `1.0000` | `0.6767` | **`changed`** | `-0.3233` |

---

## 3. Explanation of Every Change Larger Than Rounding

### 3.1 Intratrade Max Drawdown (`intratrade_max_drawdown_pct`)
- **What changed:**
  - Candidate A OOS MaxDD: `4.74%` $\rightarrow$ `4.94%` (`+0.20%`); Holdout Fold 5: `3.09%` $\rightarrow$ `3.45%` (`+0.36%`). Per-fold MaxDD: Fold 1 `4.74%` $\rightarrow$ `4.85%`, Fold 2 `4.69%` $\rightarrow$ `4.94%`, Fold 3 `1.14%` $\rightarrow$ `1.51%`, Fold 4 `3.69%` $\rightarrow$ `4.10%`.
  - Candidate B (`Trend_Donchian_120h`) OOS MaxDD: `20.42%` $\rightarrow$ `20.62%` (`+0.20%`); Holdout Fold 5: `7.30%` $\rightarrow$ `7.46%` (`+0.16%`).
  - Candidate C (`Trend_Donchian_168h`) OOS MaxDD: `23.94%` $\rightarrow$ `24.06%` (`+0.12%`); Holdout Fold 5: `9.07%` $\rightarrow$ `9.22%` (`+0.15%`).
- **Why it changed:**
  - `simulate_swing_trades_conservative()` computed `eq_lowest_fold` (marking open positions to the 1H candle's intrabar worst tick: `curr_low` for LONG and `curr_high` for SHORT), **but `run_phase_2_1_research.py` previously omitted `hourly_lowest_equity=eq_lowest_fold` when calling `compute_continuous_portfolio_metrics()`**. Consequently, the stored `intratrade_max_drawdown_pct` was previously computed on 1H bar-close equity rather than 1H intrabar worst-tick equity. Passing `hourly_lowest_equity` properly captures intrabar excursions before candle close.

### 3.2 Expectancy 95% Moving Block Bootstrap Confidence Interval (`ci_95_expectancy_bps`)
- **What changed:**
  - Candidate A OOS ($N=28$): `[29.88, 324.71] bps` $\rightarrow$ `[35.18, 314.88] bps` (Holdout $N=9$: `[-18.56, 176.74]` $\rightarrow$ `[-33.04, 150.89]`).
  - Candidate B OOS ($N=54$): `[-158.13, 185.85] bps` $\rightarrow$ `[-143.77, 172.55] bps` (Holdout $N=10$: **unchanged** at `[-79.20, 307.39]`).
  - Candidate C OOS ($N=50$): **unchanged** at `[-126.57, 296.39] bps` (Holdout $N=9$: `[-167.62, 192.87]` $\rightarrow$ `[-203.16, 231.97]`).
- **Why it changed:**
  - In commit `8ef7311` (Phase 2.2A), `compute_reconciled_trade_metrics()` in `utils/quant_evaluation.py` fixed the Moving Block Bootstrap block count from floor division `n_blocks = max(1, n_trades // block_size)` (with `block_size = 5`) to ceiling division `n_blocks = max(1, math.ceil(n_trades / eff_block))` followed by `b_idx[:n_trades]`. Under floor division, any trade count not divisible by `5` resampled fewer than $N$ trades per bootstrap draw (e.g., $28 // 5 = 5$ blocks $= 25$ trades for Candidate A OOS; $54 // 5 = 10$ blocks $= 50$ trades for Donchian 120h OOS; $9 // 5 = 1$ block $= 5$ trades for Holdout Fold 5), whereas exact multiples of `5` ($N=50$ for Donchian 168h OOS and $N=10$ for Donchian 120h Holdout) had $N // 5 == \lceil N / 5 \rceil$ and are therefore 100% identical before and after. Because `run_phase_2_1_research.py` had not been re-run after commit `8ef7311`, regenerating brings the stored JSON/Markdown artifacts in sync with `utils/quant_evaluation.py`.
  - No confidence interval changed sign (Candidate A OOS 95% CI remains strictly above zero `[+35.18, +314.88] bps`; Donchian 120h and 168h 95% CIs still span zero).

### 3.3 Deflated Sharpe Ratio (`deflated_sharpe_ratio`)
- **What changed:**
  - Candidate A OOS: `0.0000` $\rightarrow$ `0.1777` (Holdout: `0.0000` $\rightarrow$ `0.1111`).
  - Candidate B OOS: `0.0000` $\rightarrow$ `0.0228` (Holdout: `0.0000` $\rightarrow$ `0.0168`).
  - Candidate C OOS: `0.0000` $\rightarrow$ `0.0576` (Holdout: `0.0000` $\rightarrow$ `0.0133`).
  - Buy & Hold ETH OOS: `0.0000` $\rightarrow$ `0.0899` (Holdout with `n_trials=1`: `1.0000` $\rightarrow$ `0.6767`).
- **Why it changed:**
  - In commit `8ef7311` (Phase 2.2A), `compute_deflated_sharpe_ratio()` in `utils/quant_evaluation.py` was repaired to de-annualize `observed_sharpe` to the periodic return frequency (`sr_obs_periodic = observed_sharpe / sqrt(q)`) before computing the Bailey & López de Prado (2014) standard error `sr_std = sqrt((1 - skew * sr + ((kurt - 1)/4) * sr^2) / (T - 1))`, and `compute_continuous_portfolio_metrics()` updated `deflated_sharpe_ratio` to report `daily_dsr`. Previously in commit `537c51c`, annualized Sharpe was plugged directly into the periodic variance formula, inflating `sr_std` by $\sqrt{q}$ and collapsing DSR to `0.0000`. Regenerating `phase_2_1_fold_metrics.json` applies the `8ef7311` formula. All candidates remain far below the `DSR > 0.95` acceptance threshold (`0.1777`, `0.0228`, `0.0576`).

### 3.4 Secondary Artifact Updates (`volatility_transitions`, `cscv_pbo`, `candidate_a_attribution`)
1. **Pooled OOS `pnl_reconciliation_difference_usd`:** Now recorded as `0.0` across all strategy candidates after scaling pooled OOS trade PnL by each fold's compounded starting equity base.
2. **Rolling Volatility Transitions (`volatility_transitions`):** Excluding Holdout Fold 5 (`1,306` 4H bars after `2026-03-31 23:00 UTC`) reduces the sample size from `8,112` to `6,806` 4H bars (`928` $\rightarrow$ `742` compression bars) and shifts `p_expansion_given_compression` by `-1.1%` to `-1.8%` across horizons (e.g., 24h horizon `19.29%` $\rightarrow$ `17.52%`, edge `-10.96%` $\rightarrow$ `-12.95%`).
3. **CSCV PBO (`cscv_pbo`):** Changes from `0.5260` to `0.5100` (`51.00%`) because commit `8ef7311` updated `compute_probability_of_backtest_overfitting()` in `utils/quant_evaluation.py` to evaluate slice Sharpe ratios (`metric="sharpe"`) and `rank_best >= median_rank`.
4. **Candidate A Attribution (`research/candidate_a_attribution.json`, `research/candidate_a_attribution_report.md`, `research/candidate_a_trade_attribution.csv`):**
   - Previously pooled all `37` trades across Folds 1–5 (`include_holdout=True`, expectancy `+127.7 bps`, PF `1.95`, Win Rate `56.76%`, total PnL `+$1,912.97`).
   - Now restricted to Walk-Forward OOS Folds 1–4 (`include_holdout=False`, `28` trades, expectancy `+135.4 bps`, PF `1.86`, Win Rate `53.57%`, total PnL `+$1,286.76`), with Holdout Fold 5 (`9` trades, expectancy `+103.9 bps`, PF `2.65`, Win Rate `66.67%`, total PnL `+$626.21`) isolated in a separate section labeled `"HOLDOUT ALREADY USED, NOT UNTOUCHED"`.

---

## 4. Conclusion Impact Assessment

- **Did any candidate's overall verdict change?** **No.** All candidates remain `NO VALIDATED EDGE (DO NOT DEPLOY)` (`DSR = 0.1777 < 0.95` and `CAGR = 5.01%` for Candidate A; negative Fold 2 / Holdout performance and wide CIs spanning zero for Donchian 120h and 168h).
- **Did any confidence interval newly include or exclude zero?** **No.** Candidate A's Pooled OOS Moving Block Bootstrap 95% expectancy CI remains strictly positive (`[+29.88, +324.71] bps` $\rightarrow$ `[+35.18, +314.88] bps`), while Donchian 120h (`[-143.77, +172.55] bps`) and Donchian 168h (`[-126.57, +296.39] bps`) still span zero.
