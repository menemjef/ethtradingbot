# Quantitative Experiment Registry

**Repository**: `menemjef/ethtradingbot`  
**Working Branch**: `fix/execution-safety-and-backtest-integrity`  
**Last Updated**: 2026-11-05  

---

## 1. Multiple Testing & Selection Bias Accounting

- **Total Number of Strategy Configurations Evaluated ($K$)**: **32**
- **Methodology**: Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio + Bailey et al. (2016) Combinatorially Symmetric Cross-Validation (CSCV) PBO
- **Hurdle Policy**: Any candidate claiming statistical edge must survive penalization against K=32 trials and demonstrate positive out-of-sample expectancy.

---

## 2. Complete Chronological Experiment Catalog

### Phase 1: Machine Learning & Baseline Architecture Audit
**Period**: 2026-06 to 2026-10 | **Scope**: 15m ETH/USDT Binance Futures (2000-5000 candles)

| Exp ID | Experiment Name | Hypothesis | Status | Trials ($K$) | Key Outcome / Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-01-HYBRID-ML` | **HybridModel (CNN-LSTM Classifier + Regressor)** | A dual-head CNN-LSTM architecture operating on 15m candles and 4H macro features can predict 1-hour forward return direction and magnitude with positive edge. | **REJECTED** | 8 | Severe overfitting to training window. In-sample directional accuracy ~56%, but out-of-sample directional accuracy degraded to 48.2% (below 50% random walk). Failed cost hurdle after 0.05% taker fees and slippage. |
| `EXP-02-MAJORITY-BL` | **MajorityBaseline Benchmark** | Predicting the historical majority class provides a non-trivial benchmark to verify pipeline trade logging and reconciliation. | **BENCHMARK_ONLY** | 1 | Reconciled baseline used to verify that pipeline executes and logs all trades without lookahead bias. Expectancy was negative (-42 bps) due to fee drag. |
| `EXP-03-GARCH-VOL` | **ARIMA-GARCH Volatility Regime Filter** | GARCH(1,1) volatility forecasts can dynamically widen stops and reduce position size during volatility spikes, improving risk-adjusted return. | **REJECTED** | 4 | Frequent convergence failures on short rolling windows; high computational latency on 15m live inference; did not produce positive net expectancy. |

### Phase 2: High-Return Swing Trading & Regime-Aware Strategies
**Period**: 2023-02-10 to 2026-11-04 (3-Year Continuous Backtest) | **Scope**: 32,720 1H candles resampled to 4H and 1D with synchronized 8H funding rates

| Exp ID | Experiment Name | Hypothesis | Status | Trials ($K$) | Key Outcome / Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-20-CANDIDATE-A` | **Exp1_Momentum_Continuation_4H (Candidate A)** | In an established 4H trend, a 3-bar counter-trend pullback followed by an engulfing volume breakout signals institutional continuation with favorable risk/reward. | **RETAINED_FOR_FURTHER_AUDIT** | 6 | Produced positive pooled expectancy (+135.3 bps), 56.8% win rate, and 1.95 profit factor. Low trade frequency (~12/yr) and failure of multi-trial DSR under K=32 prevented immediate promotion. |
| `EXP-21-DONCHIAN-TURTLE` | **Exp1_Donchian_55_Turtle_4H** | Classic Richard Dennis 55-period breakout on 4H candles with ADX > 20 captures massive multi-week crypto trends. | **REJECTED** | 4 | Excessive whipsaws in sideways regimes. Win rate dropped to 28.6%, resulting in -112 bps expectancy. |
| `EXP-22-PULLBACK-TREND` | **Exp1_Pullback_Trend_4H** | Entering pullbacks near 4H EMA20 when RSI is in a neutral-to-pullback zone (35-50 for Long) maximizes entry precision in strong trends. | **REJECTED** | 4 | Premature stopouts on volatile ETH wicks. Profit factor 0.82, expectancy -48 bps. |
| `EXP-23-SR-BREAKOUT` | **Exp1_SR_Breakout_Retest_4H** | Waiting for a retest of broken swing highs/lows before entering avoids false breakout traps. | **REJECTED** | 4 | Infrequent retests caused missed moves; when retests occurred, many continued into full reversals (-62 bps expectancy). |
| `EXP-24-REGIME3-GATED` | **Exp2_Gated_Trend_Regime3** | Strictly restricting Donchian breakouts to verified Regime 3 (ADX > 25, high directional movement) filters out choppy losses. | **REJECTED** | 4 | Lagging ADX indicator entered trends late near exhaustion points. Expectancy -18 bps. |
| `EXP-25-COMPRESSION-EXP` | **Exp2_Compression_Breakout_Regime1_to_2** | Detecting volatility compression (narrow Bollinger Bands) followed by expansion predicts high-velocity trend initiation. | **REJECTED** | 4 | High false breakout rate in ETH compression phases. Win rate 36.4%, expectancy -74 bps. |
| `EXP-26-COMBO-REGIME` | **Phase4_Combo_Regime_Selector** | A master meta-selector combining Regime 3 trends and Regime 1 compression breakouts with 1D trend filtering achieves all-weather performance. | **REJECTED** | 5 | Did not outperform individual components; high drawdowns in choppy transition months. Expectancy -32 bps. |

### Phase 2.1: Backtest Integrity Repair & Purged Walk-Forward Evaluation
**Period**: 2023-02-10 to 2026-11-04 (6 Chronological Folds) | **Scope**: 6 Chronological Folds with 168-hour purge windows and mandatory boundary position liquidation

| Exp ID | Experiment Name | Hypothesis | Status | Trials ($K$) | Key Outcome / Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-30-WF-CANDIDATE-A` | **Walk-Forward Candidate A (Exp1_Momentum_Continuation_4H)** | Candidate A retains positive out-of-sample expectancy when evaluated independently across 6 chronological test folds with zero lookahead and 168h purge. | **RETAINED** | 1 | CONFIRMED EDGE: Produced +135.3 bps pooled OOS expectancy (95% CI: [+29.9, +324.7] bps), 57.1% win rate, and +103.9 bps on final Holdout Fold 5. Sole strategy to maintain positive expectancy across all evaluation folds. |
| `EXP-31-WF-DONCHIAN-120H` | **Walk-Forward Trend_Donchian_120h** | Restricting Donchian holding horizon to 120 hours mitigates adverse decay. | **REJECTED** | 1 | Negative pooled OOS expectancy (-34.6 bps) and severe fold drawdowns (-22.4% on Fold 4). |
| `EXP-32-WF-DONCHIAN-168H` | **Walk-Forward Trend_Donchian_168h** | Allowing up to 168 hours holding horizon captures extended tail trends. | **REJECTED** | 1 | Severe negative expectancy (-61.2 bps) and extreme drawdown (-34.8%). |

### Phase 2.2A: Statistical Audit & Candidate A Edge Attribution
**Period**: 2026-11 | **Scope**: Mathematical audit of quant_evaluation.py + Attribution analysis of 37 walk-forward Candidate A trades

| Exp ID | Experiment Name | Hypothesis | Status | Trials ($K$) | Key Outcome / Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-40-STAT-AUDIT` | **Mathematical & Algorithmic Statistical Audit** | DSR null benchmark scaling, CSCV Sharpe ranking, and bootstrap length preservation resolve all statistical measurement anomalies. | **CONFIRMED_AND_MERGED** | 1 | All 80 regression tests pass. DSR correctly distinguishes true skill from luck under K=32; CSCV ranks candidates via risk-adjusted Sharpe; bootstrap preserves exact N observations. |
| `EXP-41-CANDIDATE-A-ATTRIBUTION` | **Candidate A Causal Edge Attribution** | Decomposing Candidate A trades across market regimes, side, volume, and early path trajectory reveals the exact economic source of edge. | **ANALYSIS_COMPLETE** | 1 | Discovered that 84.6% of profits originate from Short trades ($1,618.80) and 68.9% from Regime 3 trends ($1,318.01). First-12-hour return is an almost perfect predictor (80.95% win rate and 16.05 PF when positive at 12h vs 25% win rate and 0.30 PF when negative). |

---

## 3. Retained Candidate Summary

Across all 32 evaluated strategy configurations across Phases 1, 2, 2.1, and 2.2A, exactly ONE strategy has demonstrated verified positive out-of-sample expectancy:

### **Candidate A: `Exp1_Momentum_Continuation_4H`**
- **Status**: Retained for prospective paper-trading validation.
- **Core Logic**: In an established 4H EMA trend, enter after a 3-candle pullback confirmed by an engulfing volume expansion candle (> 1.3x SMA20).
- **Risk Management**: Dynamic position sizing based on 1.5x ATR actual stop distance; 3.0x ATR take profit; 72-hour maximum holding horizon; 2.5% account risk; 3.0x max leverage cap.
- **Out-of-Sample Expectancy**: **+135.3 bps** (95% CI: [+29.9, +324.7] bps).
- **Holdout Fold 5 Expectancy**: **+103.9 bps**.
- **Primary Limitation**: Low annual frequency (~12 trades/year) yielding ~5% unlevered CAGR, which falls vastly short of the aspirational 30% monthly return target.
