# Final Quantitative Recommendation & Model Selection

## 1. Evaluation of Existing HybridModel
- **Retain or Deprecate:** **DEPRECATE FROM LIVE DEPLOYMENT; RETAIN ONLY AS BENCHMARK.**
- **Rationale:** Across 3,500 out-of-sample research bars with corrected decision alignment, the baseline HybridModel:
  - Produced 59 trades, all LONG (due to $+0.0335$ positive regression bias).
  - Achieved win rate of 47.46%, profit factor of 0.629, and net expectancy of **-25.33 bps**.
  - Generated an unlevered compounded return of **-14.27%** (-23.05% levered).
  - Failed to beat simple buy-and-hold or no-trade cash baselines after market friction (5 bps taker fee + 3 bps slippage + funding).
  - Deflated Sharpe Ratio (DSR): 0.0000 (no statistical confidence of genuine edge).

## 2. Multi-Year Strategy Research Findings (32,720 Bars / 3.7 Years)
- **Strategy A (Trend Donchian + 4h EMA + ADX):**
  - Performance: 4.7% net return, Expectancy: 3.68 bps.
- **Strategy B (Volatility Expansion):**
  - Performance: -68.45% net return, Expectancy: -19.37 bps.
- **Strategy C (Regime-Gated Mean Reversion):**
  - Performance: -28.17% net return, Expectancy: -28.87 bps.
- **Strategy D (Funding Crowding):**
  - Performance: -77.08% net return, Expectancy: -13.33 bps.
- **Strategy E (Pullback in Established Trend):**
  - Performance: -92.97% net return, Expectancy: -20.2 bps.
- **Strategy Combo (Rule-Based Regime Selector):**
  - Performance: -51.43% net return, Expectancy: -8.56 bps.

## 3. Best-Supported Strategy & Acceptance Status
- **Best Strategy Identified:** `Strategy_A_Trend_Donchian`
- **Acceptance Status:** **NO VALIDATED EDGE APPROVED FOR LIVE EXECUTION.**
- **Quantitative Rationale:**
  While `Strategy_A_Trend_Donchian` achieved superior risk-adjusted preservation compared to unconstrained momentum or deep neural network baselines, transaction friction in perpetual futures (10 bps round-trip taker fees + 6 bps round-trip slippage = 16 bps minimum round-trip hurdle) creates a severe barrier for 1h high-frequency trading. Under realistic taker fee conditions without free maker fills, no evaluated automated strategy achieves a positive 95% bootstrap lower bound for expectancy.

## 4. Remaining Risks & Recommended Next Experiments
1. **Fee Optimization:** Transitioning execution from aggressive market taker orders to calibrated post-only limit orders (maker fee: 0.02% or 0.00%) is strictly required to clear the transaction cost hurdle.
2. **Timeframe Expansion:** Hourly candles suffer from excessive noise relative to the 16 bps round-trip spread/fee hurdle. Testing 4h and daily swing strategies with wider ATR profit targets (e.g. 5x ATR) will dramatically reduce fee turnover relative to price gain.
3. **Execution Guardrails:** Live bot safety remains fail-closed and locked (`TRADING_ENABLED=False`, `MOCK_TRADING=True`).
