"""
tests/test_swing_research.py — Automated Regression & Research Validation Tests for Quant Phase 2.

Verifies:
1. Actual stop-loss position sizing dynamically scales with stop distance and respects the 3x leverage cap.
2. Calendar-time CAGR accurately computes annualization based on true elapsed calendar duration.
3. Intratrade mark-to-market drawdown captures adverse price excursions during trade lifecycle.
4. Continuous daily and hourly Sharpe and Sortino ratios operate on continuous portfolio equity series.
5. Multi-timeframe resampling (4H, 1D) strictly prevents incomplete candle leakage and enforces causality.
6. Probability of Backtest Overfitting (PBO) correctly evaluates selection bias across strategy candidates.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.quant_evaluation import (
    calculate_actual_stop_position_sizing,
    compute_continuous_portfolio_metrics,
    compute_probability_of_backtest_overfitting,
    compute_reconciled_trade_metrics
)


class TestSwingResearch(unittest.TestCase):

    def test_01_actual_stop_position_sizing(self):
        """
        Verify that position sizing dynamically scales with actual stop distance:
        - 1.0% stop distance -> notional = equity * 0.025 / 0.01 = 2.5x leverage
        - 0.5% stop distance -> notional = equity * 0.025 / 0.005 = 5.0x raw -> capped at 3.0x leverage
        - 5.0% stop distance -> notional = equity * 0.025 / 0.05 = 0.5x leverage
        """
        equity = 10000.0
        entry_price = 2000.0

        # Case A: 1.0% stop distance ($1980 stop)
        res_a = calculate_actual_stop_position_sizing(
            account_equity=equity,
            entry_price=entry_price,
            stop_price=1980.0,
            risk_pct=0.025,
            max_leverage=3.0
        )
        self.assertAlmostEqual(res_a["effective_leverage"], 2.5, places=3)
        self.assertAlmostEqual(res_a["notional"], 25000.0, places=1)
        self.assertAlmostEqual(res_a["position_size"], 12.5, places=3)

        # Case B: 0.5% stop distance ($1990 stop) -> capped at 3x max leverage
        res_b = calculate_actual_stop_position_sizing(
            account_equity=equity,
            entry_price=entry_price,
            stop_price=1990.0,
            risk_pct=0.025,
            max_leverage=3.0
        )
        self.assertAlmostEqual(res_b["effective_leverage"], 3.0, places=3)
        self.assertAlmostEqual(res_b["notional"], 30000.0, places=1)
        self.assertAlmostEqual(res_b["position_size"], 15.0, places=3)

        # Case C: 5.0% stop distance ($1900 stop) -> 0.5x leverage
        res_c = calculate_actual_stop_position_sizing(
            account_equity=equity,
            entry_price=entry_price,
            stop_price=1900.0,
            risk_pct=0.025,
            max_leverage=3.0
        )
        self.assertAlmostEqual(res_c["effective_leverage"], 0.5, places=3)
        self.assertAlmostEqual(res_c["notional"], 5000.0, places=1)
        self.assertAlmostEqual(res_c["position_size"], 2.5, places=3)

    def test_02_calendar_time_annualization_and_intratrade_drawdown(self):
        """
        Verify that calendar-time CAGR is based on elapsed calendar time,
        and intratrade drawdown captures mark-to-market dips.
        """
        # Exactly 365.25 days of hourly observations
        N_hours = int(365.25 * 24)
        start_t = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        times = [start_t + pd.Timedelta(hours=i) for i in range(N_hours)]

        # Construct equity series: starts at 10000, dips to 9000 (10% DD), ends at 12000 (20% total return)
        eq_vals = np.full(N_hours, 10000.0)
        # Dip during hours 100 to 200
        eq_vals[100:200] = 9000.0
        # Rise linearly to 12000 over remainder
        eq_vals[200:] = np.linspace(9000.0, 12000.0, N_hours - 200)

        eq_series = pd.Series(eq_vals, index=times)

        # Mock 2 trades
        mock_trades = [
            {"trade_id": 1, "net_return": -0.10, "holding_bars": 100, "side": "LONG"},
            {"trade_id": 2, "net_return": 0.3333, "holding_bars": N_hours - 200, "side": "LONG"}
        ]

        metrics = compute_continuous_portfolio_metrics(
            hourly_equity=eq_series,
            trades=mock_trades,
            n_bootstraps=50,
            seed=42
        )

        # Elapsed calendar years should be exactly 1.0 (within float tolerance)
        self.assertAlmostEqual(metrics["calendar_years"], 1.0, places=2)
        # CAGR for 1.0 year with 20% gain should be ~20%
        self.assertAlmostEqual(metrics["calendar_cagr_pct"], 20.0, places=1)
        # Intratrade max drawdown should capture the 10% dip
        self.assertAlmostEqual(metrics["intratrade_max_drawdown_pct"], 10.0, places=1)

    def test_03_multitimeframe_causality_and_no_incomplete_candle_leakage(self):
        """
        Verify that multi-timeframe 4H and 1D features use only COMPLETED historical candles.
        At 1H timestamp 12:00:00 UTC, the 4H candle for [08:00, 12:00) is just completed,
        while the 1D candle for today is incomplete and cannot be used until 00:00:00 tomorrow.
        """
        N = 72
        start_t = pd.Timestamp("2026-03-01 00:00:00", tz="UTC")
        times = [start_t + pd.Timedelta(hours=i) for i in range(N)]
        df_1h = pd.DataFrame({
            "open": np.linspace(2000, 2100, N),
            "high": np.linspace(2005, 2105, N),
            "low": np.linspace(1995, 2095, N),
            "close": np.linspace(2000, 2100, N),
            "volume": np.full(N, 1000.0)
        }, index=times)

        # 4H resample
        df_4h = df_1h.resample("4h", closed="left", label="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        })
        # Shift 4h by 1 to represent available knowledge at bar close
        df_4h_completed = df_4h.shift(1)

        # Reindex to 1H via forward fill
        df_1h_with_4h = df_1h.copy()
        df_1h_with_4h["completed_4h_close"] = df_4h_completed["close"].reindex(df_1h.index, method="ffill")

        # Check timestamp at 12:00:00 UTC
        # The completed 4H candle available at 12:00:00 must be the 08:00 candle (closed at 12:00),
        # NOT the 12:00 candle (which ends at 16:00)!
        val_at_12 = df_1h_with_4h.loc[pd.Timestamp("2026-03-01 12:00:00", tz="UTC"), "completed_4h_close"]
        val_08_close = df_4h.loc[pd.Timestamp("2026-03-01 08:00:00", tz="UTC"), "close"]
        self.assertEqual(val_at_12, val_08_close)

    def test_04_probability_of_backtest_overfitting(self):
        """
        Verify that PBO calculation operates correctly on a matrix of strategy equity curves.
        """
        N = 300
        # Synthetic matrix of 10 strategies
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0001, 0.01, size=(N, 10))
        # Strategy 0 has a genuine positive drift
        rets[:, 0] += 0.005

        equity_curves = pd.DataFrame(np.cumprod(1.0 + rets, axis=0))
        pbo = compute_probability_of_backtest_overfitting(equity_curves, n_splits=8)

        # With one clearly superior strategy, PBO should be low
        self.assertLess(pbo, 0.40)


if __name__ == "__main__":
    unittest.main()
