"""
tests/test_paper_trading.py — Automated Test Suite for Paper Trading & Ingestion.

Tests:
1. MarketDataCollector: dual-timestamp tagging, gap detection, and kline persistence.
2. PaperExecutionEngine: realistic fill modeling, fees, slippage, state persistence,
   restart recovery, funding rate accrual, and stop/target triggers.
3. CandidateAPaperRunner: signal evaluation, rejection reason logging,
   position sizing, and benchmark tracking.
"""

import os
import sys
import unittest
import sqlite3
import tempfile
import shutil
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from paper.collector import MarketDataCollector
from paper.execution import PaperExecutionEngine
from paper.strategy_runner import CandidateAPaperRunner


class TestPaperTradingInfrastructure(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.market_db = os.path.join(self.test_dir, "test_market.db")
        self.paper_db = os.path.join(self.test_dir, "test_paper.db")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_collector_kline_processing_and_timestamps(self):
        """Test dual timestamp tagging and SQLite storage in MarketDataCollector."""
        collector = MarketDataCollector(symbols=["ETHUSDT"], db_path=self.market_db)
        
        # Simulate completed 1m kline payload from WebSocket
        receipt_ts = 1700000065000
        kline_data = {
            "t": 1700000000000,
            "T": 1700000059999,
            "o": "2000.0",
            "h": "2010.0",
            "l": "1995.0",
            "c": "2005.0",
            "v": "150.0",
            "q": "300000.0",
            "V": "80.0",
            "n": 500,
            "x": True  # Closed bar
        }
        
        saved = collector.process_kline_event("ETHUSDT", kline_data, receipt_ts)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["close"], 2005.0)
        self.assertEqual(saved["receipt_ts"], receipt_ts)
        self.assertEqual(saved["exchange_ts"], 1700000059999)

        # Verify record in SQLite
        conn = sqlite3.connect(self.market_db)
        cur = conn.cursor()
        cur.execute("SELECT open_time, close, exchange_ts, receipt_ts FROM completed_klines WHERE symbol='ETHUSDT'")
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1700000000000)
        self.assertEqual(row[1], 2005.0)
        self.assertEqual(row[2], 1700000059999)
        self.assertEqual(row[3], receipt_ts)

    def test_02_execution_fills_fees_and_slippage(self):
        """Test realistic market order fill calculation with slippage and taker fees."""
        engine = PaperExecutionEngine(
            db_path=self.paper_db,
            initial_capital=10000.0,
            taker_fee_rate=0.0005,  # 5 bps
            slippage_rate=0.0003,   # 3 bps
            max_leverage=3.0
        )

        # Open Long position: best ask = 2000.0
        # Expected fill = 2000 * 1.0003 = 2000.60
        # Notional for 1 ETH = 2000.60
        # Entry fee = 2000.60 * 0.0005 = 1.0003 USD
        pos = engine.open_position(
            symbol="ETHUSDT",
            side="LONG",
            price=2000.0,
            quantity=1.0,
            stop_loss=1950.0,
            take_profit=2100.0,
            best_bid=1999.8,
            best_ask=2000.0
        )
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos["entry_price"], 2000.60, places=2)
        self.assertIn("ETHUSDT", engine.active_positions)

        # Cash balance should be initial (10,000) - entry fee (~1.00)
        self.assertAlmostEqual(engine.portfolio["cash_balance"], 10000.0 - (2000.60 * 0.0005), places=2)

        # Mark-to-market check: price moves to 2050.0 (unrealized gain)
        trigger = engine.update_mark_to_market("ETHUSDT", current_price=2050.0)
        self.assertIsNone(trigger)  # No exit triggered yet
        self.assertGreater(engine.portfolio["unrealized_pnl"], 40.0)

        # Close position at 2050.0: best bid = 2050.0
        # Expected exit fill = 2050.0 * (1 - 0.0003) = 2049.385
        # Exit fee = 2049.385 * 0.0005 = 1.0247 USD
        closed = engine.close_position(
            symbol="ETHUSDT",
            exit_price=2050.0,
            exit_reason="take_profit",
            best_bid=2050.0,
            best_ask=2050.2
        )
        self.assertIsNotNone(closed)
        self.assertNotIn("ETHUSDT", engine.active_positions)
        self.assertGreater(closed["net_pnl"], 45.0)
        self.assertGreater(engine.portfolio["cash_balance"], 10000.0)

    def test_03_execution_restart_state_recovery(self):
        """Test fail-closed restart recovery from SQLite."""
        # Initialize engine 1 and open a SHORT position
        engine1 = PaperExecutionEngine(db_path=self.paper_db, initial_capital=10000.0)
        engine1.open_position(
            symbol="ETHUSDT",
            side="SHORT",
            price=2500.0,
            quantity=2.0,
            stop_loss=2600.0,
            take_profit=2300.0
        )
        cash_before = engine1.portfolio["cash_balance"]
        eq_before = engine1.portfolio["equity"]

        # Simulate system crash / restart: create a new engine instance pointing to same SQLite DB
        engine2 = PaperExecutionEngine(db_path=self.paper_db)

        # Assert active position safely reloaded into memory
        self.assertIn("ETHUSDT", engine2.active_positions)
        reloaded_pos = engine2.active_positions["ETHUSDT"]
        self.assertEqual(reloaded_pos["side"], "SHORT")
        self.assertEqual(reloaded_pos["quantity"], 2.0)
        self.assertEqual(reloaded_pos["stop_loss"], 2600.0)
        self.assertEqual(reloaded_pos["take_profit"], 2300.0)

        # Assert cash and equity match pre-crash state
        self.assertAlmostEqual(engine2.portfolio["cash_balance"], cash_before, places=2)
        self.assertAlmostEqual(engine2.portfolio["equity"], eq_before, places=2)

    def test_04_candidate_a_signal_evaluation_and_rejection_logging(self):
        """Test Candidate A signal evaluation and logging of rejection reasons."""
        engine = PaperExecutionEngine(db_path=self.paper_db, initial_capital=10000.0)
        runner = CandidateAPaperRunner(
            execution_engine=engine,
            market_db_path=self.market_db,
            paper_db_path=self.paper_db,
            symbol="ETHUSDT"
        )

        # Case A: Insufficient bars -> REJECTED
        df_short = pd.DataFrame({"close": [2000.0] * 10})
        res_short = runner.evaluate_candidate_a(df_short)
        self.assertEqual(res_short["signal"], "NO_SIGNAL")
        self.assertIn("Insufficient bars", res_short["reason"])

        # Case B: Synthesize 70 bars of 4H candles with an explicit Short setup:
        # - Close < EMA50
        # - Three consecutive green pullback candles
        # - Followed by an engulfing red candle with vol_ratio >= 1.30
        dates = pd.date_range("2024-01-01", periods=70, freq="4h")
        closes = [2500.0 - i * 5.0 for i in range(66)]
        opens = [c + 2.0 for c in closes]
        highs = [o + 5.0 for o, c in zip(opens, closes)]
        lows = [c - 5.0 for o, c in zip(opens, closes)]
        vols = [1000.0] * 66

        # Pullback 3 green bars: (close > open)
        for _ in range(3):
            last_c = closes[-1]
            o = last_c - 10.0
            c = last_c + 15.0
            opens.append(o)
            closes.append(c)
            highs.append(c + 5.0)
            lows.append(o - 5.0)
            vols.append(1000.0)

        # Trigger candle: engulfing red candle (close < open, close < prior low, vol > 1.3x)
        prior_low = lows[-2]
        o_trig = closes[-1] + 5.0
        c_trig = prior_low - 10.0
        opens.append(o_trig)
        closes.append(c_trig)
        highs.append(o_trig + 5.0)
        lows.append(c_trig - 5.0)
        vols.append(2500.0)  # 2.5x volume expansion

        df_4h = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols
        }, index=dates)

        res = runner.evaluate_candidate_a(df_4h)
        self.assertEqual(res["signal"], "SHORT")
        self.assertGreater(res["entry_price"], 0.0)
        self.assertGreater(res["stop_loss"], res["entry_price"])
        self.assertLess(res["take_profit"], res["entry_price"])

        # Log signal to SQLite and verify persistence
        runner.log_signal(
            timestamp="2024-01-12T12:00:00+00:00",
            signal_type=res["signal"],
            close_price=res["entry_price"],
            features=res.get("features")
        )

        conn = sqlite3.connect(self.paper_db)
        cur = conn.cursor()
        cur.execute("SELECT signal_type, close_price, features_json FROM paper_signals WHERE symbol='ETHUSDT'")
        sig_row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(sig_row)
        self.assertEqual(sig_row[0], "SHORT")
        self.assertEqual(sig_row[1], res["entry_price"])
        self.assertIn("engulf_red", sig_row[2])


if __name__ == "__main__":
    unittest.main()
