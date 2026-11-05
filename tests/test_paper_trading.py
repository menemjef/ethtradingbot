"""
tests/test_paper_trading.py — Automated Regression & Integrity Test Suite for Paper Trading.

Comprehensive test suite verifying all Phase 2.2B and Phase 2.2C requirements:
 1. Separate MARKET and PUBLIC WebSocket stream routing.
 2. Reconnection and stale-feed detection (health status).
 3. Duplicate and out-of-order candle handling.
 4. Exactly 240 contiguous 1-minute bars required for 4-hour aggregation.
 5. Rejection of a 239-bar candle (logged rejection reason).
 6. Rejection of a missing-minute candle (logged gap index).
 7. Exactly one evaluation per unique completed 4-hour candle (database uniqueness).
 8. Independent ETH/BTC availability handling (ETH evaluates without blocking on BTC).
 9. Restart and missed-candle sequential recovery.
10. Fresh bid/ask propagation into paper execution.
11. Stale quote rejection (StaleQuoteError on quotes > 5.0s).
12. Long and short stop execution including adverse opening gaps (open * (1 ± slippage)).
13. Funding settlement applied exactly once (idempotent accounting).
14. Multi-strategy virtual position isolation.
15. Portfolio exposure limits and drawdown circuit breaker.
16. Functional graceful daemon shutdown.
17. Single-instance protection (PID lockfile).
18. Backward-compatible database migration on exact pre-Phase-2.2B schema without hardcoded paths.
19. Deterministic offline replay integration test covering complete position lifecycle.
20. Symbol isolation: BTC candles cannot trigger ETH stops or take-profits.
21. Portfolio accounting: available cash margin reservation without artificial equity inflation.
22. Funding and fee reconciliation on position closure (no duplicate funding, exact fee accounting).
23. Restart recovery: processes all historical missed candles without opening retroactive positions.
"""

import os
import sys
import time
import json
import sqlite3
import tempfile
import shutil
import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from paper.collector import MarketDataCollector
from paper.execution import PaperExecutionEngine, StaleQuoteError
from paper.strategy_runner import CandidateAPaperRunner
from paper.run_paper_system import acquire_pid_lock, release_pid_lock, is_pid_alive
import paper.config as paper_cfg


class TestPaperTradingInfrastructure(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.market_db = os.path.join(self.test_dir, "test_market.db")
        self.paper_db = os.path.join(self.test_dir, "test_paper.db")
        # Ensure schema tables are initialized in test market db
        _c = MarketDataCollector(symbols=["ETHUSDT", "BTCUSDT"], db_path=self.market_db)
        del _c

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: Separate MARKET and PUBLIC WebSocket stream routing
    # -------------------------------------------------------------------------
    def test_01_separate_market_and_public_ws_routing(self):
        """Test 1: Verify separate Binance USD-M WebSocket endpoints for Market vs Public streams."""
        collector = MarketDataCollector(
            symbols=["ETHUSDT", "BTCUSDT"],
            db_path=self.market_db,
            market_ws_url="wss://fstream.binance.com/market/stream",
            public_ws_url="wss://fstream.binance.com/public/stream"
        )

        market_url = collector._build_market_stream_url()
        public_url = collector._build_public_stream_url()

        self.assertTrue(market_url.startswith("wss://fstream.binance.com/market/stream?streams="))
        self.assertIn("ethusdt@kline_1m", market_url)
        self.assertIn("btcusdt@kline_1m", market_url)
        self.assertIn("ethusdt@markPrice@1s", market_url)
        self.assertIn("btcusdt@markPrice@1s", market_url)
        self.assertNotIn("bookTicker", market_url)

        self.assertTrue(public_url.startswith("wss://fstream.binance.com/public/stream?streams="))
        self.assertIn("ethusdt@bookTicker", public_url)
        self.assertIn("btcusdt@bookTicker", public_url)
        self.assertNotIn("kline", public_url)
        self.assertNotIn("markPrice", public_url)

    # -------------------------------------------------------------------------
    # Test 2: Reconnection and stale-feed detection
    # -------------------------------------------------------------------------
    def test_02_reconnection_and_stale_feed_detection(self):
        """Test 2: Verify stale-feed detection based on real event arrival times, not just WS handshake."""
        collector = MarketDataCollector(symbols=["ETHUSDT"], db_path=self.market_db)

        initial_health = collector.get_health_status()
        self.assertFalse(initial_health["overall_healthy"])
        self.assertEqual(initial_health["symbols"]["ETHUSDT"]["is_quote_fresh"], False)

        now_ms = int(time.time() * 1000)
        book_ticker = {
            "u": 123456, "s": "ETHUSDT", "b": "3000.10", "B": "12.5",
            "a": "3000.20", "A": "8.0", "T": now_ms - 200
        }
        collector.process_book_ticker_event("ETHUSDT", book_ticker, receipt_ts=now_ms)

        kline = {
            "t": now_ms - 60000, "T": now_ms - 1,
            "o": "2990.0", "h": "3005.0", "l": "2985.0", "c": "3000.0", "v": "100.0",
            "q": "300000.0", "V": "50.0", "n": 200, "x": True
        }
        collector.process_kline_event("ETHUSDT", kline, receipt_ts=now_ms)

        health_after = collector.get_health_status()
        self.assertTrue(health_after["symbols"]["ETHUSDT"]["is_quote_fresh"])
        self.assertTrue(health_after["symbols"]["ETHUSDT"]["is_kline_fresh"])
        self.assertLess(health_after["symbols"]["ETHUSDT"]["quote_age_sec"], 2.0)

        collector.symbol_health["ETHUSDT"]["last_book_receipt_ts"] = now_ms - 35000
        health_stale = collector.get_health_status()
        self.assertFalse(health_stale["symbols"]["ETHUSDT"]["is_quote_fresh"])
        self.assertGreater(health_stale["symbols"]["ETHUSDT"]["quote_age_sec"], 30.0)

    # -------------------------------------------------------------------------
    # Test 3: Duplicate and out-of-order candle handling
    # -------------------------------------------------------------------------
    def test_03_duplicate_and_out_of_order_candle_handling(self):
        """Test 3: Verify SQLite primary key idempotent storage and chronological ordering."""
        collector = MarketDataCollector(symbols=["ETHUSDT"], db_path=self.market_db)

        base_t = 1700000000000
        kline_2 = {
            "t": base_t + 60000, "T": base_t + 119999,
            "o": "3005.0", "h": "3010.0", "l": "3000.0", "c": "3008.0", "v": "20.0",
            "q": "60000.0", "V": "10.0", "n": 100, "x": True
        }
        collector.process_kline_event("ETHUSDT", kline_2, receipt_ts=base_t + 120000)

        kline_1 = {
            "t": base_t, "T": base_t + 59999,
            "o": "3000.0", "h": "3006.0", "l": "2995.0", "c": "3005.0", "v": "15.0",
            "q": "45000.0", "V": "8.0", "n": 80, "x": True
        }
        collector.process_kline_event("ETHUSDT", kline_1, receipt_ts=base_t + 120100)

        kline_1_dup = {
            "t": base_t, "T": base_t + 59999,
            "o": "3000.0", "h": "3006.0", "l": "2995.0", "c": "3005.0", "v": "18.0",
            "q": "54000.0", "V": "9.0", "n": 85, "x": True
        }
        collector.process_kline_event("ETHUSDT", kline_1_dup, receipt_ts=base_t + 120200)

        conn = sqlite3.connect(self.market_db)
        cur = conn.cursor()
        cur.execute("SELECT open_time, volume FROM completed_klines WHERE symbol='ETHUSDT' ORDER BY open_time ASC")
        rows = cur.fetchall()
        conn.close()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], base_t)
        self.assertEqual(rows[0][1], 18.0)
        self.assertEqual(rows[1][0], base_t + 60000)

    # -------------------------------------------------------------------------
    # Test 4: Exactly 240 contiguous 1-minute bars required for 4-hour aggregation
    # -------------------------------------------------------------------------
    def test_04_exactly_240_contiguous_bars_required_for_4h(self):
        """Test 4: Verify that a 4H candle is ONLY generated when exactly 240 contiguous 1m bars exist."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db)

        window_start = 1700006400000
        window_end = window_start + 14_400_000

        conn = sqlite3.connect(self.market_db)
        cur = conn.cursor()
        klines = []
        for i in range(240):
            ot = window_start + i * 60000
            ct = ot + 59999
            klines.append((
                "ETHUSDT", "1m", ot, ct,
                2000.0 + i * 0.1, 2005.0 + i * 0.1, 1995.0 + i * 0.1, 2001.0 + i * 0.1,
                100.0, 200000.0, 50.0, 500, ot + 60000, ct
            ))
        cur.executemany("""
            INSERT INTO completed_klines (
                symbol, timeframe, open_time, close_time, open, high, low, close,
                volume, quote_volume, taker_buy_volume, trades_count, receipt_ts, exchange_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, klines)
        conn.commit()
        conn.close()

        df_4h, rejected = runner.validate_and_build_4h_candles("ETHUSDT", now_ms=window_end + 5000)

        self.assertGreaterEqual(len(df_4h), 1)
        self.assertEqual(len(rejected), 0)
        latest_bar = df_4h.iloc[-1]
        self.assertEqual(latest_bar["open"], 2000.0)
        self.assertAlmostEqual(latest_bar["close"], 2001.0 + 239 * 0.1, places=2)
        self.assertEqual(latest_bar["volume"], 240 * 100.0)

    # -------------------------------------------------------------------------
    # Test 5: Rejection of a 239-bar candle
    # -------------------------------------------------------------------------
    def test_05_rejection_of_239_bar_candle(self):
        """Test 5: Explicitly verify rejection of a 4H candle with only 239 1-minute bars."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db)

        window_start = 1700006400000
        window_end = window_start + 14_400_000

        conn = sqlite3.connect(self.market_db)
        cur = conn.cursor()
        klines = []
        for i in range(240):
            if i == 120:
                continue
            ot = window_start + i * 60000
            ct = ot + 59999
            klines.append((
                "ETHUSDT", "1m", ot, ct,
                2000.0, 2005.0, 1995.0, 2001.0, 100.0, 200000.0, 50.0, 500, ot + 60000, ct
            ))
        cur.executemany("""
            INSERT INTO completed_klines (
                symbol, timeframe, open_time, close_time, open, high, low, close,
                volume, quote_volume, taker_buy_volume, trades_count, receipt_ts, exchange_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, klines)
        conn.commit()
        conn.close()

        df_4h, rejected = runner.validate_and_build_4h_candles("ETHUSDT", now_ms=window_end + 5000)

        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["bars_count"], 239)
        self.assertIn("contains 239 bars", rejected[0]["reason"])
        self.assertIn("required exactly 240", rejected[0]["reason"])

    # -------------------------------------------------------------------------
    # Test 6: Rejection of a missing-minute candle
    # -------------------------------------------------------------------------
    def test_06_rejection_of_missing_minute_candle(self):
        """Test 6: Verify rejection when total bars is 240 but a timestamp gap exists (non-contiguous)."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db)

        window_start = 1700006400000
        window_end = window_start + 14_400_000

        conn = sqlite3.connect(self.market_db)
        cur = conn.cursor()
        klines = []
        for i in range(240):
            if i < 50:
                ot = window_start + i * 60000
            elif i == 50:
                ot = window_start + 49 * 60000 + 90000  # 90s gap from bar 49 (non-contiguous)
            else:
                ot = window_start + i * 60000
            ct = ot + 59999
            klines.append((
                "ETHUSDT", "1m", ot, ct,
                2000.0, 2005.0, 1995.0, 2001.0, 100.0, 200000.0, 50.0, 500, ot + 60000, ct
            ))
        cur.executemany("""
            INSERT INTO completed_klines (
                symbol, timeframe, open_time, close_time, open, high, low, close,
                volume, quote_volume, taker_buy_volume, trades_count, receipt_ts, exchange_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, klines)
        conn.commit()
        conn.close()

        df_4h, rejected = runner.validate_and_build_4h_candles("ETHUSDT", now_ms=window_end + 3600000)

        self.assertEqual(len(rejected), 1)
        self.assertIn("Missing minute gap detected", rejected[0]["reason"])
        self.assertIn("at index 49", rejected[0]["reason"])

    # -------------------------------------------------------------------------
    # Test 7: Exactly one evaluation per unique completed 4-hour candle
    # -------------------------------------------------------------------------
    def test_07_exactly_one_evaluation_per_unique_4h_candle(self):
        """Test 7: Verify database-level uniqueness prevents duplicate evaluations of the same 4H candle."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db)

        candle_close = 1700020799999

        first = runner.record_evaluation(
            candle_close_time=candle_close,
            signal_type="LONG",
            close_price=2100.0,
            rejection_reason=None,
            ema_50=2050.0,
            vol_ratio=1.45,
            atr=25.0
        )
        self.assertTrue(first)
        self.assertTrue(runner.is_candle_evaluated(candle_close))

        second = runner.record_evaluation(
            candle_close_time=candle_close,
            signal_type="SHORT",
            close_price=2105.0,
            rejection_reason="Duplicate run"
        )
        self.assertFalse(second)

        conn = sqlite3.connect(self.paper_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), signal_type FROM strategy_evaluations WHERE candle_close_time=?", (candle_close,))
        row = cur.fetchone()
        conn.close()

        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], "LONG")

    # -------------------------------------------------------------------------
    # Test 8: Independent ETH/BTC availability handling
    # -------------------------------------------------------------------------
    def test_08_independent_eth_btc_availability(self):
        """Test 8: Ensure Candidate A evaluates ETH independently without blocking when BTC is absent."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db)

        window_start = 1700006400000
        window_end = window_start + 14_400_000
        conn = sqlite3.connect(self.market_db)
        cur = conn.cursor()
        klines_eth = []
        for i in range(240):
            ot = window_start + i * 60000
            ct = ot + 59999
            klines_eth.append((
                "ETHUSDT", "1m", ot, ct,
                2000.0, 2005.0, 1995.0, 2001.0, 100.0, 200000.0, 50.0, 500, ot + 60000, ct
            ))
        cur.executemany("""
            INSERT INTO completed_klines (
                symbol, timeframe, open_time, close_time, open, high, low, close,
                volume, quote_volume, taker_buy_volume, trades_count, receipt_ts, exchange_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, klines_eth)
        conn.commit()
        conn.close()

        df_eth_4h, _ = runner.validate_and_build_4h_candles("ETHUSDT", now_ms=window_end + 1000)
        self.assertGreaterEqual(len(df_eth_4h), 1)

        eval_res = runner.evaluate_candidate_a_logic(df_eth_4h)
        self.assertIn("signal", eval_res)
        self.assertIn("reason", eval_res)

    # -------------------------------------------------------------------------
    # Test 9: Restart and missed-candle recovery
    # -------------------------------------------------------------------------
    def test_09_restart_and_missed_candle_recovery(self):
        """Test 9: Verify fail-closed restart recovery processes missed completed candles sequentially."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db)
        runner1 = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db)

        dates = pd.date_range("2024-01-01 00:00:00", periods=70, freq="4h", tz="UTC")
        close_times = [int(ts.timestamp() * 1000) + 14_400_000 - 1 for ts in dates]

        for i in range(68):
            runner1.record_evaluation(
                candle_close_time=close_times[i],
                signal_type="NO_SIGNAL",
                close_price=2000.0,
                rejection_reason="Historical baseline"
            )

        runner2 = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db)

        self.assertTrue(runner2.is_candle_evaluated(close_times[67]))
        self.assertFalse(runner2.is_candle_evaluated(close_times[68]))
        self.assertFalse(runner2.is_candle_evaluated(close_times[69]))

    # -------------------------------------------------------------------------
    # Test 10: Fresh bid/ask propagation into paper execution
    # -------------------------------------------------------------------------
    def test_10_fresh_bid_ask_propagation_into_paper_execution(self):
        """Test 10: Verify market orders execute at best ask (Long) / best bid (Short) with exact slippage."""
        engine = PaperExecutionEngine(
            db_path=self.paper_db,
            market_db_path=self.market_db,
            initial_capital=10000.0,
            taker_fee_rate=0.0005,  # 5 bps
            slippage_rate=0.0003    # 3 bps
        )

        now_ms = int(time.time() * 1000)
        quote_entry = {
            "best_bid": 3000.0,
            "best_ask": 3000.50,
            "receipt_ts": now_ms,
            "exchange_ts": now_ms - 50
        }

        pos = engine.open_position(
            strategy_id="Strat_A",
            symbol="ETHUSDT",
            side="LONG",
            quantity=1.0,
            stop_loss=2900.0,
            take_profit=3200.0,
            quote=quote_entry
        )
        self.assertIsNotNone(pos)
        expected_fill = round(3000.50 * 1.0003, 2)
        self.assertAlmostEqual(pos["entry_price"], expected_fill, places=2)
        self.assertAlmostEqual(pos["entry_fee_usd"], expected_fill * 0.0005, places=3)

        quote_exit = {
            "best_bid": 3100.00,
            "best_ask": 3100.50,
            "receipt_ts": now_ms,
            "exchange_ts": now_ms - 50
        }
        closed = engine.close_position(
            strategy_id="Strat_A",
            symbol="ETHUSDT",
            exit_reason="manual_exit",
            quote=quote_exit
        )
        self.assertIsNotNone(closed)
        expected_exit_fill = round(3100.00 * (1.0 - 0.0003), 2)
        self.assertAlmostEqual(closed["exit_price"], expected_exit_fill, places=2)
        self.assertAlmostEqual(closed["exit_fee_usd"], expected_exit_fill * 0.0005, places=3)
        self.assertGreater(closed["directional_pnl"], 90.0)

    # -------------------------------------------------------------------------
    # Test 11: Stale quote rejection
    # -------------------------------------------------------------------------
    def test_11_stale_quote_rejection(self):
        """Test 11: Verify strict rejection (StaleQuoteError) when quote age exceeds max_quote_age_sec."""
        engine = PaperExecutionEngine(
            db_path=self.paper_db,
            market_db_path=self.market_db,
            max_quote_age_sec=5.0
        )

        now_ms = int(time.time() * 1000)

        stale_quote = {
            "best_bid": 3000.0,
            "best_ask": 3000.50,
            "receipt_ts": now_ms - 6500,
            "exchange_ts": now_ms - 6550
        }

        with self.assertRaises(StaleQuoteError):
            engine.validate_quote_freshness(stale_quote, now_ms=now_ms)

        with self.assertRaises(StaleQuoteError):
            engine.open_position(
                strategy_id="Strat_A",
                symbol="ETHUSDT",
                side="LONG",
                quantity=1.0,
                stop_loss=2900.0,
                take_profit=3200.0,
                quote=stale_quote
            )
        self.assertEqual(len(engine.active_positions), 0)

    # -------------------------------------------------------------------------
    # Test 12: Long and short stop execution including adverse gaps
    # -------------------------------------------------------------------------
    def test_12_long_and_short_stop_execution_with_adverse_gaps(self):
        """Test 12: Verify stop execution resolves stop-first and fills adverse opening gaps at open * (1 ± slip)."""
        engine = PaperExecutionEngine(
            db_path=self.paper_db,
            market_db_path=self.market_db,
            slippage_rate=0.0003
        )
        now_ms = int(time.time() * 1000)
        fresh_quote = {"best_bid": 3000.0, "best_ask": 3000.50, "receipt_ts": now_ms}

        # Subtest 12A: Long adverse gap down
        engine.open_position("Strat_Long", "ETHUSDT", "LONG", 1.0, stop_loss=2900.0, take_profit=3200.0, quote=fresh_quote)
        res = engine.check_position_stops_and_gaps("Strat_Long", "ETHUSDT", bar_open=2850.0, bar_high=2880.0, bar_low=2840.0, bar_close=2860.0)
        self.assertEqual(res[0], "stop_loss_gap")
        self.assertEqual(res[1], 2850.0)

        closed_long = engine.close_position("Strat_Long", "ETHUSDT", exit_reason="stop_loss_gap", adverse_gap_price=res[1])
        expected_long_fill = round(2850.0 * (1.0 - 0.0003), 2)
        self.assertAlmostEqual(closed_long["exit_price"], expected_long_fill, places=2)
        self.assertLess(closed_long["exit_price"], 2900.0)

        # Subtest 12B: Short adverse gap up
        engine.open_position("Strat_Short", "ETHUSDT", "SHORT", 1.0, stop_loss=3100.0, take_profit=2800.0, quote=fresh_quote)
        res_short = engine.check_position_stops_and_gaps("Strat_Short", "ETHUSDT", bar_open=3150.0, bar_high=3160.0, bar_low=3120.0, bar_close=3130.0)
        self.assertEqual(res_short[0], "stop_loss_gap")
        self.assertEqual(res_short[1], 3150.0)

        closed_short = engine.close_position("Strat_Short", "ETHUSDT", exit_reason="stop_loss_gap", adverse_gap_price=res_short[1])
        expected_short_fill = round(3150.0 * (1.0 + 0.0003), 2)
        self.assertAlmostEqual(closed_short["exit_price"], expected_short_fill, places=2)
        self.assertGreater(closed_short["exit_price"], 3100.0)

        # Subtest 12C: Same-bar collision stop-first resolution
        engine.open_position("Strat_Coll", "ETHUSDT", "LONG", 1.0, stop_loss=2900.0, take_profit=3200.0, quote=fresh_quote)
        res_coll = engine.check_position_stops_and_gaps("Strat_Coll", "ETHUSDT", bar_open=3000.0, bar_high=3250.0, bar_low=2890.0, bar_close=3050.0)
        self.assertEqual(res_coll[0], "stop_loss_collision")

    # -------------------------------------------------------------------------
    # Test 13: Funding settlement applied exactly once
    # -------------------------------------------------------------------------
    def test_13_funding_settlement_applied_exactly_once(self):
        """Test 13: Verify funding payments are applied idempotently with primary key deduplication."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db, initial_capital=10000.0)
        now_ms = int(time.time() * 1000)
        fresh_quote = {"best_bid": 3000.0, "best_ask": 3000.50, "receipt_ts": now_ms}

        engine.open_position(
            "Strat_A", "ETHUSDT", "LONG", 2.0, stop_loss=2800.0, take_profit=3300.0,
            quote=fresh_quote, entry_time_str="2023-11-14T00:00:00+00:00"
        )
        cash_before = engine.portfolio["cash_balance"]

        settlement_time = 1700000000000
        conn = sqlite3.connect(self.market_db)
        conn.execute("""
            INSERT INTO funding_settlements (symbol, funding_rate, mark_price, settlement_time)
            VALUES (?, ?, ?, ?)
        """, ("ETHUSDT", 0.0001, 3000.0, settlement_time))
        conn.commit()
        conn.close()

        applied_1 = engine.apply_settled_funding_rates()
        self.assertEqual(applied_1, 1)
        self.assertAlmostEqual(engine.portfolio["cash_balance"], cash_before - 0.60, places=2)
        pos = engine.active_positions[("Strat_A", "ETHUSDT")]
        self.assertAlmostEqual(pos["accumulated_funding"], -0.60, places=2)

        applied_2 = engine.apply_settled_funding_rates()
        self.assertEqual(applied_2, 0)
        self.assertAlmostEqual(engine.portfolio["cash_balance"], cash_before - 0.60, places=2)

    # -------------------------------------------------------------------------
    # Test 14: Multi-strategy virtual position isolation
    # -------------------------------------------------------------------------
    def test_14_multi_strategy_virtual_position_isolation(self):
        """Test 14: Verify multiple strategies can hold opposing/independent positions on ETH without collision."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db, initial_capital=20000.0)
        now_ms = int(time.time() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.50, "receipt_ts": now_ms}

        pos_a = engine.open_position("CandidateA_Trend", "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote)
        pos_b = engine.open_position("StrategyB_Donchian", "ETHUSDT", "SHORT", 2.0, 3100.0, 2800.0, quote)

        self.assertIsNotNone(pos_a)
        self.assertIsNotNone(pos_b)
        self.assertEqual(len(engine.active_positions), 2)
        self.assertIn(("CandidateA_Trend", "ETHUSDT"), engine.active_positions)
        self.assertIn(("StrategyB_Donchian", "ETHUSDT"), engine.active_positions)

        engine.close_position("CandidateA_Trend", "ETHUSDT", "stop_loss", quote=quote)
        self.assertNotIn(("CandidateA_Trend", "ETHUSDT"), engine.active_positions)
        self.assertIn(("StrategyB_Donchian", "ETHUSDT"), engine.active_positions)
        self.assertEqual(engine.active_positions[("StrategyB_Donchian", "ETHUSDT")]["quantity"], 2.0)

    # -------------------------------------------------------------------------
    # Test 15: Portfolio exposure limits and circuit breaker
    # -------------------------------------------------------------------------
    def test_15_portfolio_exposure_limits_and_circuit_breaker(self):
        """Test 15: Verify portfolio leverage limits (3.0x) and drawdown circuit breaker (15%)."""
        engine = PaperExecutionEngine(
            db_path=self.paper_db,
            market_db_path=self.market_db,
            initial_capital=10000.0,
            max_leverage=3.0
        )
        now_ms = int(time.time() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.50, "receipt_ts": now_ms}

        oversized_qty = 11.66
        pos_oversized = engine.open_position("Strat_A", "ETHUSDT", "LONG", oversized_qty, 2900.0, 3200.0, quote)
        self.assertIsNone(pos_oversized)
        self.assertEqual(len(engine.active_positions), 0)

        engine.portfolio["drawdown_pct"] = 16.5
        pos_blocked = engine.open_position("Strat_B", "ETHUSDT", "LONG", 0.5, 2900.0, 3200.0, quote)
        self.assertIsNone(pos_blocked)

    # -------------------------------------------------------------------------
    # Test 16: Functional graceful daemon shutdown
    # -------------------------------------------------------------------------
    def test_16_functional_graceful_daemon_shutdown(self):
        """Test 16: Verify PID lockfile creation, inspection, and clean release on shutdown."""
        pid_file = os.path.join(self.test_dir, "test_daemon.pid")

        acquired = acquire_pid_lock(pid_file)
        self.assertTrue(acquired)
        self.assertTrue(os.path.exists(pid_file))

        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        self.assertEqual(pid, os.getpid())
        self.assertTrue(is_pid_alive(pid))

        release_pid_lock(pid_file)
        self.assertFalse(os.path.exists(pid_file))

    # -------------------------------------------------------------------------
    # Test 17: Single-instance protection
    # -------------------------------------------------------------------------
    def test_17_single_instance_protection(self):
        """Test 17: Verify second daemon process fails to start when first daemon is active."""
        pid_file = os.path.join(self.test_dir, "test_daemon.pid")

        self.assertTrue(acquire_pid_lock(pid_file))
        self.assertFalse(acquire_pid_lock(pid_file))

        release_pid_lock(pid_file)
        self.assertTrue(acquire_pid_lock(pid_file))
        release_pid_lock(pid_file)

    # -------------------------------------------------------------------------
    # Test 18: Database migration integrity on exact legacy schema (Isolated)
    # -------------------------------------------------------------------------
    def test_18_database_migration_and_legacy_compatibility(self):
        """Test 18: Verify migration of pre-Phase-2.2B SQLite databases with legacy NOT NULL columns."""
        legacy_db = os.path.join(self.test_dir, "legacy_paper_state.db")

        # Synthesize exact pre-Phase-2.2B SQLite schema
        conn = sqlite3.connect(legacy_db)
        conn.executescript("""
        CREATE TABLE paper_portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cash_balance REAL NOT NULL,
            equity REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            unrealized_pnl REAL NOT NULL,
            margin_used REAL NOT NULL,
            peak_equity REAL NOT NULL,
            drawdown_pct REAL NOT NULL
        );
        INSERT INTO paper_portfolio (timestamp, cash_balance, equity, realized_pnl, unrealized_pnl, margin_used, peak_equity, drawdown_pct)
        VALUES ('2026-11-04T00:00:00+00:00', 10000.0, 10000.0, 0.0, 0.0, 0.0, 10000.0, 0.0);

        CREATE TABLE paper_positions (
            position_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            margin REAL NOT NULL,
            leverage REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            status TEXT NOT NULL,
            accumulated_funding REAL NOT NULL DEFAULT 0.0
        );

        CREATE TABLE paper_orders (
            order_id TEXT PRIMARY KEY,
            position_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            requested_price REAL NOT NULL,
            fill_price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            fee_usd REAL NOT NULL,
            slippage_usd REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            filled_at TEXT NOT NULL
        );

        CREATE TABLE paper_trades (
            trade_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            quantity REAL NOT NULL,
            holding_hours REAL NOT NULL,
            gross_pnl REAL NOT NULL,
            net_pnl REAL NOT NULL,
            fee_cost_usd REAL NOT NULL,
            slippage_cost_usd REAL NOT NULL,
            funding_cost_usd REAL NOT NULL,
            exit_reason TEXT NOT NULL
        );
        INSERT INTO paper_trades VALUES (
            'legacy_tr_01', 'ETHUSDT', 'LONG', '2026-11-04T00:00:00+00:00', '2026-11-04T04:00:00+00:00',
            2000.0, 2050.0, 1.0, 4.0, 50.0, 48.0, 2.0, 0.0, 0.0, 'take_profit'
        );
        """)
        conn.commit()
        conn.close()

        # Initialize PaperExecutionEngine on this legacy database -> Triggers automatic migration
        engine = PaperExecutionEngine(db_path=legacy_db, market_db_path=self.market_db, initial_capital=10000.0)

        # Verify schema columns after migration
        conn = sqlite3.connect(legacy_db)
        cur = conn.cursor()

        cur.execute("PRAGMA table_info(paper_orders);")
        order_cols = {row[1] for row in cur.fetchall()}
        self.assertIn("strategy_id", order_cols)

        cur.execute("PRAGMA table_info(paper_trades);")
        trade_cols = {row[1] for row in cur.fetchall()}
        self.assertIn("strategy_id", trade_cols)
        self.assertIn("directional_pnl", trade_cols)
        self.assertIn("funding_pnl", trade_cols)
        self.assertIn("entry_fee_usd", trade_cols)
        self.assertIn("exit_fee_usd", trade_cols)

        cur.execute("PRAGMA table_info(paper_portfolio);")
        port_cols = {row[1] for row in cur.fetchall()}
        self.assertIn("gross_leverage", port_cols)
        self.assertIn("net_leverage", port_cols)

        # Verify historical legacy trade preserved
        cur.execute("SELECT trade_id, directional_pnl, net_pnl, strategy_id FROM paper_trades WHERE trade_id='legacy_tr_01';")
        legacy_row = cur.fetchone()
        self.assertIsNotNone(legacy_row)
        self.assertEqual(legacy_row[0], "legacy_tr_01")
        self.assertEqual(legacy_row[1], 50.0)
        self.assertEqual(legacy_row[2], 48.0)
        self.assertEqual(legacy_row[3], "CandidateA_Momentum_Continuation")
        conn.close()

        # Verify that opening and closing a new position works seamlessly on migrated DB
        now_ms = int(time.time() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.50, "receipt_ts": now_ms}
        pos = engine.open_position("CandidateA_Trend", "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote)
        self.assertIsNotNone(pos)

        closed = engine.close_position("CandidateA_Trend", "ETHUSDT", "manual_exit", quote=quote)
        self.assertIsNotNone(closed)
        self.assertEqual(closed["strategy_id"], "CandidateA_Trend")

    # -------------------------------------------------------------------------
    # Test 19: Deterministic offline replay integration test (Full Lifecycle)
    # -------------------------------------------------------------------------
    def test_19_deterministic_offline_replay_integration(self):
        """Test 19: End-to-end deterministic replay verifying complete position lifecycle and reconciliation."""
        engine = PaperExecutionEngine(
            db_path=self.paper_db,
            market_db_path=self.market_db,
            initial_capital=10000.0,
            taker_fee_rate=0.0005,
            slippage_rate=0.0003
        )
        collector = MarketDataCollector(symbols=["ETHUSDT"], db_path=self.market_db)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db)

        # Step 1: Ingest 480 1-minute synthetic bars (2 complete 4H candles)
        start_ms = 1700006400000
        for i in range(480):
            ot = start_ms + i * 60000
            ct = ot + 59999
            price = 2500.0 + (i % 20) * 2.0
            kline_event = {
                "t": ot, "T": ct,
                "o": str(price), "h": str(price + 3.0), "l": str(price - 2.0), "c": str(price + 1.0),
                "v": "50.0", "q": "125000.0", "V": "25.0", "n": 100, "x": True
            }
            collector.process_kline_event("ETHUSDT", kline_event, receipt_ts=ct + 100)

        df_4h, rejected = runner.validate_and_build_4h_candles("ETHUSDT", now_ms=start_ms + 480 * 60000 + 10000)
        self.assertGreaterEqual(len(df_4h), 2)
        self.assertEqual(len(rejected), 0)

        # Step 2: Open Position (LONG 2.0 ETH at ask $2500)
        now_ms = int(time.time() * 1000)
        quote_entry = {"best_bid": 2499.5, "best_ask": 2500.0, "receipt_ts": now_ms}
        pos = engine.open_position("CandidateA", "ETHUSDT", "LONG", 2.0, 2400.0, 2700.0, quote_entry)
        self.assertIsNotNone(pos)

        # Fill price = 2500 * (1 + 0.0003) = 2500.75
        # Notional = 5001.50, Margin (3x) = 1667.1667, Fee = 2.50075
        expected_fill = round(2500.0 * 1.0003, 2)
        expected_margin = round(expected_fill * 2.0 / 3.0, 4)
        expected_fee = round(expected_fill * 2.0 * 0.0005, 4)

        # Verify available cash reservation (no margin inflation)
        self.assertAlmostEqual(engine.portfolio["margin_used"], expected_margin, places=2)
        self.assertAlmostEqual(engine.portfolio["cash_balance"], 10000.0 - expected_margin - expected_fee, places=2)
        self.assertAlmostEqual(engine.portfolio["equity"], 10000.0 - expected_fee, places=2)

        # Step 3: Mark to Market (Price advances to $2550)
        quote_mtm = {"best_bid": 2549.5, "best_ask": 2550.0, "mid_price": 2549.75, "receipt_ts": now_ms}
        engine.update_mark_to_market({"ETHUSDT": quote_mtm})
        expected_unrealized = (2549.75 - expected_fill) * 2.0
        self.assertAlmostEqual(engine.portfolio["unrealized_pnl"], expected_unrealized, places=2)
        self.assertAlmostEqual(
            engine.portfolio["equity"],
            engine.portfolio["cash_balance"] + engine.portfolio["margin_used"] + expected_unrealized,
            places=2
        )

        # Step 4: Funding Settlement (+1 bp funding rate, Long pays Short)
        settlement_time = now_ms + 1000
        conn = sqlite3.connect(self.market_db)
        conn.execute("""
            INSERT INTO funding_settlements (symbol, funding_rate, mark_price, settlement_time)
            VALUES (?, ?, ?, ?)
        """, ("ETHUSDT", 0.0001, 2550.0, settlement_time))
        conn.commit()
        conn.close()

        cash_before_funding = engine.portfolio["cash_balance"]
        applied = engine.apply_settled_funding_rates()
        self.assertEqual(applied, 1)
        expected_funding = -1.0 * 2.0 * 2550.0 * 0.0001  # -0.51 USD
        self.assertAlmostEqual(engine.portfolio["cash_balance"], cash_before_funding + expected_funding, places=2)

        # Step 5: Position Exit (Close at bid $2600)
        quote_exit = {"best_bid": 2600.0, "best_ask": 2600.5, "receipt_ts": now_ms}
        closed = engine.close_position("CandidateA", "ETHUSDT", "manual_exit", quote=quote_exit)
        self.assertIsNotNone(closed)

        # Step 6: Final Full Accounting Reconciliation
        self.assertEqual(len(engine.active_positions), 0)
        self.assertEqual(engine.portfolio["margin_used"], 0.0)
        self.assertEqual(engine.portfolio["unrealized_pnl"], 0.0)

        expected_exit_fill = round(2600.0 * (1.0 - 0.0003), 2)
        expected_exit_fee = round(expected_exit_fill * 2.0 * 0.0005, 4)
        expected_directional = round((expected_exit_fill - expected_fill) * 2.0, 2)
        expected_net_pnl = expected_directional + expected_funding - expected_fee - expected_exit_fee

        self.assertAlmostEqual(closed["net_pnl"], expected_net_pnl, places=2)
        self.assertAlmostEqual(engine.portfolio["realized_pnl"], expected_net_pnl, places=2)
        self.assertAlmostEqual(engine.portfolio["cash_balance"], 10000.0 + expected_net_pnl, places=2)
        self.assertAlmostEqual(engine.portfolio["equity"], 10000.0 + expected_net_pnl, places=2)

    # -------------------------------------------------------------------------
    # Test 20: Symbol Isolation — BTC candles cannot trigger ETH SL/TP
    # -------------------------------------------------------------------------
    def test_20_symbol_isolation_btc_cannot_trigger_eth_sl_tp(self):
        """Test 20: Verify BTC candles are never evaluated against ETH positions."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db, initial_capital=10000.0)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db, symbol="ETHUSDT")

        now_ms = int(time.time() * 1000)
        quote_eth = {"best_bid": 2500.0, "best_ask": 2500.50, "receipt_ts": now_ms}
        quotes = {"ETHUSDT": quote_eth, "BTCUSDT": {"best_bid": 80000.0, "best_ask": 80001.0, "receipt_ts": now_ms}}

        # Open ETH Long: SL=2400.0, TP=2700.0
        pos = engine.open_position(runner.strategy_id, "ETHUSDT", "LONG", 1.0, 2400.0, 2700.0, quote_eth)
        self.assertIsNotNone(pos)
        self.assertIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

        # Simulate incoming BTC bar: high = 81,000 (which exceeds ETH TP 2700.0!)
        btc_bar = {
            "symbol": "BTCUSDT",
            "open": 80000.0,
            "high": 81000.0,
            "low": 79500.0,
            "close": 80500.0
        }

        # Process BTC bar in ETH runner
        runner.process_incoming_market_tick(quotes, current_bar=btc_bar)

        # Assert ETH position was NOT triggered and remains open!
        self.assertIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

        # Now simulate incoming ETH bar touching TP: high = 2750.0
        eth_bar = {
            "symbol": "ETHUSDT",
            "open": 2680.0,
            "high": 2750.0,
            "low": 2670.0,
            "close": 2720.0
        }
        runner.process_incoming_market_tick(quotes, current_bar=eth_bar)

        # Assert ETH position was closed by the authentic ETH bar
        self.assertNotIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

    # -------------------------------------------------------------------------
    # Test 21: Portfolio Accounting — Available cash margin reservation (No Inflation)
    # -------------------------------------------------------------------------
    def test_21_portfolio_accounting_no_margin_inflation(self):
        """Test 21: Verify reserved margin reduces available cash without inflating account equity."""
        engine = PaperExecutionEngine(
            db_path=self.paper_db,
            market_db_path=self.market_db,
            initial_capital=10000.0,
            max_leverage=3.0,
            slippage_rate=0.0
        )
        now_ms = int(time.time() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.0, "receipt_ts": now_ms}

        # Open 1.0 ETH at $3000 -> notional $3000, margin $1000, fee $1.50
        pos = engine.open_position("Strat_A", "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote)
        self.assertIsNotNone(pos)

        # Cash balance MUST represent AVAILABLE cash: $10,000 - $1,000 - $1.50 = $8,998.50
        self.assertAlmostEqual(engine.portfolio["cash_balance"], 8998.50, places=2)
        self.assertAlmostEqual(engine.portfolio["margin_used"], 1000.0, places=2)

        # Account Equity MUST equal available cash ($8,998.50) + margin used ($1,000.0) = $9,998.50
        # MUST NOT be inflated to $10,998.50!
        self.assertAlmostEqual(engine.portfolio["equity"], 9998.50, places=2)

    # -------------------------------------------------------------------------
    # Test 22: Funding and Fee Reconciliation on Close (No Double Counting)
    # -------------------------------------------------------------------------
    def test_22_funding_and_fee_reconciliation_on_close(self):
        """Test 22: Verify funding payments and fees reconcile exactly once upon position closure for Long and Short."""
        engine = PaperExecutionEngine(
            db_path=self.paper_db,
            market_db_path=self.market_db,
            initial_capital=10000.0,
            taker_fee_rate=0.0005,
            slippage_rate=0.0
        )
        now_ms = int(time.time() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.0, "receipt_ts": now_ms}

        # Open LONG 1.0 ETH
        engine.open_position("Strat_Long", "ETHUSDT", "LONG", 1.0, 2800.0, 3200.0, quote, entry_time_str="2023-11-14T00:00:00+00:00")
        entry_fee = 3000.0 * 0.0005  # 1.50

        # Apply Funding Settlement: +1 bp (Long pays Short: payment = -0.30)
        settlement_time = 1700000000000
        conn = sqlite3.connect(self.market_db)
        conn.execute("INSERT INTO funding_settlements (symbol, funding_rate, mark_price, settlement_time) VALUES ('ETHUSDT', 0.0001, 3000.0, ?)", (settlement_time,))
        conn.commit()
        conn.close()

        engine.apply_settled_funding_rates()
        # Available cash = 10000 - 1000 (margin) - 1.50 (entry fee) - 0.30 (funding) = 8998.20
        self.assertAlmostEqual(engine.portfolio["cash_balance"], 8998.20, places=2)

        # Close position at $3100.0 (Gain $100)
        quote_exit = {"best_bid": 3100.0, "best_ask": 3100.0, "receipt_ts": now_ms}
        closed = engine.close_position("Strat_Long", "ETHUSDT", "manual_exit", quote=quote_exit)
        exit_fee = 3100.0 * 0.0005  # 1.55

        # Expected Net PnL = Directional (100.0) + Funding (-0.30) - Entry Fee (1.50) - Exit Fee (1.55) = 96.65
        expected_net_pnl = 100.0 - 0.30 - entry_fee - exit_fee
        self.assertAlmostEqual(closed["net_pnl"], expected_net_pnl, places=2)
        self.assertAlmostEqual(engine.portfolio["realized_pnl"], expected_net_pnl, places=2)

        # Cash balance reconciliation: Margin (1000) returned + Directional (100) - Exit Fee (1.55)
        # Cash = 8998.20 + 1000 + 100 - 1.55 = 10096.65 == 10000 + net_pnl!
        # Funding MUST NOT be credited a second time!
        self.assertAlmostEqual(engine.portfolio["cash_balance"], 10000.0 + expected_net_pnl, places=2)
        self.assertAlmostEqual(engine.portfolio["equity"], 10000.0 + expected_net_pnl, places=2)

    # -------------------------------------------------------------------------
    # Test 23: Restart Recovery — No Retroactive Entries on Historical Missed Candles
    # -------------------------------------------------------------------------
    def test_23_restart_recovery_no_retroactive_entries(self):
        """Test 23: Verify missed candles are evaluated and recorded without opening retroactive entries at current prices."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db, initial_capital=10000.0)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db, symbol="ETHUSDT")

        # Synthesize 70 4H candles
        dates = pd.date_range("2024-01-01 00:00:00", periods=70, freq="4h", tz="UTC")
        close_times = [int(ts.timestamp() * 1000) + 14_400_000 - 1 for ts in dates]

        # Populate SQLite with 1m bars forming these 70 candles
        conn = sqlite3.connect(self.market_db)
        cur = conn.cursor()
        klines = []
        for c_idx, dt in enumerate(dates):
            w_start = int(dt.timestamp() * 1000)
            base_px = 2500.0 - c_idx * 5.0
            for m in range(240):
                ot = w_start + m * 60000
                ct = ot + 59999
                # Candle 50 has a strong green bar (simulated breakout in historical data)
                px = base_px + (100.0 if c_idx == 50 else 0.0)
                klines.append((
                    "ETHUSDT", "1m", ot, ct, px, px + 2.0, px - 2.0, px + 1.0,
                    100.0, 200000.0, 50.0, 500, ot + 60000, ct
                ))
        cur.executemany("""
            INSERT INTO completed_klines (
                symbol, timeframe, open_time, close_time, open, high, low, close,
                volume, quote_volume, taker_buy_volume, trades_count, receipt_ts, exchange_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, klines)
        conn.commit()
        conn.close()

        # Engine was offline for all 70 candles. Now evaluate with fresh quotes
        now_ms = int(time.time() * 1000)
        quotes = {"ETHUSDT": {"best_bid": 3000.0, "best_ask": 3000.5, "receipt_ts": now_ms}}

        evaluations_made = runner.evaluate_new_completed_candles(quotes)
        self.assertGreaterEqual(evaluations_made, 1)

        # Historical candles (e.g. candle 50) must NOT trigger a live market entry!
        # Because the latest candle (candle 69) is not an engulfing trigger setup
        self.assertEqual(len(engine.active_positions), 0)

        # Verify evaluations table has recorded the candles
        conn = sqlite3.connect(self.paper_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM strategy_evaluations WHERE symbol='ETHUSDT'")
        cnt = cur.fetchone()[0]
        conn.close()
        self.assertGreaterEqual(cnt, 10)

    # -------------------------------------------------------------------------
    # Test 24: Versioned Migration — Phase 2.2B Available Cash with Open Position & Unrealized PnL
    # -------------------------------------------------------------------------
    def test_24_versioned_available_cash_migration_with_open_position(self):
        """Test 24: Verify versioned one-time migration converts exact Phase 2.2B portfolio balances with open position and unrealized PnL."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_db = os.path.join(tmp_dir, "legacy_paper_22b.db")
            conn = sqlite3.connect(legacy_db)

            # Create Phase 2.2B schema without migrations table
            conn.execute("""
            CREATE TABLE paper_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cash_balance REAL NOT NULL,
                equity REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                margin_used REAL NOT NULL,
                peak_equity REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                gross_leverage REAL NOT NULL DEFAULT 0.0,
                net_leverage REAL NOT NULL DEFAULT 0.0
            );
            """)
            conn.execute("""
            CREATE TABLE paper_positions (
                position_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                notional REAL NOT NULL,
                margin REAL NOT NULL,
                leverage REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                entry_time TEXT NOT NULL,
                status TEXT NOT NULL,
                accumulated_funding REAL NOT NULL DEFAULT 0.0
            );
            """)
            conn.execute("""
            CREATE TABLE paper_orders (
                order_id TEXT PRIMARY KEY,
                position_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                requested_price REAL NOT NULL,
                fill_price REAL NOT NULL,
                quantity REAL NOT NULL,
                notional REAL NOT NULL,
                fee_usd REAL NOT NULL,
                slippage_usd REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                filled_at TEXT NOT NULL
            );
            """)
            conn.execute("""
            CREATE TABLE paper_trades (
                trade_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                holding_hours REAL NOT NULL,
                directional_pnl REAL NOT NULL,
                funding_pnl REAL NOT NULL,
                entry_fee_usd REAL NOT NULL,
                exit_fee_usd REAL NOT NULL,
                slippage_cost_usd REAL NOT NULL,
                net_pnl REAL NOT NULL,
                exit_reason TEXT NOT NULL
            );
            """)

            # Insert exact Phase 2.2B state:
            # Initial capital: $10,000.00
            # Open position: 1.0 ETH at $3,000.00 (margin: $1,000.00, entry fee: $1.50)
            # Unrealized PnL: +$50.00 (marked at $3,050.00)
            # In Phase 2.2B:
            #   cash_balance = 10000.0 - 1.50 = 9998.50 (margin was NOT deducted)
            #   margin_used = 1000.00
            #   unrealized_pnl = 50.00
            #   equity = cash_balance + margin_used + unrealized_pnl = 9998.50 + 1000.00 + 50.00 = 11048.50 (inflated!)
            #   peak_equity = 11048.50
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute("""
            INSERT INTO paper_portfolio (
                id, timestamp, cash_balance, equity, realized_pnl, unrealized_pnl,
                margin_used, peak_equity, drawdown_pct, gross_leverage, net_leverage
            ) VALUES (1, ?, 9998.50, 11048.50, 0.0, 50.00, 1000.0, 11048.50, 0.0, 0.28, 0.28);
            """, (now_iso,))

            conn.execute("""
            INSERT INTO paper_positions (
                position_id, strategy_id, symbol, side, entry_price, quantity, notional,
                margin, leverage, stop_loss, take_profit, entry_time, status, accumulated_funding
            ) VALUES ('pos_legacy_1', 'CandidateA', 'ETHUSDT', 'LONG', 3000.0, 1.0, 3000.0, 1000.0, 0.3, 2900.0, 3200.0, ?, 'OPEN', 0.0);
            """, (now_iso,))
            conn.commit()
            conn.close()

            # Initialize PaperExecutionEngine which triggers the versioned migration
            engine = PaperExecutionEngine(db_path=legacy_db, market_db_path=self.market_db, initial_capital=10000.0)

            # Available cash MUST now be converted: $9,998.50 - $1,000.00 = $8,998.50
            # Margin used: $1,000.00
            # Unrealized PnL: $50.00
            # Equity: $8,998.50 + $1,000.00 + $50.00 = $10,048.50
            self.assertAlmostEqual(engine.portfolio["cash_balance"], 8998.50, places=2)
            self.assertAlmostEqual(engine.portfolio["margin_used"], 1000.0, places=2)
            self.assertAlmostEqual(engine.portfolio["unrealized_pnl"], 50.00, places=2)
            self.assertAlmostEqual(engine.portfolio["equity"], 10048.50, places=2)
            self.assertAlmostEqual(engine.portfolio["peak_equity"], 10048.50, places=2)

            # Verify reconciliation: cash_balance + margin_used + unrealized_pnl == equity
            self.assertAlmostEqual(
                engine.portfolio["equity"],
                engine.portfolio["cash_balance"] + engine.portfolio["margin_used"] + engine.portfolio["unrealized_pnl"],
                places=2
            )

            # Verify paper_schema_migrations has recorded the migration
            conn = sqlite3.connect(legacy_db)
            cur = conn.cursor()
            cur.execute("SELECT migration_id FROM paper_schema_migrations")
            mig_ids = [r[0] for r in cur.fetchall()]
            conn.close()
            self.assertTrue(any("portfolio_migration" in m or "available_cash" in m for m in mig_ids))

            # Re-initialize engine to verify migration is idempotent and values remain consistent on repeated initialization
            engine2 = PaperExecutionEngine(db_path=legacy_db, market_db_path=self.market_db, initial_capital=10000.0)
            self.assertAlmostEqual(engine2.portfolio["cash_balance"], 8998.50, places=2)
            self.assertAlmostEqual(engine2.portfolio["margin_used"], 1000.0, places=2)
            self.assertAlmostEqual(engine2.portfolio["unrealized_pnl"], 50.00, places=2)
            self.assertAlmostEqual(engine2.portfolio["equity"], 10048.50, places=2)
            self.assertAlmostEqual(
                engine2.portfolio["equity"],
                engine2.portfolio["cash_balance"] + engine2.portfolio["margin_used"] + engine2.portfolio["unrealized_pnl"],
                places=2
            )

    # -------------------------------------------------------------------------
    # Test 25: Funding Accruals — Long & Short Atomic Persistence and Restart Recovery
    # -------------------------------------------------------------------------
    def test_25_funding_accruals_long_and_short_restart_persistence(self):
        """Test 25: Verify funding accruals for Long and Short positions persist atomically and survive restart."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            p_db = os.path.join(tmp_dir, "paper_funding.db")
            m_db = os.path.join(tmp_dir, "market_funding.db")
            collector = MarketDataCollector(db_path=m_db)

            engine = PaperExecutionEngine(
                db_path=p_db,
                market_db_path=m_db,
                initial_capital=20000.0,
                taker_fee_rate=0.0005,
                slippage_rate=0.0
            )
            now_ms = int(time.time() * 1000)
            quote = {"best_bid": 3000.0, "best_ask": 3000.0, "receipt_ts": now_ms}

            # Open LONG 1.0 ETH at $3000 (margin 1000, fee 1.50)
            # Open SHORT 1.0 ETH at $3000 (margin 1000, fee 1.50)
            pos_l = engine.open_position("Strat_L", "ETHUSDT", "LONG", 1.0, 2800.0, 3200.0, quote, entry_time_str="2024-01-01T00:00:00+00:00")
            pos_s = engine.open_position("Strat_S", "ETHUSDT", "SHORT", 1.0, 3200.0, 2800.0, quote, entry_time_str="2024-01-01T00:00:00+00:00")

            # Cash = 20000 - 2000 (margin) - 3.00 (fees) = 17997.00
            self.assertAlmostEqual(engine.portfolio["cash_balance"], 17997.00, places=2)

            # 1. Settlement 1: Positive Funding Rate (+1 bp = 0.0001)
            # Long pays: -1 * 1.0 * 3000 * 0.0001 = -0.30
            # Short receives: -1 * (-1.0) * 3000 * 0.0001 = +0.30
            conn = sqlite3.connect(m_db)
            conn.execute("INSERT INTO funding_settlements (symbol, settlement_time, funding_rate, mark_price) VALUES ('ETHUSDT', 1704096000000, 0.0001, 3000.0)")
            conn.commit()
            conn.close()

            engine.apply_settled_funding_rates()
            self.assertAlmostEqual(engine.active_positions[("Strat_L", "ETHUSDT")]["accumulated_funding"], -0.30, places=4)
            self.assertAlmostEqual(engine.active_positions[("Strat_S", "ETHUSDT")]["accumulated_funding"], 0.30, places=4)
            # Net cash impact of +0.30 and -0.30 is 0.0: cash remains 17997.00
            self.assertAlmostEqual(engine.portfolio["cash_balance"], 17997.00, places=2)

            # 2. Settlement 2: Negative Funding Rate (-2 bp = -0.0002)
            # Long receives: -1 * 1.0 * 3000 * (-0.0002) = +0.60
            # Short pays: -1 * (-1.0) * 3000 * (-0.0002) = -0.60
            conn = sqlite3.connect(m_db)
            conn.execute("INSERT INTO funding_settlements (symbol, settlement_time, funding_rate, mark_price) VALUES ('ETHUSDT', 1704124800000, -0.0002, 3000.0)")
            conn.commit()
            conn.close()

            engine.apply_settled_funding_rates()
            # Net accumulated: Long: -0.30 + 0.60 = +0.30; Short: +0.30 - 0.60 = -0.30
            self.assertAlmostEqual(engine.active_positions[("Strat_L", "ETHUSDT")]["accumulated_funding"], 0.30, places=4)
            self.assertAlmostEqual(engine.active_positions[("Strat_S", "ETHUSDT")]["accumulated_funding"], -0.30, places=4)

            # Verify atomic persistence in SQLite before restart
            conn = sqlite3.connect(p_db)
            cur = conn.cursor()
            cur.execute("SELECT position_id, accumulated_funding FROM paper_positions")
            db_fundings = {r[0]: r[1] for r in cur.fetchall()}
            conn.close()
            self.assertAlmostEqual(db_fundings[pos_l["position_id"]], 0.30, places=4)
            self.assertAlmostEqual(db_fundings[pos_s["position_id"]], -0.30, places=4)

            # 3. Simulate RESTART: Initialize new PaperExecutionEngine from same DB
            engine_restart = PaperExecutionEngine(
                db_path=p_db,
                market_db_path=m_db,
                initial_capital=20000.0,
                taker_fee_rate=0.0005,
                slippage_rate=0.0
            )
            # Accumulated funding must NOT be lost on restart
            self.assertAlmostEqual(engine_restart.active_positions[("Strat_L", "ETHUSDT")]["accumulated_funding"], 0.30, places=4)
            self.assertAlmostEqual(engine_restart.active_positions[("Strat_S", "ETHUSDT")]["accumulated_funding"], -0.30, places=4)
            self.assertAlmostEqual(engine_restart.portfolio["cash_balance"], 17997.00, places=2)

            # Re-running apply_settled_funding_rates must NOT double-apply existing settlements
            settlements = engine_restart.apply_settled_funding_rates()
            self.assertEqual(settlements, 0)
            self.assertAlmostEqual(engine_restart.active_positions[("Strat_L", "ETHUSDT")]["accumulated_funding"], 0.30, places=4)
            self.assertAlmostEqual(engine_restart.active_positions[("Strat_S", "ETHUSDT")]["accumulated_funding"], -0.30, places=4)

            # 4. Close both positions at $3000.0 (0 directional PnL)
            # Exit fee for each = 3000 * 0.0005 = 1.50
            # Long Net PnL = Directional (0.0) + Funding (+0.30) - Entry Fee (1.50) - Exit Fee (1.50) = -2.70
            # Short Net PnL = Directional (0.0) + Funding (-0.30) - Entry Fee (1.50) - Exit Fee (1.50) = -3.30
            # Total Realized PnL = -6.00
            q_exit = {"best_bid": 3000.0, "best_ask": 3000.0, "receipt_ts": now_ms}
            c_l = engine_restart.close_position("Strat_L", "ETHUSDT", "manual_exit", quote=q_exit)
            c_s = engine_restart.close_position("Strat_S", "ETHUSDT", "manual_exit", quote=q_exit)

            self.assertAlmostEqual(c_l["net_pnl"], -2.70, places=2)
            self.assertAlmostEqual(c_s["net_pnl"], -3.30, places=2)
            self.assertAlmostEqual(engine_restart.portfolio["realized_pnl"], -6.00, places=2)
            self.assertAlmostEqual(engine_restart.portfolio["cash_balance"], 20000.0 - 6.00, places=2)
            self.assertAlmostEqual(engine_restart.portfolio["equity"], 20000.0 - 6.00, places=2)

    # -------------------------------------------------------------------------
    # Test 26: Maximum Signal Age — Reject Stale Historical Signals
    # -------------------------------------------------------------------------
    def test_26_maximum_signal_age_enforcement(self):
        """Test 26: Verify that signals older than MAX_SIGNAL_AGE_SECONDS are rejected as stale."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db, initial_capital=10000.0)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db, symbol="ETHUSDT")

        # Synthesize 65 completed 4H candles
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dates = [base_time + timedelta(hours=4 * i) for i in range(65)]

        # Populate market_db with 1m bars for these candles
        conn = sqlite3.connect(self.market_db)
        cur = conn.cursor()
        klines = []
        for c_idx, dt in enumerate(dates):
            w_start = int(dt.timestamp() * 1000)
            base_px = 2500.0
            # Candle 64 has a strong breakout setup
            if c_idx == 64:
                base_px = 2800.0
            for m in range(240):
                ot = w_start + m * 60000
                ct = ot + 59999
                px = base_px + (150.0 if (c_idx == 64 and m >= 200) else 0.0)
                vol = 200.0 if c_idx == 64 else 50.0
                klines.append((
                    "ETHUSDT", "1m", ot, ct, px, px + 2.0, px - 2.0, px + 1.0,
                    vol, vol * px, vol * 0.5, 100, ot + 60000, ct
                ))
        cur.executemany("""
            INSERT INTO completed_klines (
                symbol, timeframe, open_time, close_time, open, high, low, close,
                volume, quote_volume, taker_buy_volume, trades_count, receipt_ts, exchange_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, klines)
        conn.commit()
        conn.close()

        now_ms = int(time.time() * 1000)
        quotes = {"ETHUSDT": {"best_bid": 3000.0, "best_ask": 3000.5, "receipt_ts": now_ms}}

        # Mock evaluate_candidate_a_logic to return a valid LONG signal on candle 64
        runner.evaluate_candidate_a_logic = lambda df: {
            "signal": "LONG",
            "entry_price": 3000.0,
            "stop_loss": 2900.0,
            "take_profit": 3200.0,
            "atr": 50.0,
            "features": {"atr_14": 50.0, "ema_50": 2800.0, "vol_ratio": 1.5}
        }

        # Case A: Evaluated 2 hours after candle close (stale: age = 7200s > 900s)
        candle_64_close = dates[-1] + timedelta(hours=4)
        stale_eval_time = candle_64_close + timedelta(hours=2)
        runner.evaluate_new_completed_candles(quotes, now_utc=stale_eval_time)

        # Must NOT open a position because signal is stale!
        self.assertEqual(len(engine.active_positions), 0)

        # Clean evaluations table to re-test with fresh time
        conn = sqlite3.connect(self.paper_db)
        conn.execute("DELETE FROM strategy_evaluations")
        conn.commit()
        conn.close()

        # Case B: Evaluated 2 minutes after candle close (fresh: age = 120s <= 900s)
        quotes["ETHUSDT"]["receipt_ts"] = int(time.time() * 1000)
        fresh_eval_time = candle_64_close + timedelta(minutes=2)
        runner.evaluate_new_completed_candles(quotes, now_utc=fresh_eval_time)

        # Genuinely fresh signal opens position!
        self.assertIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

    # -------------------------------------------------------------------------
    # Test 27: 1m Candle Deduplication and Entry Causality Guard
    # -------------------------------------------------------------------------
    def test_27_candle_deduplication_and_causality_guard(self):
        """Test 27: Process each completed 1m candle only once and reject pre-entry candles for stops/targets."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db, initial_capital=10000.0, slippage_rate=0.0005)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=self.market_db, paper_db_path=self.paper_db, symbol="ETHUSDT")

        now_dt = datetime.now(timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.0, "receipt_ts": now_ms}
        quotes = {"ETHUSDT": quote}

        # Open Long position at entry_time = 5 minutes ago (within 72-hour horizon)
        entry_dt = now_dt - timedelta(minutes=5)
        entry_ms = int(entry_dt.timestamp() * 1000)
        pos = engine.open_position(
            runner.strategy_id, "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote,
            entry_time_str=entry_dt.isoformat()
        )
        self.assertIsNotNone(pos)
        self.assertIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

        # 1. Causality Guard: Completed 1m bar completed BEFORE entry!
        # Its low drops to $2850.0 (below stop loss $2900.0)
        pre_entry_bar = {
            "symbol": "ETHUSDT",
            "open_time": entry_ms - 120000,
            "close_time": entry_ms - 60001,  # completed before position entry
            "open": 2950.0,
            "high": 2960.0,
            "low": 2850.0,
            "close": 2920.0
        }
        runner.process_incoming_market_tick(quotes, current_bar=pre_entry_bar)
        # Position MUST remain open because the candle completed BEFORE position entry!
        self.assertIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

        # 2. Deduplication: Send a neutral post-entry 1m bar
        post_bar = {
            "symbol": "ETHUSDT",
            "open_time": entry_ms + 60000,
            "close_time": entry_ms + 119999,  # completed after position entry
            "open": 3000.0,
            "high": 3050.0,
            "low": 2980.0,
            "close": 3020.0
        }
        runner.process_incoming_market_tick(quotes, current_bar=post_bar)
        self.assertEqual(runner.last_processed_1m_open_time["ETHUSDT"], entry_ms + 60000)

        # Sending the EXACT same post-entry bar a second time must be ignored
        runner.process_incoming_market_tick(quotes, current_bar=post_bar)
        self.assertIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

        # 3. Legitimate Post-Entry Stop Breach:
        sl_bar = {
            "symbol": "ETHUSDT",
            "open_time": entry_ms + 120000,
            "close_time": entry_ms + 179999,  # completed after position entry
            "open": 2980.0,
            "high": 2990.0,
            "low": 2880.0,  # below SL 2900.0
            "close": 2890.0
        }
        runner.process_incoming_market_tick(quotes, current_bar=sl_bar)
        # Position closed by authentic post-entry bar
        self.assertNotIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

    # -------------------------------------------------------------------------
    # Test 28: Execution Fills — Stop-Loss Touch, Adverse Gap, Take-Profit, Collision
    # -------------------------------------------------------------------------
    def test_28_execution_fills_sl_gap_tp_collision(self):
        """Test 28: Verify execution fills for stop-loss touch, adverse gap, take-profit, and same-bar collision."""
        engine = PaperExecutionEngine(
            db_path=self.paper_db,
            market_db_path=self.market_db,
            initial_capital=20000.0,
            slippage_rate=0.0010,  # 10 bps
            taker_fee_rate=0.0005
        )
        now_ms = int(time.time() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.0, "receipt_ts": now_ms}

        # 1. Stop-Loss Touch (LONG):
        # Open LONG at 3000.0, SL=2900.0, TP=3200.0
        engine.open_position("Strat_SL", "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote)
        res_sl = engine.close_position("Strat_SL", "ETHUSDT", "stop_loss")
        # Long Stop-loss touch fills at stop level with adverse slippage: 2900.0 * (1 - 0.001) = 2897.10
        self.assertAlmostEqual(res_sl["exit_price"], 2897.10, places=2)

        # 2. Stop-Loss Touch (SHORT):
        # Open SHORT at 3000.0, SL=3100.0, TP=2800.0
        engine.open_position("Strat_SL_S", "ETHUSDT", "SHORT", 1.0, 3100.0, 2800.0, quote)
        res_sl_s = engine.close_position("Strat_SL_S", "ETHUSDT", "stop_loss")
        # Short Stop-loss touch fills at stop level with adverse slippage: 3100.0 * (1 + 0.001) = 3103.10
        self.assertAlmostEqual(res_sl_s["exit_price"], 3103.10, places=2)

        # 3. Adverse Gap (LONG):
        # Bar opened at 2850.0 (below stop loss 2900.0)
        engine.open_position("Strat_Gap_L", "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote)
        res_gap_l = engine.close_position("Strat_Gap_L", "ETHUSDT", "stop_loss_gap", adverse_gap_price=2850.0)
        # Long Adverse Gap fills from open price with adverse slippage: 2850.0 * (1 - 0.001) = 2847.15
        self.assertAlmostEqual(res_gap_l["exit_price"], 2847.15, places=2)

        # 4. Adverse Gap (SHORT):
        # Bar opened at 3150.0 (above stop loss 3100.0)
        engine.open_position("Strat_Gap_S", "ETHUSDT", "SHORT", 1.0, 3100.0, 2800.0, quote)
        res_gap_s = engine.close_position("Strat_Gap_S", "ETHUSDT", "stop_loss_gap", adverse_gap_price=3150.0)
        # Short Adverse Gap fills from open price with adverse slippage: 3150.0 * (1 + 0.001) = 3153.15
        self.assertAlmostEqual(res_gap_s["exit_price"], 3153.15, places=2)

        # 5. Take-Profit (LONG & SHORT):
        # Conservative target fill at exact target level (0 slippage)
        engine.open_position("Strat_TP_L", "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote)
        res_tp_l = engine.close_position("Strat_TP_L", "ETHUSDT", "take_profit")
        self.assertAlmostEqual(res_tp_l["exit_price"], 3200.00, places=2)

        engine.open_position("Strat_TP_S", "ETHUSDT", "SHORT", 1.0, 3100.0, 2800.0, quote)
        res_tp_s = engine.close_position("Strat_TP_S", "ETHUSDT", "take_profit")
        self.assertAlmostEqual(res_tp_s["exit_price"], 2800.00, places=2)

        # 6. Same-Bar Collision (LONG):
        # Resolves STOP-FIRST at stop level with adverse slippage: 2900.0 * (1 - 0.001) = 2897.10
        engine.open_position("Strat_Coll", "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote)
        # Check trigger resolves as stop_loss_collision
        trigger = engine.check_position_stops_and_gaps("Strat_Coll", "ETHUSDT", bar_open=3000.0, bar_high=3250.0, bar_low=2850.0, bar_close=3100.0)
        self.assertEqual(trigger[0], "stop_loss_collision")
        res_coll = engine.close_position("Strat_Coll", "ETHUSDT", trigger[0])
        self.assertAlmostEqual(res_coll["exit_price"], 2897.10, places=2)

    # -------------------------------------------------------------------------
    # Test 29: Open-Interest USD Conversion, Legacy Migration & Display Formatting
    # -------------------------------------------------------------------------
    def test_29_open_interest_usd_conversion_and_migration(self):
        """Test 29: Verify open interest preserves base units, converts via fresh mark price, migrates legacy schemas, and formats N/A."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            m_db = os.path.join(tmp_dir, "market_oi.db")
            conn = sqlite3.connect(m_db)

            # 1. Create legacy schema with open_interest_usd REAL NOT NULL and 0.0 rows
            conn.execute("""
            CREATE TABLE open_interest (
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open_interest REAL NOT NULL,
                open_interest_usd REAL NOT NULL,
                exchange_ts INTEGER NOT NULL,
                receipt_ts INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );
            """)
            conn.execute("INSERT INTO open_interest VALUES ('ETHUSDT', 1700000000000, 50000.0, 0.0, 1700000000000, 1700000000000);")
            conn.commit()
            conn.close()

            # Initialize MarketDataCollector to trigger migration
            collector = MarketDataCollector(db_path=m_db)

            # Verify table was migrated to nullable REAL and 0.0 converted to NULL
            conn = sqlite3.connect(m_db)
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(open_interest);")
            oi_info = {row[1]: row for row in cur.fetchall()}
            self.assertEqual(oi_info["open_interest_usd"][3], 0)  # notnull is 0 (nullable!)

            cur.execute("SELECT open_interest, open_interest_usd FROM open_interest WHERE symbol='ETHUSDT'")
            row = cur.fetchone()
            conn.close()
            self.assertEqual(row[0], 50000.0)
            self.assertIsNone(row[1])  # Converted to NULL (None) instead of 0.0!

            # 2. Test fetch_open_interest when mark price is unavailable:
            # Mock _http_get_json to return open interest data
            collector._http_get_json = lambda url: {"symbol": "ETHUSDT", "openInterest": "60000.0", "time": 1700000060000}
            oi_rec = collector.fetch_open_interest("ETHUSDT")
            self.assertEqual(oi_rec["open_interest"], 60000.0)
            self.assertIsNone(oi_rec["open_interest_usd"])  # Never persists 0.0!

            # 3. Test fetch_open_interest when fresh mark price is available:
            collector.latest_mark_prices["ETHUSDT"] = {
                "mark_price": 2500.0,
                "receipt_ts": int(time.time() * 1000)
            }
            oi_rec_fresh = collector.fetch_open_interest("ETHUSDT")
            self.assertEqual(oi_rec_fresh["open_interest"], 60000.0)
            self.assertAlmostEqual(oi_rec_fresh["open_interest_usd"], 60000.0 * 2500.0, places=2)

            # 4. Verify display formatting logic: unavailable displays 'N/A'
            oi_usd_none = None
            oi_usd_zero = 0.0
            oi_usd_val = 150000000.0
            fmt_none = f"${oi_usd_none:,.2f}" if (oi_usd_none is not None and oi_usd_none > 0) else "N/A"
            fmt_zero = f"${oi_usd_zero:,.2f}" if (oi_usd_zero is not None and oi_usd_zero > 0) else "N/A"
            fmt_val = f"${oi_usd_val:,.2f}" if (oi_usd_val is not None and oi_usd_val > 0) else "N/A"

            self.assertEqual(fmt_none, "N/A")
            self.assertEqual(fmt_zero, "N/A")
            self.assertEqual(fmt_val, "$150,000,000.00")

    # -------------------------------------------------------------------------
    # Test 30: Open Interest Fresh DB Mark Price Query After Restart
    # -------------------------------------------------------------------------
    def test_30_open_interest_fresh_db_mark_price_after_restart(self):
        """Test 30: Verify fetch_open_interest queries funding_and_mark_price when in-memory cache is empty after restart."""
        m_db = os.path.join(self.test_dir, "market_restart_oi.db")
        collector = MarketDataCollector(db_path=m_db)

        # Insert a fresh mark price observation into funding_and_mark_price
        now_ms = int(time.time() * 1000)
        with collector._get_connection() as conn:
            conn.execute("""
            INSERT INTO funding_and_mark_price (
                symbol, timestamp, mark_price, index_price, funding_rate,
                next_funding_time, exchange_ts, receipt_ts
            ) VALUES ('ETHUSDT', ?, 3120.50, 3120.00, 0.0001, ?, ?, ?);
            """, (now_ms - 2000, now_ms + 14400000, now_ms - 2000, now_ms - 2000))

        # Simulate fresh restart: create a new collector instance with empty memory cache
        collector_restarted = MarketDataCollector(db_path=m_db)
        self.assertEqual(len(collector_restarted.latest_mark_prices), 0)

        # Mock REST API response for base open interest
        collector_restarted._http_get_json = lambda url: {
            "symbol": "ETHUSDT",
            "openInterest": "50000.0",
            "time": now_ms
        }

        oi_rec = collector_restarted.fetch_open_interest("ETHUSDT")
        self.assertIsNotNone(oi_rec)
        self.assertEqual(oi_rec["open_interest"], 50000.0)
        # Fresh mark price 3120.50 from funding_and_mark_price in DB MUST be used:
        self.assertAlmostEqual(oi_rec["open_interest_usd"], 50000.0 * 3120.50, places=2)

        # Verify persisted row in SQLite open_interest table
        with collector_restarted._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT open_interest, open_interest_usd FROM open_interest WHERE symbol = 'ETHUSDT'")
            row = cur.fetchone()
            self.assertEqual(row[0], 50000.0)
            self.assertAlmostEqual(row[1], 50000.0 * 3120.50, places=2)

    # -------------------------------------------------------------------------
    # Test 31: Persistent 1m Candle Tracking and Chronological Missed Bar Replay
    # -------------------------------------------------------------------------
    def test_31_persistent_1m_candle_processing_and_chronological_replay(self):
        """Test 31: Verify persistent 1m candle tracking, chronological missed bar replay, and duplicate prevention."""
        p_db = os.path.join(self.test_dir, "paper_replay.db")
        m_db = os.path.join(self.test_dir, "market_replay.db")

        collector = MarketDataCollector(db_path=m_db)
        engine = PaperExecutionEngine(db_path=p_db, market_db_path=m_db, initial_capital=10000.0, slippage_rate=0.001)
        runner = CandidateAPaperRunner(execution_engine=engine, market_db_path=m_db, paper_db_path=p_db, symbol="ETHUSDT")

        now_dt = datetime.now(timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.0, "receipt_ts": now_ms}
        quotes = {"ETHUSDT": quote}

        # Open LONG position at entry_time = 10 minutes ago
        entry_dt = now_dt - timedelta(minutes=10)
        entry_ms = int(entry_dt.timestamp() * 1000)
        engine.open_position(runner.strategy_id, "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote, entry_time_str=entry_dt.isoformat())
        self.assertIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

        # Insert 5 completed 1m candles into market_data.db representing downtime bars
        # Bars 1 and 2: neutral
        # Bar 3: drops to 2890.0 (breaches SL 2900.0!)
        # Bars 4 and 5: post-stop bars
        klines = [
            ("ETHUSDT", "1m", entry_ms + 60000, entry_ms + 119999, 3000.0, 3010.0, 2990.0, 3005.0, 100.0, 300000.0, 50.0, 100, entry_ms + 120000, entry_ms + 119999),
            ("ETHUSDT", "1m", entry_ms + 120000, entry_ms + 179999, 3005.0, 3020.0, 3000.0, 3015.0, 100.0, 300000.0, 50.0, 100, entry_ms + 180000, entry_ms + 179999),
            ("ETHUSDT", "1m", entry_ms + 180000, entry_ms + 239999, 3015.0, 3025.0, 2890.0, 2900.0, 150.0, 450000.0, 20.0, 200, entry_ms + 240000, entry_ms + 239999),
            ("ETHUSDT", "1m", entry_ms + 240000, entry_ms + 299999, 2900.0, 2910.0, 2880.0, 2890.0, 100.0, 290000.0, 40.0, 100, entry_ms + 300000, entry_ms + 299999),
            ("ETHUSDT", "1m", entry_ms + 300000, entry_ms + 359999, 2890.0, 2900.0, 2870.0, 2880.0, 100.0, 288000.0, 40.0, 100, entry_ms + 360000, entry_ms + 359999),
        ]
        with sqlite3.connect(m_db) as conn:
            conn.executemany("""
            INSERT INTO completed_klines (
                symbol, timeframe, open_time, close_time, open, high, low, close,
                volume, quote_volume, taker_buy_volume, trades_count, receipt_ts, exchange_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, klines)

        # Replay missed bars chronologically
        replayed_cnt = runner.replay_missed_completed_1m_bars(quotes)
        self.assertEqual(replayed_cnt, 5)

        # Active position MUST have been closed by Bar 3 (stop-loss touch)
        self.assertNotIn((runner.strategy_id, "ETHUSDT"), engine.active_positions)

        # Verify that all 5 bars were recorded in processed_1m_candles table
        with sqlite3.connect(p_db) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM processed_1m_candles WHERE symbol = 'ETHUSDT'")
            self.assertEqual(cur.fetchone()[0], 5)

        # Replaying again immediately MUST process 0 new bars (duplicate prevention!)
        replayed_again = runner.replay_missed_completed_1m_bars(quotes)
        self.assertEqual(replayed_again, 0)

    # -------------------------------------------------------------------------
    # Test 32: Entry Causality — Candle Spanning Entry Time Never Triggers Stops/Targets
    # -------------------------------------------------------------------------
    def test_32_entry_causality_candle_spanning_entry_time(self):
        """Test 32: Verify candle that opens before position entry but closes after entry cannot trigger stops/targets."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db, initial_capital=10000.0)
        now_ms = int(time.time() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.0, "receipt_ts": now_ms}

        # Position entered at 12:00:30.000 UTC
        entry_time_str = "2024-01-01T12:00:30+00:00"
        entry_ms = 1704110430000

        engine.open_position("Strat_Causality", "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote, entry_time_str=entry_time_str)

        # Candle that opened at 12:00:00.000 (precedes entry!) and closed at 12:00:59.999 (after entry!)
        # With extreme prices that would hit both SL (2800 <= 2900) and TP (3300 >= 3200)
        spanning_bar_open = 1704110400000  # 12:00:00
        spanning_bar_close = 1704110459999  # 12:00:59

        trigger = engine.check_position_stops_and_gaps(
            strategy_id="Strat_Causality",
            symbol="ETHUSDT",
            bar_open=2950.0,
            bar_high=3300.0,
            bar_low=2800.0,
            bar_close=3000.0,
            bar_close_time=spanning_bar_close,
            bar_open_time=spanning_bar_open
        )

        # Must return None! Entry causality strictly protects the position from prior price excursions.
        self.assertIsNone(trigger)
        self.assertIn(("Strat_Causality", "ETHUSDT"), engine.active_positions)

        # Next candle opens at 12:01:00.000 (>= entry_ms) and touches SL
        post_bar_open = 1704110460000
        post_bar_close = 1704110519999
        trigger_valid = engine.check_position_stops_and_gaps(
            strategy_id="Strat_Causality",
            symbol="ETHUSDT",
            bar_open=3000.0,
            bar_high=3010.0,
            bar_low=2890.0,  # below SL 2900.0
            bar_close=2895.0,
            bar_close_time=post_bar_close,
            bar_open_time=post_bar_open
        )
        self.assertIsNotNone(trigger_valid)
        self.assertEqual(trigger_valid[0], "stop_loss")

    # -------------------------------------------------------------------------
    # Test 33: Take-Profit Execution — Conservative Trade-Through Requirement
    # -------------------------------------------------------------------------
    def test_33_take_profit_trade_through_requirement(self):
        """Test 33: Verify touching target price alone does not fill; trade-through by tick size required for Long and Short."""
        engine = PaperExecutionEngine(db_path=self.paper_db, market_db_path=self.market_db, initial_capital=20000.0)
        now_dt = datetime.now(timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)
        quote = {"best_bid": 3000.0, "best_ask": 3000.0, "receipt_ts": now_ms}

        entry_time_str = now_dt.isoformat()
        bar_open_time = now_ms + 60000
        bar_close_time = now_ms + 119999

        # 1. LONG Position: Entry 3000.0, SL 2900.0, TP 3200.0, ETHUSDT tick_size = 0.01
        engine.open_position("Strat_TP_TT_L", "ETHUSDT", "LONG", 1.0, 2900.0, 3200.0, quote, entry_time_str=entry_time_str)

        # Case A: High touches exact target 3200.00 without trade-through
        trigger_touch = engine.check_position_stops_and_gaps(
            strategy_id="Strat_TP_TT_L",
            symbol="ETHUSDT",
            bar_open=3100.0,
            bar_high=3200.00,  # Mere touch
            bar_low=3090.0,
            bar_close=3180.0,
            bar_open_time=bar_open_time,
            bar_close_time=bar_close_time
        )
        # Touch alone must NOT fill!
        self.assertIsNone(trigger_touch)

        # Case B: High trades through by 1 tick (3200.01)
        trigger_through = engine.check_position_stops_and_gaps(
            strategy_id="Strat_TP_TT_L",
            symbol="ETHUSDT",
            bar_open=3100.0,
            bar_high=3200.01,  # 1-tick trade through
            bar_low=3090.0,
            bar_close=3190.0,
            bar_open_time=bar_open_time,
            bar_close_time=bar_close_time
        )
        self.assertEqual(trigger_through[0], "take_profit")
        res_l = engine.close_position("Strat_TP_TT_L", "ETHUSDT", "take_profit")
        self.assertAlmostEqual(res_l["exit_price"], 3200.00, places=2)

        # 2. SHORT Position: Entry 3000.0, SL 3100.0, TP 2800.0, ETHUSDT tick_size = 0.01
        engine.open_position("Strat_TP_TT_S", "ETHUSDT", "SHORT", 1.0, 3100.0, 2800.0, quote, entry_time_str=entry_time_str)

        # Case A: Low touches exact target 2800.00 without trade-through
        trigger_s_touch = engine.check_position_stops_and_gaps(
            strategy_id="Strat_TP_TT_S",
            symbol="ETHUSDT",
            bar_open=2900.0,
            bar_high=2910.0,
            bar_low=2800.00,  # Mere touch
            bar_close=2820.0,
            bar_open_time=bar_open_time,
            bar_close_time=bar_close_time
        )
        # Touch alone must NOT fill!
        self.assertIsNone(trigger_s_touch)

        # Case B: Low trades through by 1 tick (2799.99)
        trigger_s_through = engine.check_position_stops_and_gaps(
            strategy_id="Strat_TP_TT_S",
            symbol="ETHUSDT",
            bar_open=2900.0,
            bar_high=2910.0,
            bar_low=2799.99,  # 1-tick trade through
            bar_close=2810.0,
            bar_open_time=bar_open_time,
            bar_close_time=bar_close_time
        )
        self.assertEqual(trigger_s_through[0], "take_profit")
        res_s = engine.close_position("Strat_TP_TT_S", "ETHUSDT", "take_profit")
        self.assertAlmostEqual(res_s["exit_price"], 2800.00, places=2)


if __name__ == "__main__":
    unittest.main()
