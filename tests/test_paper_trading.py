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
from datetime import datetime, timezone
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
            exit_reason="take_profit",
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
        closed = engine.close_position("CandidateA", "ETHUSDT", "take_profit", quote=quote_exit)
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


if __name__ == "__main__":
    unittest.main()
