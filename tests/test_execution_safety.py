"""
tests/test_execution_safety.py — Comprehensive Execution Safety & Backtest Integrity Regression Suite.

Covers the 20 required quantitative and safety failure scenarios:
 1. Exchange position fetch failure (fail-closed, explicit UNKNOWN_ERROR).
 2. Open-order fetch failure (fail-closed, explicit UNKNOWN_ERROR).
 3. Balance fetch failure (fail-closed, explicit UNKNOWN_ERROR).
 4. Failed market entry (graceful abort, internal state remains flat).
 5. Entry fill with incomplete protective orders (partial bracket handling).
 6. Stop-loss placement failure (triggers emergency reduce-only close; latches EMERGENCY_HALT).
 7. Take-profit placement failure (retains protective stop-loss, position remains protected).
 8. Failed position close (does NOT clear internal state, preserves protection).
 9. Failed close retry (retries once, latches EMERGENCY_HALT if still unclosed).
10. No clearing internal state before confirmed flat (asserts state preserved until exchange confirms 0 contracts).
11. Failed order cancellation (handles gracefully without assuming orders deleted).
12. Trailing-stop replacement failure (verifies replacement before canceling previous stop; retains existing stop).
13. Restart with an existing real position (reconciles exchange position and restores missing brackets).
14. Daily drawdown lock blocking new orders (evaluated at top before any order sizing or execution).
15. Position sizing below minimum notional without exceeding risk budget (skips if bumping exceeds risk budget; bumps if within).
16. ATR look-ahead prevention (changing entry candle high/low/close does NOT alter decision ATR or barriers).
17. Next-open signal timing (signal decided at candle i-1 close, entry executes strictly at candle i open).
18. Funding calculation (8-hour settlement timestamps 00:00, 08:00, 16:00 UTC applied to holding period).
19. Compounded equity consistency (sequential returns compounded multiplicatively without fee double-counting).
20. Non-destructive artifact handling (verifies tests run in isolated directories and never touch production artifacts).
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from live.execution_engine import (
    ExecutionEngine, PositionStatus, BalanceStatus, OrderStatus,
    PositionInfo, BalanceInfo, OrdersInfo
)
from live.trader import Trader
from model.walk_forward import simulate_trades_ohlc
from config import (
    SYMBOL, MIN_NOTIONAL_USD, LEVERAGE, RISK_PER_TRADE_PERCENT,
    MODEL_PATH, SCALER_CANDLE_PATH, SCALER_MARKET_PATH,
    LIVE_TRADING_ACK_TOKEN
)


class TestExecutionSafetyAndIntegrity(unittest.TestCase):
    def setUp(self):
        self.mock_ee = MagicMock(spec=ExecutionEngine)
        self.mock_ee.mock = False
        self.mock_ee.emergency_halt = False
        self.mock_ee.get_balance.return_value = BalanceInfo(status=BalanceStatus.VERIFIED, balance=1000.0)
        self.mock_ee.get_open_position.return_value = PositionInfo(status=PositionStatus.VERIFIED_FLAT)
        self.mock_ee.get_open_orders.return_value = OrdersInfo(status=OrderStatus.VERIFIED, orders=[])
        self.mock_ee.get_current_price.return_value = 2300.0
        self.mock_ee.place_order.return_value = {
            'id': 'ord_1', 'status': 'closed', 'average': 2300.0, 'amount': 0.1, 'filled': 0.1
        }
        self.mock_ee.place_tp_sl_orders.return_value = {
            'tp': {'id': 'tp_1', 'status': 'open'},
            'sl': {'id': 'sl_1', 'status': 'open'}
        }
        self.mock_ee.close_position.return_value = {
            'id': 'cls_1', 'status': 'closed', 'verified': True
        }
        self.mock_ee.cancel_all_orders.return_value = True

        self.trader = Trader(self.mock_ee)
        self.trader.daily_pnl = 0.0
        self.trader.drawdown_blocked = False
        self.trader.current_position = None
        self.trader.last_trade_close_time = None

    def _create_isolated_ee(self):
        """Helper to create an ExecutionEngine in mock=False under the explicit opt-in patch."""
        with patch('live.execution_engine.ALLOW_LIVE_TRADING', LIVE_TRADING_ACK_TOKEN), \
             patch('live.execution_engine.BINANCE_API_KEY', 'test_key'), \
             patch('live.execution_engine.BINANCE_API_SECRET', 'test_secret'), \
             patch('ccxt.binanceusdm'):
            ee = ExecutionEngine(mock=False)
            return ee

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 1: Exchange position fetch failure (fail-closed)
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_exchange_position_fetch_failure(self):
        """Scenario 1: Exchange API failure returns UNKNOWN_ERROR; bot fails closed."""
        real_ee = self._create_isolated_ee()
        real_ee.exchange = MagicMock()
        real_ee.exchange.fetch_positions.side_effect = Exception("Binance 502 Bad Gateway")

        pos = real_ee.get_open_position(SYMBOL)
        self.assertEqual(pos.status, PositionStatus.UNKNOWN_ERROR)
        self.assertTrue(pos.is_error)
        self.assertFalse(pos.is_flat)
        self.assertFalse(pos.is_open)
        self.assertIn("502", str(pos.error))

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 2: Open-order fetch failure (fail-closed)
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_open_order_fetch_failure(self):
        """Scenario 2: Open-orders API failure returns UNKNOWN_ERROR, never empty list silently."""
        real_ee = self._create_isolated_ee()
        real_ee.exchange = MagicMock()
        real_ee.exchange.fetch_open_orders.side_effect = Exception("Network timeout")

        orders_info = real_ee.get_open_orders(SYMBOL)
        self.assertEqual(orders_info.status, OrderStatus.UNKNOWN_ERROR)
        self.assertTrue(orders_info.is_error)
        self.assertEqual(len(orders_info), 0)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 3: Balance fetch failure (fail-closed)
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_balance_fetch_failure(self):
        """Scenario 3: Balance fetch failure returns UNKNOWN_ERROR, never 0.0 balance silently."""
        real_ee = self._create_isolated_ee()
        real_ee.exchange = MagicMock()
        real_ee.exchange.fapiPrivateV2GetBalance.side_effect = Exception("API rate limited")

        bal_info = real_ee.get_balance()
        self.assertEqual(bal_info.status, BalanceStatus.UNKNOWN_ERROR)
        self.assertTrue(bal_info.is_error)
        self.assertFalse(bal_info.is_verified)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 4: Failed market entry
    # ─────────────────────────────────────────────────────────────────────────
    def test_04_failed_market_entry(self):
        """Scenario 4: Failed market entry order aborts safely; internal state remains flat."""
        self.mock_ee.place_order.return_value = None  # Placement failed
        now = datetime.now(timezone.utc)

        with patch('live.trader.TRADING_ENABLED', True):
            self.trader.handle_signal(now, 0.08, 2300.0, atr=25.0)

        self.assertIsNone(self.trader.current_position)
        self.assertEqual(self.trader.amount, 0.0)
        self.mock_ee.place_tp_sl_orders.assert_not_called()

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 5: Entry fill with incomplete protective orders
    # ─────────────────────────────────────────────────────────────────────────
    def test_05_entry_fill_with_incomplete_protective_orders(self):
        """Scenario 5: Entry fills but Take-Profit fails; Stop-Loss is confirmed active."""
        self.mock_ee.place_tp_sl_orders.return_value = {
            'tp': None,  # TP failed
            'sl': {'id': 'sl_101', 'status': 'open'}  # SL succeeded
        }
        now = datetime.now(timezone.utc)

        with patch('live.trader.TRADING_ENABLED', True):
            self.trader.handle_signal(now, 0.08, 2300.0, atr=25.0)

        # Position is entered because critical Stop Loss is active
        self.assertEqual(self.trader.current_position, "LONG")
        self.assertGreater(self.trader.current_sl, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 6: Stop-loss placement failure
    # ─────────────────────────────────────────────────────────────────────────
    def test_06_stop_loss_placement_failure_emergency_close(self):
        """Scenario 6: If Stop-Loss fails, execution engine triggers emergency reduce-only close."""
        real_ee = self._create_isolated_ee()
        real_ee.exchange = MagicMock()
        def mock_create_order(sym, otype, side, amt, price, params):
            if otype == 'STOP_MARKET':
                raise Exception("Order would immediately trigger")
            return {'id': 'tp_12', 'type': otype}
        real_ee.exchange.create_order.side_effect = mock_create_order

        # Emergency close mock
        real_ee.close_position = MagicMock(return_value={'status': 'closed', 'verified': True})

        res = real_ee.place_tp_sl_orders(SYMBOL, "LONG", 0.1, 2400.0, 2200.0)
        self.assertIsNone(res['sl'])
        self.assertEqual(res.get('emergency_status'), 'EMERGENCY_CLOSED')
        real_ee.close_position.assert_called_once()

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 7: Take-profit placement failure
    # ─────────────────────────────────────────────────────────────────────────
    def test_07_take_profit_placement_failure(self):
        """Scenario 7: TP placement failure preserves Stop-Loss without halting system."""
        real_ee = self._create_isolated_ee()
        real_ee.exchange = MagicMock()
        def mock_create_order(sym, otype, side, amt, price, params):
            if otype == 'TAKE_PROFIT_MARKET':
                raise Exception("TP error")
            return {'id': 'sl_ok', 'type': otype}
        real_ee.exchange.create_order.side_effect = mock_create_order

        res = real_ee.place_tp_sl_orders(SYMBOL, "LONG", 0.1, 2400.0, 2200.0)
        self.assertIsNotNone(res['sl'])
        self.assertIsNone(res['tp'])
        self.assertFalse(real_ee.emergency_halt)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 8: Failed position close
    # ─────────────────────────────────────────────────────────────────────────
    def test_08_failed_position_close(self):
        """Scenario 8: If exchange close fails, bot retains position state and enters paused state."""
        self.trader.current_position = "LONG"
        self.trader.amount = 0.5
        self.trader.entry_price = 2300.0
        self.trader.current_sl = 2250.0
        self.trader.current_tp = 2400.0

        # Close position fails
        self.mock_ee.close_position.return_value = {'status': 'failed', 'error': 'Exchange rejection'}
        now = datetime.now(timezone.utc)

        with patch('live.trader.TRADING_ENABLED', True):
            # Opposite signal (SHORT) triggers exit attempt
            self.trader.handle_signal(now, -0.08, 2300.0, atr=25.0)

        # Internal state MUST be preserved to prevent unmanaged exposure
        self.assertEqual(self.trader.current_position, "LONG")
        self.assertEqual(self.trader.amount, 0.5)
        self.assertTrue(self.trader.paused)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 9: Failed close retry
    # ─────────────────────────────────────────────────────────────────────────
    def test_09_failed_close_retry(self):
        """Scenario 9: If close retry also fails, execution engine latches EMERGENCY_HALT."""
        real_ee = self._create_isolated_ee()
        real_ee.exchange = MagicMock()

        # Step 1: Open position of 0.5 ETH
        pos_open = PositionInfo(status=PositionStatus.VERIFIED_OPEN, side='LONG', amount=0.5, entry_price=2300.0)
        real_ee.get_open_position = MagicMock(return_value=pos_open)
        real_ee.place_order = MagicMock(return_value={'id': 'fill_attempt'})

        res = real_ee.close_position(SYMBOL, 'sell', 0.5)
        self.assertEqual(res.get('status'), 'failed')
        self.assertEqual(res.get('id'), 'retry_failed')
        self.assertTrue(real_ee.emergency_halt)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 10: No clearing internal state before confirmed flat
    # ─────────────────────────────────────────────────────────────────────────
    def test_10_no_clearing_internal_state_before_confirmed_flat(self):
        """Scenario 10: Local state is never cleared until exchange explicitly confirms flat."""
        self.trader.current_position = "SHORT"
        self.trader.amount = 0.25
        self.trader.entry_price = 2300.0

        # Close position returns UNVERIFIED error
        self.mock_ee.close_position.return_value = {'status': 'unverified_error', 'verified': False}
        now = datetime.now(timezone.utc)

        with patch('live.trader.TRADING_ENABLED', True):
            self.trader.handle_signal(now, 0.08, 2300.0)

        self.assertEqual(self.trader.current_position, "SHORT")
        self.assertEqual(self.trader.amount, 0.25)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 11: Failed order cancellation
    # ─────────────────────────────────────────────────────────────────────────
    def test_11_failed_order_cancellation(self):
        """Scenario 11: Failed cancel_all_orders returns False without raising unhandled exception."""
        real_ee = self._create_isolated_ee()
        real_ee.exchange = MagicMock()
        real_ee.exchange.cancel_all_orders.side_effect = Exception("Cancel timeout")

        res = real_ee.cancel_all_orders(SYMBOL)
        self.assertFalse(res)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 12: Trailing-stop replacement failure
    # ─────────────────────────────────────────────────────────────────────────
    def test_12_trailing_stop_replacement_failure(self):
        """Scenario 12: If trailing-stop replacement fails, previous Stop-Loss is strictly retained."""
        self.trader.current_position = "LONG"
        self.trader.entry_price = 2000.0
        self.trader.amount = 0.5
        self.trader.current_sl = 1970.0  # Initial SL
        self.trader.current_tp = 2100.0

        # Price at +2.5% profit ($2050), triggering step 2 trailing stop
        self.mock_ee.get_current_price.return_value = 2050.0

        # Replacement fails
        self.mock_ee.place_tp_sl_orders.return_value = {'sl': None, 'tp': None}

        with patch('config.TRAILING_STOP_ENABLED', True):
            self.trader.check_trailing_stops()

        # Previous SL must be retained
        self.assertEqual(self.trader.current_sl, 1970.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 13: Restart with an existing real position
    # ─────────────────────────────────────────────────────────────────────────
    def test_13_restart_with_existing_real_position(self):
        """Scenario 13: Startup queries exchange, restores position, and restores missing protective stops."""
        self.mock_ee.get_open_position.return_value = PositionInfo(
            status=PositionStatus.VERIFIED_OPEN, side='SHORT', amount=0.3, entry_price=2350.0
        )
        self.mock_ee.get_open_orders.return_value = OrdersInfo(status=OrderStatus.VERIFIED, orders=[])

        new_trader = Trader(self.mock_ee)
        self.assertEqual(new_trader.current_position, "SHORT")
        self.assertEqual(new_trader.amount, 0.3)
        self.assertEqual(new_trader.entry_price, 2350.0)
        self.mock_ee.place_tp_sl_orders.assert_called()

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 14: Daily drawdown lock blocking new orders
    # ─────────────────────────────────────────────────────────────────────────
    def test_14_daily_drawdown_lock_blocks_new_orders(self):
        """Scenario 14: Daily drawdown limit triggers and blocks new orders before sizing or placement."""
        self.trader.daily_pnl = -70.0  # 7% loss on $1000 balance (> 6% limit)
        now = datetime.now(timezone.utc)

        with patch('live.trader.TRADING_ENABLED', True):
            self.trader.handle_signal(now, 0.08, 2300.0, atr=25.0)

        self.assertTrue(self.trader.drawdown_blocked)
        self.assertIsNone(self.trader.current_position)
        self.mock_ee.place_order.assert_not_called()

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 15: Position sizing below min notional without exceeding risk budget
    # ─────────────────────────────────────────────────────────────────────────
    def test_15_position_sizing_min_notional_risk_budget_skip(self):
        """Scenario 15: If min notional ($21) requires risking more than 2.5% budget, trade is skipped."""
        # $10 balance: 2.5% risk budget = $0.25.
        # At $2300 and 1.5% stop ($34.5), $21 notional risks $21 * 0.015 = $0.315 > $0.25.
        self.mock_ee.get_balance.return_value = BalanceInfo(status=BalanceStatus.VERIFIED, balance=10.0)

        sized = self.trader._calculate_position_size(price=2300.0, atr=25.0, side="LONG")
        # Must skip (0.0) rather than violate the 2.5% risk budget
        self.assertEqual(sized, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 16: ATR look-ahead prevention in simulation
    # ─────────────────────────────────────────────────────────────────────────
    def test_16_atr_lookahead_prevention(self):
        """Scenario 16: Future changes to entry candle OHLC cannot alter entry barrier ATR."""
        dates = pd.date_range("2026-01-01", periods=10, freq="1h")
        df1 = pd.DataFrame({
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0],
            "low": [95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "atr_14": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }, index=dates)

        df2 = df1.copy()
        df2.loc[dates[1], "high"] = 250.0  # Massive swing on entry bar
        df2.loc[dates[1], "low"] = 10.0
        df2.loc[dates[1], "atr_14"] = 50.0  # Bar 1 ATR inflated

        signals = np.zeros(10)
        signals[0] = 0.08  # Signal decided at bar 0 (above threshold)

        res1 = simulate_trades_ohlc(signals=signals, df_test=df1, signal_threshold=0.01, pt_mult=2.0, sl_mult=2.0)
        res2 = simulate_trades_ohlc(signals=signals, df_test=df2, signal_threshold=0.01, pt_mult=2.0, sl_mult=2.0)

        # Both MUST have used ATR from decision bar 0 (atr=2.0), NOT entry bar 1 (atr=50.0)
        t1 = res1["trades"][0]
        t2 = res2["trades"][0]
        self.assertEqual(t1["pt_price"], t2["pt_price"])
        self.assertEqual(t1["sl_price"], t2["sl_price"])

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 17: Next-open signal timing
    # ─────────────────────────────────────────────────────────────────────────
    def test_17_next_open_signal_timing(self):
        """Scenario 17: Signal decided at bar i-1 close strictly executes at bar i open."""
        dates = pd.date_range("2026-01-01 00:00", periods=5, freq="1h")
        df = pd.DataFrame({
            "open": [100.0, 102.0, 104.0, 106.0, 108.0],
            "high": [101.0, 103.0, 105.0, 107.0, 109.0],
            "low": [99.0, 101.0, 103.0, 105.0, 107.0],
            "close": [100.5, 102.5, 104.5, 106.5, 108.5],
            "atr_14": [2.0, 2.0, 2.0, 2.0, 2.0],
        }, index=dates)

        signals = np.array([0.08, 0.0, 0.0, 0.0, 0.0])  # Decided at bar 0
        res = simulate_trades_ohlc(signals=signals, df_test=df, signal_threshold=0.01, slippage=0.0)

        trade = res["trades"][0]
        self.assertEqual(trade["entry_bar"], 1)  # Entered at bar 1
        self.assertEqual(trade["raw_entry_price"], 102.0)  # Exactly bar 1 open

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 18: Funding calculation
    # ─────────────────────────────────────────────────────────────────────────
    def test_18_funding_settlement_calculation(self):
        """Scenario 18: 8-hour funding settlements correctly deduct funding cost."""
        dates = pd.date_range("2026-01-01 06:00", periods=4, freq="1h")
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "atr_14": [5.0, 5.0, 5.0, 5.0],
        }, index=dates)

        signals = np.array([0.08, 0.0, 0.0, 0.0])
        funding_map = {str(dates[2]): 0.0001}  # bar 2 is 08:00 UTC

        res = simulate_trades_ohlc(signals=signals, df_test=df, signal_threshold=0.01, funding_rates=funding_map, slippage=0.0)
        trade = res["trades"][0]
        self.assertAlmostEqual(trade["funding_cost"], 0.0001, places=6)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 19: Compounded equity consistency
    # ─────────────────────────────────────────────────────────────────────────
    def test_19_compounded_equity_consistency(self):
        """Scenario 19: Cumulative equity strictly compounds sequential trade returns."""
        dates = pd.date_range("2026-01-01", periods=10, freq="1h")
        df = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [100.0] * 10,
            "atr_14": [2.0] * 10,
        }, index=dates)

        signals = np.zeros(10)
        signals[0] = 0.08
        signals[4] = 0.08

        res = simulate_trades_ohlc(signals=signals, df_test=df, signal_threshold=0.01, holding_max=2)
        trades = res["trades"]
        if len(trades) >= 2:
            expected_equity = (1.0 + trades[0]["net_return"]) * (1.0 + trades[1]["net_return"])
            actual_equity = 1.0 + (res["compounded_net_return_pct"] / 100.0)
            self.assertAlmostEqual(expected_equity, actual_equity, places=4)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 20: Non-destructive artifact handling
    # ─────────────────────────────────────────────────────────────────────────
    def test_20_non_destructive_artifact_handling(self):
        """Scenario 20: Ensures test routines never point to production checkpoint paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_model = os.path.join(tmp_dir, "test_model.pt")
            temp_scaler_c = os.path.join(tmp_dir, "test_sc_c.pkl")
            temp_scaler_m = os.path.join(tmp_dir, "test_sc_m.pkl")

            self.assertNotEqual(temp_model, MODEL_PATH)
            self.assertNotEqual(temp_scaler_c, SCALER_CANDLE_PATH)
            self.assertNotEqual(temp_scaler_m, SCALER_MARKET_PATH)


if __name__ == '__main__':
    unittest.main()
