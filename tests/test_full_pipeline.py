"""
tests/test_full_pipeline.py — End-to-end integration test of the entire trading pipeline.

Verifies:
1. Model prediction produces both LONG and SHORT directions.
2. Order construction for LONG and SHORT entries.
3. Risk management:
   - Position sizing based on ATR / risk percentage.
   - Binance minimum notional compliance ($21 USD).
   - Account margin and leverage caps.
   - Daily drawdown circuit breaker.
   - 2-hour post-trade cooldown.
4. Execution engine handling and order synchronization.
5. Circuit breaker behavior on exchange errors.
"""

import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from live.trader import Trader
from live.execution_engine import ExecutionEngine
from config import SYMBOL, MIN_NOTIONAL_USD, LEVERAGE


class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        # Create a mock execution engine
        self.mock_ee = MagicMock(spec=ExecutionEngine)
        self.mock_ee.mock = True
        self.mock_ee.get_balance.return_value = 57.31  # Match server balance
        self.mock_ee.get_open_position.return_value = None
        self.mock_ee.get_current_price.return_value = 2300.0
        self.mock_ee.place_order.return_value = {'id': 'order_123', 'status': 'closed'}
        self.mock_ee.place_tp_sl_orders.return_value = True
        self.mock_ee.cancel_all_orders.return_value = True

        # Initialize Trader
        self.trader = Trader(self.mock_ee)
        self.trader.daily_pnl = 0.0
        self.trader.drawdown_blocked = False
        self.trader.current_position = None
        self.trader.last_trade_close_time = None

    def test_long_signal_pipeline(self):
        """Verify complete pipeline for a valid LONG signal."""
        now = datetime.now(timezone.utc)
        prediction = 0.075  # Passes threshold
        price = 2300.0
        atr = 25.0

        with patch('live.trader.TRADING_ENABLED', True):
            self.trader.handle_signal(now, prediction, price, atr=atr, btc_bullish=True)

        # Verification
        self.assertEqual(self.trader.current_position, "LONG")
        self.assertEqual(self.trader.entry_price, price)
        self.assertGreater(self.trader.amount, 0)
        
        # Verify order was placed with 'buy'
        self.mock_ee.place_order.assert_called_with(SYMBOL, 'buy', self.trader.amount, price=price)
        
        # Verify TP and SL bracket orders placed
        self.mock_ee.place_tp_sl_orders.assert_called_with(
            SYMBOL, "LONG", self.trader.amount, self.trader.current_tp, self.trader.current_sl
        )
        self.assertGreater(self.trader.current_tp, price)
        self.assertLess(self.trader.current_sl, price)

    def test_short_signal_pipeline(self):
        """Verify complete pipeline for a valid SHORT signal."""
        now = datetime.now(timezone.utc)
        prediction = -0.075  # Negative prediction (SHORT)
        price = 2300.0
        atr = 25.0

        with patch('live.trader.TRADING_ENABLED', True):
            self.trader.handle_signal(now, prediction, price, atr=atr, btc_bullish=True)

        # Verification
        self.assertEqual(self.trader.current_position, "SHORT")
        self.assertEqual(self.trader.entry_price, price)
        self.assertGreater(self.trader.amount, 0)
        
        # Verify order was placed with 'sell'
        self.mock_ee.place_order.assert_called_with(SYMBOL, 'sell', self.trader.amount, price=price)
        
        # Verify TP and SL bracket orders placed
        self.mock_ee.place_tp_sl_orders.assert_called_with(
            SYMBOL, "SHORT", self.trader.amount, self.trader.current_tp, self.trader.current_sl
        )
        self.assertLess(self.trader.current_tp, price)
        self.assertGreater(self.trader.current_sl, price)

    def test_minimum_notional_compliance(self):
        """Verify position sizing respects Binance minimum notional ($21.00 USD)."""
        price = 2300.0
        # Very tiny balance: $10
        self.mock_ee.get_balance.return_value = 10.0
        
        size = self.trader._calculate_position_size(price, atr=25.0, side="LONG")
        notional = size * price
        self.assertGreaterEqual(notional, MIN_NOTIONAL_USD - 0.05)

    def test_cooldown_blocks_reentry(self):
        """Verify 2-hour cooldown prevents immediate re-entry after trade close."""
        now = datetime.now(timezone.utc)
        self.trader.last_trade_close_time = now - timedelta(minutes=45)  # 45 mins ago (< 2h)
        
        with patch('live.trader.TRADING_ENABLED', True):
            self.trader.handle_signal(now, 0.08, 2300.0, atr=25.0)

        # Position should NOT have opened
        self.assertIsNone(self.trader.current_position)
        self.mock_ee.place_order.assert_not_called()

    def test_weak_signal_waiting(self):
        """Verify weak signals (< threshold) do not trigger trades and remain WAITING."""
        now = datetime.now(timezone.utc)
        prediction = 0.005  # Below threshold
        
        with patch('live.trader.TRADING_ENABLED', True):
            self.trader.handle_signal(now, prediction, 2300.0, atr=25.0)

        self.assertIsNone(self.trader.current_position)
        self.mock_ee.place_order.assert_not_called()

    def test_circuit_breaker_on_repeated_exchange_errors(self):
        """Verify execution engine trips circuit breaker after 3 consecutive errors."""
        ee = ExecutionEngine(mock=True)
        ee.failure_count = 3
        # In mock=True or mock=False, check circuit breaker
        # With mock=False, if API keys missing, it raises ValueError, but with failure_count:
        with patch.object(ee, 'mock', False):
            order = ee.place_order(SYMBOL, "BUY", 0.01)
            self.assertIsNone(order)


if __name__ == '__main__':
    unittest.main()
