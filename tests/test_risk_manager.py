import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from live.trader import Trader
from config import SYMBOL, DAILY_DRAWDOWN_LIMIT

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.mock_execution = MagicMock()
        self.mock_execution.get_balance.return_value = 1000.0
        self.mock_execution.close_position.return_value = {'id': 'close_123', 'status': 'closed', 'verified': True}
        self.mock_execution.cancel_all_orders.return_value = True
        
        # Force TRADING_ENABLED to True for tests
        patcher = patch('live.trader.TRADING_ENABLED', True)
        patcher.start()
        self.addCleanup(patcher.stop)
        
        self.trader = Trader(self.mock_execution)
        # Ensure reset
        self.trader.daily_pnl = 0.0
        self.trader.drawdown_blocked = False

    def test_drawdown_limit_long(self):
        """Test that a large loss triggers the drawdown hard cap for LONG positions."""
        # 7% loss on $1000 balance with 6% limit
        self.trader.current_position = "LONG"
        self.trader.entry_price = 2000.0
        self.trader.amount = 1.0  # $2000 position
        
        # Price drops to $1970 -> $30 loss (3%)
        current_price = 1930.0
        # Use a very strong prediction to avoid SIGNAL WEAK exit
        self.trader.handle_signal(datetime.now(), 0.05, current_price)
        
        # Verify position was closed
        self.assertTrue(self.trader.drawdown_blocked)
        self.assertIsNone(self.trader.current_position)
        self.mock_execution.close_position.assert_called()

    def test_leverage_cap_scaling(self):
        """Test that position size is scaled down to LEVERAGE cap (3x)."""
        from config import LEVERAGE
        balance = 1000.0
        self.mock_execution.get_balance.return_value = balance
        
        # Attempt to size for 10x leverage
        price = 2000.0
        # amount = (risk_usd / sl_dist) -> let's say risk is high
        # We'll just test the scaling logic directly
        
        # Test 10x leverage request
        requested_amount = (balance * 10) / price # 5 ETH ($10,000)
        
        sized_amount = self.trader._calculate_position_size(price, atr=None, side="LONG")
        
        # sized_amount * price should be <= balance * LEVERAGE
        self.assertLessEqual(sized_amount * price, balance * LEVERAGE + 0.1)

    def test_daily_reset(self):
        """Test that drawdown block resets on a new day."""
        self.trader.drawdown_blocked = True
        self.trader.last_pnl_reset = datetime(2020, 1, 1).date()
        
        # Next signal handles reset
        self.trader.handle_signal(datetime.now(), 0.0, 2000.0)
        
        self.assertFalse(self.trader.drawdown_blocked)
        self.assertEqual(self.trader.daily_pnl, 0.0)

if __name__ == '__main__':
    unittest.main()
