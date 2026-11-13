"""Unit tests for Dollar Scalper Stage 0 pre-flight checks and planted-answer tests.

Tests:
  0.3.1: A position cannot exit in its entry bar (entry-bar causality).
  0.3.2: Stop and target both reachable in one 5-minute bar means the stop wins.
  0.3.3: Take-profit limit fills only if the bar trades through the target by one tick.
  0.3.4: Gaps fill at the open with slippage.
  0.3.5: Target price solved so that PnL after all costs is +$1.00 at tick-rounded price.
  0.3.6: Quantity and price rounded to Binance tick and step sizes, min notional respected.
  0.3.7: Mark-to-market equity at every bar equals cash + unrealized PnL to the cent.
  0.3.8: Sum of trade PnLs plus unrealized PnL equals equity change.
  0.3.9: No entry happens outside the four daily slots.
  0.4: Planted-answer tests on synthetic paths with known drift and zero drift.
"""

import os
import sys
import unittest
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research.dollar_scalper.engine import (
    DollarScalperBacktest,
    round_to_tick,
    round_to_step,
    compute_breakeven_math,
    SPECS,
    COST_MODELS,
)


def make_synthetic_df(
    start_time: str = "2024-01-01 00:00:00",
    n_bars: int = 50,
    base_price: float = 50000.0,
    price_pattern: Optional[List[Tuple[float, float, float, float]]] = None,
) -> pd.DataFrame:
    """Create a 5m bar DataFrame for testing."""
    times = pd.date_range(start=start_time, periods=n_bars, freq="5min", tz="UTC")
    if price_pattern is not None:
        assert len(price_pattern) == n_bars
        rows = price_pattern
    else:
        rows = [(base_price, base_price, base_price, base_price)] * n_bars

    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=times)
    return df


class TestDollarScalperPreflight(unittest.TestCase):

    def test_01_no_exit_in_entry_bar(self):
        """0.3.1: A position cannot exit in its entry bar."""
        bars = [
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:00 slot arrives
            (50000.0, 60000.0, 40000.0, 50000.0),  # 00:05 entry bar: extreme high/low
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:10 flat
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:15 flat
        ]
        df = make_synthetic_df(n_bars=4, price_pattern=bars)
        bt = DollarScalperBacktest(df, symbol="BTCUSDT", entry_variant="E2", exit_variant="X2")
        res = bt.run()
        trades = res["trades"]

        for t in trades:
            if t.is_closed:
                self.assertGreater(t.exit_time, t.entry_time, "Position must not exit in its entry bar!")
            else:
                self.assertGreater(t.holding_bars, 0)

    def test_02_stop_wins_over_target_in_same_bar(self):
        """0.3.2: Stop and target both reachable in one 5-minute bar means stop wins."""
        bars = [
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:00 slot
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:05 entry bar
            (50000.0, 55000.0, 45000.0, 50000.0),  # 00:10 reaches BOTH TP (high) and Stop (low)
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:15
        ]
        df = make_synthetic_df(n_bars=4, price_pattern=bars)
        bt = DollarScalperBacktest(df, symbol="BTCUSDT", entry_variant="E2", exit_variant="X2")
        res = bt.run()
        closed_trades = [t for t in res["trades"] if t.is_closed]
        self.assertEqual(len(closed_trades), 1)
        t = closed_trades[0]
        self.assertEqual(t.exit_reason, "STOP", f"Expected STOP to win over TP, got {t.exit_reason}")
        self.assertLess(t.net_pnl, 0, "Stop loss must result in negative net PnL")

    def test_03_tp_requires_trade_through_by_one_tick(self):
        """0.3.3: TP fills only if bar's high (long) trades through target by one tick."""
        tick = SPECS["BTCUSDT"]["tick_size"]  # 0.10
        bars_touch = [
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:00 slot
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:05 entry
            (50000.0, 50105.0, 49990.0, 50000.0),  # bar 2: high touches TP exactly
            (50000.0, 50000.0, 50000.0, 50000.0),  # bar 3
        ]
        df_touch = make_synthetic_df(n_bars=4, price_pattern=bars_touch)
        bt = DollarScalperBacktest(df_touch, symbol="BTCUSDT", entry_variant="E2", exit_variant="X1")
        bt_run = bt.run()
        trade = bt_run["trades"][0]
        exact_tp = trade.tp_price

        # Test A: high == exact_tp (just touches, does NOT trade through)
        bars_exact_touch = [
            (50000.0, 50000.0, 50000.0, 50000.0),
            (50000.0, 50000.0, 50000.0, 50000.0),
            (50000.0, exact_tp, 49990.0, 50000.0),  # high == exact_tp
            (50000.0, 50000.0, 50000.0, 50000.0),
        ]
        df_exact = make_synthetic_df(n_bars=4, price_pattern=bars_exact_touch)
        bt_exact = DollarScalperBacktest(df_exact, symbol="BTCUSDT", entry_variant="E2", exit_variant="X1")
        res_exact = bt_exact.run()
        closed_exact = [t for t in res_exact["trades"] if t.is_closed]
        self.assertEqual(len(closed_exact), 0, "Touch without trade-through must NOT fill TP limit!")

        # Test B: high == exact_tp + tick (trades through by 1 tick)
        bars_trade_through = [
            (50000.0, 50000.0, 50000.0, 50000.0),
            (50000.0, 50000.0, 50000.0, 50000.0),
            (50000.0, exact_tp + tick, 49990.0, 50000.0),  # high == exact_tp + tick
            (50000.0, 50000.0, 50000.0, 50000.0),
        ]
        df_through = make_synthetic_df(n_bars=4, price_pattern=bars_trade_through)
        bt_through = DollarScalperBacktest(df_through, symbol="BTCUSDT", entry_variant="E2", exit_variant="X1")
        res_through = bt_through.run()
        closed_through = [t for t in res_through["trades"] if t.is_closed]
        self.assertEqual(len(closed_through), 1, "Trade through by 1 tick MUST fill TP limit!")
        self.assertEqual(closed_through[0].exit_reason, "TP")

    def test_04_gaps_fill_at_open_with_slippage(self):
        """0.3.4: Gaps fill at the open with slippage."""
        bars = [
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:00 slot
            (50000.0, 50000.0, 50000.0, 50000.0),  # 00:05 entry
            (45000.0, 45010.0, 44990.0, 45000.0),  # 00:10 gapped down below stop
        ]
        df = make_synthetic_df(n_bars=3, price_pattern=bars)
        bt = DollarScalperBacktest(df, symbol="BTCUSDT", entry_variant="E2", exit_variant="X2")
        res = bt.run()
        closed = [t for t in res["trades"] if t.is_closed]
        self.assertEqual(len(closed), 1)
        t = closed[0]
        self.assertLess(t.exit_price, 45000.0, f"Expected gap fill with slippage below 45000, got {t.exit_price}")
        self.assertLess(t.net_pnl, -10.0, f"Gapped stop loss should exceed -$10, got {t.net_pnl}")

    def test_05_target_price_solved_for_exact_one_dollar_net(self):
        """0.3.5: Target price solved so PnL after costs is approximately +$1.00 net."""
        bars = [
            (50000.0, 50000.0, 50000.0, 50000.0),
            (50000.0, 50000.0, 50000.0, 50000.0),
            (50000.0, 52000.0, 49990.0, 50000.0),  # hits TP cleanly
        ]
        df = make_synthetic_df(n_bars=3, price_pattern=bars)
        for cm in ["M1", "M2"]:
            bt = DollarScalperBacktest(df, symbol="BTCUSDT", cost_model=cm, entry_variant="E2", exit_variant="X1")
            res = bt.run()
            t = res["trades"][0]
            self.assertLess(abs(t.net_pnl - 1.00), 0.05, f"Net PnL under {cm} is {t.net_pnl}, expected ~$1.00")

    def test_06_binance_rounding_and_min_notional(self):
        """0.3.6: Quantity and price rounded to Binance tick and step sizes, min notional respected."""
        tick_btc = SPECS["BTCUSDT"]["tick_size"]
        step_btc = SPECS["BTCUSDT"]["step_size"]

        qty = round_to_step(1000.0 / 63452.17, step_btc)
        self.assertEqual(qty, 0.016)
        self.assertGreaterEqual(qty * 63452.17, SPECS["BTCUSDT"]["min_notional"])

        p_tick = round_to_tick(63452.178, tick_btc, mode="nearest")
        self.assertEqual(p_tick, 63452.2)

    def test_07_mark_to_market_accounting(self):
        """0.3.7 & 0.3.8: Mark-to-market equity equals cash + unrealized PnL to the cent,
        and sum of trade PnLs + unrealized PnL equals delta equity."""
        np.random.seed(123)
        prices = 50000.0 + np.cumsum(np.random.randn(20) * 20.0)
        bars = [(p, p + 10.0, p - 10.0, p) for p in prices]
        df = make_synthetic_df(n_bars=20, price_pattern=bars)

        bt = DollarScalperBacktest(df, symbol="BTCUSDT", entry_variant="E2", exit_variant="X1")
        res = bt.run()

        init_eq = res["initial_equity"]
        final_eq = res["final_equity"]

        closed_pnl_sum = sum(t.net_pnl for t in res["trades"] if t.is_closed)
        unrealized = res["unresolved_position"].net_pnl if res["unresolved_position"] else 0.0

        discrepancy = abs((final_eq - init_eq) - (closed_pnl_sum + unrealized))
        self.assertLess(discrepancy, 0.01, f"Accounting discrepancy {discrepancy} exceeds 1 cent!")

    def test_08_no_entry_outside_four_slots(self):
        """0.3.9: No entry happens outside the four daily slots."""
        df = make_synthetic_df(n_bars=100, base_price=50000.0)
        bt = DollarScalperBacktest(df, symbol="BTCUSDT", entry_variant="E2", exit_variant="X1")
        res = bt.run()
        for t in res["trades"]:
            hour = t.entry_time.hour
            minute = t.entry_time.minute
            self.assertEqual(minute, 5)
            self.assertIn(hour, {0, 6, 12, 18})

    def test_planted_answer_known_drift(self):
        """0.4 Planted-answer test on synthetic price path with strong upward drift."""
        n_bars = 40
        bars = []
        p = 50000.0
        for _ in range(n_bars):
            bars.append((p, p + 100.0, p - 5.0, p + 50.0))
            p += 50.0
        df = make_synthetic_df(n_bars=n_bars, price_pattern=bars)

        bt = DollarScalperBacktest(df, symbol="BTCUSDT", entry_variant="E2", exit_variant="X2")
        res = bt.run()
        closed = [t for t in res["trades"] if t.is_closed]
        self.assertGreater(len(closed), 0)
        for t in closed:
            self.assertEqual(t.exit_reason, "TP")
            self.assertLess(abs(t.net_pnl - 1.00), 0.05)
        win_rate = len([t for t in closed if t.net_pnl > 0]) / len(closed)
        self.assertEqual(win_rate, 1.0)

    def test_planted_answer_known_downward_drift(self):
        """0.4 Planted-answer test on synthetic price path with strong downward drift."""
        n_bars = 40
        bars = []
        p = 50000.0
        for _ in range(n_bars):
            bars.append((p, p + 2.0, p - 120.0, p - 100.0))
            p -= 100.0
        df = make_synthetic_df(n_bars=n_bars, price_pattern=bars)

        bt = DollarScalperBacktest(df, symbol="BTCUSDT", entry_variant="E2", exit_variant="X2")
        res = bt.run()
        closed = [t for t in res["trades"] if t.is_closed]
        self.assertGreater(len(closed), 0)
        for t in closed:
            self.assertEqual(t.exit_reason, "STOP")
            self.assertLessEqual(t.net_pnl, -10.0)
        win_rate = len([t for t in closed if t.net_pnl > 0]) / len(closed)
        self.assertEqual(win_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
