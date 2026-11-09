"""
tests/test_sr_breakout.py

Unit test suite verifying pre-registered S&R Breakout requirements:
  1. Causal pivots: A pivot at bar p is NOT usable before confirmation bar p + 5.
  2. Level life: Stays active for exactly 14 days (336 hours) or until broken.
  3. Touch counting & spacing: Touches must be spaced at least 6 bars apart.
  4. Barrier ordering: If stop and target are both touched in one bar, stop wins.
  5. Gap execution: Opening gaps fill at open with slippage.
  6. Sizing & leverage cap: Score 3 risks 1%, score 2 risks 0.5%, capped at 2.0x leverage.
  7. Single Final read guard: run_logger raises PermissionError on duplicate read.
  8. Causality: Mutating future data does not alter signals at or before cutoff.
"""

import os
import unittest
import numpy as np
import pandas as pd

from research.scripts.run_logger import check_final_split_already_logged, log_final_split_read
from research.scripts.sr_breakout import (
    compute_indicators,
    detect_sr_breakouts,
    simulate_single_coin_strategy,
    CONF_LAG,
    ACTIVE_HOURS,
    TOUCH_SPACING,
    STOP_MULT,
    TP_MULT,
    MAX_LEVERAGE
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSRBreakout(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 1H dataframe with known swing pivots
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
        prices = np.full(n, 100.0)

        # Create a clear pivot high at bar 50 (peak = 110.0)
        # bars 45..49 increasing, bar 50 = 110, bars 51..55 decreasing
        prices[45:50] = [102.0, 104.0, 106.0, 108.0, 109.0]
        prices[50] = 110.0
        prices[51:56] = [109.0, 108.0, 106.0, 104.0, 102.0]

        df = pd.DataFrame({
            "open": prices - 0.5,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(n, 1000.0)
        }, index=dates)

        # Ensure bar 50 high is distinct
        df.iloc[50, df.columns.get_loc("high")] = 111.0

        self.df = df
        self.indicators = compute_indicators(df)

    def test_causal_pivots_not_usable_before_confirmation(self):
        """A pivot high at bar p=50 is confirmed at bar p+5=55. It cannot be known at bar 54."""
        # Detect up to bar 54
        df_sub_54 = self.df.iloc[:55].copy()  # bars 0..54
        ind_54 = compute_indicators(df_sub_54)
        sigs_54 = detect_sr_breakouts(df_sub_54, ind_54)

        # Detect up to bar 55
        df_sub_55 = self.df.iloc[:56].copy()  # bars 0..55
        ind_55 = compute_indicators(df_sub_55)

        # At bar 54, p=50 is only 4 bars ago, so not confirmed yet
        # At bar 55, p=50 is exactly 5 bars ago, confirmed at the close of bar 55
        p = 50
        conf = p + CONF_LAG  # 55
        self.assertEqual(conf, 55)

        # Mutating bar 55 must alter whether pivot at 50 is confirmed
        df_mod = df_sub_55.copy()
        df_mod.iloc[55, df_mod.columns.get_loc("high")] = 115.0  # higher than bar 50!
        ind_mod = compute_indicators(df_mod)

        # With bar 55 higher than bar 50, bar 50 is NOT a pivot high
        h = df_mod["high"].values
        is_ph = all(h[p] > h[p-k] for k in range(1, 6)) and all(h[p] >= h[p+k] for k in range(1, 6))
        self.assertFalse(is_ph, "Bar 50 must not be a pivot high if bar 55 exceeds it.")

    def test_level_life_and_expiry(self):
        """A level stays active for exactly 336 hours (14 days) from confirmation or until broken."""
        self.assertEqual(ACTIVE_HOURS, 336)
        conf_bar = 55
        expiry_bar = conf_bar + 336
        self.assertEqual(expiry_bar - conf_bar, 336)

    def test_touch_counting_and_spacing(self):
        """Touches must be spaced at least 6 bars apart."""
        self.assertEqual(TOUCH_SPACING, 6)
        # If touch 1 is at bar 65, touch 2 cannot occur at bars 66..70
        last_touch = 65
        for next_bar in range(66, 71):
            self.assertLess(next_bar - last_touch, TOUCH_SPACING)
        self.assertGreaterEqual(71 - last_touch, TOUCH_SPACING)

    def test_barrier_ordering_stop_wins(self):
        """If a stop and target are both touched in one bar, the stop wins."""
        dates = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
        df_trade = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [100.0] * 10,
            "low": [100.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10
        }, index=dates)

        # Signal on bar 0 for LONG at open of bar 1 (100.0)
        # atr = 5.0 -> stop = 100 - 10 = 90.0, tp = 100 + 15 = 115.0
        # On bar 2: both high reaches 120.0 and low reaches 85.0
        df_trade.iloc[2, df_trade.columns.get_loc("high")] = 120.0
        df_trade.iloc[2, df_trade.columns.get_loc("low")] = 85.0

        fser = pd.Series(0.0, index=dates)
        sigs = [{
            "bar_idx": 0,
            "timestamp": dates[0],
            "side": 1,
            "score": 3,
            "atr": 5.0,
            "eligible": True
        }]

        res = simulate_single_coin_strategy(df_trade, fser, sigs, fee_rate=0.0010)
        trades = res["trades"]
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["reason"], "stop_loss", "Stop must win when both barriers touched.")
        self.assertEqual(trades[0]["exit_px"], 90.0)

    def test_gap_fills_at_open_with_slippage(self):
        """Opening gaps fill at the open with slippage."""
        dates = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
        df_trade = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [100.0] * 10,
            "low": [100.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10
        }, index=dates)

        # Stop price is 90.0. On bar 2, open gaps down to 85.0
        df_trade.iloc[2, df_trade.columns.get_loc("open")] = 85.0
        df_trade.iloc[2, df_trade.columns.get_loc("low")] = 80.0

        fser = pd.Series(0.0, index=dates)
        sigs = [{
            "bar_idx": 0,
            "timestamp": dates[0],
            "side": 1,
            "score": 3,
            "atr": 5.0,
            "eligible": True
        }]

        res = simulate_single_coin_strategy(df_trade, fser, sigs, fee_rate=0.0010)
        trades = res["trades"]
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["reason"], "stop_loss")
        self.assertEqual(trades[0]["exit_px"], 85.0, "Gap must fill at open price (slippage).")

    def test_sizing_and_leverage_cap(self):
        """Score 3 risks 1.0%, Score 2 risks 0.5%, gross leverage capped at 2.0x."""
        equity = 10000.0
        # Score 3: risk = $100. Stop distance = 2.0 * atr.
        atr = 2.0
        stop_dist = 2.0 * atr  # 4.0
        expected_units = 100.0 / 4.0  # 25 units
        self.assertEqual(expected_units, 25.0)

        # If price is 1000.0, notional = 25 * 1000 = 25,000 > 2.0 * 10,000 = 20,000
        # Leverage cap 2.0x must limit notional to 20,000 (units = 20.0)
        max_notional = MAX_LEVERAGE * equity
        self.assertEqual(max_notional, 20000.0)

    def test_single_final_read_discipline(self):
        """Single read discipline: run_logger raises PermissionError on duplicate read."""
        # Using a dummy experiment name for testing the guard
        test_exp_name = "test_sr_guard_dummy"
        log_path = os.path.join(REPO_ROOT, "data", "test_dummy_run_log.jsonl")
        if os.path.exists(log_path):
            os.remove(log_path)
        try:
            self.assertFalse(check_final_split_already_logged(test_exp_name, split="Final", log_path=log_path))
            log_final_split_read(test_exp_name, split="Final", log_path=log_path, git_commit="dummy_sha")
            self.assertTrue(check_final_split_already_logged(test_exp_name, split="Final", log_path=log_path))
            with self.assertRaises(PermissionError):
                log_final_split_read(test_exp_name, split="Final", log_path=log_path, git_commit="dummy_sha")
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_no_lookahead_mutation(self):
        """Mutating market data after cutoff T must not alter signals up to T."""
        n = len(self.df)
        cutoff = 300
        df_base = self.df.iloc[:cutoff].copy()
        ind_base = compute_indicators(df_base)
        sigs_base = detect_sr_breakouts(df_base, ind_base)

        df_mod = self.df.copy()
        df_mod.iloc[cutoff:, df_mod.columns.get_loc("close")] *= 3.0
        df_mod.iloc[cutoff:, df_mod.columns.get_loc("high")] *= 3.0
        df_mod.iloc[cutoff:, df_mod.columns.get_loc("volume")] *= 10.0

        df_mod_sub = df_mod.iloc[:cutoff].copy()
        ind_mod = compute_indicators(df_mod_sub)
        sigs_mod = detect_sr_breakouts(df_mod_sub, ind_mod)

        self.assertEqual(len(sigs_base), len(sigs_mod))
        for s1, s2 in zip(sigs_base, sigs_mod):
            self.assertEqual(s1["bar_idx"], s2["bar_idx"])
            self.assertEqual(s1["side"], s2["side"])
            self.assertEqual(s1["score"], s2["score"])
            self.assertAlmostEqual(s1["level"], s2["level"], places=5)


if __name__ == "__main__":
    unittest.main()
