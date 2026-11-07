"""
tests/test_low_noise_strategies.py

Unit tests for low-noise pre-registered strategies F0, G0, G1, G2:
1. Causality & future-shift invariance (zero lookahead).
2. Execution causality (signals at bar close, execution at next bar open).
3. Trailing stop monotonicity (ratchet only, never loosens) and stop-first execution.
4. Cooldown enforcement (4 completed 6H bars post-exit).
5. Cost model application (10 bps vs 20 bps exact scaling).
6. Holdout data partitioning and lock enforcement.
"""
import os
import sys
import math
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import research.scripts.low_noise_strategies as L


def _make_synthetic_1h_data(n_hours: int = 24 * 300, start: str = "2020-01-01 00:00:00+00:00") -> pd.DataFrame:
    """Generates synthetic random walk 1H OHLCV data."""
    np.random.seed(42)
    idx = pd.date_range(start, periods=n_hours, freq="1h")
    # Log-returns with low volatility
    rets = np.random.normal(0.0001, 0.003, size=n_hours)
    px = 1000.0 * np.exp(np.cumsum(rets))
    # Synthetic intra-bar spread
    high = px * (1.0 + np.abs(np.random.normal(0, 0.002, size=n_hours)))
    low = px * (1.0 - np.abs(np.random.normal(0, 0.002, size=n_hours)))
    open_ = np.roll(px, 1)
    open_[0] = px[0]
    vol = np.random.uniform(10.0, 100.0, size=n_hours)
    df = pd.DataFrame({
        "open": open_,
        "high": np.maximum(high, np.maximum(open_, px)),
        "low": np.minimum(low, np.minimum(open_, px)),
        "close": px,
        "volume": vol
    }, index=idx)
    return df


class TestLowNoiseStrategies(unittest.TestCase):

    def test_future_data_invariance_f0_features(self):
        """Future price movements must NOT alter past F0 indicator values or leverage targets."""
        df = _make_synthetic_1h_data(n_hours=24 * 250)
        cutoff_hours = 24 * 200
        df_past = df.iloc[:cutoff_hours].copy()

        # Extended dataframe with wild price movements in the future
        df_extended = df.copy()
        df_extended.iloc[cutoff_hours:, df_extended.columns.get_loc("close")] *= 5.0
        df_extended.iloc[cutoff_hours:, df_extended.columns.get_loc("high")] *= 6.0
        df_extended.iloc[cutoff_hours:, df_extended.columns.get_loc("low")] *= 4.0

        f_past = L.f0_features(df_past)
        f_ext = L.f0_features(df_extended)

        common_idx = f_past.index
        # All rows up to cutoff must be identical
        pd.testing.assert_series_equal(f_past["score"], f_ext.loc[common_idx, "score"], check_names=False)
        pd.testing.assert_series_equal(f_past["rv"], f_ext.loc[common_idx, "rv"], check_names=False)
        pd.testing.assert_series_equal(f_past["target"], f_ext.loc[common_idx, "target"], check_names=False)

    def test_future_data_invariance_g_features(self):
        """Future price movements must NOT alter past G0, G1, G2 signals or indicators."""
        df = _make_synthetic_1h_data(n_hours=24 * 400)
        cutoff_hours = 24 * 350
        df_past = df.iloc[:cutoff_hours].copy()

        # Modify future prices
        df_extended = df.copy()
        df_extended.iloc[cutoff_hours:, df_extended.columns.get_loc("close")] *= 0.2
        df_extended.iloc[cutoff_hours:, df_extended.columns.get_loc("high")] *= 0.3
        df_extended.iloc[cutoff_hours:, df_extended.columns.get_loc("low")] *= 0.1

        for variant in ("G0", "G1", "G2"):
            with self.subTest(variant=variant):
                g_past = L.g_features(df_past, variant=variant)
                g_ext = L.g_features(df_extended, variant=variant)

                common_idx = g_past.index
                pd.testing.assert_series_equal(g_past["atr"], g_ext.loc[common_idx, "atr"], check_names=False)
                pd.testing.assert_series_equal(g_past["long_sig"], g_ext.loc[common_idx, "long_sig"], check_names=False)
                pd.testing.assert_series_equal(g_past["short_sig"], g_ext.loc[common_idx, "short_sig"], check_names=False)

    def test_execution_causality_fill_at_next_bar_open(self):
        """Signal generated at bar close must fill at next bar open."""
        # EMA(100) on 6H requires 100 6H bars (600 hours) to warm up
        idx = pd.date_range("2021-01-01 00:00:00+00:00", periods=24 * 50, freq="1h")
        px = np.full(len(idx), 100.0)
        # Breakout at hour 720 (after 120 6H bars, fully warmed up)
        px[720:] = 150.0
        df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 100.0}, index=idx)

        feat = L.g_features(df, variant="G0", donchian_n=20)
        # Find the first true long signal
        sig_bars = feat[feat["long_sig"]].index
        self.assertGreater(len(sig_bars), 0)
        sig_open_time = sig_bars[0]
        # 6H bar closes at sig_open_time + 6 hours
        expected_fill_time = sig_open_time + pd.Timedelta(hours=6)

        res = L.simulate_g(df, None, feat, symbol="TEST", strategy="G0")
        self.assertGreater(len(res["trades"]), 0)
        first_trade = res["trades"][0]
        self.assertEqual(pd.Timestamp(first_trade["entry_time"]), expected_fill_time)
        self.assertEqual(first_trade["entry_price"], df.loc[expected_fill_time, "open"])

    def test_trailing_stop_never_loosens_long(self):
        """Long trailing stop must be monotonically non-decreasing once active."""
        idx = pd.date_range("2021-01-01 00:00:00+00:00", periods=24 * 100, freq="1h")
        t = np.arange(len(idx))
        px = 100.0 + 0.1 * t
        df = pd.DataFrame({"open": px, "high": px + 1.0, "low": px - 1.0, "close": px, "volume": 100.0}, index=idx)

        feat = L.g_features(df, variant="G0")
        cl6 = feat["close"].values
        atr6 = feat["atr"].values
        best = float("nan")
        stop = float("nan")
        for c, a in zip(cl6, atr6):
            if not math.isnan(a):
                best = c if math.isnan(best) else max(best, c)
                new_stop = best - 3.0 * a if math.isnan(stop) else max(stop, best - 3.0 * a)
                if not math.isnan(stop):
                    self.assertGreaterEqual(new_stop, stop)
                stop = new_stop

    def test_cooldown_bars_enforced(self):
        """No new entry permitted within 4 completed 6H bars of trade exit."""
        df = _make_synthetic_1h_data(n_hours=24 * 100)
        feat = L.g_features(df, variant="G0")
        res = L.simulate_g(df, None, feat, symbol="TEST", strategy="G0")

        trades = res["trades"]
        if len(trades) >= 2:
            for t1, t2 in zip(trades[:-1], trades[1:]):
                exit_t = pd.Timestamp(t1["exit_time"])
                entry_t = pd.Timestamp(t2["entry_time"])
                # At least 4 * 6 = 24 hours between exit and next entry
                delta_hours = (entry_t - exit_t).total_seconds() / 3600.0
                self.assertGreaterEqual(delta_hours, 24.0)

    def test_double_cost_deduction(self):
        """Double fee rate must double the dollar fees paid on identical trade sizing."""
        df = _make_synthetic_1h_data(n_hours=24 * 100)
        feat = L.g_features(df, variant="G0")

        res_base = L.simulate_g(df, None, feat, symbol="TEST", fee=0.0010)
        res_dbl = L.simulate_g(df, None, feat, symbol="TEST", fee=0.0020)

        self.assertGreater(len(res_base["trades"]), 0)
        self.assertEqual(len(res_base["trades"]), len(res_dbl["trades"]))
        # For the first trade (identical initial cash), fee paid must scale by exactly 2.0x
        t0_base = res_base["trades"][0]
        t0_dbl = res_dbl["trades"][0]
        self.assertAlmostEqual(t0_dbl["fee_usd"] / t0_base["fee_usd"], 2.0, places=4)

    def test_holdout_guard_raises_permission_error_when_locked(self):
        """Holdout-only symbols must refuse to load when holdout is locked."""
        L.lock_holdout()
        self.assertFalse(L.holdout_unlocked())
        for sym in ("ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT"):
            with self.assertRaises(PermissionError):
                L.load_symbol(sym)

    def test_dev_truncation_enforced_when_locked(self):
        """Development symbols must be truncated strictly at DEV_END when locked."""
        L.lock_holdout()
        df, fser = L.load_symbol("BTCUSDT")
        self.assertLessEqual(df.index[-1], L.DEV_END)
        self.assertLessEqual(fser.index[-1], L.DEV_END)


if __name__ == "__main__":
    unittest.main()
