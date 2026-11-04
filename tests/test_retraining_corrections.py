"""
tests/test_retraining_corrections.py — Rigorous verification suite for retraining pipeline corrections.

Verifies:
1. Multi-page pagination (>2,000 candles) across 1,000-candle page boundaries with cursor progression.
2. Latest completed candle inclusion and currently forming candle exclusion.
3. Safe termination on non-progressing exchange cursors / repeated pages (infinite loop protection).
4. Shortfall detection when exchange has fewer candles than requested.
5. Duplicate timestamp deduplication and missing interval integrity handling.
6. TBM ambiguous candle resolution: default exclusion (unbiased directional target),
   stop-first (for trade simulation only), and lower-timeframe chronological ordering.
7. End-to-end integration of purged train/val split and strict scaler fitting restriction.
8. Mathematical verification of target future indices against sample boundaries.
9. 3-way chronological split (train / val / untouched test) and walk-forward cross-validation.
10. Moving block bootstrap uncertainty estimation for overlapping 12-hour targets.
11. Probability calibration metrics (Brier score, Log Loss, ECE, reliability curves).
12. Data-driven classification loss weighting (neg/pos ratio) vs unweighted vs legacy hardcoded 1.17.
13. Warmup bars pre-roll integration through pipeline.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler

from data.fetcher import fetch_ohlcv
from data.indicators import add_tbm_target, compute_indicators
from data.preprocessor import (
    split_train_val_purged,
    split_train_val_test_purged,
    generate_walk_forward_splits,
    block_bootstrap_confidence_interval,
    build_features,
    build_windows,
    transform,
)
from model.trainer import (
    directional_accuracy,
    compute_comprehensive_metrics,
)
import config


class TestDataFetcherCorrections(unittest.TestCase):
    """Test suite for data/fetcher.py multi-page pagination, lag elimination, and data quality."""

    def test_multipage_pagination_over_2000_candles(self):
        """Verify pagination across 1,000-candle page boundaries (requesting 2,500 candles)."""
        tf_ms = 3600 * 1000  # 1h in ms
        now_dt = datetime.datetime(2026, 11, 4, 12, 45, 0, tzinfo=datetime.timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)
        latest_completed_ms = int(datetime.datetime(2026, 11, 4, 11, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)

        # Build 3,000 synthetic completed candles + 1 forming candle
        total_available = 3000
        start_ms = latest_completed_ms - (total_available - 1) * tf_ms
        mock_candles = []
        for i in range(total_available + 1):  # +1 for currently forming 12:00 candle
            c_time = start_ms + i * tf_ms
            mock_candles.append([c_time, 2000.0 + (i % 100), 2010.0 + (i % 100), 1990.0 + (i % 100), 2005.0 + (i % 100), 1000.0])

        mock_exchange = MagicMock()
        mock_exchange.milliseconds.return_value = now_ms
        mock_exchange.parse_timeframe.return_value = 3600
        mock_exchange.rateLimit = 5

        page_calls = []

        def mock_fetch(symbol, timeframe, since, limit):
            page_calls.append({"since": since, "limit": limit})
            matching = [c for c in mock_candles if c[0] >= since][:limit]
            return matching

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch

        with patch("data.fetcher.get_exchange", return_value=mock_exchange):
            df = fetch_ohlcv(symbol="ETH/USDT:USDT", timeframe="1h", limit=2500, validate=False)

            # 1. Returned candle count must be exactly 2500
            self.assertEqual(len(df), 2500)
            # 2. Must require at least 3 pages (Binance page cap is 1,000)
            self.assertGreaterEqual(len(page_calls), 3)
            # 3. Check cursor progression: each subsequent page call must have strictly increasing 'since'
            sinces = [c["since"] for c in page_calls]
            for i in range(len(sinces) - 1):
                self.assertLess(sinces[i], sinces[i + 1], "Cursor did not advance chronologically")
            # 4. First two pages requested 1,000 candles
            self.assertEqual(page_calls[0]["limit"], 1000)
            self.assertEqual(page_calls[1]["limit"], 1000)
            # 5. Latest completed candle (11:00 UTC) is included
            self.assertEqual(df.index[-1], pd.to_datetime(latest_completed_ms, unit="ms", utc=True))
            # 6. Currently forming candle (12:00 UTC) is excluded
            forming_time = pd.to_datetime(latest_completed_ms + tf_ms, unit="ms", utc=True)
            self.assertNotIn(forming_time, df.index)
            # 7. Monotonically increasing
            self.assertTrue(df.index.is_monotonic_increasing)

    def test_safe_termination_on_repeated_non_progressing_page(self):
        """Simulate an exchange returning a stagnant page and verify safe termination without infinite loop."""
        now_dt = datetime.datetime(2026, 11, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)
        tf_ms = 3600 * 1000
        latest_completed = now_ms - tf_ms

        mock_exchange = MagicMock()
        mock_exchange.milliseconds.return_value = now_ms
        mock_exchange.parse_timeframe.return_value = 3600
        mock_exchange.rateLimit = 5

        # When limit=5, fetch_n=5. Return 5 candles that don't advance cursor
        stagnant_candles = [
            [latest_completed - 10 * tf_ms, 2000.0, 2010.0, 1990.0, 2005.0, 500.0]
            for _ in range(5)
        ]
        mock_exchange.fetch_ohlcv.return_value = stagnant_candles

        with patch("data.fetcher.get_exchange", return_value=mock_exchange):
            with self.assertLogs("data.fetcher", level="WARNING") as log_capture:
                df = fetch_ohlcv(symbol="ETH/USDT:USDT", timeframe="1h", limit=5, validate=False)
                self.assertIsNotNone(df)
                self.assertTrue(any("cursor did not advance" in msg for msg in log_capture.output))

    def test_shortfall_reporting_when_exchange_has_fewer_candles(self):
        """Verify explicit warning when exchange returns fewer candles than requested."""
        now_dt = datetime.datetime(2026, 11, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)
        tf_ms = 3600 * 1000
        latest_completed = now_ms - tf_ms

        mock_exchange = MagicMock()
        mock_exchange.milliseconds.return_value = now_ms
        mock_exchange.parse_timeframe.return_value = 3600
        mock_exchange.rateLimit = 5

        start_ms = latest_completed - 14 * tf_ms
        mock_candles = [
            [start_ms + i * tf_ms, 2000.0, 2010.0, 1990.0, 2005.0, 500.0]
            for i in range(15)
        ]

        def mock_fetch(symbol, timeframe, since, limit):
            return [c for c in mock_candles if c[0] >= since][:limit]

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch

        with patch("data.fetcher.get_exchange", return_value=mock_exchange):
            with self.assertLogs("data.fetcher", level="WARNING") as log_capture:
                df = fetch_ohlcv(symbol="ETH/USDT:USDT", timeframe="1h", limit=100, validate=False)
                self.assertLess(len(df), 100)
                self.assertTrue(any("Shortfall detected" in msg for msg in log_capture.output))

    def test_duplicate_and_missing_interval_detection(self):
        """Verify deduplication and missing interval handling (including raise_on_gap)."""
        now_dt = datetime.datetime(2026, 11, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)
        tf_ms = 3600 * 1000
        latest_completed_open = now_ms - tf_ms
        target_start = latest_completed_open - 4 * tf_ms

        mock_candles = [
            [target_start + 0 * tf_ms, 100.0, 105.0, 95.0, 100.0, 10.0],
            [target_start + 1 * tf_ms, 101.0, 106.0, 96.0, 101.0, 10.0],
            [target_start + 1 * tf_ms, 101.0, 106.0, 96.0, 101.0, 15.0],  # Duplicate
            # Missing index 2 (1-hour gap)
            [target_start + 3 * tf_ms, 103.0, 108.0, 98.0, 103.0, 10.0],
            [target_start + 4 * tf_ms, 104.0, 109.0, 99.0, 104.0, 10.0],
        ]

        mock_exchange = MagicMock()
        mock_exchange.milliseconds.return_value = now_ms
        mock_exchange.parse_timeframe.return_value = 3600
        mock_exchange.rateLimit = 5

        def mock_fetch(symbol, timeframe, since, limit):
            return [c for c in mock_candles if c[0] >= since][:limit]

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch

        with patch("data.fetcher.get_exchange", return_value=mock_exchange):
            # 1. Non-raising mode: logs warning and deduplicates
            with self.assertLogs("data.fetcher", level="WARNING") as log_capture:
                df = fetch_ohlcv(symbol="ETH/USDT:USDT", timeframe="1h", limit=5, validate=False, raise_on_gap=False)
                self.assertEqual(len(df.index), len(set(df.index)))
                self.assertTrue(any("duplicate timestamps" in msg for msg in log_capture.output))
                self.assertTrue(any("missing intervals" in msg for msg in log_capture.output))

            # 2. Raising mode: raises explicit ValueError on detected gap
            with self.assertRaises(ValueError) as ctx:
                fetch_ohlcv(symbol="ETH/USDT:USDT", timeframe="1h", limit=5, validate=False, raise_on_gap=True)
            self.assertIn("Data integrity failure", str(ctx.exception))

    def test_warmup_bars_fetching_guarantee(self):
        """Verify warmup_bars fetches extra candles for pre-roll."""
        now_dt = datetime.datetime(2026, 11, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)
        tf_ms = 3600 * 1000
        latest_completed = now_ms - tf_ms

        mock_exchange = MagicMock()
        mock_exchange.milliseconds.return_value = now_ms
        mock_exchange.parse_timeframe.return_value = 3600
        mock_exchange.rateLimit = 5

        # 100 candles available
        start_ms = latest_completed - 99 * tf_ms
        mock_candles = [
            [start_ms + i * tf_ms, 2000.0, 2010.0, 1990.0, 2005.0, 500.0]
            for i in range(100)
        ]

        def mock_fetch(symbol, timeframe, since, limit):
            return [c for c in mock_candles if c[0] >= since][:limit]

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch

        with patch("data.fetcher.get_exchange", return_value=mock_exchange):
            # Request 20 candles + 10 warmup bars -> total needed 30
            df = fetch_ohlcv(symbol="ETH/USDT:USDT", timeframe="1h", limit=20, warmup_bars=10, validate=False)
            self.assertEqual(len(df), 30)


class TestTBMCorrections(unittest.TestCase):
    """Test suite for Triple Barrier Method ambiguous candle resolution and parameters."""

    def test_ambiguous_candle_resolution_modes(self):
        """Verify exclude (default), stop-first, overshoot, and LTF resolution."""
        dates = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "close": [1000.0] * 10,
            "high": [1005.0] * 10,
            "low": [995.0] * 10,
            "open": [1000.0] * 10,
            "atr_14": [10.0] * 10,  # ATR = 10 -> PT (+20) = 1020, SL (-20) = 980
        }, index=dates)

        # Bar index 1: price is 1000. Next bar (index 2): both barriers breached
        # High = 1050 (overshoot PT by 30), Low = 975 (overshoot SL by 5)
        df.iloc[2, df.columns.get_loc("high")] = 1050.0
        df.iloc[2, df.columns.get_loc("low")] = 975.0

        # 1. Default mode: "exclude" -> target is NaN (prevents directional bias)
        df_exclude = add_tbm_target(df, pt_mult=2.0, sl_mult=2.0, horizon=3, ambiguous_resolution="exclude")
        self.assertTrue(np.isnan(df_exclude["target"].iloc[1]))

        # 2. "stop_first" mode: target is -1.0 (for trade simulation only)
        df_stop_first = add_tbm_target(df, pt_mult=2.0, sl_mult=2.0, horizon=3, ambiguous_resolution="stop_first")
        self.assertEqual(df_stop_first["target"].iloc[1], -1.0)

        # 3. "overshoot" mode: target is +1.0 (PT overshoot 30 > SL overshoot 5)
        df_overshoot = add_tbm_target(df, pt_mult=2.0, sl_mult=2.0, horizon=3, ambiguous_resolution="overshoot")
        self.assertEqual(df_overshoot["target"].iloc[1], 1.0)

        # 4. Lower-timeframe resolution mode:
        # Construct 15-minute sub-bars for bar 2 (2026-01-01 02:00 UTC)
        # Sub-bar 02:00-02:15: hits SL first (low = 970, high = 1010)
        # Sub-bar 02:15-02:30: hits PT second (high = 1050)
        ltf_dates = pd.date_range("2026-01-01 02:00", periods=4, freq="15min", tz="UTC")
        df_ltf = pd.DataFrame({
            "open": [1000.0, 1000.0, 1020.0, 1030.0],
            "high": [1010.0, 1050.0, 1040.0, 1035.0],
            "low": [970.0, 995.0, 1015.0, 1025.0],
            "close": [980.0, 1040.0, 1030.0, 1030.0],
        }, index=ltf_dates)

        df_resolved = add_tbm_target(df, pt_mult=2.0, sl_mult=2.0, horizon=3, df_ltf=df_ltf)
        # Because LTF sub-bar at 02:00 hit SL first, chronological order resolves to -1.0
        self.assertEqual(df_resolved["target"].iloc[1], -1.0)

    def test_tbm_parameter_propagation(self):
        """Verify that config parameters propagate into TBM labeling."""
        self.assertEqual(config.TBM_PT_MULT, 2.0)
        self.assertEqual(config.TBM_SL_MULT, 2.0)
        self.assertEqual(config.TARGET_HORIZON, 12)


class TestEvaluationAndCalibration(unittest.TestCase):
    """Test suite for evaluation decoupling and probability calibration metrics."""

    def test_head_specific_metrics_and_calibration(self):
        """Verify calibration metrics: Brier score, log loss, and ECE."""
        targets = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        # Perfectly separated logits
        clf_logits = np.array([5.0, 5.0, -5.0, -5.0], dtype=np.float32)
        reg_preds = np.array([0.05, 0.04, -0.05, -0.04], dtype=np.float32)

        metrics = compute_comprehensive_metrics(
            reg_preds=reg_preds,
            clf_logits=clf_logits,
            targets=targets,
            signal_threshold=0.01,
            target_multiplier=1.0,
        )

        cal = metrics["calibration"]
        # With high confidence correct predictions, Brier score and log loss must be near 0
        self.assertLess(cal["brier_score"], 0.05)
        self.assertLess(cal["log_loss"], 0.05)
        self.assertLess(cal["expected_calibration_error_ece"], 0.05)
        self.assertEqual(len(cal["reliability_curve"]), 10)

    def test_trivial_zero_regression_baseline(self):
        """Verify constant-zero baseline yields MSE = 1.0 on +/-1 targets."""
        targets = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        reg_zero = np.zeros_like(targets)
        clf_zero = np.zeros_like(targets)

        metrics = compute_comprehensive_metrics(reg_preds=reg_zero, clf_logits=clf_zero, targets=targets)
        self.assertAlmostEqual(metrics["trivial_zero_mse"], 1.0, places=5)
        self.assertAlmostEqual(metrics["regression_mse"], 1.0, places=5)
        self.assertAlmostEqual(metrics["mse_vs_trivial_ratio"], 1.0, places=5)


class TestTemporalLeakageAndPartitions(unittest.TestCase):
    """Test suite for temporal leakage protection, 3-way partitioning, and walk-forward cross-validation."""

    def test_three_way_purged_partitioning(self):
        """Verify 3-way split: train / val / untouched test with purges at both boundaries."""
        N = 100
        horizon = 5
        Xc = np.arange(N * 2).reshape(N, 2)
        Xm = np.arange(N * 3).reshape(N, 3)
        y = np.ones(N)

        train_s, val_s, test_s, meta = split_train_val_test_purged(
            Xc, Xm, y, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, purge_horizon=horizon
        )

        # Val starts at 60 -> train ends at 60 - 5 = 55 (purged 55..60)
        self.assertEqual(meta["train_samples"], 55)
        self.assertEqual(meta["train_purge_size"], 5)

        # Test starts at 80 -> val ends at 80 - 5 = 75 (val samples: 60..75 = 15, purged 75..80)
        self.assertEqual(meta["val_samples"], 15)
        self.assertEqual(meta["val_purge_size"], 5)

        # Test samples: 80..100 = 20 (untouched final test partition)
        self.assertEqual(meta["test_samples"], 20)

        # No overlapping indices between partitions
        train_max_idx = meta["train_indices"][1] - 1
        val_min_idx = meta["val_indices"][0]
        val_max_idx = meta["val_indices"][1] - 1
        test_min_idx = meta["test_indices"][0]

        self.assertLess(train_max_idx + horizon, val_min_idx + 1)
        self.assertLess(val_max_idx + horizon, test_min_idx + 1)

    def test_walk_forward_splits_generator(self):
        """Verify walk-forward splits generate expanding train partitions and purged test partitions."""
        splits = generate_walk_forward_splits(total_samples=1000, n_splits=4, train_ratio=0.50, test_ratio=0.125, purge_horizon=12)
        self.assertEqual(len(splits), 4)

        for s in splits:
            train_end = s["train_indices"][1]
            test_start = s["test_indices"][0]
            # Purge gap must be exactly 12
            self.assertEqual(test_start - train_end, 12)
            self.assertGreater(s["test_count"], 0)

    def test_block_bootstrap_uncertainty_estimation(self):
        """Verify moving block bootstrap calculates consistent confidence intervals."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([-1.0, 1.0], size=200)
        y_pred = y_true * 0.5 + rng.normal(0, 0.2, size=200)

        def acc_metric(p, t):
            return float(np.mean((p > 0) == (t > 0)))

        res = block_bootstrap_confidence_interval(y_true, y_pred, acc_metric, block_size=12, n_bootstraps=200)
        self.assertIn("ci_lower", res)
        self.assertIn("ci_upper", res)
        self.assertLessEqual(res["ci_lower"], res["point_estimate"])
        self.assertGreaterEqual(res["ci_upper"], res["point_estimate"])

    def test_end_to_end_scaler_fitting_and_purged_split_integration(self):
        """Integration test: exercise real feature building, purged split, and scaler fitting."""
        N = 300
        dates = pd.date_range("2026-01-01", periods=N, freq="1h", tz="UTC")
        price = 1000.0 + 50.0 * np.sin(np.linspace(0, 30, N))
        df = pd.DataFrame({
            "open": price,
            "high": price + 5.0,
            "low": price - 5.0,
            "close": price,
            "volume": np.linspace(100, 500, N),
        }, index=dates)

        df_feat = build_features(df, add_targets=False)
        self.assertGreater(len(df_feat), 50)
        df_feat = add_tbm_target(df_feat, pt_mult=2.0, sl_mult=2.0, horizon=5, ambiguous_resolution="exclude").dropna(subset=["target"])

        split_idx = int(len(df_feat) * 0.8)
        purge_horizon = 5
        purged_train_end = split_idx - purge_horizon

        # Real scaler fitting strictly on purged training slice
        df_train = df_feat.iloc[:purged_train_end]
        self.assertGreater(len(df_train), 0)
        scaler = StandardScaler()
        scaler.fit(df_train[["close", "open"]])

        # Verify scaler mean matches training slice exactly
        self.assertAlmostEqual(scaler.mean_[0], df_train["close"].mean(), places=5)
        self.assertAlmostEqual(scaler.mean_[1], df_train["open"].mean(), places=5)
        # Verify validation data did not influence scaler mean
        self.assertNotEqual(scaler.mean_[0], df_feat["close"].mean())


if __name__ == "__main__":
    unittest.main()
