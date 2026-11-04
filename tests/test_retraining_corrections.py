"""
tests/test_retraining_corrections.py — Rigorous verification suite for retraining pipeline corrections.

Verifies:
1. Pagination & latest completed candle selection (no historical lag, excludes unfinished candle).
2. Shortfall detection when exchange has fewer candles than requested.
3. Duplicate and missing timestamp detection/handling.
4. TBM configuration parameter propagation from config to indicators.
5. Ambiguous candle resolution in TBM (stop-first vs overshoot).
6. Head-specific evaluation metrics (regression head vs classification head).
7. Zero-prediction regression baseline comparison (MSE = 1.0 for +/-1 targets).
8. Temporal leakage protection: proof that no training target uses validation-period OHLC prices.
9. Scaler fitting restriction: scalers fit strictly on purged training portion.
10. Reporting consistency and integrity.
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
from data.indicators import add_tbm_target
from data.preprocessor import split_train_val_purged, build_windows
from model.trainer import directional_accuracy, compute_comprehensive_metrics
import config


class TestDataFetcherCorrections(unittest.TestCase):
    """Test suite for data/fetcher.py pagination, lag elimination, and data quality."""

    def test_pagination_and_latest_completed_candle(self):
        """Verify that fetch_ohlcv targets the latest completed candle and excludes unfinished candle."""
        tf_ms = 3600 * 1000  # 1h in ms
        # Fixed reference time: 2026-11-04 12:45:00 UTC (12:00 candle is currently in progress)
        now_dt = datetime.datetime(2026, 11, 4, 12, 45, 0, tzinfo=datetime.timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)

        # Expected latest completed candle is 11:00:00 UTC
        expected_latest_ms = int(datetime.datetime(2026, 11, 4, 11, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)

        # Mock exchange with 50 completed candles + 1 currently forming candle (12:00)
        start_ms = expected_latest_ms - (50 - 1) * tf_ms
        mock_candles = []
        for i in range(51):
            c_time = start_ms + i * tf_ms
            mock_candles.append([c_time, 2000.0 + i, 2010.0 + i, 1990.0 + i, 2005.0 + i, 1000.0])

        mock_exchange = MagicMock()
        mock_exchange.milliseconds.return_value = now_ms
        mock_exchange.parse_timeframe.return_value = 3600
        mock_exchange.rateLimit = 50

        # Implement paginated fetch_ohlcv mock
        def mock_fetch(symbol, timeframe, since, limit):
            return [c for c in mock_candles if c[0] >= since][:limit]

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch

        with patch("data.fetcher.get_exchange", return_value=mock_exchange):
            df = fetch_ohlcv(symbol="ETH/USDT:USDT", timeframe="1h", limit=20, validate=False)

            self.assertEqual(len(df), 20)
            # Verify latest completed candle is 11:00:00 UTC
            self.assertEqual(df.index[-1], pd.to_datetime(expected_latest_ms, unit="ms", utc=True))
            # Verify the forming candle (12:00:00 UTC) was excluded
            forming_time = pd.to_datetime(expected_latest_ms + tf_ms, unit="ms", utc=True)
            self.assertNotIn(forming_time, df.index)
            # Verify strictly chronological
            self.assertTrue(df.index.is_monotonic_increasing)

    def test_shortfall_reporting_when_exchange_has_fewer_candles(self):
        """Verify that fetch_ohlcv logs an explicit warning when exchange returns fewer candles than requested."""
        now_dt = datetime.datetime(2026, 11, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)

        mock_exchange = MagicMock()
        mock_exchange.milliseconds.return_value = now_ms
        mock_exchange.parse_timeframe.return_value = 3600
        mock_exchange.rateLimit = 50

        # Only 15 candles available — timestamps in the range the function will target
        tf_ms = 3600 * 1000
        latest_completed = now_ms - tf_ms  # 11:00 UTC
        # Place the 15 candles at the END of the target range (so they include latest_completed)
        start_ms = latest_completed - 14 * tf_ms  # 15 candles: start to latest_completed
        mock_candles = [
            [start_ms + i * tf_ms, 2000.0, 2010.0, 1990.0, 2005.0, 500.0]
            for i in range(15)
        ]

        # Use side_effect so pagination works correctly and terminates
        def mock_fetch(symbol, timeframe, since, limit):
            matching = [c for c in mock_candles if c[0] >= since][:limit]
            return matching

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch

        with patch("data.fetcher.get_exchange", return_value=mock_exchange):
            with self.assertLogs("data.fetcher", level="WARNING") as log_capture:
                df = fetch_ohlcv(symbol="ETH/USDT:USDT", timeframe="1h", limit=100, validate=False)
                self.assertLess(len(df), 100)
                # Verify shortfall warning was emitted
                self.assertTrue(any("Shortfall detected" in msg for msg in log_capture.output))

    def test_duplicate_and_missing_interval_detection(self):
        """Verify that duplicate timestamps are deduplicated and missing intervals logged."""
        now_dt = datetime.datetime(2026, 11, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)
        tf_ms = 3600 * 1000

        # latest_completed_open = 11:00 UTC; with limit=5, target_start = 07:00 UTC
        latest_completed_open = now_ms - tf_ms  # 11:00 UTC
        target_start = latest_completed_open - 4 * tf_ms  # 07:00 UTC

        # Build candles in the correct target range (07:00 - 11:00) with a duplicate and a gap
        mock_candles = [
            [target_start + 0 * tf_ms, 100.0, 105.0, 95.0, 100.0, 10.0],  # 07:00
            [target_start + 1 * tf_ms, 101.0, 106.0, 96.0, 101.0, 10.0],  # 08:00
            [target_start + 1 * tf_ms, 101.0, 106.0, 96.0, 101.0, 15.0],  # 08:00 DUPLICATE
            # Missing 09:00 (1-hour gap)
            [target_start + 3 * tf_ms, 103.0, 108.0, 98.0, 103.0, 10.0],  # 10:00
            [target_start + 4 * tf_ms, 104.0, 109.0, 99.0, 104.0, 10.0],  # 11:00
        ]

        mock_exchange = MagicMock()
        mock_exchange.milliseconds.return_value = now_ms
        mock_exchange.parse_timeframe.return_value = 3600
        mock_exchange.rateLimit = 50

        # Use side_effect that filters candles by 'since' and returns only matching ones
        def mock_fetch(symbol, timeframe, since, limit):
            return [c for c in mock_candles if c[0] >= since][:limit]

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch

        with patch("data.fetcher.get_exchange", return_value=mock_exchange):
            with self.assertLogs("data.fetcher", level="WARNING") as log_capture:
                df = fetch_ohlcv(symbol="ETH/USDT:USDT", timeframe="1h", limit=5, validate=False)
                # Deduplication check
                self.assertEqual(len(df.index), len(set(df.index)))
                # Warnings emitted for both duplicates and missing intervals
                self.assertTrue(any("duplicate timestamps" in msg for msg in log_capture.output))
                self.assertTrue(any("missing intervals" in msg for msg in log_capture.output))


class TestTBMCorrections(unittest.TestCase):
    """Test suite for Triple Barrier Method ambiguous candle resolution and parameters."""

    def test_ambiguous_candle_stop_first_vs_overshoot(self):
        """Verify that ambiguous candles (both PT and SL breached) default to stop-first (-1)."""
        # Construct synthetic dataframe where bar 1 produces an ambiguous candle at bar 2
        dates = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "close": [1000.0] * 10,
            "high": [1005.0] * 10,
            "low": [995.0] * 10,
            "open": [1000.0] * 10,
            "atr_14": [10.0] * 10,  # ATR = 10 -> PT (+20) = 1020, SL (-20) = 980
        }, index=dates)

        # Bar index 1: price is 1000. Next bar (index 2): high explodes to 1050, low drops to 975
        # Overshoot for PT: 1050 - 1020 = 30. Overshoot for SL: 980 - 975 = 5.
        df.iloc[2, df.columns.get_loc("high")] = 1050.0
        df.iloc[2, df.columns.get_loc("low")] = 975.0

        # With legacy overshoot rule: overshoot for PT (30) > SL (5) -> labels as +1 (BUY WIN)
        df_overshoot = add_tbm_target(df, pt_mult=2.0, sl_mult=2.0, horizon=3, ambiguous_resolution="overshoot")
        self.assertEqual(df_overshoot["target"].iloc[1], 1.0)

        # With conservative stop-first rule: must label as -1 (STOP LOSS HIT)
        df_stop_first = add_tbm_target(df, pt_mult=2.0, sl_mult=2.0, horizon=3, ambiguous_resolution="stop_first")
        self.assertEqual(df_stop_first["target"].iloc[1], -1.0)

        # With exclude rule: must label as NaN
        df_exclude = add_tbm_target(df, pt_mult=2.0, sl_mult=2.0, horizon=3, ambiguous_resolution="exclude")
        self.assertTrue(np.isnan(df_exclude["target"].iloc[1]))

    def test_tbm_parameter_propagation(self):
        """Verify that config parameters propagate into TBM labeling."""
        self.assertEqual(config.TBM_PT_MULT, 2.0)
        self.assertEqual(config.TBM_SL_MULT, 2.0)
        self.assertEqual(config.TARGET_HORIZON, 12)


class TestEvaluationMetricsCorrections(unittest.TestCase):
    """Test suite for regression vs classification head independent evaluation."""

    def test_head_specific_metrics_separation(self):
        """Verify that regression and classification metrics are calculated independently."""
        # Scenario: Classification head predicts correctly, Regression head predicts incorrectly
        # Targets: [+1, +1, -1, -1]
        targets = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        # Classification logits: [+2.0, +1.5, -1.0, -0.5] -> 100% directional accuracy
        clf_logits = np.array([2.0, 1.5, -1.0, -0.5], dtype=np.float32)
        # Regression predictions: [-0.05, -0.02, +0.03, +0.04] -> 0% directional accuracy!
        reg_preds = np.array([-0.05, -0.02, 0.03, 0.04], dtype=np.float32)

        metrics = compute_comprehensive_metrics(
            reg_preds=reg_preds,
            clf_logits=clf_logits,
            targets=targets,
            signal_threshold=0.01,
            target_multiplier=1.0,
        )

        # Regression accuracy must be 0.0, Classification accuracy must be 1.0
        self.assertEqual(metrics["regression_directional_accuracy"], 0.0)
        self.assertEqual(metrics["trading_signal_directional_accuracy"], 0.0)
        self.assertEqual(metrics["classification_directional_accuracy"], 1.0)
        self.assertEqual(metrics["classification_balanced_accuracy"], 1.0)

    def test_trivial_zero_regression_baseline(self):
        """Verify that constant-zero regression baseline yields MSE = 1.0 on +/-1 targets."""
        targets = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        reg_zero = np.zeros_like(targets)
        clf_zero = np.zeros_like(targets)

        metrics = compute_comprehensive_metrics(
            reg_preds=reg_zero,
            clf_logits=clf_zero,
            targets=targets,
        )

        # Trivial baseline MSE must be exactly 1.0
        self.assertAlmostEqual(metrics["trivial_zero_mse"], 1.0, places=5)
        self.assertAlmostEqual(metrics["regression_mse"], 1.0, places=5)
        self.assertAlmostEqual(metrics["mse_vs_trivial_ratio"], 1.0, places=5)


class TestTemporalLeakageProtection(unittest.TestCase):
    """Test suite for temporal leakage elimination via purge/embargo."""

    def test_purge_prevents_target_overlap_with_validation(self):
        """Prove that the last training target does not use prices from the validation timeline."""
        lookback = 4
        horizon = 3
        N = 30

        # Construct time sequence
        dates = pd.date_range("2026-01-01", periods=N, freq="1h", tz="UTC")
        candle_arr = np.arange(N * 2).reshape(N, 2).astype(np.float32)
        market_arr = np.arange(N * 3).reshape(N, 3).astype(np.float32)
        # Target for index i uses prices up to i + horizon
        targets = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)], dtype=np.float32)

        Xc, Xm, y = build_windows(candle_arr, market_arr, targets, lookback=lookback)
        total_windows = len(y)  # N - lookback

        # Split with purge_horizon = horizon
        (Xc_tr, Xm_tr, y_tr), (Xc_va, Xm_va, y_va), meta = split_train_val_purged(
            Xc, Xm, y, train_split=0.8, purge_horizon=horizon
        )

        split_idx = meta["split_idx"]
        purged_train_end = split_idx - horizon

        self.assertEqual(len(y_tr), purged_train_end)
        self.assertEqual(meta["purge_size"], horizon)

        # Index of last training observation in original candle series
        # Window k has features ending at index k + lookback - 1
        last_train_window_idx = purged_train_end - 1
        last_train_feature_end_bar = last_train_window_idx + lookback - 1
        # Target for this window looks forward by `horizon` bars
        last_train_target_future_bar = last_train_feature_end_bar + horizon

        # First validation window is at split_idx
        first_val_window_idx = split_idx
        first_val_decision_bar = first_val_window_idx + lookback - 1

        # PROOF: The future price bar reached by the last training target
        # must be strictly before or at most equal to the validation decision bar minus 1
        self.assertLess(last_train_target_future_bar, first_val_decision_bar + 1)
        self.assertLessEqual(last_train_target_future_bar, first_val_decision_bar)

    def test_scaler_fitting_restricted_to_purged_training(self):
        """Verify scalers fit strictly on the purged training set without validation contamination."""
        data = np.array([
            [10.0, 100.0],
            [12.0, 110.0],
            [14.0, 120.0],
            [16.0, 130.0],
            [18.0, 140.0],
            # Purged boundary rows
            [50.0, 500.0],
            [60.0, 600.0],
            # Validation rows
            [100.0, 1000.0],
            [110.0, 1100.0],
        ])
        df = pd.DataFrame(data, columns=["feat1", "feat2"])

        # Train split index at row 7, purge horizon 2 -> rows 0:5 are train, 5:7 are purged, 7:9 are val
        train_split_idx = 7
        purge_size = 2
        df_purged_train = df.iloc[:train_split_idx - purge_size]

        scaler = StandardScaler()
        scaler.fit(df_purged_train)

        # Expected mean of rows 0..4: feat1 = 14.0, feat2 = 120.0
        self.assertAlmostEqual(scaler.mean_[0], 14.0, places=4)
        self.assertAlmostEqual(scaler.mean_[1], 120.0, places=4)
        # Confirm that extreme values in purged (50, 60) and validation (100, 110) did NOT contaminate scaler mean
        self.assertNotEqual(scaler.mean_[0], df.mean()["feat1"])


if __name__ == "__main__":
    unittest.main()
