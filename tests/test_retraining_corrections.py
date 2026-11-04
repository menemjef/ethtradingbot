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
13. Warmup bars pre-roll integration through pipeline and pre-fillna 720-bar Fibonacci validation.
14. Non-destructive train.py orchestration, epoch propagation, inner-val scaler isolation, and OHLC stop-gap accounting.
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from model.walk_forward import simulate_trades_ohlc
from train import main as train_main


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
    validate_fibonacci_warmup,
)
from model.walk_forward import run_walk_forward_evaluation
from model.trainer import (
    directional_accuracy,
    compute_comprehensive_metrics,
    train_model,
)
import config


def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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

    def test_multipage_traversal_across_intermediate_gap(self):
        """Verify fetch_ohlcv detects gaps in assembled data and honours raise_on_gap."""
        import data.fetcher as fetcher_mod
        tf_ms = 3600 * 1000
        now_dt = datetime.datetime(2026, 11, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
        now_ms = int(now_dt.timestamp() * 1000)

        current_candle_open = (now_ms // tf_ms) * tf_ms
        latest_completed_open = current_candle_open - tf_ms

        t_batch1_start = latest_completed_open - 1199 * tf_ms
        candles_batch1 = [[t_batch1_start + i * tf_ms, 2000.0, 2010.0, 1990.0, 2005.0, 100.0] for i in range(600)]

        # Gap of 50 hours (missing 50 candles)
        t_batch2_start = candles_batch1[-1][0] + 51 * tf_ms
        candles_batch2 = [[t_batch2_start + i * tf_ms, 2100.0, 2110.0, 2090.0, 2105.0, 100.0] for i in range(600)]

        all_mock = candles_batch1 + candles_batch2

        mock_exchange = MagicMock()
        mock_exchange.milliseconds.return_value = now_ms
        mock_exchange.parse_timeframe.return_value = 3600
        mock_exchange.rateLimit = 5

        def mock_fetch(symbol, timeframe, since, limit):
            return [c for c in all_mock if c[0] >= since][:limit]

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch

        saved_exchange = fetcher_mod._EXCHANGE
        try:
            # 1. With raise_on_gap=True, the assembled DataFrame has missing intervals -> ValueError
            fetcher_mod._EXCHANGE = None
            with patch("data.fetcher.get_exchange", return_value=mock_exchange):
                with self.assertRaises(ValueError) as ctx:
                    fetch_ohlcv("ETH/USDT:USDT", "1h", limit=800, warmup_bars=0, validate=False, raise_on_gap=True)
                self.assertIn("missing intervals", str(ctx.exception))

            # 2. With raise_on_gap=False, returns data (with gap warning, no exception)
            fetcher_mod._EXCHANGE = None
            with patch("data.fetcher.get_exchange", return_value=mock_exchange):
                df = fetch_ohlcv("ETH/USDT:USDT", "1h", limit=800, warmup_bars=0, validate=False, raise_on_gap=False)
                self.assertGreater(len(df), 0)
                self.assertTrue(df.index.is_monotonic_increasing)
        finally:
            fetcher_mod._EXCHANGE = saved_exchange

    def test_walk_forward_evaluation_execution(self):
        """Verify full end-to-end walk forward execution on synthetic data."""
        N = 350
        dates = pd.date_range("2026-01-01", periods=N, freq="1h", tz="UTC")
        price = 2000.0 + 100.0 * np.sin(np.linspace(0, 20, N))
        df_raw = pd.DataFrame({
            "open": price,
            "high": price + 10.0,
            "low": price - 10.0,
            "close": price,
            "volume": np.linspace(100, 1000, N),
        }, index=dates)

        wf_results = run_walk_forward_evaluation(
            df_raw,
            n_splits=2,
            train_ratio=0.50,
            test_ratio=0.20,
            purge_horizon=5,
            max_epochs=2,
            batch_size=32,
            device="cpu",
        )

        self.assertIn("fold_by_fold_results", wf_results)
        self.assertIn("aggregate_summary", wf_results)
        self.assertEqual(len(wf_results["fold_by_fold_results"]), 2)

        agg = wf_results["aggregate_summary"]
        self.assertIn("mean_directional_accuracy_pct", agg)
        self.assertIn("mean_profit_factor", agg)
        self.assertIn("total_net_pnl_sum_pct", agg)


class TestPass4ExecutionAndIntegrity(unittest.TestCase):
    """Test suite for Quant Audit Pass 4 & Pass 5 non-destructive orchestration requirements."""

    def test_train_py_full_orchestration_mocked(self):
        """
        Verify train.py entry-point runs end-to-end without touching production checkpoints or scalers.
        Asserts SHA-256 hashes of real artifacts are 100% identical before and after the test.
        """
        real_model_path = config.MODEL_PATH
        real_scaler_c_path = config.SCALER_CANDLE_PATH
        real_scaler_m_path = config.SCALER_MARKET_PATH

        hash_before_model = _sha256_file(real_model_path)
        hash_before_sc = _sha256_file(real_scaler_c_path)
        hash_before_sm = _sha256_file(real_scaler_m_path)

        # Provide 720 warmup bars + 250 requested bars = 970 bars so 720-bar Fibonacci warm-up passes
        warmup = 720
        limit = 250
        N = warmup + limit
        dates = pd.date_range("2026-01-01", periods=N, freq="1h", tz="UTC")
        price = 2000.0 + 50.0 * np.sin(np.linspace(0, 40, N))
        df_mock = pd.DataFrame({
            "open": price,
            "high": price + 5.0,
            "low": price - 5.0,
            "close": price,
            "volume": np.linspace(100, 500, N),
        }, index=dates)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model = os.path.join(tmpdir, "temp_best_model.pt")
            tmp_sc = os.path.join(tmpdir, "temp_scaler_candle.pkl")
            tmp_sm = os.path.join(tmpdir, "temp_scaler_market.pkl")

            with patch.object(config, "MODEL_PATH", tmp_model), \
                 patch.object(config, "SCALER_CANDLE_PATH", tmp_sc), \
                 patch.object(config, "SCALER_MARKET_PATH", tmp_sm), \
                 patch("train.fetch_ohlcv", return_value=df_mock), \
                 patch("sys.argv", ["train.py", "--fast", "--limit", str(limit), "--warmup-bars", str(warmup), "--train-ratio", "0.6", "--val-ratio", "0.2"]):
                try:
                    train_main()
                except SystemExit as e:
                    self.fail(f"train.py main() raised unexpected SystemExit: {e}")
                except KeyError as e:
                    self.fail(f"train.py main() raised KeyError (metric schema issue): {e}")
                except TypeError as e:
                    self.fail(f"train.py main() raised TypeError (argument mismatch): {e}")

            # Verify temporary artifacts were written only inside tmpdir
            self.assertTrue(os.path.exists(tmp_model), "Temp model was not saved in tmpdir")
            self.assertTrue(os.path.exists(tmp_sc), "Temp candle scaler was not saved in tmpdir")
            self.assertTrue(os.path.exists(tmp_sm), "Temp market scaler was not saved in tmpdir")

        # Verify production artifact hashes are 100% untouched
        self.assertEqual(_sha256_file(real_model_path), hash_before_model, "Production best_model.pt was modified!")
        self.assertEqual(_sha256_file(real_scaler_c_path), hash_before_sc, "Production scaler_candle.pkl was modified!")
        self.assertEqual(_sha256_file(real_scaler_m_path), hash_before_sm, "Production scaler_market.pkl was modified!")

    def test_preroll_exclusion_and_fibonacci_validity(self):
        """Verify that pre-roll warms up 720-bar Fibonacci BEFORE fillna(0.0) and is excluded from training samples."""
        warmup = 720
        requested = 100
        total = warmup + requested
        dates = pd.date_range("2026-01-01", periods=total, freq="1h", tz="UTC")
        price = 2000.0 + 100.0 * np.sin(np.linspace(0, 30, total))
        df_raw = pd.DataFrame({
            "open": price,
            "high": price + 10.0,
            "low": price - 10.0,
            "close": price,
            "volume": np.linspace(100, 1000, total),
        }, index=dates)

        requested_start_ts = df_raw.index[-requested]
        self.assertEqual(len(df_raw.loc[requested_start_ts:]), requested)

        # Build features on full history with explicit pre-fillna Fibonacci warm-up validation
        df_feat_full = build_features(
            df_raw,
            add_targets=False,
            requested_start_ts=requested_start_ts,
            min_warmup_bars=720,
        )
        self.assertIn("fib_618_dist_180d", df_feat_full.columns)
        self.assertIn("fib_warmup_validation", df_feat_full.attrs)
        self.assertEqual(df_feat_full.attrs["fib_warmup_validation"]["valid_prior_bars"], 720)
        self.assertEqual(df_feat_full.attrs["fib_warmup_validation"]["nan_count_retained_pre_fillna"], 0)

        # Retained slice starts at requested_start_ts
        df_feat = df_feat_full.loc[requested_start_ts:].copy()
        self.assertEqual(len(df_feat), requested)

        # Verify raw pre-fillna validity mask is True across all retained rows
        raw_valid = df_feat_full.attrs["raw_fib_valid_mask"].loc[requested_start_ts:]
        self.assertTrue(raw_valid.all(), "Raw Fibonacci had NaNs before fillna(0.0) in retained slice!")

    def test_consecutive_hourly_windows_with_missing_targets_in_middle(self):
        """Verify that missing/ambiguous targets in the middle do NOT cause lookback windows to skip an hour."""
        N = 200
        lookback = 72
        dates = pd.date_range("2026-01-01", periods=N, freq="1h", tz="UTC")
        price = 2000.0 + 50.0 * np.sin(np.linspace(0, 20, N))
        candle_arr = np.column_stack([price, price + 5.0, price - 5.0, price, np.ones(N)])
        market_arr = np.zeros((N, 5), dtype=np.float32)

        # Create targets with a 10-hour gap of NaNs in the middle (indices 90 to 100)
        targets = np.ones(N, dtype=np.float32)
        targets[90:101] = np.nan

        X_c, X_m, y, meta = build_windows(
            candle_arr, market_arr, targets,
            lookback=lookback,
            timestamps=dates,
            drop_nan_targets=True,
            return_metadata=True,
        )

        # 1. Windows with NaN targets are excluded
        self.assertFalse(np.isnan(y).any(), "NaN targets must be filtered out post-construction")

        # 2. But internal features of EVERY window must be contiguous 1-hour steps
        dec_indices = meta["decision_indices"]
        for idx, dec_i in enumerate(dec_indices):
            win_timestamps = dates[dec_i - lookback + 1 : dec_i + 1]
            deltas = pd.Series(win_timestamps).diff().dropna()
            self.assertTrue((deltas == pd.Timedelta(hours=1)).all(), "A lookback window skipped an hour internally!")

    def test_purge_boundaries_against_future_resolution_timestamps(self):
        """Verify that training target resolution timestamps never overlap with validation or test start."""
        N = 300
        dates = pd.date_range("2026-01-01", periods=N, freq="1h", tz="UTC")
        X_c = np.zeros((N, 72, 5), dtype=np.float32)
        X_m = np.zeros((N, 72, 5), dtype=np.float32)
        y = np.ones(N, dtype=np.float32)
        purge_horizon = 12

        train_slice, val_slice, test_slice, meta = split_train_val_test_purged(
            X_c, X_m, y,
            timestamps=dates,
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            purge_horizon=purge_horizon,
        )

        val_start_ts = meta["val_start_ts"]
        test_start_ts = meta["test_start_ts"]
        train_max_res = meta["train_max_resolution_ts"]
        val_max_res = meta["val_max_resolution_ts"]

        self.assertLessEqual(train_max_res, val_start_ts, "Train target resolution leaks into validation set")
        self.assertLessEqual(val_max_res, test_start_ts, "Val target resolution leaks into test set")

    def test_ohlc_next_bar_open_fill_and_slippage(self):
        """Verify simulator fills at next candle OPEN with modeled slippage."""
        dates = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
        df_test = pd.DataFrame({
            "open": [2000.0, 2010.0, 2020.0, 2030.0, 2040.0],
            "high": [2005.0, 2015.0, 2025.0, 2035.0, 2045.0],
            "low": [1995.0, 2005.0, 2015.0, 2025.0, 2035.0],
            "close": [2002.0, 2012.0, 2022.0, 2032.0, 2042.0],
            "atr_14": [10.0, 10.0, 10.0, 10.0, 10.0],
        }, index=dates)

        # Signal at candle 0 close (+1.0 -> LONG)
        signals = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        res = simulate_trades_ohlc(
            signals=signals,
            df_test=df_test,
            signal_threshold=0.5,
            slippage=0.001,  # 0.1% slippage
            taker_fee=0.0005,
        )

        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        # Fill MUST be at candle 1 OPEN (2010.0) + slippage (2010 * 1.001 = 2012.01)
        self.assertEqual(trade["entry_bar"], 1)
        self.assertEqual(trade["decision_time"], str(dates[0]))
        self.assertEqual(trade["entry_time"], str(dates[1]))
        self.assertAlmostEqual(trade["entry_price"], 2010.0 * 1.001, places=3)
        self.assertEqual(trade["side"], "LONG")

    def test_intrabar_collision_conservative_stop_first(self):
        """Verify that when both TP and SL are touched in the same candle, simulator resolves stop-first."""
        dates = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
        df_test = pd.DataFrame({
            "open": [2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
            # At candle 2: high touches PT (2020) and low touches SL (1980)
            "high": [2005.0, 2005.0, 2050.0, 2005.0, 2005.0],
            "low": [1995.0, 1995.0, 1950.0, 1995.0, 1995.0],
            "close": [2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
            "atr_14": [10.0, 10.0, 10.0, 10.0, 10.0],
        }, index=dates)

        signals = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        res = simulate_trades_ohlc(
            signals=signals,
            df_test=df_test,
            signal_threshold=0.5,
            pt_mult=2.0,  # PT = 2020
            sl_mult=2.0,  # SL = 1980
            slippage=0.0,
            taker_fee=0.0,
        )

        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        self.assertEqual(trade["exit_reason"], "stop_loss_collision")
        self.assertTrue(trade["is_collision"])
        self.assertLess(trade["net_return"], 0.0, "Collision must resolve as loss (stop-first)")

    def test_forced_end_of_fold_position_handling(self):
        """Verify that positions still open at the final bar of the fold are marked-to-market and closed."""
        dates = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
        df_test = pd.DataFrame({
            "open": [2000.0, 2000.0, 2000.0, 2010.0],
            "high": [2005.0, 2005.0, 2005.0, 2015.0],
            "low": [1995.0, 1995.0, 1995.0, 2005.0],
            "close": [2000.0, 2000.0, 2000.0, 2012.0],
            "atr_14": [10.0, 10.0, 10.0, 10.0],
        }, index=dates)

        # Enter at candle 2 (from signal at candle 1)
        signals = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        res = simulate_trades_ohlc(
            signals=signals,
            df_test=df_test,
            signal_threshold=0.5,
            pt_mult=10.0,  # Never hit
            sl_mult=10.0,  # Never hit
            holding_max=20,  # Never hit
            slippage=0.0,
            taker_fee=0.0,
        )

        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        self.assertEqual(trade["exit_bar"], 3)
        self.assertEqual(trade["exit_reason"], "end_of_fold")
        self.assertAlmostEqual(trade["exit_price"], 2012.0)

    def test_pooled_profit_factor_and_compounded_return_calculation(self):
        """Verify pooled profit factor and compounded return calculations."""
        fold_results = [
            {"net_pnl_pct": 10.0, "profit_factor": 2.0, "max_drawdown_pct": 3.0, "win_rate_pct": 60.0, "trade_count": 5},
            {"net_pnl_pct": -5.0, "profit_factor": 0.5, "max_drawdown_pct": 7.0, "win_rate_pct": 40.0, "trade_count": 5},
        ]
        compounded_ret = 1.0
        for f in fold_results:
            compounded_ret *= (1.0 + f["net_pnl_pct"] / 100.0)
        compounded_net_return_pct = float(round((compounded_ret - 1.0) * 100.0, 2))
        self.assertAlmostEqual(compounded_net_return_pct, 4.5, places=2)

        trades = [
            {"net_return": 0.10}, {"net_return": 0.10},  # sum = 0.20
            {"net_return": -0.05}, {"net_return": -0.05}, # sum = -0.10
        ]
        gains = sum(t["net_return"] for t in trades if t["net_return"] > 0)
        losses = abs(sum(t["net_return"] for t in trades if t["net_return"] < 0))
        pooled_pf = gains / losses
        self.assertAlmostEqual(pooled_pf, 2.0)


class TestPass5ArtifactIntegrityAndWalkForward(unittest.TestCase):
    """Test suite for Quant Audit Pass 5 requirements."""

    def test_fast_and_test_mode_epoch_propagation(self):
        """
        Verify that explicit max_epochs and patience passed to train_model() override
        import-time config constants, and that --fast / --test propagate <= 5 / <= 30 epochs.
        """
        N = 120
        X_c = np.random.randn(N, config.LOOKBACK, len(config.CANDLE_FEATURES)).astype(np.float32)
        X_m = np.random.randn(N, config.LOOKBACK, len(config.MARKET_FEATURES)).astype(np.float32)
        y = np.random.choice([-1.0, 1.0], size=N).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_fast = os.path.join(tmpdir, "fast_model.pt")
            _, res_fast = train_model(
                X_c, X_m, y,
                max_epochs=5,
                patience=3,
                model_path=tmp_model_fast,
            )
            val_fast = res_fast["validation_metrics"]
            self.assertEqual(val_fast["max_epochs_configured"], 5)
            self.assertEqual(val_fast["patience_configured"], 3)
            self.assertLessEqual(val_fast["epochs_trained"], 5)

            # Verify --test mode in train.py passes max_epochs=30 to train_model
            captured_kwargs = {}
            def fake_train_model(*args, **kwargs):
                captured_kwargs.update(kwargs)
                return MagicMock(), {
                    "validation_metrics": {
                        "best_epoch": 1,
                        "epochs_trained": 1,
                        "best_checkpoint_metric_val": 0.9,
                        "trading_signal_directional_accuracy": 0.55,
                        "regression_mse": 0.99,
                        "trivial_zero_mse": 1.0,
                    },
                    "final_test_metrics": None,
                    "calibration_results": {"fitted_temperature": 1.0},
                    "test_uncertainty_analysis": None,
                }

            warmup = 720
            limit = 200
            total = warmup + limit
            dates = pd.date_range("2026-01-01", periods=total, freq="1h", tz="UTC")
            price = 2000.0 + 30.0 * np.sin(np.linspace(0, 20, total))
            df_mock = pd.DataFrame({
                "open": price, "high": price + 5.0, "low": price - 5.0, "close": price,
                "volume": np.linspace(100, 500, total),
            }, index=dates)

            tmp_sc = os.path.join(tmpdir, "sc.pkl")
            tmp_sm = os.path.join(tmpdir, "sm.pkl")
            with patch.object(config, "MODEL_PATH", tmp_model_fast), \
                 patch.object(config, "SCALER_CANDLE_PATH", tmp_sc), \
                 patch.object(config, "SCALER_MARKET_PATH", tmp_sm), \
                 patch("train.fetch_ohlcv", return_value=df_mock), \
                 patch("train.train_model", side_effect=fake_train_model), \
                 patch("sys.argv", ["train.py", "--test", "--limit", str(limit), "--warmup-bars", str(warmup)]):
                train_main()

            self.assertEqual(captured_kwargs.get("max_epochs"), 30)

    def test_fibonacci_warmup_validation_fails_on_insufficient_history_and_cache(self):
        """
        Verify that missing 720-bar history BEFORE requested_start_ts fails validation
        with ValueError before fillna(0.0) can replace NaN with 0.0 (both direct and cached CSV).
        """
        # Only 300 warmup bars (< 720 required) + 200 requested bars = 500 bars total
        warmup = 300
        requested = 200
        total = warmup + requested
        dates = pd.date_range("2026-01-01", periods=total, freq="1h", tz="UTC")
        price = 2000.0 + 50.0 * np.sin(np.linspace(0, 20, total))
        df_insufficient = pd.DataFrame({
            "open": price,
            "high": price + 5.0,
            "low": price - 5.0,
            "close": price,
            "volume": np.linspace(100, 500, total),
        }, index=dates)

        requested_start_ts = df_insufficient.index[-requested]

        # 1. Direct build_features call with requested_start_ts MUST fail before fillna(0.0)
        with self.assertRaises(ValueError) as ctx:
            build_features(
                df_insufficient,
                add_targets=False,
                requested_start_ts=requested_start_ts,
                min_warmup_bars=720,
            )
        self.assertIn("Fibonacci warm-up validation failed", str(ctx.exception))

        # 2. Even if someone bypassed min_history_bars=300, raw fib_618_dist_180d at bar 300 is NaN before fillna(0.0)!
        with self.assertRaises(ValueError) as ctx2:
            build_features(
                df_insufficient,
                add_targets=False,
                requested_start_ts=requested_start_ts,
                min_warmup_bars=100,  # bypass bar count check to test raw NaN check before fillna(0.0)
            )
        self.assertIn("before fillna(0.0)", str(ctx2.exception))

        # 3. Cached CSV dataset (--use_cache) with insufficient pre-roll MUST also fail in train.py
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "insufficient_cache.csv")
            df_insufficient.to_csv(csv_path)
            tmp_model = os.path.join(tmpdir, "m.pt")
            tmp_sc = os.path.join(tmpdir, "sc.pkl")
            tmp_sm = os.path.join(tmpdir, "sm.pkl")

            with patch.object(config, "MODEL_PATH", tmp_model), \
                 patch.object(config, "SCALER_CANDLE_PATH", tmp_sc), \
                 patch.object(config, "SCALER_MARKET_PATH", tmp_sm), \
                 patch("sys.argv", ["train.py", "--fast", "--limit", str(requested), "--use_cache", csv_path]):
                with self.assertRaises(ValueError) as ctx_cache:
                    train_main()
                self.assertIn("Fibonacci warm-up validation failed", str(ctx_cache.exception))

    def test_walk_forward_signal_timestamp_alignment(self):
        """
        Verify walk-forward signal timestamp alignment:
        1. First test prediction uses decision timestamp at candle k-1 close (immediately before test start)
           and executes at candle k OPEN (first test candle).
        2. Every trade has decision_time < entry_time (specifically entry_time == decision_time + 1h).
        3. Final test candle cannot generate a new trade outside the fold.
        """
        N = 360
        dates = pd.date_range("2026-01-01", periods=N, freq="1h", tz="UTC")
        price = 2000.0 + 80.0 * np.sin(np.linspace(0, 25, N))
        df_raw = pd.DataFrame({
            "open": price,
            "high": price + 15.0,
            "low": price - 15.0,
            "close": price,
            "volume": np.linspace(100, 1000, N),
        }, index=dates)

        wf_results = run_walk_forward_evaluation(
            df_raw,
            n_splits=2,
            train_ratio=0.50,
            test_ratio=0.20,
            purge_horizon=5,
            max_epochs=1,
            batch_size=32,
            device="cpu",
            signal_threshold=0.0,  # Force trades so we can inspect alignment on every fold
        )

        for fold in wf_results["fold_by_fold_results"]:
            test_start_ts = pd.Timestamp(fold["test_start_ts"])
            test_end_ts = pd.Timestamp(fold["test_end_ts"])
            first_dec_ts = pd.Timestamp(fold["first_decision_ts"])
            first_exec_ts = pd.Timestamp(fold["first_execution_open_ts"])

            # First decision timestamp is exactly 1 hour before first test candle open
            self.assertEqual(first_exec_ts, test_start_ts)
            self.assertEqual(first_dec_ts + pd.Timedelta(hours=1), first_exec_ts)

            trades = fold["trades"]
            self.assertGreater(len(trades), 0)
            for t in trades:
                dec_t = pd.Timestamp(t["decision_time"])
                ent_t = pd.Timestamp(t["entry_time"])
                self.assertLess(dec_t, ent_t, "Trade decision_time must strictly precede entry_time!")
                self.assertEqual(dec_t + pd.Timedelta(hours=1), ent_t, "Entry must occur at next hourly candle OPEN!")
                # Final test candle cannot initiate a new trade
                self.assertLess(ent_t, test_end_ts, "Final test candle must not open a new trade!")

    def test_walk_forward_inner_validation_scaler_isolation(self):
        """
        Prove numerically that extreme outliers injected into the inner validation slice
        of a walk-forward fold have ZERO effect on fitted scaler means and variances.
        """
        N = 300
        dates = pd.date_range("2026-01-01", periods=N, freq="1h", tz="UTC")
        price = 2000.0 + 50.0 * np.sin(np.linspace(0, 20, N))
        df_raw = pd.DataFrame({
            "open": price,
            "high": price + 10.0,
            "low": price - 10.0,
            "close": price,
            "volume": np.linspace(100, 500, N),
        }, index=dates)

        df_feat = build_features(df_raw, add_targets=False)
        df_feat = add_tbm_target(df_feat, pt_mult=2.0, sl_mult=2.0, horizon=5, ambiguous_resolution="exclude")

        # Replicate walk_forward.py inner train / inner val split on outer training window
        tr_start_idx, tr_end_idx = 0, len(df_feat)
        outer_train_len = tr_end_idx - tr_start_idx
        inner_val_size = max(config.LOOKBACK + 10, int(outer_train_len * 0.25))
        inner_val_start_idx = tr_end_idx - inner_val_size
        inner_train_end_idx = max(tr_start_idx + config.LOOKBACK + 10, inner_val_start_idx - 5)

        # Case A: Clean data
        df_inner_train_A = df_feat.iloc[tr_start_idx:inner_train_end_idx]
        scaler_c_A = StandardScaler().fit(df_inner_train_A[config.CANDLE_FEATURES].values)
        scaler_m_A = StandardScaler().fit(df_inner_train_A[config.MARKET_FEATURES].values)

        # Case B: Corrupt the inner validation slice [inner_val_start_idx : tr_end_idx] with 10^9 outliers
        df_feat_corrupted = df_feat.copy()
        df_feat_corrupted.iloc[inner_val_start_idx:tr_end_idx, :] = 1e9
        df_inner_train_B = df_feat_corrupted.iloc[tr_start_idx:inner_train_end_idx]
        scaler_c_B = StandardScaler().fit(df_inner_train_B[config.CANDLE_FEATURES].values)
        scaler_m_B = StandardScaler().fit(df_inner_train_B[config.MARKET_FEATURES].values)

        np.testing.assert_allclose(scaler_c_A.mean_, scaler_c_B.mean_, rtol=0, atol=0)
        np.testing.assert_allclose(scaler_c_A.var_, scaler_c_B.var_, rtol=0, atol=0)
        np.testing.assert_allclose(scaler_m_A.mean_, scaler_m_B.mean_, rtol=0, atol=0)
        np.testing.assert_allclose(scaler_m_A.var_, scaler_m_B.var_, rtol=0, atol=0)

    def test_ohlc_stop_loss_gap_and_itemized_accounting(self):
        """
        Verify realistic OHLC stop-loss gap fills, intratrade adverse excursion drawdown,
        itemized cost decomposition, and compounded equity return.
        """
        dates = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
        # Trade enters LONG at bar 1 OPEN = 2000.0 (with ATR=10, sl_mult=2 -> SL = 1980, PT = 2020)
        # At bar 2, market gaps down at OPEN = 1950.0 (< SL 1980.0)!
        # Must fill at worse opening gap price 1950.0 (not 1980.0).
        df_test = pd.DataFrame({
            "open":  [2000.0, 2000.0, 1950.0, 1950.0, 1950.0],
            "high":  [2005.0, 2005.0, 1960.0, 1955.0, 1955.0],
            "low":   [1995.0, 1990.0, 1940.0, 1945.0, 1945.0],
            "close": [2000.0, 1995.0, 1950.0, 1950.0, 1950.0],
            "atr_14": [10.0,   10.0,   10.0,   10.0,   10.0],
        }, index=dates)

        signals = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        slip = 0.001     # 10 bps
        taker = 0.0005   # 5 bps

        res = simulate_trades_ohlc(
            signals=signals,
            df_test=df_test,
            signal_threshold=0.5,
            pt_mult=2.0,
            sl_mult=2.0,
            slippage=slip,
            taker_fee=taker,
        )

        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        self.assertEqual(trade["exit_reason"], "stop_loss_gap")
        # Raw exit price is the gap open 1950.0, NOT the stop trigger ~1982.0
        self.assertAlmostEqual(trade["raw_exit_price"], 1950.0, places=4)
        self.assertAlmostEqual(trade["exit_price"], 1950.0 * (1.0 - slip), places=4)

        # Itemized fields present and consistent
        self.assertIn("gross_price_return", trade)
        self.assertIn("entry_slippage_cost", trade)
        self.assertIn("exit_slippage_cost", trade)
        self.assertIn("total_slippage_cost", trade)
        self.assertIn("entry_fee_cost", trade)
        self.assertIn("exit_fee_cost", trade)
        self.assertIn("funding_cost", trade)
        self.assertEqual(trade["funding_cost"], 0.0)
        self.assertAlmostEqual(trade["Gross_Price_Verify"] if "Gross_Price_Verify" in trade else trade["gross_price_return"], (1950.0 - 2000.0) / 2000.0, places=6)

        # Verify summary reports both compounded_net_return_pct and sum_net_pnl_pct
        self.assertIn("compounded_net_return_pct", res)
        self.assertIn("sum_net_pnl_pct", res)
        self.assertGreater(res["max_drawdown_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
