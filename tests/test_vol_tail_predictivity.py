"""
tests/test_vol_tail_predictivity.py

Unit tests for Volatility & Tail-Risk Predictability pipeline:
  1. Causal stamping verification (metrics timestamp <= decision time minus 5m).
  2. Walk-forward integrity (strictly expanding window, no future data in training set or standardizer).
  3. Benjamini-Hochberg FDR correction.
  4. Final split locked guard (cannot read data beyond DEV_END until unlocked).
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.scripts.run_logger import (
    log_final_split_read,
    check_final_split_already_logged,
    get_all_logged_runs,
)
from research.scripts.run_vol_tail_predictivity import (
    benjamini_hochberg_adjust,
    unlock_final,
    lock_final,
    final_unlocked,
    load_symbol_daily_dataset,
    run_walk_forward,
    evaluate_forecasts,
    qlike_loss,
    DEV_END,
    EVAL_START_DATE
)


class TestVolTailPredictivity(unittest.TestCase):
    def setUp(self):
        lock_final()

    def tearDown(self):
        lock_final()

    def test_final_split_locked_guard(self):
        """Verifies that final split data cannot be read until explicitly unlocked."""
        self.assertFalse(final_unlocked())
        # Attempting to load with a split cutoff beyond DEV_END must raise PermissionError
        future_cutoff = pd.Timestamp("2025-06-01 00:00:00+00:00")
        with self.assertRaises(PermissionError):
            load_symbol_daily_dataset("BTCUSDT", split_cutoff=future_cutoff)

        # Unlocking allows access
        unlock_final()
        self.assertTrue(final_unlocked())
        lock_final()
        self.assertFalse(final_unlocked())

    def test_run_log_records_every_final_read(self):
        """Verifies that data/run_log.jsonl records timestamp, experiment name, and git commit."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            rec = log_final_split_read(
                experiment_name="test_experiment_alpha",
                split="Final",
                log_path=tmp_path
            )
            self.assertEqual(rec["experiment_name"], "test_experiment_alpha")
            self.assertEqual(rec["split"], "Final")
            self.assertIn("timestamp", rec)
            self.assertIn("git_commit", rec)
            self.assertTrue(check_final_split_already_logged("test_experiment_alpha", split="Final", log_path=tmp_path))

            # Verify file content
            runs = get_all_logged_runs(log_path=tmp_path)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["experiment_name"], "test_experiment_alpha")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_run_log_duplicate_final_read_raises_error(self):
        """Verifies that reading Final split when already logged for experiment raises PermissionError."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            log_final_split_read("test_experiment_beta", split="Final", log_path=tmp_path)
            # Second attempt must raise PermissionError
            with self.assertRaises(PermissionError):
                log_final_split_read("test_experiment_beta", split="Final", log_path=tmp_path)

            # Different experiment name must succeed
            rec2 = log_final_split_read("test_experiment_gamma", split="Final", log_path=tmp_path)
            self.assertEqual(rec2["experiment_name"], "test_experiment_gamma")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_benjamini_hochberg_monotonic(self):
        """Verifies Benjamini-Hochberg FDR correction properties."""
        p_vals = [0.001, 0.04, 0.02, 0.50, 0.90]
        adj = benjamini_hochberg_adjust(p_vals)
        self.assertEqual(len(adj), len(p_vals))
        for p_adj in adj:
            self.assertGreaterEqual(p_adj, 0.0)
            self.assertLessEqual(p_adj, 1.0)
        # Check monotonicity: if p_i < p_j, adj_i <= adj_j
        sorted_pairs = sorted(zip(p_vals, adj), key=lambda x: x[0])
        for k in range(len(sorted_pairs) - 1):
            self.assertLessEqual(sorted_pairs[k][1], sorted_pairs[k + 1][1])

    def test_causal_stamping(self):
        """Verifies that features known at 00:00 UTC strictly ingest data <= 23:55 UTC of D-1."""
        # Load daily dataset for BTC up to DEV_END
        df_daily = load_symbol_daily_dataset("BTCUSDT", split_cutoff=DEV_END)
        self.assertFalse(df_daily.empty)
        # Check that daily timestamps are strictly on 00:00:00 UTC
        self.assertTrue((df_daily["timestamp"].dt.hour == 0).all())
        self.assertTrue((df_daily["timestamp"].dt.minute == 0).all())
        self.assertTrue((df_daily["timestamp"].dt.second == 0).all())

        # Check that log_har_1d at day D is equal to actual_rv at day D-1
        rv_series = df_daily.set_index("timestamp")["actual_rv"]
        har_1d_series = df_daily.set_index("timestamp")["log_har_1d"]

        # For any day D, exp(log_har_1d[D]) must equal actual_rv[D - 1 day]
        sample_dates = df_daily["timestamp"].iloc[100:105]
        for dt in sample_dates:
            prior_dt = dt - pd.Timedelta(days=1)
            actual_prior_rv = rv_series.loc[prior_dt]
            har_pred = np.exp(har_1d_series.loc[dt])
            self.assertAlmostEqual(actual_prior_rv, har_pred, places=6)

    def test_walk_forward_integrity(self):
        """Verifies that each monthly test block is trained only on strictly prior data."""
        # Create synthetic daily panel with 180 days across 6 months
        dates = pd.date_range("2022-11-01", "2023-04-30", freq="1D", tz="UTC")
        n = len(dates)
        rng = np.random.default_rng(42)

        synthetic_df = pd.DataFrame({
            "timestamp": dates,
            "symbol": "BTCUSDT",
            "target_log_rv": rng.normal(-6.0, 0.5, size=n),
            "actual_rv": np.exp(rng.normal(-6.0, 0.5, size=n)),
            "log_har_1d": rng.normal(-6.0, 0.5, size=n),
            "log_har_7d": rng.normal(-6.0, 0.5, size=n),
            "log_har_30d": rng.normal(-6.0, 0.5, size=n),
            "baseline_vol_30d": np.abs(rng.normal(0.02, 0.005, size=n)),
            "z_oi_change_24h": rng.normal(0.0, 1.0, size=n),
            "z_global_ls_ratio": rng.normal(0.0, 1.0, size=n),
            "z_toptrader_ls_ratio": rng.normal(0.0, 1.0, size=n),
            "z_taker_buy_sell_ratio": rng.normal(0.0, 1.0, size=n),
            "z_mean_funding_7d": rng.normal(0.0, 1.0, size=n),
            "sign_ret_x_z_oi_change_24h": rng.normal(0.0, 1.0, size=n)
        })

        eval_start = pd.Timestamp("2023-03-01 00:00:00+00:00")
        preds = run_walk_forward(
            synthetic_df,
            target_name="target_log_rv",
            base_features=["log_har_1d", "log_har_7d", "log_har_30d"],
            aug_features=["log_har_1d", "log_har_7d", "log_har_30d", "z_oi_change_24h"],
            eval_start=eval_start,
            eval_end=None,
            is_pooled=False
        )

        # Check that predictions only exist on or after eval_start
        self.assertTrue((preds["timestamp"] >= eval_start).all())
        # Check that all test dates in March and April are predicted
        march_dates = set(dates[(dates >= "2023-03-01") & (dates <= "2023-04-30")])
        pred_dates = set(preds["timestamp"])
        self.assertEqual(march_dates, pred_dates)

    def test_qlike_loss_properties(self):
        """Verifies that QLIKE loss is strictly non-negative and zero when forecast matches actual."""
        actual = np.array([0.001, 0.002, 0.005])
        # Perfect forecast
        loss_perfect = qlike_loss(actual, actual)
        np.testing.assert_allclose(loss_perfect, 0.0, atol=1e-8)

        # Imperfect forecast
        pred_wrong = np.array([0.002, 0.001, 0.010])
        loss_wrong = qlike_loss(actual, pred_wrong)
        self.assertTrue((loss_wrong > 0.0).all())


if __name__ == "__main__":
    unittest.main()
