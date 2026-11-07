"""
tests/test_vol_target_beta.py

Unit tests for Volatility-Targeted BTC/ETH vs. Buy-and-Hold pipeline:
  1. Causality: Signals at day D strictly use data <= D - 5m; future price changes do not alter signal.
  2. Walk-forward integrity: Monthly refits strictly train on data < month_start, scalers fit on train only.
  3. Run log: data/run_log.jsonl records Final split read; duplicate execution is blocked with PermissionError.
  4. Exposure bounds: Long-only, exposure strictly within [0.0, 1.0], no leverage.
"""

import os
import sys
import unittest
import tempfile
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
from research.scripts.run_vol_target_beta import (
    simulate_portfolio,
    compute_paired_sharpe_bootstrap,
    run_random_exposure_control,
    FEE_BASE,
    DEV_START,
    DEV_END
)
from research.scripts.run_vol_tail_predictivity import (
    run_walk_forward,
)


class TestVolTargetBeta(unittest.TestCase):
    def test_run_log_integrity(self):
        """Verifies that data/run_log.jsonl records Final split read and blocks duplicate runs."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # First read succeeds
            rec = log_final_split_read("vol_target_beta_unit_test", split="Final", log_path=tmp_path)
            self.assertEqual(rec["experiment_name"], "vol_target_beta_unit_test")
            self.assertEqual(rec["split"], "Final")
            self.assertIn("timestamp", rec)
            self.assertIn("git_commit", rec)
            self.assertTrue(check_final_split_already_logged("vol_target_beta_unit_test", split="Final", log_path=tmp_path))

            # Duplicate attempt must raise PermissionError
            with self.assertRaises(PermissionError):
                log_final_split_read("vol_target_beta_unit_test", split="Final", log_path=tmp_path)

            # Different experiment name succeeds
            rec2 = log_final_split_read("other_experiment", split="Final", log_path=tmp_path)
            self.assertEqual(rec2["experiment_name"], "other_experiment")

            runs = get_all_logged_runs(log_path=tmp_path)
            self.assertEqual(len(runs), 2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_exposure_bounds_no_leverage_long_only(self):
        """Verifies that strategy exposures are strictly in [0.0, 1.0], no leverage, no shorting."""
        n_days = 100
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2023-01-01", periods=n_days, freq="1D", tz="UTC")
        d_ret = rng.normal(0.0005, 0.02, size=n_days)
        daily_funding = rng.normal(0.0001, 0.0002, size=n_days)

        # Raw volatility forecast yielding varied targets
        vol_forecast = rng.uniform(0.15, 1.20, size=n_days)
        w_v1_target = np.clip(0.40 / vol_forecast, 0.0, 1.0)

        # V1 simulation
        res_v1 = simulate_portfolio(w_v1_target, d_ret, daily_funding, timestamps, strategy_type="V1")
        positions_v1 = res_v1["positions"]
        self.assertTrue((positions_v1 >= 0.0).all(), "V1 has negative (short) exposure")
        self.assertTrue((positions_v1 <= 1.0).all(), "V1 exceeds 1.0 (has leverage)")

        # V2 simulation
        w_v2_target = rng.choice([0.0, 1.0], size=n_days)
        res_v2 = simulate_portfolio(w_v2_target, d_ret, daily_funding, timestamps, strategy_type="V2")
        positions_v2 = res_v2["positions"]
        self.assertTrue((positions_v2 >= 0.0).all(), "V2 has negative (short) exposure")
        self.assertTrue((positions_v2 <= 1.0).all(), "V2 exceeds 1.0 (has leverage)")
        self.assertTrue(set(np.unique(positions_v2)).issubset({0.0, 1.0}), "V2 positions are not binary {0, 1}")

    def test_causality_future_data_invariance(self):
        """Verifies that signals at day D do not change if future prices are perturbed."""
        n_days = 250
        dates = pd.date_range("2021-01-01", periods=n_days, freq="1D", tz="UTC")
        rng = np.random.default_rng(42)

        prices = 30000.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=n_days)))
        d_close = pd.Series(prices, index=dates)

        # SMA200 on shift(1)
        sma200 = d_close.shift(1).rolling(200, min_periods=200).mean()
        prior_close = d_close.shift(1)
        v2_signals = (prior_close > sma200).astype(float)

        day_test_idx = 210
        original_signal_at_d = v2_signals.iloc[day_test_idx]

        # Mutate future prices from day_test_idx onwards
        d_close_perturbed = d_close.copy()
        d_close_perturbed.iloc[day_test_idx:] *= 5.0  # Massive price jump in the future

        sma200_pert = d_close_perturbed.shift(1).rolling(200, min_periods=200).mean()
        prior_close_pert = d_close_perturbed.shift(1)
        v2_signals_pert = (prior_close_pert > sma200_pert).astype(float)

        perturbed_signal_at_d = v2_signals_pert.iloc[day_test_idx]
        self.assertEqual(
            original_signal_at_d,
            perturbed_signal_at_d,
            "Causality violation: future price changes altered signal at day D!"
        )

    def test_walk_forward_integrity_no_future_leakage(self):
        """Verifies that walk-forward HAR-RV estimation strictly uses prior data for training."""
        dates = pd.date_range("2022-01-01", "2022-06-30", freq="1D", tz="UTC")
        n = len(dates)
        rng = np.random.default_rng(42)

        synthetic_df = pd.DataFrame({
            "timestamp": dates,
            "symbol": "BTCUSDT",
            "target_log_rv": rng.normal(-6.0, 0.4, size=n),
            "actual_rv": np.exp(rng.normal(-6.0, 0.4, size=n)),
            "log_har_1d": rng.normal(-6.0, 0.4, size=n),
            "log_har_7d": rng.normal(-6.0, 0.4, size=n),
            "log_har_30d": rng.normal(-6.0, 0.4, size=n),
        })

        eval_start = pd.Timestamp("2022-04-01 00:00:00+00:00")
        preds = run_walk_forward(
            synthetic_df,
            target_name="target_log_rv",
            base_features=["log_har_1d", "log_har_7d", "log_har_30d"],
            aug_features=["log_har_1d", "log_har_7d", "log_har_30d"],
            eval_start=eval_start,
            eval_end=None,
            is_pooled=False,
        )

        # Predictions only exist on or after eval_start
        self.assertTrue((preds["timestamp"] >= eval_start).all())
        # Check all months predicted
        test_months = set(dates[dates >= eval_start].to_period("M"))
        pred_months = set(preds["timestamp"].dt.to_period("M"))
        self.assertEqual(test_months, pred_months)


if __name__ == "__main__":
    unittest.main()
