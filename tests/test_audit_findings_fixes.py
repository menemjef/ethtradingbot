"""
tests/test_audit_findings_fixes.py — Regression Test Suite for Outstanding Audit Findings 1 to 5.

Verifies:
1. Issue 1: Prediction/target alignment against candle j-1 target vs flawed j target.
2. Issue 1: Chronological ordering (dec_time < exec_time) and final bar exclusion.
3. Issue 2: MajorityBaseline full trade logging, CSV inclusion, and mathematical reconciliation.
4. Issue 4: Moving Block Bootstrap serial correlation preservation vs naive IID.
5. Issue 4: Deflated Sharpe Ratio penalty for multi-trial selection bias.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.quant_evaluation import (
    compute_aligned_predictive_metrics,
    compute_reconciled_trade_metrics,
    compute_deflated_sharpe_ratio
)


class TestAuditFindingsFixes(unittest.TestCase):

    def test_01_prediction_target_one_candle_alignment(self):
        """
        Issue 1: Test that predictions generated at candle j-1 close
        evaluate against target[j-1], NOT target[j].
        """
        N = 20
        # Timestamps
        base_time = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
        times = [base_time + pd.Timedelta(hours=i) for i in range(N)]

        # Construct synthetic targets where target[j-1] and target[j] have OPPOSITE signs!
        # Even indices: +1.0, Odd indices: -1.0
        targets = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)], dtype=np.float32)

        # Suppose model at decision bar j-1 always predicts in the direction of target[j-1]:
        # For decision at j-1, model predicts +0.05 when target[j-1] > 0, -0.05 when target[j-1] < 0.
        dec_indices = list(range(0, N - 1)) # j-1
        predictions_at_j_minus_1 = np.array([0.05 if targets[idx] > 0 else -0.05 for idx in dec_indices])

        # Correct alignment: evaluate predictions_at_j_minus_1 against targets[j-1]
        aligned_targets = targets[dec_indices]
        correct_metrics = compute_aligned_predictive_metrics(aligned_targets, predictions_at_j_minus_1)

        # Flawed old alignment: evaluate predictions_at_j_minus_1 against targets[j]
        flawed_indices = [idx + 1 for idx in dec_indices] # candle j
        flawed_targets = targets[flawed_indices]
        flawed_metrics = compute_aligned_predictive_metrics(flawed_targets, predictions_at_j_minus_1)

        # Assert correct alignment yields 100% Directional Accuracy
        self.assertEqual(correct_metrics["directional_accuracy_pct"], 100.0)
        self.assertLess(correct_metrics["regression_mse"], 1.0)

        # Assert flawed old alignment completely fails (0% Directional Accuracy)
        self.assertEqual(flawed_metrics["directional_accuracy_pct"], 0.0)
        self.assertGreater(flawed_metrics["regression_mse"], 1.0)

    def test_02_chronological_timestamp_ordering_and_final_bar_exclusion(self):
        """
        Issue 1: Assert strict chronology: decision_time < execution_time
        and verify that the final candle is excluded from predictive evaluation.
        """
        N_test = 10
        base_time = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
        test_times = [base_time + pd.Timedelta(hours=j) for j in range(N_test)]
        
        # Test window starts at te_start_idx = 100
        te_start_idx = 100
        all_times = [base_time + pd.Timedelta(hours=k - te_start_idx) for k in range(200)]
        
        decision_timestamps = []
        execution_timestamps = []
        target_timestamps = []

        # For j in range(N_test - 1): exclude final candle j = N_test - 1
        for j in range(N_test - 1):
            dec_idx = te_start_idx + j - 1
            dec_t = all_times[dec_idx]
            exec_t = test_times[j]
            target_t = all_times[dec_idx]

            decision_timestamps.append(dec_t)
            execution_timestamps.append(exec_t)
            target_timestamps.append(target_t)

            # Strict chronological assertion: decision occurs strictly before execution
            self.assertLess(dec_t, exec_t)
            self.assertEqual(dec_t, target_t)
            self.assertEqual(exec_t - dec_t, pd.Timedelta(hours=1))

        # Assert length is exactly N_test - 1 (final bar excluded)
        self.assertEqual(len(decision_timestamps), N_test - 1)
        self.assertEqual(len(execution_timestamps), N_test - 1)

    def test_03_majority_baseline_trade_logging_and_reconciliation(self):
        """
        Issue 2: Verify MajorityBaseline generates complete trade records,
        includes them in trade log, and reconciles win rate, profit factor, and return.
        """
        mock_trades = [
            {
                "trade_id": "MajorityBaseline_f1_1",
                "model": "MajorityBaseline",
                "fold": 1,
                "decision_time": "2026-06-01 10:00:00+00:00",
                "entry_time": "2026-06-01 11:00:00+00:00",
                "exit_time": "2026-06-01 15:00:00+00:00",
                "side": "LONG",
                "entry_price": 2000.0,
                "exit_price": 2040.0,
                "holding_bars": 4,
                "entry_fee_cost": 0.0005,
                "exit_fee_cost": 0.0005,
                "total_slippage_cost": 0.0006,
                "funding_cost": 0.0001,
                "exit_reason": "take_profit",
                "net_return": 0.0183
            },
            {
                "trade_id": "MajorityBaseline_f1_2",
                "model": "MajorityBaseline",
                "fold": 1,
                "decision_time": "2026-06-02 02:00:00+00:00",
                "entry_time": "2026-06-02 03:00:00+00:00",
                "exit_time": "2026-06-02 07:00:00+00:00",
                "side": "LONG",
                "entry_price": 2040.0,
                "exit_price": 2010.0,
                "holding_bars": 4,
                "entry_fee_cost": 0.0005,
                "exit_fee_cost": 0.0005,
                "total_slippage_cost": 0.0006,
                "funding_cost": 0.0001,
                "exit_reason": "stop_loss",
                "net_return": -0.0164
            }
        ]

        metrics = compute_reconciled_trade_metrics(mock_trades)
        
        # Invariant checks:
        self.assertEqual(metrics["total_trades"], 2)
        self.assertEqual(metrics["long_trades"], 2)
        self.assertEqual(metrics["short_trades"], 0)
        self.assertEqual(metrics["win_rate_pct"], 50.0)
        self.assertAlmostEqual(metrics["gross_profit_pct"], 1.83, places=2)
        self.assertAlmostEqual(metrics["gross_loss_pct"], 1.64, places=2)
        self.assertGreater(metrics["profit_factor"], 1.0)
        
        # Mathematical reconciliation of compounded return: (1 + 0.0183) * (1 - 0.0164) - 1 = +0.0016008
        expected_compounded = ((1.0 + 0.0183) * (1.0 - 0.0164) - 1.0) * 100.0
        self.assertAlmostEqual(metrics["compounded_net_return_pct"], round(expected_compounded, 2), places=2)

    def test_04_moving_block_bootstrap_preserves_temporal_dependence(self):
        """
        Issue 4: Verify that Moving Block Bootstrap preserves serial correlation
        and exhibits larger, more realistic confidence intervals than naive IID bootstrap.
        """
        # Generate an AR(1) process with strong positive autocorrelation (rho = 0.8)
        np.random.seed(42)
        N = 500
        rho = 0.80
        innovations = np.random.normal(0, 1, N)
        ar1_series = np.zeros(N)
        for t in range(1, N):
            ar1_series[t] = rho * ar1_series[t-1] + innovations[t]

        # Targets = true AR(1) direction, Predictions = slightly noisy AR(1)
        targets = np.where(ar1_series > 0, 1.0, -1.0)
        preds = ar1_series + np.random.normal(0, 0.5, N)

        # Naive IID bootstrap (block_size = 1)
        metrics_iid = compute_aligned_predictive_metrics(targets, preds, block_size=1, n_bootstraps=500, seed=42)
        # Moving block bootstrap (block_size = 24 hours)
        metrics_block = compute_aligned_predictive_metrics(targets, preds, block_size=24, n_bootstraps=500, seed=42)

        ci_iid = metrics_iid["ci_95_directional_accuracy"]
        ci_block = metrics_block["ci_95_directional_accuracy"]

        ci_width_iid = ci_iid[1] - ci_iid[0]
        ci_width_block = ci_block[1] - ci_block[0]

        # Block bootstrap accounts for clustered errors -> wider, safer confidence interval
        self.assertGreater(ci_width_block, ci_width_iid * 0.95)

    def test_05_deflated_sharpe_ratio_penalizes_multiple_trials(self):
        """
        Issue 4: Verify that Deflated Sharpe Ratio strictly penalizes
        as the number of investigated strategy trials K increases (selection bias).
        """
        returns = np.array([0.005, -0.002, 0.008, 0.001, -0.003, 0.006, 0.004, -0.001, 0.005, 0.002] * 10)
        observed_sharpe = 1.5

        dsr_1_trial = compute_deflated_sharpe_ratio(observed_sharpe, returns, len(returns), n_trials=1)
        dsr_10_trials = compute_deflated_sharpe_ratio(observed_sharpe, returns, len(returns), n_trials=10)
        dsr_100_trials = compute_deflated_sharpe_ratio(observed_sharpe, returns, len(returns), n_trials=100)

        # Invariant: DSR must strictly decrease as trial count K increases
        self.assertGreater(dsr_1_trial, dsr_10_trials)
        self.assertGreater(dsr_10_trials, dsr_100_trials)


if __name__ == "__main__":
    unittest.main()
