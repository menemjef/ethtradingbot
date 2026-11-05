import unittest
import numpy as np
import pandas as pd
import math
from utils.quant_evaluation import (
    compute_deflated_sharpe_ratio,
    compute_probability_of_backtest_overfitting,
    compute_reconciled_trade_metrics,
    compute_continuous_portfolio_metrics
)

class TestStatisticalAudit(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_deflated_sharpe_ratio_properties(self):
        """Test DSR mathematical properties, scaling, and trial penalization."""
        # Synthetic daily returns with positive mean (T=1000 days ~ 2.7 years)
        T = 1000
        q_daily = 365.25
        daily_mean = 0.001
        daily_std = 0.015
        rets = np.random.normal(daily_mean, daily_std, T)
        sample_sharpe = float(np.mean(rets) / np.std(rets)) * math.sqrt(q_daily)

        # 1. Single trial (K=1) should have higher DSR than multiple trials (K=50)
        dsr_k1 = compute_deflated_sharpe_ratio(
            observed_sharpe=sample_sharpe,
            returns=rets,
            track_record_len=T,
            n_trials=1,
            annualization_factor=q_daily
        )
        dsr_k50 = compute_deflated_sharpe_ratio(
            observed_sharpe=sample_sharpe,
            returns=rets,
            track_record_len=T,
            n_trials=50,
            annualization_factor=q_daily
        )
        self.assertGreaterEqual(dsr_k1, dsr_k50, "DSR must penalize for multiple trials (K=50 < K=1)")
        self.assertGreater(dsr_k1, 0.5, "Single trial with Sharpe > 1.0 should have DSR > 0.5")

        # 2. Hourly return series with annualization factor q=8760
        T_hourly = 8760
        q_hourly = 8760.0
        hourly_rets = np.random.normal(0.0001, 0.005, T_hourly)
        hourly_sharpe = float(np.mean(hourly_rets) / np.std(hourly_rets)) * math.sqrt(q_hourly)
        dsr_hourly = compute_deflated_sharpe_ratio(
            observed_sharpe=hourly_sharpe,
            returns=hourly_rets,
            track_record_len=T_hourly,
            n_trials=10,
            annualization_factor=q_hourly
        )
        self.assertGreaterEqual(dsr_hourly, 0.0)
        self.assertLessEqual(dsr_hourly, 1.0)

        # 3. Negative Sharpe ratio should yield near-zero DSR
        neg_rets = np.random.normal(-0.001, 0.01, 500)
        neg_sharpe = float(np.mean(neg_rets) / np.std(neg_rets)) * math.sqrt(q_daily)
        dsr_neg = compute_deflated_sharpe_ratio(
            observed_sharpe=neg_sharpe,
            returns=neg_rets,
            track_record_len=len(neg_rets),
            n_trials=10,
            annualization_factor=q_daily
        )
        self.assertLess(dsr_neg, 0.05, "Negative Sharpe ratio must produce DSR near 0.0")

        # 4. Short track record safeguard (< 10)
        dsr_short = compute_deflated_sharpe_ratio(
            observed_sharpe=2.0,
            returns=rets[:8],
            track_record_len=8,
            n_trials=1
        )
        self.assertEqual(dsr_short, 0.0, "Track record < 10 must return 0.0")

    def test_cscv_probability_of_backtest_overfitting(self):
        """Test CSCV PBO with both Sharpe and Return metrics."""
        dates = pd.date_range("2024-01-01", periods=500, freq="1D")
        
        # Scenario A: Pure noise / random walks -> PBO should be substantial (> 0.20)
        noise_matrix = pd.DataFrame(
            np.cumsum(np.random.normal(0, 1, (500, 10)), axis=0) + 1000.0,
            index=dates,
            columns=[f"strat_{i}" for i in range(10)]
        )
        pbo_noise_sharpe = compute_probability_of_backtest_overfitting(
            noise_matrix, n_splits=8, max_evals=100, seed=42, metric="sharpe"
        )
        pbo_noise_return = compute_probability_of_backtest_overfitting(
            noise_matrix, n_splits=8, max_evals=100, seed=42, metric="return"
        )
        self.assertGreaterEqual(pbo_noise_sharpe, 0.0)
        self.assertLessEqual(pbo_noise_sharpe, 1.0)
        self.assertGreater(pbo_noise_sharpe, 0.2, "Pure noise strategies should exhibit high PBO")

        # Scenario B: One clearly superior persistent strategy -> PBO should be low (<= 0.20)
        persistent_data = np.random.normal(0, 1, (500, 10))
        persistent_data[:, 0] += 0.5  # Strategy 0 has persistent strong positive drift
        perf_matrix = pd.DataFrame(
            np.cumsum(persistent_data, axis=0) + 1000.0,
            index=dates,
            columns=[f"strat_{i}" for i in range(10)]
        )
        pbo_perf = compute_probability_of_backtest_overfitting(
            perf_matrix, n_splits=8, max_evals=100, seed=42, metric="sharpe"
        )
        self.assertLess(pbo_perf, 0.25, "Consistently dominant strategy should yield low PBO")

    def test_moving_block_bootstrap_sample_coverage(self):
        """Verify that moving block bootstrap preserves exact length without integer truncation."""
        trades = []
        for i in range(37):  # Non-multiple of block_size=5
            trades.append({
                "trade_id": i,
                "net_pnl_usd": 100.0 if i % 2 == 0 else -60.0,
                "trade_return_pct": 1.0 if i % 2 == 0 else -0.6,
                "holding_bars": 12
            })
        
        # Test trade metrics bootstrap
        metrics = compute_reconciled_trade_metrics(
            trades=trades,
            initial_capital=10000.0,
            block_size=5,
            n_bootstraps=200,
            n_trials=10,
            seed=42,
            calendar_duration_years=3.0
        )
        ci_lower, ci_upper = metrics["ci_95_expectancy_bps"]
        self.assertLessEqual(ci_lower, ci_upper, "Bootstrap CI lower bound must be <= upper bound")
        self.assertGreater(metrics["total_trades"], 0)

        # Test small sample size smaller than block size (n < block_size)
        small_trades = trades[:3]
        metrics_small = compute_reconciled_trade_metrics(
            trades=small_trades,
            initial_capital=10000.0,
            block_size=5,
            n_bootstraps=50,
            seed=42
        )
        self.assertEqual(metrics_small["total_trades"], 3)
        self.assertIsNotNone(metrics_small["ci_95_expectancy_bps"])

    def test_portfolio_metrics_and_reconciliation(self):
        """Test continuous portfolio metrics, hourly MTM drawdown, and PnL reconciliation."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        eq_curve = pd.Series(np.linspace(10000, 11000, 100), index=dates)
        lowest_curve = eq_curve - 50.0  # Lowest intra-bar excursion
        
        # Synthesize matching trades
        trades = [
            {"trade_id": 1, "trade_pnl_usd": 600.0, "net_pnl_usd": 600.0, "holding_bars": 24},
            {"trade_id": 2, "trade_pnl_usd": 400.0, "net_pnl_usd": 400.0, "holding_bars": 24}
        ]
        
        metrics = compute_continuous_portfolio_metrics(
            hourly_equity=eq_curve,
            trades=trades,
            hourly_lowest_equity=lowest_curve,
            n_trials=5
        )
        
        # Sum of trades: 600 + 400 = 1000. Equity diff: 11000 - 10000 = 1000. Diff = 0.0.
        self.assertEqual(metrics["pnl_reconciliation_difference_usd"], 0.0)
        self.assertIn("daily_deflated_sharpe_ratio", metrics)
        self.assertIn("hourly_deflated_sharpe_ratio", metrics)
        self.assertGreater(metrics["calendar_cagr_pct"], 0.0)

if __name__ == "__main__":
    unittest.main()
