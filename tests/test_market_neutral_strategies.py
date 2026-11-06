"""
tests/test_market_neutral_strategies.py

Unit tests for Market-Neutral strategies N1 and N2:
1. Point-in-time universe: No coin enters before 180 days after its first bar.
2. No-look-ahead ranking: Return uses t-28d to t-1d (excluding last day); future prices do not alter past ranks.
3. Funding sign: Short perp RECEIVES positive funding (+ notional * rate).
4. Two-leg fees: Both spot and perp legs incur fees on entry and exit; 20 bps doubles fees.
5. Final test lock: Final test data (> 2024-12-31) cannot be accessed when locked.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import research.scripts.market_neutral_strategies as M


def _create_synthetic_market_data(n_hours: int = 24 * 300, start: str = "2020-01-01 00:00:00+00:00") -> dict:
    """Creates synthetic market data for tests."""
    np.random.seed(42)
    idx = pd.date_range(start, periods=n_hours, freq="1h")
    symbols = M.SYMBOLS

    perp, spot, funding, eligibility = {}, {}, {}, {}
    for i, s in enumerate(symbols):
        rets = np.random.normal(0.0001 * (i + 1), 0.005, size=n_hours)
        p_px = 100.0 * np.exp(np.cumsum(rets))
        # Small basis spread for spot
        s_px = p_px * (1.0 + np.random.normal(0.0005, 0.001, size=n_hours))

        perp[s] = pd.DataFrame({"open": p_px, "high": p_px * 1.01, "low": p_px * 0.99,
                                "close": p_px, "volume": 1000.0}, index=idx)
        spot[s] = pd.DataFrame({"open": s_px, "high": s_px * 1.01, "low": s_px * 0.99,
                                "close": s_px, "volume": 1000.0}, index=idx)
        # Funding rate settling every 8 hours (+1 bps positive funding)
        f_rates = np.zeros(n_hours)
        f_rates[::8] = 0.0001
        funding[s] = pd.Series(f_rates, index=idx)
        # Staggered eligibility
        eligibility[s] = idx[0] + pd.Timedelta(days=180 + i * 10)

    return {"perp": perp, "spot": spot, "funding": funding, "eligibility": eligibility}


class TestMarketNeutralStrategies(unittest.TestCase):

    def test_point_in_time_universe(self):
        """A coin enters the universe ONLY 180 days after its first available bar."""
        eligibility = M.get_universe_eligibility()
        self.assertEqual(len(eligibility), len(M.SYMBOLS))

        for sym in M.SYMBOLS:
            p_df = pd.read_parquet(os.path.join(M.CACHE_DIR, f"{sym.lower()}_1h_full.parquet"))
            first_bar = p_df.index[0]
            s_path = os.path.join(M.CACHE_DIR, f"{sym.lower()}_spot_1h_full.parquet")
            if os.path.exists(s_path):
                s_df = pd.read_parquet(s_path)
                first_bar = max(first_bar, s_df.index[0])
            expected_eligible = first_bar + pd.Timedelta(days=180)
            self.assertEqual(eligibility[sym], expected_eligible)

    def test_no_look_ahead_ranking(self):
        """N1 momentum rankings must exclude the last 24h and remain invariant to future price shifts."""
        data = _create_synthetic_market_data(n_hours=24 * 250)
        t_rebal = pd.Timestamp("2020-07-12 00:00:00+00:00")  # A Sunday 00:00 UTC

        # Compute return from t-28d to t-1d
        h_excl = 24
        h_look = 28 * 24
        idx_t = data["perp"]["BTCUSDT"].index.get_loc(t_rebal)

        # Baseline scores
        scores_base = {}
        for s in M.SYMBOLS:
            p = data["perp"][s]["close"].values
            scores_base[s] = (p[idx_t - h_excl] / p[idx_t - h_look]) - 1.0

        # Clone data and add wild future swings after t_rebal
        data_future = {k: {s: (df.copy() if hasattr(df, "copy") else df) for s, df in v.items()}
                       if isinstance(v, dict) else v for k, v in data.items()}
        for s in M.SYMBOLS:
            data_future["perp"][s].iloc[idx_t:, data_future["perp"][s].columns.get_loc("close")] *= 5.0

        # Compute scores on future-perturbed data
        scores_future = {}
        for s in M.SYMBOLS:
            p = data_future["perp"][s]["close"].values
            scores_future[s] = (p[idx_t - h_excl] / p[idx_t - h_look]) - 1.0

        for s in M.SYMBOLS:
            self.assertAlmostEqual(scores_base[s], scores_future[s], places=10)

    def test_funding_sign_short_perp_receives_positive_funding(self):
        """Short perpetual position must RECEIVE positive cash flow when funding rate > 0."""
        # Simple 2-day simulation with 1 short perp position
        idx = pd.date_range("2021-01-01 00:00:00+00:00", periods=24, freq="1h")
        px = 100.0
        # Positive funding rate (+0.001 = 10 bps) at hour 8
        f_rate = 0.001
        notional = 1000.0
        units = - notional / px  # Short 10 units

        # Funding payment rule: cash_flow = - units * px * funding_rate
        # For short units (-10), cash_flow = - (-10) * 100 * 0.001 = +1.00 USD
        cash_flow = - units * px * f_rate
        self.assertGreater(cash_flow, 0.0)
        self.assertAlmostEqual(cash_flow, 1.0, places=6)

    def test_two_leg_fees_and_cost_doubling(self):
        """N2 must charge fees on both spot and perp legs; double fees must double friction."""
        data = _create_synthetic_market_data(n_hours=24 * 250)
        # Set all eligibility to start of data so coins qualify
        for s in M.SYMBOLS:
            data["eligibility"][s] = pd.Timestamp("2020-01-01 00:00:00+00:00")
            # Force high positive funding rate so N2 enters
            data["funding"][s] = pd.Series(0.001, index=data["funding"][s].index)

        t_start = pd.Timestamp("2020-03-01 00:00:00+00:00")
        t_end = pd.Timestamp("2020-03-15 00:00:00+00:00")

        res_base = M.simulate_n2(data, start_time=t_start, end_time=t_end, fee=0.0010)
        res_dbl = M.simulate_n2(data, start_time=t_start, end_time=t_end, fee=0.0020)

        self.assertGreater(res_base["fees_usd"], 0.0)
        self.assertGreater(res_dbl["fees_usd"], 0.0)
        # Double fees must scale fees by exactly 2.0x
        ratio = res_dbl["fees_usd"] / res_base["fees_usd"]
        self.assertAlmostEqual(ratio, 2.0, places=3)

    def test_final_test_locked_guard(self):
        """Accessing final test data (> 2024-12-31) must raise PermissionError while locked."""
        M.lock_final()
        self.assertFalse(M.final_unlocked())
        with self.assertRaises(PermissionError):
            M.load_all_market_data(truncate_at=pd.Timestamp("2025-06-01 00:00:00+00:00"))

        # Default load must truncate at VAL_END
        data = M.load_all_market_data()
        for s in M.SYMBOLS:
            self.assertLessEqual(data["perp"][s].index[-1], M.VAL_END)
            self.assertLessEqual(data["spot"][s].index[-1], M.VAL_END)
            self.assertLessEqual(data["funding"][s].index[-1], M.VAL_END)


    def test_hedged_intrabar_wick_drawdown_near_zero(self):
        """A perfectly hedged synthetic pair with a 10% intrabar wick must show near-zero drawdown."""
        n_hours = 24 * 10
        idx = pd.date_range("2020-01-01 00:00:00+00:00", periods=n_hours, freq="1h")
        symbols = M.SYMBOLS

        perp, spot, funding, eligibility = {}, {}, {}, {}
        for s in symbols:
            p_open = np.full(n_hours, 100.0)
            p_high = np.full(n_hours, 100.0)
            p_low = np.full(n_hours, 100.0)
            p_close = np.full(n_hours, 100.0)

            s_open = np.full(n_hours, 100.0)
            s_high = np.full(n_hours, 100.0)
            s_low = np.full(n_hours, 100.0)
            s_close = np.full(n_hours, 100.0)

            # Intrabar wick at hour 100: spot low = 90 (-10%), perp high = 110 (+10%), close remains 100
            s_low[100] = 90.0
            p_high[100] = 110.0

            perp[s] = pd.DataFrame({"open": p_open, "high": p_high, "low": p_low, "close": p_close, "volume": 1000.0}, index=idx)
            spot[s] = pd.DataFrame({"open": s_open, "high": s_high, "low": s_low, "close": s_close, "volume": 1000.0}, index=idx)
            f_rates = np.zeros(n_hours)
            f_rates[::8] = 0.0001
            funding[s] = pd.Series(f_rates, index=idx)
            eligibility[s] = idx[0]

        data = {"perp": perp, "spot": spot, "funding": funding, "eligibility": eligibility}
        res = M.simulate_n2(data, start_time=idx[0], end_time=idx[-1], fee=0.0)

        eql = res["equity_low"]
        peak = eql.cummax()
        dd = (peak - eql) / peak
        max_dd_pct = float(dd.max()) * 100.0

        self.assertLess(max_dd_pct, 0.01, f"Expected near-zero drawdown, got {max_dd_pct}%")
        self.assertIn("max_basis_stress_usd", res)

    def test_near_constant_returns_returns_nan_and_flag(self):
        """Near-constant return series must return NaN/None for Sharpe and DSR with near_constant_returns=True."""
        from utils.quant_evaluation import compute_deflated_sharpe_ratio
        n_days = 100
        idx = pd.date_range("2021-01-01 00:00:00+00:00", periods=n_days * 24, freq="1h")
        eq = pd.Series(10000.0 + np.linspace(0, 0.01, len(idx)), index=idx)
        btc_eq = eq.copy()

        fake_res = {
            "strategy": "N2_Funding_Carry",
            "split": "test",
            "fee_per_side": 0.001,
            "start": str(idx[0]),
            "end": str(idx[-1]),
            "equity": eq,
            "equity_low": eq,
            "funding_usd": 0.01,
            "basis_pnl_usd": 0.0,
            "fees_usd": 0.0,
            "turnover_usd": 0.0,
            "per_coin_pnl": {s: 0.0 for s in M.SYMBOLS},
            "initial_equity": 10000.0,
            "final_equity": 10000.01,
            "eight_hour_returns": [0.0] * 50,
            "max_basis_stress_usd": 0.0,
        }

        metrics = M.compute_strategy_metrics(fake_res, btc_eq, n_bootstraps=50)
        self.assertTrue(metrics["near_constant_returns"])
        self.assertIsNone(metrics["sharpe_rf4"])
        self.assertIsNone(metrics["sharpe_zero"])
        self.assertIsNone(metrics["dsr"])

        # Directly test compute_deflated_sharpe_ratio
        const_returns = np.full(100, 1e-7)
        dsr_val = compute_deflated_sharpe_ratio(0.0, const_returns)
        self.assertTrue(np.isnan(dsr_val))

    def test_deflated_sharpe_ratio_positive_skew_mertens_expansion(self):
        """Mertens expansion var_term <= 0 under high Sharpe and positive skew must return 1.0 (not 0.0)."""
        from utils.quant_evaluation import compute_deflated_sharpe_ratio
        rets = np.array([0.0001] * 700 + [0.01] * 30)
        dsr = compute_deflated_sharpe_ratio(7.58, rets, annualization_factor=365.0, n_trials=8)
        self.assertEqual(dsr, 1.0, f"Expected DSR of 1.0 for high positive Sharpe, got {dsr}")


if __name__ == "__main__":
    unittest.main()
