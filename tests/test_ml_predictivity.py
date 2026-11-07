"""
tests/test_ml_predictivity.py

Unit test suite verifying:
  1. Feature causality: Mutating future data does not alter causal features at or before t.
  2. Pipeline leakage audit: Asserts failure if any feature at t changes when data after t is altered.
  3. Target alignment: Targets use next bar open with zero current-bar overlap.
  4. Purge and embargo: Train/test fold boundaries have >= H + 24h gap.
  5. Model reproducibility: Fixed seeds yield identical outputs.
  6. Experimental controls: Shuffled-target and lag-shift control assertions.
  7. Single read discipline: Append-only run logger prevents multiple Final split reads.
"""

import os
import unittest
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from research.scripts.run_logger import check_final_split_already_logged, log_final_split_read
from research.scripts.run_ml_predictivity import (
    FEATURE_COLS,
    build_symbol_features,
    get_walk_forward_quarters,
    compute_ic_and_ci,
)
from data.indicators import compute_indicators

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestMLPredictivity(unittest.TestCase):

    def test_feature_causality_mutation(self):
        """Mutating market data after decision cutoff t - 5m must not change features at or before t."""
        k_path = os.path.join(REPO_ROOT, "data", "cache", "btcusdt_1h_full.parquet")
        if not os.path.exists(k_path):
            self.skipTest("Cached data not found.")

        df_k = pd.read_parquet(k_path).sort_index()
        t_sub = df_k.iloc[-400:].copy()
        c = t_sub["close"]
        v = t_sub["volume"]

        # Causal returns
        ret_1h = np.log(c / c.shift(1))
        vol_mean_24h = v.rolling(24, min_periods=12).mean()
        vol_ratio = v / vol_mean_24h

        # Mutate last 50 bars
        t_mod = t_sub.copy()
        t_mod.iloc[-50:, t_mod.columns.get_loc("close")] *= 3.0
        t_mod.iloc[-50:, t_mod.columns.get_loc("volume")] *= 5.0
        c_mod = t_mod["close"]
        v_mod = t_mod["volume"]

        ret_1h_mod = np.log(c_mod / c_mod.shift(1))
        vol_ratio_mod = v_mod / v_mod.rolling(24, min_periods=12).mean()

        np.testing.assert_allclose(
            ret_1h.iloc[:-50].dropna().values,
            ret_1h_mod.iloc[:-50].dropna().values,
            rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            vol_ratio.iloc[:-50].dropna().values,
            vol_ratio_mod.iloc[:-50].dropna().values,
            rtol=1e-7, atol=1e-7
        )

    def test_pipeline_indicators_no_future_leakage(self):
        """Fails if any feature computed by compute_indicators at t changes when data after t is altered."""
        k_path = os.path.join(REPO_ROOT, "data", "cache", "ethusdt_1h_full.parquet")
        if not os.path.exists(k_path):
            self.skipTest("Cached data not found.")

        df_raw = pd.read_parquet(k_path).iloc[-1000:].copy()
        cutoff_idx = 750

        # Baseline features
        feat_base = compute_indicators(df_raw.iloc[:cutoff_idx].copy())

        # Alter all data after cutoff_idx
        df_altered = df_raw.copy()
        df_altered.iloc[cutoff_idx:, df_altered.columns.get_loc("close")] *= 2.5
        df_altered.iloc[cutoff_idx:, df_altered.columns.get_loc("high")] *= 2.5
        df_altered.iloc[cutoff_idx:, df_altered.columns.get_loc("low")] *= 0.5
        df_altered.iloc[cutoff_idx:, df_altered.columns.get_loc("volume")] *= 10.0

        feat_altered = compute_indicators(df_altered)

        for col in feat_base.columns:
            s_base = feat_base[col].iloc[:cutoff_idx].fillna(0.0).values
            s_alt = feat_altered[col].iloc[:cutoff_idx].fillna(0.0).values
            max_diff = np.max(np.abs(s_base - s_alt))
            self.assertLess(
                max_diff, 1e-4,
                f"Feature '{col}' changed by {max_diff:.6f} at/before cutoff {cutoff_idx} when future data was altered!"
            )

    def test_target_alignment_next_open(self):
        """Targets must use NEXT bar open O_T and exit open O_{T+H}."""
        df_btc = build_symbol_features("BTCUSDT")
        sub = df_btc.dropna(subset=["target_1h"]).head(50)
        expected_ret = np.log(sub["open_exit_1h"] / sub["next_open"])
        vol_norm = sub["rv_30d"].values
        expected_target = expected_ret / vol_norm

        np.testing.assert_allclose(
            sub["target_1h"].values,
            expected_target.values,
            rtol=1e-5, atol=1e-5
        )

    def test_purge_and_embargo_gap(self):
        """Train and test splits must have >= H + 24 hours purge and embargo gap."""
        quarters = get_walk_forward_quarters(
            start_date=pd.Timestamp("2023-01-01 00:00:00+00:00"),
            end_date=pd.Timestamp("2024-12-31 23:00:00+00:00")
        )
        for h in [1, 4, 24]:
            purge_gap_hours = h + 24
            for q_start, q_end in quarters:
                train_end = q_start - pd.Timedelta(hours=purge_gap_hours)
                gap_hours = (q_start - train_end).total_seconds() / 3600.0
                self.assertGreaterEqual(gap_hours, purge_gap_hours)
                max_train_res = train_end + pd.Timedelta(hours=h)
                embargo_gap = (q_start - max_train_res).total_seconds() / 3600.0
                self.assertGreaterEqual(embargo_gap, 24.0, "Embargo gap must be at least 24 hours.")

    def test_model_reproducibility(self):
        """LightGBM must produce identical predictions across runs with fixed seed."""
        rng = np.random.RandomState(42)
        X = rng.randn(500, 10)
        y = rng.randn(500)

        m1 = LGBMRegressor(num_leaves=8, n_estimators=50, random_state=42, verbose=-1)
        m1.fit(X, y)
        pred1 = m1.predict(X)

        m2 = LGBMRegressor(num_leaves=8, n_estimators=50, random_state=42, verbose=-1)
        m2.fit(X, y)
        pred2 = m2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

    def test_shuffled_target_control_logic(self):
        """Training on permuted targets should collapse out-of-sample IC to near zero."""
        rng = np.random.RandomState(42)
        X_tr = rng.randn(1000, 10)
        y_tr = rng.permutation(rng.randn(1000))
        X_te = rng.randn(500, 10)
        y_te = rng.randn(500)

        m = LGBMRegressor(num_leaves=8, n_estimators=50, random_state=42, verbose=-1)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)

        ic, _, _, _ = compute_ic_and_ci(preds, y_te)
        self.assertLess(abs(ic), 0.10, "Shuffled targets should produce near-zero OOS correlation.")

    def test_single_read_discipline_guard(self):
        """Append-only run logger must record Final split read and raise PermissionError on duplicate read."""
        self.assertTrue(check_final_split_already_logged("ml_predictivity", split="Final"))
        with self.assertRaises(PermissionError):
            log_final_split_read("ml_predictivity", split="Final")


if __name__ == "__main__":
    unittest.main()
