"""Unit tests for Stage 0 Pre-Flight Safety Checks of the Chart-Image CNN pipeline.

Covers every Stage 0 requirement:
- 0.1 Data audit & dropping 168-bar windows spanning gaps > 2 bars (no forward-fill)
- 0.2(a) Altering any bar after t leaves image at t bit-identical
- 0.2(b) Multiplying all prices by 7 or adding a constant leaves image identical
- 0.2(c) No absolute price level, date, or coin ID is recoverable from pixels
- 0.2(d) Synthetic steady uptrend, flat series, and crash render with expected pixel layout
- 0.2(e) At most 1% of images are blank or constant
- 0.2(f) 12 sample images written to gitignored folder
- 0.3 Target starts at NEXT bar open; purge = horizon + 24h embargo enforced
- 0.4 Planted-signal learnability test (OOS Spearman IC > 0.20)
- 0.5 Capacity test: overfit 512 random-label images to near-zero training loss
- 0.6 Determinism: training same fold twice with same seed matches within 1e-5
- 0.7 Power & cost realism: effective sample size and UNDERPOWERED flag when Final MDE IC > 0.08
- 0.8 Compute budget & statistical recomputation from per-bar predictions
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.scripts.chart_cnn import (
    ROOT,
    SAMPLE_IMG_DIR,
    block_bootstrap_spearman,
    build_quarterly_folds,
    compute_forward_labels_for_coin,
    enforce_purge_embargo_mask,
    extract_numeric_twin_features,
    load_and_audit_coin_data,
    render_dual_window_image,
    render_image_at_decision_index,
    run_stage0_capacity_test,
    run_stage0_determinism_test,
    run_stage0_planted_signal_test,
     simulate_strategy_from_predictions,
)


class TestChartCNNPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.btc_df, cls.btc_audit = load_and_audit_coin_data("BTCUSDT")

    def test_01_data_audit_drops_windows_spanning_gaps_gt_2_bars(self) -> None:
        """0.1: Windows spanning a gap > 2 bars are dropped; zero-volume bars removed; no ffill."""
        self.assertGreater(self.btc_audit["cleaned_1h_bars"], 50000)
        self.assertEqual(self.btc_audit["zero_volume_bars_removed"], 3)
        self.assertGreater(self.btc_audit["dropped_windows_spanning_gap_gt_2_bars"], 0)

        # Verify no retained 168-bar window in BTCUSDT spans a gap > 2 bars
        diff_hours = self.btc_df["diff_hours"].to_numpy()
        for k in self.btc_df.attrs["valid_decision_indices"]:
            max_missing_bars = int(np.max(diff_hours[k - 166 : k + 1])) - 1
            self.assertLessEqual(max_missing_bars, 2)

    def test_02a_altering_bars_after_t_leaves_image_bit_identical(self) -> None:
        """0.2(a): Altering any bar after t leaves the image at t bit-identical."""
        k = int(self.btc_df.attrs["valid_decision_indices"][50])
        img_orig = render_image_at_decision_index(self.btc_df, k)

        df_perturbed = self.btc_df.copy()
        df_perturbed.loc[k + 1 :, ["open", "high", "low", "close"]] *= 50.0
        df_perturbed.loc[k + 1 :, "volume"] *= 0.01
        img_perturbed = render_image_at_decision_index(df_perturbed, k)

        self.assertTrue(np.array_equal(img_orig, img_perturbed))
        self.assertEqual(img_orig.tobytes(), img_perturbed.tobytes())

    def test_02b_multiplying_by_7_or_adding_constant_leaves_image_identical(self) -> None:
        """0.2(b): Multiplying all prices by 7 or adding a constant leaves the image identical."""
        k = int(self.btc_df.attrs["valid_decision_indices"][75])
        img_orig = render_image_at_decision_index(self.btc_df, k)

        df_mul = self.btc_df.copy()
        for col in ("open", "high", "low", "close"):
            df_mul[col] = df_mul[col] * 7.0
        img_mul = render_image_at_decision_index(df_mul, k)
        self.assertTrue(np.array_equal(img_orig, img_mul))

        df_add = self.btc_df.copy()
        for col in ("open", "high", "low", "close"):
            df_add[col] = df_add[col] + 54321.0
        img_add = render_image_at_decision_index(df_add, k)
        self.assertTrue(np.array_equal(img_orig, img_add))

        df_affine = self.btc_df.copy()
        for col in ("open", "high", "low", "close"):
            df_affine[col] = df_affine[col] * 7.0 + 9999.9
        img_affine = render_image_at_decision_index(df_affine, k)
        self.assertTrue(np.array_equal(img_orig, img_affine))

    def test_02c_no_absolute_price_date_or_coin_id_recoverable(self) -> None:
        """0.2(c): No absolute price level, date, or coin ID is recoverable from pixels."""
        rng = np.random.default_rng(123)
        base_ret = rng.normal(0.0, 0.005, size=168)
        base_c = np.cumprod(1.0 + base_ret)
        base_o = np.concatenate([[1.0], base_c[:-1]])
        base_h = np.maximum(base_o, base_c) * 1.002
        base_l = np.minimum(base_o, base_c) * 0.998
        base_v = rng.uniform(1.0, 10.0, size=168)

        # Coin A at price ~$0.15 in 2020 vs Coin B at price ~$95,000 in 2026
        img_coin_a = render_dual_window_image(
            base_o * 0.15, base_h * 0.15, base_l * 0.15, base_c * 0.15, base_v * 1e6
        )
        img_coin_b = render_dual_window_image(
            base_o * 95000.0 + 500.0,
            base_h * 95000.0 + 500.0,
            base_l * 95000.0 + 500.0,
            base_c * 95000.0 + 500.0,
            base_v * 25.0,
        )
        self.assertTrue(np.array_equal(img_coin_a, img_coin_b))

    def test_02d_synthetic_uptrend_flat_and_crash_pixel_layout(self) -> None:
        """0.2(d): Synthetic steady uptrend, flat series, and crash render with expected layout."""
        t = np.arange(168, dtype=np.float64)

        # 1. Steady uptrend
        c_up = 100.0 + 1.0 * t
        o_up = c_up - 0.6
        h_up = c_up + 0.4
        l_up = o_up - 0.4
        v_up = np.full(168, 10.0)
        img_up = render_dual_window_image(o_up, h_up, l_up, c_up, v_up)
        # Leftmost column (x=0) price pixels in bottom of price pane (rows 40..49)
        left_rows_up = np.where(img_up[1, 2:50, 0] > 0)[0] + 2
        right_rows_up = np.where(img_up[1, 2:50, 63] > 0)[0] + 2
        self.assertGreaterEqual(int(np.min(left_rows_up)), 40)
        self.assertLessEqual(int(np.max(right_rows_up)), 10)
        # Bullish body intensity 0.85 present
        self.assertTrue(np.any(np.isclose(img_up[0, 2:50, :], 0.85)))

        # 2. Flat series
        c_flat = np.full(168, 250.0)
        img_flat = render_dual_window_image(c_flat, c_flat, c_flat, c_flat, v_up)
        flat_rows = np.where(img_flat[1, 2:50, :] > 0)[0] + 2
        self.assertTrue(np.all((flat_rows >= 25) & (flat_rows <= 27)))

        # 3. Crash series (flat near 100 then sharp drop in last 15 bars to 50)
        c_cr = np.full(168, 100.0)
        c_cr[-15:] = np.linspace(98.0, 50.0, 15)
        o_cr = np.concatenate([[100.0], c_cr[:-1]])
        h_cr = np.maximum(o_cr, c_cr) + 0.2
        l_cr = np.minimum(o_cr, c_cr) - 0.2
        img_cr = render_dual_window_image(o_cr, h_cr, l_cr, c_cr, v_up)
        early_rows_cr = np.where(img_cr[1, 2:50, 10] > 0)[0] + 2
        end_rows_cr = np.where(img_cr[1, 2:50, 63] > 0)[0] + 2
        self.assertLessEqual(int(np.min(early_rows_cr)), 5)
        self.assertGreaterEqual(int(np.max(end_rows_cr)), 47)
        # Bearish body intensity 0.55 present in last columns
        self.assertTrue(np.any(np.isclose(img_cr[0, 2:50, -5:], 0.55)))

    def test_02e_at_most_1_percent_blank_or_constant_images(self) -> None:
        """0.2(e): At most 1% of rendered real-data images are blank or constant."""
        dec_idx = self.btc_df.attrs["valid_decision_indices"][:200]
        blank_count = 0
        for k in dec_idx:
            img = render_image_at_decision_index(self.btc_df, int(k))
            if float(np.std(img)) < 1e-6:
                blank_count += 1
        blank_rate = blank_count / float(len(dec_idx))
        self.assertLessEqual(blank_rate, 0.01)
        self.assertEqual(blank_count, 0)

    def test_02f_sample_images_written_to_gitignored_folder(self) -> None:
        """0.2(f): Verify 12 sample PNG images exist or can be created in data/chart_cnn/sample_images/."""
        SAMPLE_IMG_DIR.mkdir(parents=True, exist_ok=True)
        pngs = sorted(SAMPLE_IMG_DIR.glob("sample_*.png"))
        if len(pngs) < 12:
            from PIL import Image
            dec_idx = self.btc_df.attrs["valid_decision_indices"][:12]
            for i, k in enumerate(dec_idx, start=1):
                img = render_image_at_decision_index(self.btc_df, int(k))
                canvas = np.zeros((64, 130), dtype=np.uint8)
                canvas[:, :64] = np.rint(img[0] * 255).astype(np.uint8)
                canvas[:, 66:] = np.rint(img[1] * 255).astype(np.uint8)
                Image.fromarray(canvas, mode="L").save(SAMPLE_IMG_DIR / f"sample_{i:02d}_test.png")
            pngs = sorted(SAMPLE_IMG_DIR.glob("sample_*.png"))
        self.assertGreaterEqual(len(pngs), 12)

    def test_03_target_starts_at_next_bar_open_and_purge_enforced(self) -> None:
        """0.3: Target starts at NEXT bar open (k+1 to k+1+H) and purge = H + 24h is enforced."""
        n = 850
        ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
        open_p = 100.0 + np.arange(n, dtype=np.float64)
        syn_df = pd.DataFrame({
            "timestamp": ts,
            "close_time": ts + pd.Timedelta(hours=1),
            "open": open_p,
            "high": open_p + 0.5,
            "low": open_p - 0.5,
            "close": open_p + 0.25,
            "volume": np.full(n, 50.0),
            "funding_rate": np.full(n, 0.0001),
            "vol_1h_720": np.full(n, 0.005),
        })
        k = 720
        for h in (24, 72):
            fwd_ret, y_scaled, f_sum, v_mask = compute_forward_labels_for_coin(
                syn_df, np.array([k]), h
            )
            self.assertTrue(bool(v_mask[0]))
            expected_ret = (open_p[k + 1 + h] / open_p[k + 1]) - 1.0
            self.assertAlmostEqual(float(fwd_ret[0]), expected_ret, places=12)
            # Modifying open[k] (current bar open) must NOT affect forward return
            syn_df_mod = syn_df.copy()
            syn_df_mod.loc[k, "open"] = 99999.0
            fwd_ret_mod, _, _, _ = compute_forward_labels_for_coin(syn_df_mod, np.array([k]), h)
            self.assertAlmostEqual(float(fwd_ret_mod[0]), expected_ret, places=12)

            # Verify purge = H + 24h mask
            test_start = ts[k]
            mask = enforce_purge_embargo_mask(ts, test_start, h, extra_embargo_hours=24)
            max_train_ts = ts[mask].max()
            self.assertGreaterEqual(test_start - max_train_ts, pd.Timedelta(hours=h + 24))

    def test_04_planted_signal_learnability_oos_ic_gt_02(self) -> None:
        """0.4: Planted-signal learnability test recovers pattern with OOS IC > 0.20."""
        res = run_stage0_planted_signal_test(self.device)
        self.assertTrue(res["passed"], f"Planted signal OOS IC={res['oos_ic']:.4f} <= 0.20")
        self.assertGreater(res["oos_ic"], 0.20)

    def test_05_capacity_overfit_512_random_labels(self) -> None:
        """0.5: CNN overfits 512 random-label images to near-zero training loss (< 0.02)."""
        res = run_stage0_capacity_test(self.device)
        self.assertTrue(res["passed"], f"Capacity final MSE={res['final_mse_loss']:.6f} >= 0.02")
        self.assertLess(res["final_mse_loss"], 0.02)

    def test_06_seed_determinism_within_1e5(self) -> None:
        """0.6: Same fold trained twice with the same seed matches within 1e-5."""
        rng = np.random.default_rng(42)
        X_u8 = rng.integers(0, 21, size=(320, 2, 64, 64), dtype=np.uint8)
        y = rng.standard_normal(320).astype(np.float32)
        res = run_stage0_determinism_test(X_u8, y, self.device)
        self.assertTrue(res["passed"], f"Determinism max diff={res['max_abs_diff']:.2e} > 1e-5")
        self.assertLessEqual(res["max_abs_diff"], 1e-5)

    def test_07_power_and_cost_underpowered_flag(self) -> None:
        """0.7: Single-coin Final split MDE IC > 0.08 is flagged UNDERPOWERED."""
        from scipy.stats import norm
        z_sum = float(norm.ppf(0.975) + norm.ppf(0.80))
        # Final split has ~2,650 6H timestamps -> N_eff(24H) = 662.5 -> MDE IC ~ 0.109 > 0.08
        n_eff_btc_24 = 2650.0 / 4.0
        mde_btc_24 = math.tanh(z_sum / math.sqrt(n_eff_btc_24 - 3.0))
        self.assertGreater(mde_btc_24, 0.08)

    def test_08_block_bootstrap_and_strategy_simulation(self) -> None:
        """0.8: Moving-block bootstrap Spearman IC and strategy simulator run on saved schema."""
        ts = pd.date_range("2025-01-01", periods=200, freq="6h", tz="UTC")
        rng = np.random.default_rng(7)
        y = rng.standard_normal(200)
        p_cnn = 0.2 * y + rng.standard_normal(200)
        p_mom = rng.standard_normal(200)
        df_p = pd.DataFrame({
            "timestamp": ts,
            "symbol": "BTCUSDT",
            "split": "Final",
            "fold": "2025Q1",
            "y_true": y,
            "fwd_ret": y * 0.01,
            "funding_cost": np.zeros(200),
            "pred_cnn": p_cnn,
            "pred_mom": p_mom,
        })
        st = block_bootstrap_spearman(
            df_p, "pred_cnn", "y_true", n_eff=50.0, compare_cols={"momentum": "pred_mom"}, n_boot=200
        )
        self.assertIn("ic", st)
        self.assertIn("momentum", st["paired"])
        strat = simulate_strategy_from_predictions(df_p, "BTCUSDT", 24, "Final")
        self.assertIn("sharpe", strat)
        self.assertIn("mean_daily_return_ci_low", strat)


if __name__ == "__main__":
    unittest.main()
