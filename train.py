"""
train.py — Full training pipeline entry point.

WHAT THIS FILE DOES:
This is the machine learning pipeline. It grabs historical crypto data, 
calculates indicators, scales the data, and trains the Neural Network to 
predict future price movements. It saves the resulting "brain" to the checkpoints folder.

Usage:
    python train.py                     # Full training (2500 candles)
    python train.py --fast              # Smoke-test (500 candles, 5 epochs)
"""

import argparse
import io
import os
import sys
import logging
import numpy as np
import pandas as pd
from utils.logger import setup_logging
from utils.seeding import set_seed
setup_logging()
set_seed(42)  # Ensure reproducibility across all runs

import config
from data.fetcher import fetch_ohlcv
from data.preprocessor import (
    build_features,
    fit_and_save_scalers,
    transform,
    build_windows,
)
from data.indicators import add_tbm_target
from model.trainer import train_model
from config import (
    TBM_PT_MULT,
    TBM_SL_MULT,
    TARGET_HORIZON,
    SYMBOL,
    TIMEFRAME,
    FETCH_LIMIT,
    LOOKBACK,
)

# ── Logging setup ──────────────────────────────────────────────────────────
# Handled by utils.logger.setup_logging()
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="ETH Trading Signal — Training Script")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke-test mode: 500 candles, 5 epochs max",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: 30 epochs max",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=FETCH_LIMIT,
        help=f"Number of historical candles to fetch (default: {FETCH_LIMIT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for saved model/scaler files",
    )
    parser.add_argument(
        "--use_cache",
        type=str,
        default="",
        help="Path to a local CSV candle cache (skips Binance fetch — useful for local GPU training)",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=720,
        help="Extra pre-roll candles to fetch for indicator warm-up (default: 720 for 30d Fibonacci)",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Allow historical data gaps without raising ValueError (default: False, gaps raise error)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.60,
        help="Fraction of data for training partition (default: 0.60)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Fraction of data for validation/calibration partition (default: 0.20)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run full out-of-sample walk-forward cross-validation",
    )
    parser.add_argument(
        "--wf-folds",
        type=int,
        default=4,
        help="Number of folds for walk-forward evaluation (default: 4)",
    )
    parser.add_argument(
        "--pos-weight-mode",
        type=str,
        default="balanced",
        choices=["balanced", "unweighted", "legacy_1.17"],
        help="Positive class weighting strategy for BCE (default: balanced from training data)",
    )
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="regression_mse",
        choices=["regression_mse", "classification_loss", "directional_accuracy"],
        help="Validation metric for checkpoint selection (default: regression_mse)",
    )
    parser.add_argument(
        "--ambiguous-resolution",
        type=str,
        default="exclude",
        choices=["exclude", "stop_first", "overshoot"],
        help="Resolution for candles breaching both PT and SL (default: exclude for directional training)",
    )
    args = parser.parse_args()

    # Apply seed
    set_seed(args.seed)

    # Resolve artifact paths dynamically from config (allowing safe test patching & --suffix)
    model_path = config.MODEL_PATH
    scaler_candle_path = config.SCALER_CANDLE_PATH
    scaler_market_path = config.SCALER_MARKET_PATH
    if args.suffix:
        model_path = model_path.replace(".pt", f"_{args.suffix}.pt")
        scaler_candle_path = scaler_candle_path.replace(".pkl", f"_{args.suffix}.pkl")
        scaler_market_path = scaler_market_path.replace(".pkl", f"_{args.suffix}.pkl")

    limit = 500 if (args.fast and args.limit == FETCH_LIMIT) else args.limit

    # Explicitly determine max_epochs and patience and pass them to train_model()
    if args.fast:
        max_epochs = 5
        patience = 3
        logger.info("FAST MODE: capping at %d candles, %d epochs (patience=%d)", limit, max_epochs, patience)
    elif args.test:
        max_epochs = 30
        patience = config.PATIENCE
        logger.info("TEST MODE: running for %d epochs (patience=%d)", max_epochs, patience)
    else:
        max_epochs = config.MAX_EPOCHS
        patience = config.PATIENCE

    # ── 1. Fetch data ────────────────────────────────────────────────────────
    # STEP 1: Download historical crypto candles (Open, High, Low, Close, Volume)
    logger.info("=" * 60)
    if args.use_cache and os.path.exists(args.use_cache):
        logger.info("Step 1/5 — Loading candles from local cache: %s", args.use_cache)
        df_raw = pd.read_csv(args.use_cache, index_col=0, parse_dates=True)
        if df_raw.index.tz is None:
            df_raw.index = df_raw.index.tz_localize('UTC')
        else:
            df_raw.index = df_raw.index.tz_convert('UTC')
        df_raw = df_raw.sort_index()
        dup_count = int(df_raw.index.duplicated().sum())
        if dup_count > 0:
            logger.warning("Detected %d duplicate timestamps in cache. Deduplicating.", dup_count)
            df_raw = df_raw[~df_raw.index.duplicated(keep="last")]
        if len(df_raw) > 1 and not args.allow_gaps:
            expected_range = pd.date_range(start=df_raw.index[0], end=df_raw.index[-1], freq="1h")
            missing = expected_range.difference(df_raw.index)
            if len(missing) > 0:
                raise ValueError(
                    f"Cache data integrity failure: Detected {len(missing)} missing intervals "
                    f"between {df_raw.index[0]} and {df_raw.index[-1]}"
                )
        if len(df_raw) >= limit:
            requested_start_ts = df_raw.index[-limit]
        else:
            requested_start_ts = df_raw.index[0]
    else:
        logger.info("Step 1/5 — Fetching %d x %s candles for %s (warmup_bars: %d)", limit, TIMEFRAME, SYMBOL, args.warmup_bars)
        df_raw = fetch_ohlcv(
            SYMBOL,
            TIMEFRAME,
            limit=limit,
            warmup_bars=args.warmup_bars,
            raise_on_gap=not args.allow_gaps,
        )
        requested_start_ts = df_raw.index[-limit] if len(df_raw) >= limit else df_raw.index[0]

    logger.info("Raw data shape: %s | Date range: %s -> %s",
                df_raw.shape, df_raw.index[0], df_raw.index[-1])


    # ── Walk-Forward Mode (if requested) ───────────────────────────────────
    if args.walk_forward:
        logger.info("=" * 60)
        logger.info("Running Walk-Forward Cross-Validation (%d folds)...", args.wf_folds)
        from model.walk_forward import run_walk_forward_evaluation
        wf_results = run_walk_forward_evaluation(
            df_raw,
            n_splits=args.wf_folds,
            train_ratio=0.50,
            test_ratio=0.125,
            purge_horizon=TARGET_HORIZON,
            max_epochs=max_epochs,
        )
        logger.info("=" * 60)
        logger.info("WALK-FORWARD EVALUATION COMPLETE:")
        agg = wf_results["aggregate_summary"]
        logger.info("  Mean Directional Accuracy: %.2f%% (std: %.2f%%)", agg["mean_directional_accuracy_pct"], agg["std_directional_accuracy_pct"])
        logger.info("  Mean Trade Win Rate:      %.2f%% (Pooled: %.2f%%)", agg.get("mean_win_rate_pct", 0.0), agg.get("pooled_win_rate_pct", 0.0))
        logger.info("  Pooled Profit Factor:     %.2f (Fold Mean: %.2f)", agg.get("pooled_profit_factor", 0.0), agg.get("mean_profit_factor", 0.0))
        logger.info("  Compounded Net Return:    %+.2f%% (Sum: %+.2f%%)", agg.get("compounded_net_return_pct", 0.0), agg["total_net_pnl_sum_pct"])
        logger.info("  Worst Max Drawdown:       %.2f%%", agg.get("worst_max_drawdown_pct", 0.0))
        logger.info("  Total Out-of-Sample Trades: %d", agg.get("total_trades_all_folds", 0))
        logger.info("=" * 60)
        return

    # ── 2. Compute indicators & target ───────────────────────────────────────
    # STEP 2: Turn raw candles into mathematical indicators (RSI, MACD, etc)
    # Compute features on the COMPLETE fetched history (including pre-roll bars)
    # and validate 720-bar Fibonacci warm-up BEFORE fillna(0.0) is applied.
    logger.info("Step 2/5 — Computing technical indicators on complete history (%d bars)", len(df_raw))
    df_feat_full = build_features(
        df_raw,
        add_targets=False,
        requested_start_ts=requested_start_ts,
        min_warmup_bars=720,
    )
    
    # Apply Triple Barrier Labeling
    df_feat_full = add_tbm_target(
        df_feat_full,
        TBM_PT_MULT,
        TBM_SL_MULT,
        TARGET_HORIZON,
        ambiguous_resolution=args.ambiguous_resolution,
    )

    # Retain the requested training date range ONLY AFTER feature computation:
    df_feat = df_feat_full.loc[requested_start_ts:].copy()

    logger.info(
        "Retained Dataset Range: %s -> %s (%d continuous hourly bars). "
        "Pre-roll observations (%d bars) strictly excluded from training samples.",
        df_feat.index[0], df_feat.index[-1], len(df_feat), len(df_raw) - len(df_feat),
    )

    if len(df_feat) < LOOKBACK + 50:
        logger.error(
            "Insufficient data after indicator warm-up: %d rows (need > %d). "
            "Increase --limit.",
            len(df_feat), LOOKBACK + 50,
        )
        sys.exit(1)

    # ── 3. Fit & save scalers ────────────────────────────────────────────────
    # STEP 3: Fit scalers ONLY on training partition to prevent lookahead leakage
    train_end_idx = int(len(df_feat) * args.train_ratio)
    purged_train_end = max(0, train_end_idx - TARGET_HORIZON)
    df_train = df_feat.iloc[:purged_train_end]

    val_end_idx = int(len(df_feat) * (args.train_ratio + args.val_ratio))
    purged_val_end = max(train_end_idx, val_end_idx - TARGET_HORIZON)

    logger.info("Step 3/5 — Partition Boundaries & Scaler Fitting:")
    logger.info("  Candles Requested: %d | Pre-roll Warmup: %d | Total Fetched: %d", limit, args.warmup_bars, len(df_raw))
    logger.info("  Usable Feature Rows (after indicator warmup & NaNs): %d", len(df_feat))
    logger.info("  Training Partition (Fit Scalers Here): [0 : %d] (%d bars)", purged_train_end, len(df_train))
    logger.info("  Purge Gap 1 (Train -> Val): [%d : %d] (%d bars)", purged_train_end, train_end_idx, train_end_idx - purged_train_end)
    logger.info("  Validation Partition (Model Selection / Calibration): [%d : %d] (%d bars)", train_end_idx, purged_val_end, purged_val_end - train_end_idx)
    logger.info("  Purge Gap 2 (Val -> Test):  [%d : %d] (%d bars)", purged_val_end, val_end_idx, val_end_idx - purged_val_end)
    logger.info("  Final Test Partition (Frozen Evaluation): [%d : %d] (%d bars)", val_end_idx, len(df_feat), len(df_feat) - val_end_idx)

    scaler_candle, scaler_market = fit_and_save_scalers(
        df_train,
        scaler_candle_path=scaler_candle_path,
        scaler_market_path=scaler_market_path,
    )

    # ── 4. Build sliding windows ─────────────────────────────────────────────
    # STEP 4: Build continuous lookback windows on unbroken hourly features.
    # Exclude windows with invalid/NaN targets post-construction so that no window skips an hour.
    logger.info("Step 4/5 — Building continuous sliding-window dataset")

    candle_arr, market_arr = transform(df_feat, scaler_candle, scaler_market)
    targets = df_feat["target"].values.astype("float32")

    X_candle, X_market, y, window_meta = build_windows(
        candle_arr,
        market_arr,
        targets,
        lookback=LOOKBACK,
        timestamps=df_feat.index,
        purge_horizon=TARGET_HORIZON,
        drop_nan_targets=True,
        return_metadata=True,
    )
    logger.info(
        "Continuous Windows built | X_candle: %s | X_market: %s | y: %s (Filtered NaN targets post-construction)",
        X_candle.shape, X_market.shape, y.shape,
    )

    # ── 5. Train model ───────────────────────────────────────────────────────
    # STEP 5: Feed the windows into the CNN + LSTM model and let it learn!
    # It will loop over the data (Epochs) trying to minimize its errors.
    logger.info("Step 5/5 — Training HybridModel with 3-Way Purged Partitioning")
    model, results = train_model(
        X_candle, X_market, y,
        timestamps=window_meta["decision_timestamps"],
        resolution_timestamps=window_meta["resolution_timestamps"],
        pos_weight_mode=args.pos_weight_mode,
        checkpoint_metric=args.checkpoint_metric,
        use_three_way_split=True,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        purge_horizon=TARGET_HORIZON,
        max_epochs=max_epochs,
        patience=patience,
        model_path=model_path,
    )

    val_metrics = results["validation_metrics"]
    test_metrics = results.get("final_test_metrics")
    cal_res = results.get("calibration_results", {})
    unc_res = results.get("test_uncertainty_analysis")

    logger.info("=" * 60)
    logger.info("Training finished. Artifacts saved:")
    logger.info("  Model   -> %s", model_path)
    logger.info("  Scaler1 -> %s", scaler_candle_path)
    logger.info("  Scaler2 -> %s", scaler_market_path)
    logger.info("Run  python live.py  to start the live signal engine.")
    logger.info("=" * 60)
    logger.info("VALIDATION (MODEL SELECTION & CALIBRATION PARTITION):")
    logger.info("  Best Epoch: %d / %d (Criterion: %s = %.4f)", val_metrics.get("best_epoch", 0), val_metrics.get("epochs_trained", 0), args.checkpoint_metric, val_metrics.get("best_checkpoint_metric_val", 0.0))
    logger.info("  Directional Accuracy: %.2f%%", val_metrics["trading_signal_directional_accuracy"] * 100)
    logger.info("  Regression MSE:       %.4f (Trivial Zero Baseline: %.4f)", val_metrics["regression_mse"], val_metrics["trivial_zero_mse"])
    logger.info("  Fitted Temperature:   T = %.4f", cal_res.get("fitted_temperature", 1.0))

    if test_metrics:
        logger.info("=" * 60)
        logger.info("UNTOUCHED FINAL TEST PARTITION EVALUATION (FROZEN MODEL):")
        test_acc = test_metrics["trading_signal_directional_accuracy"]
        logger.info("  Directional Accuracy (Regression Head): %.2f%%", test_acc * 100)
        logger.info("  Classification Head Directional Acc:   %.2f%%", test_metrics["classification_directional_accuracy"] * 100)
        logger.info("  Regression MSE:                        %.4f (Trivial Zero: %.4f)", test_metrics["regression_mse"], test_metrics["trivial_zero_mse"])
        
        cal_test = test_metrics.get("calibrated_with_val_temperature", {})
        if cal_test:
            logger.info("  Calibrated Brier Score:                %.4f (Prior Baseline: %.4f, Diff: %+.4f)",
                        cal_test.get("brier_score", 0.0), cal_test.get("brier_baseline_constant_prior", 0.0),
                        cal_test.get("brier_improvement_over_baseline", 0.0))
            logger.info("  Calibrated Log Loss:                   %.4f (Prior Baseline: %.4f)",
                        cal_test.get("log_loss", 0.0), cal_test.get("log_loss_baseline_constant_prior", 0.0))

        tf = test_metrics["threshold_filtered"]
        logger.info("  Threshold (|pred| >= %.4f) Passing:    %d/%d (%.2f%%) | Winrate: %.2f%%",
                    tf["threshold"], tf["samples_passing"], test_metrics["sample_count"], tf["pct_passing"], tf["trading_signal_accuracy"] * 100)

        if unc_res and "block_evaluations" in unc_res:
            logger.info("-" * 60)
            logger.info("BLOCK BOOTSTRAP UNCERTAZZINTY ESTIMATION (95%% CIs):")
            for b_name, b_data in unc_res["block_evaluations"].items():
                acc_ci = b_data["directional_accuracy"]
                mse_ci = b_data["regression_mse"]
                brier_ci = b_data["brier_score"]
                logger.info("  %s: Acc [%.2f%% .. %.2f%%] (se: %.4f) | MSE [%.4f .. %.4f] | Brier [%.4f .. %.4f]",
                            b_name, acc_ci["ci_lower"] * 100, acc_ci["ci_upper"] * 100, acc_ci["std_error"],
                            mse_ci["ci_lower"], mse_ci["ci_upper"], brier_ci["ci_lower"], brier_ci["ci_upper"])


if __name__ == "__main__":
    main()
