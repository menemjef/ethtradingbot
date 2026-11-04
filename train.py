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
from utils.logger import setup_logging
from utils.seeding import set_seed
setup_logging()
set_seed(42)  # Ensure reproducibility across all runs

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
        default=50,
        help="Extra pre-roll candles to fetch for indicator warm-up (default: 50)",
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
    
    # Update paths if suffix is provided
    import config
    if args.suffix:
        config.MODEL_PATH = config.MODEL_PATH.replace(".pt", f"_{args.suffix}.pt")
        config.SCALER_CANDLE_PATH = config.SCALER_CANDLE_PATH.replace(".pkl", f"_{args.suffix}.pkl")
        config.SCALER_MARKET_PATH = config.SCALER_MARKET_PATH.replace(".pkl", f"_{args.suffix}.pkl")
    
    # Re-import updated paths
    from config import MODEL_PATH, SCALER_CANDLE_PATH, SCALER_MARKET_PATH

    limit = 500 if args.fast else args.limit

    if args.fast:
        # Override epochs for smoke test
        import config
        config.MAX_EPOCHS = 5
        config.PATIENCE = 3
        logger.info("FAST MODE: capping at %d candles, 5 epochs", limit)
    elif args.test:
        import config
        config.MAX_EPOCHS = 30
        logger.info("TEST MODE: running for 30 epochs")

    # ── 1. Fetch data ────────────────────────────────────────────────────────
    # STEP 1: Download historical crypto candles (Open, High, Low, Close, Volume)
    logger.info("=" * 60)
    if args.use_cache and os.path.exists(args.use_cache):
        import pandas as pd
        logger.info("Step 1/5 — Loading candles from local cache: %s", args.use_cache)
        df_raw = pd.read_csv(args.use_cache, index_col=0, parse_dates=True)
        if df_raw.index.tz is None:
            df_raw.index = df_raw.index.tz_localize('UTC')
    else:
        logger.info("Step 1/5 — Fetching %d x %s candles for %s (warmup_bars: %d)", limit, TIMEFRAME, SYMBOL, args.warmup_bars)
        df_raw = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit, warmup_bars=args.warmup_bars)
    logger.info("Raw data shape: %s | Date range: %s -> %s",
                df_raw.shape, df_raw.index[0], df_raw.index[-1])

    # ── 2. Compute indicators & target ───────────────────────────────────────
    # STEP 2: Turn raw candles into mathematical indicators (RSI, MACD, etc)
    # These become the "features" the AI will learn from.
    logger.info("Step 2/5 — Computing technical indicators (TBM)")
    df_feat = build_features(df_raw, add_targets=False)
    
    # Apply Triple Barrier Labeling
    # This creates the "Target" (what the AI tries to predict). It simulates 
    # taking a trade and seeing if it hits Profit Target (PT) or Stop Loss (SL).
    df_feat = add_tbm_target(df_feat, TBM_PT_MULT, TBM_SL_MULT, TARGET_HORIZON, ambiguous_resolution=args.ambiguous_resolution)
    df_feat = df_feat.dropna(subset=["target"])
    
    logger.info("Feature data shape: %s", df_feat.shape)

    if len(df_feat) < LOOKBACK + 50:
        logger.error(
            "Insufficient data after indicator warm-up: %d rows (need > %d). "
            "Increase --limit.",
            len(df_feat), LOOKBACK + 50,
        )
        sys.exit(1)

    # ── 3. Fit & save scalers ────────────────────────────────────────────────
    # STEP 3: Fit scalers ONLY on training data to prevent lookahead bias (Data Leakage)
    from config import TRAIN_SPLIT, TARGET_HORIZON
    split_idx = int(len(df_feat) * TRAIN_SPLIT)
    # Temporal leakage protection: fit scalers strictly on purged training samples
    purged_train_end = max(0, split_idx - TARGET_HORIZON)
    df_train = df_feat.iloc[:purged_train_end]
    
    logger.info("Step 3/5 — Fitting and saving StandardScalers (on purged train split: %d rows)", len(df_train))
    scaler_candle, scaler_market = fit_and_save_scalers(df_train)

    # ── 4. Build sliding windows ─────────────────────────────────────────────
    # STEP 4: The AI needs to see a "window" of time (e.g. 5 days of history) 
    # to predict the next move. We chop the data into overlapping windows.
    logger.info("Step 4/5 — Building sliding-window dataset")
    from config import CANDLE_FEATURES, MARKET_FEATURES
    import numpy as np

    candle_arr, market_arr = transform(df_feat, scaler_candle, scaler_market)
    targets = df_feat["target"].values.astype("float32")

    X_candle, X_market, y = build_windows(candle_arr, market_arr, targets)
    logger.info(
        "Windows built | X_candle: %s | X_market: %s | y: %s",
        X_candle.shape, X_market.shape, y.shape,
    )

    # ── 5. Train model ───────────────────────────────────────────────────────
    # STEP 5: Feed the windows into the CNN + LSTM model and let it learn!
    # It will loop over the data (Epochs) trying to minimize its errors.
    logger.info("Step 5/5 — Training HybridModel")
    model, metrics = train_model(
        X_candle, X_market, y,
        pos_weight_mode=args.pos_weight_mode,
        checkpoint_metric=args.checkpoint_metric,
    )
    val_acc = metrics["trading_signal_directional_accuracy"]
    val_acc_thresh = metrics["threshold_filtered"]["trading_signal_accuracy"]

    logger.info("=" * 60)
    logger.info("Training finished. Artifacts saved:")
    from config import MODEL_PATH, SCALER_CANDLE_PATH, SCALER_MARKET_PATH
    logger.info("  Model   -> %s", MODEL_PATH)
    logger.info("  Scaler1 -> %s", SCALER_CANDLE_PATH)
    logger.info("  Scaler2 -> %s", SCALER_MARKET_PATH)
    logger.info("Run  python live.py  to start the live signal engine.")
    logger.info("=" * 60)
    logger.info("RIGOROUS TEST RESULTS:")
    logger.info("  Trading Signal Accuracy (Regression Head): %.2f%%", val_acc * 100)
    logger.info("  Classification Head Directional Accuracy:  %.2f%%", metrics["classification_directional_accuracy"] * 100)
    logger.info("  Regression MSE: %.4f (Trivial Zero Baseline: %.4f)", metrics["regression_mse"], metrics["trivial_zero_mse"])
    tf = metrics["threshold_filtered"]
    logger.info("  Threshold (|pred| >= %.4f) Passing: %d/%d (%.2f%%) | Winrate: %.2f%%",
                tf["threshold"], tf["samples_passing"], metrics["validation_samples"], tf["pct_passing"], val_acc_thresh * 100)


if __name__ == "__main__":
    main()
