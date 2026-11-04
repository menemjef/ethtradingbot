"""
evaluate.py — Evaluate the trained HybridModel on historical data.

Usage:
    python evaluate.py --limit 1000
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import joblib

from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger("evaluate")

from config import (
    SYMBOL,
    TIMEFRAME,
    LOOKBACK,
    MODEL_PATH,
    SCALER_CANDLE_PATH,
    SCALER_MARKET_PATH,
    SIGNAL_THRESHOLD,
    TBM_PT_MULT,
    TBM_SL_MULT,
    TARGET_HORIZON,
)
from data.fetcher import fetch_ohlcv
from data.preprocessor import (
    build_features,
    transform,
    build_windows,
)
from model.architecture import HybridModel
from model.ensemble import EnsembleModel

def directional_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of predictions where sign(pred) == sign(target)."""
    if len(preds) == 0:
        return 0.0
    return float(np.mean(np.sign(preds) == np.sign(targets)))

def calculate_trading_metrics(preds, targets, thresh, commission, slippage):
    """
    Calculate realistic profit/loss metrics.
    Targets are log-returns.
    """
    mask = np.abs(preds) >= thresh
    if mask.sum() == 0:
        return {}

    # Signal direction (1 for LONG, -1 for SHORT)
    dir_preds = np.sign(preds[mask])
    actual_moves = targets[mask]
    
    # Gross returns = side * actual_move
    gross_returns = dir_preds * actual_moves
    
    # Net returns = gross - (commission + slippage) * 2 (for entry and exit)
    # Actually, commission is usually applied to the notional, so on returns it's additive.
    cost_per_trade = commission + slippage
    net_returns = gross_returns - (cost_per_trade * 2)
    
    cum_returns = np.cumsum(net_returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = peak - cum_returns
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
    
    win_rate = np.mean(net_returns > 0)
    total_net_return = np.sum(net_returns)
    
    # Sharpe Ratio (assuming no risk-free rate for crypto)
    sharpe = (np.mean(net_returns) / np.std(net_returns)) * np.sqrt(365 * 24) if np.std(net_returns) > 0 else 0
    
    return {
        "net_profit": total_net_return,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "trades": len(net_returns)
    }

def main():
    parser = argparse.ArgumentParser(description="ETH Trading Signal — Evaluation Script")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of historical candles to fetch for evaluation",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use 3-model ensemble (42, 1337, 7)",
    )
    args = parser.parse_args()

    # ── 1. Check for artifacts ───────────────────────────────────────────────
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_CANDLE_PATH, SCALER_MARKET_PATH]):
        logger.error("Missing model or scalers. Please run python train.py first.")
        sys.exit(1)

    # ── 2. Load artifacts ────────────────────────────────────────────────────
    logger.info("Loading model and scalers...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for ensemble vs single model
    if args.ensemble:
        logger.info("Initializing 3-Model Ensemble (42, 1337, 7)...")
        model = EnsembleModel(suffixes=["42", "1337", "7"], device=device)
    else:
        model = HybridModel().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    model.eval()

    scaler_candle = joblib.load(SCALER_CANDLE_PATH)
    scaler_market = joblib.load(SCALER_MARKET_PATH)

    # ── 3. Fetch data ────────────────────────────────────────────────────────
    logger.info("Fetching %d candles for evaluation...", args.limit)
    df_raw = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=args.limit)
    
    # ── 4. Preprocess ────────────────────────────────────────────────────────
    logger.info("Preprocessing data (TBM Evaluation)...")
    df_feat = build_features(df_raw, add_targets=False)
    from data.indicators import add_tbm_target
    df_feat = add_tbm_target(df_feat, TBM_PT_MULT, TBM_SL_MULT, TARGET_HORIZON)
    df_feat = df_feat.dropna(subset=["target"])
    
    if len(df_feat) < LOOKBACK + 1:
        logger.error("Insufficient data for lookback. Increase --limit.")
        sys.exit(1)

    candle_arr, market_arr = transform(df_feat, scaler_candle, scaler_market)
    targets = df_feat["target"].values.astype("float32")

    X_candle, X_market, y = build_windows(candle_arr, market_arr, targets)

    # ── 5. Run Inference ─────────────────────────────────────────────────────
    logger.info("Running inference on %d samples...", len(y))
    with torch.no_grad():
        x_c_tensor = torch.tensor(X_candle, dtype=torch.float32).to(device)
        x_m_tensor = torch.tensor(X_market, dtype=torch.float32).to(device)
        reg_preds, clf_preds = model(x_c_tensor, x_m_tensor)
        
        preds = reg_preds.squeeze(-1).cpu().numpy()

    # ── 6. Calculate Metrics ────────────────────────────────────────────────
    mse = np.mean((preds - y)**2)
    dir_acc = directional_accuracy(preds, y)

    from config import COMMISSION_RATE, SLIPPAGE_RATE
    trading_results = calculate_trading_metrics(
        preds, y, SIGNAL_THRESHOLD, COMMISSION_RATE, SLIPPAGE_RATE
    )

    # ── 7. Report ───────────────────────────────────────────────────────────
    logger.info("="*60)
    logger.info(f"EVALUATION RESULTS ({'ENSEMBLE' if args.ensemble else 'SINGLE'})")
    logger.info("-"*60)
    logger.info(f"Samples Evaluated: {len(y)}")
    logger.info(f"Overall Directional Accuracy: {dir_acc*100:.2f}%")
    
    # Sigmoid Confidence Benchmarking
    probs = torch.sigmoid(torch.tensor(all_clf_preds)).numpy()
    conf = np.abs(probs - 0.5) + 0.5  # Confidence probability
    
    for c_thresh in [0.55, 0.60, 0.70, 0.80]:
        c_mask = conf >= c_thresh
        if c_mask.sum() > 0:
            c_acc = directional_accuracy(all_clf_preds[c_mask], all_targets[c_mask])
            logger.info(f"Confidence >= {c_thresh:.2f}: {c_acc*100:5.2f}% Win Rate | {c_mask.sum():5d} Trades")
    
    if trading_results:
        logger.info("-" * 60)
        logger.info(f"Trades Executed (Reg Thresh {SIGNAL_THRESHOLD}): {trading_results['trades']}")
        logger.info(f"Win Rate (Net): {trading_results['win_rate']*100:.2f}%")
        logger.info(f"Cumulative Net Return: {trading_results['net_profit']*100:.2f}%")
        logger.info(f"Max Drawdown: {trading_results['max_dd']*100:.2f}%")
        logger.info(f"Estimated Sharpe Ratio: {trading_results['sharpe']:.2f}")
    else:
        logger.info(f"No signals exceeded the threshold of {SIGNAL_THRESHOLD}")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()
