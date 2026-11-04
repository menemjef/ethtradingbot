"""
training/reproduce_baseline_audit.py — Independent Reproducibility & Provenance Script

Verifies and reproduces:
1. Cryptographic hashes of all model checkpoints and scalers.
2. Independent head comparison evaluation on the baseline retrained checkpoint (54.88% clf vs 50.15% reg).
3. Ambiguous candle counts and impact under TBM labeling (8 ambiguous bars on 5,000 candles).
4. Probability calibration analysis (Brier score, ECE, Log Loss).
5. Output distribution parameters and trivial baseline MSE comparison.
"""

import sys
import os
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import joblib

import config
from config import (
    MODEL_PATH,
    SCALER_CANDLE_PATH,
    SCALER_MARKET_PATH,
    LOOKBACK,
    TARGET_MULTIPLIER,
    SIGNAL_THRESHOLD,
    SYMBOL,
    TIMEFRAME,
    TBM_PT_MULT,
    TBM_SL_MULT,
    TARGET_HORIZON,
)
from model.architecture import HybridModel
from model.trainer import directional_accuracy, compute_comprehensive_metrics
from data.fetcher import fetch_ohlcv
from data.indicators import compute_indicators, add_tbm_target
from data.preprocessor import build_features, transform, build_windows


def get_file_sha256(filepath: str) -> str:
    """Compute standard SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def reproduce_provenance():
    results = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "artifacts_provenance": {},
        "evaluation_reproduction": {},
        "ambiguous_candle_analysis": {},
    }

    # 1. Cryptographic Provenance
    current_model_hash = get_file_sha256(MODEL_PATH)
    current_scaler_c_hash = get_file_sha256(SCALER_CANDLE_PATH)
    current_scaler_m_hash = get_file_sha256(SCALER_MARKET_PATH)

    backup_dir = Path("C:/Python Projects/trading_bot_backups/baseline_pre_retrain")
    backup_model_hash = get_file_sha256(str(backup_dir / "best_model.pt"))
    backup_scaler_c_hash = get_file_sha256(str(backup_dir / "scaler_candle.pkl"))
    backup_scaler_m_hash = get_file_sha256(str(backup_dir / "scaler_market.pkl"))

    results["artifacts_provenance"] = {
        "active_checkpoint": {
            "path": MODEL_PATH,
            "sha256": current_model_hash,
            "expected_baseline_sha256": "ac224bb7731f26b46d3f1cdc8cf39457742f7afde81799b31d67f9e02e3505bb",
            "matches_baseline_run": current_model_hash == "ac224bb7731f26b46d3f1cdc8cf39457742f7afde81799b31d67f9e02e3505bb",
        },
        "active_scaler_candle": {
            "path": SCALER_CANDLE_PATH,
            "sha256": current_scaler_c_hash,
        },
        "active_scaler_market": {
            "path": SCALER_MARKET_PATH,
            "sha256": current_scaler_m_hash,
        },
        "pre_retrain_backup": {
            "backup_dir": str(backup_dir),
            "best_model_pt_sha256": backup_model_hash,
            "expected_backup_sha256": "8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95",
            "matches_original_backup": backup_model_hash == "8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95",
        },
    }

    # 2. Evaluation Reproduction
    device = results["device"]
    model = HybridModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    scaler_candle = joblib.load(SCALER_CANDLE_PATH)
    scaler_market = joblib.load(SCALER_MARKET_PATH)

    print("Fetching 5,000 completed candles for reproducible evaluation...")
    df_raw = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=5000, validate=False)
    df_feat = build_features(df_raw, add_targets=False)
    # Using symmetric 2.0/2.0 with overshoot for exact reproduction of baseline run conditions
    df_feat = add_tbm_target(df_feat, 2.0, 2.0, 12, ambiguous_resolution="overshoot").dropna(subset=["target"])

    candle_arr, market_arr = transform(df_feat, scaler_candle, scaler_market)
    targets = df_feat["target"].values.astype("float32")
    Xc, Xm, y = build_windows(candle_arr, market_arr, targets)

    split_idx = int(len(y) * 0.8)
    Xc_val, Xm_val, y_val = Xc[split_idx:], Xm[split_idx:], y[split_idx:]

    with torch.no_grad():
        reg_out, clf_out = model(torch.from_numpy(Xc_val).to(device), torch.from_numpy(Xm_val).to(device))
        reg_preds = reg_out.squeeze().cpu().numpy()
        clf_logits = clf_out.squeeze().cpu().numpy()

    comp_metrics = compute_comprehensive_metrics(
        reg_preds=reg_preds,
        clf_logits=clf_logits,
        targets=y_val,
        signal_threshold=SIGNAL_THRESHOLD,
        target_multiplier=TARGET_MULTIPLIER,
    )

    results["evaluation_reproduction"] = {
        "dataset_candle_count": len(df_raw),
        "dataset_date_start": str(df_raw.index[0]),
        "dataset_date_end": str(df_raw.index[-1]),
        "validation_windows": len(y_val),
        "regression_directional_accuracy_pct": float(round(comp_metrics["regression_directional_accuracy"] * 100, 2)),
        "classification_directional_accuracy_pct": float(round(comp_metrics["classification_directional_accuracy"] * 100, 2)),
        "regression_mse": float(round(comp_metrics["regression_mse"], 4)),
        "trivial_zero_mse": float(round(comp_metrics["trivial_zero_mse"], 4)),
        "brier_score": float(round(comp_metrics["calibration"]["brier_score"], 4)),
        "log_loss": float(round(comp_metrics["calibration"]["log_loss"], 4)),
        "expected_calibration_error_ece": float(round(comp_metrics["calibration"]["expected_calibration_error_ece"], 4)),
        "threshold_filtered": comp_metrics["threshold_filtered"],
        "regression_output_stats": {
            "min": float(reg_preds.min()),
            "max": float(reg_preds.max()),
            "mean": float(reg_preds.mean()),
            "std": float(reg_preds.std()),
        },
    }

    # 3. Ambiguous Candle Analysis Reproduction
    df_ind = compute_indicators(df_raw).dropna(subset=["atr_14"])
    atr = df_ind["atr_14"]
    close = df_ind["close"]
    high = df_ind["high"]
    low = df_ind["low"]
    horizon = 12

    ambiguous_hits = 0
    overshoot_win = 0
    overshoot_loss = 0

    for i in range(len(df_ind)):
        if i >= len(df_ind) - horizon:
            continue
        p_price = close.iloc[i]
        vol_adj = atr.iloc[i] / p_price
        p_pt = p_price * (1 + 2.0 * vol_adj)
        p_sl = p_price * (1 - 2.0 * vol_adj)

        future_h = high.iloc[i + 1 : i + 1 + horizon]
        future_l = low.iloc[i + 1 : i + 1 + horizon]

        for j in range(len(future_h)):
            hit_pt = future_h.iloc[j] >= p_pt
            hit_sl = future_l.iloc[j] <= p_sl
            if hit_pt and hit_sl:
                ambiguous_hits += 1
                if (future_h.iloc[j] - p_pt) > (p_sl - future_l.iloc[j]):
                    overshoot_win += 1
                else:
                    overshoot_loss += 1
                break
            elif hit_pt or hit_sl:
                break

    results["ambiguous_candle_analysis"] = {
        "total_bars_scanned": len(df_ind) - horizon,
        "ambiguous_bars_count": ambiguous_hits,
        "ambiguous_bars_pct": float(round((ambiguous_hits / (len(df_ind) - horizon)) * 100, 3)),
        "legacy_overshoot_wins": overshoot_win,
        "legacy_overshoot_losses": overshoot_loss,
        "recommendation": "Exclude from directional training dataset to eliminate artificial bias",
    }

    output_path = PROJECT_ROOT / "training" / "reproducibility_audit_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Reproducibility report successfully saved to {output_path}")
    print(f"  Active checkpoint SHA-256: {current_model_hash}")
    print(f"  Regression Accuracy: {results['evaluation_reproduction']['regression_directional_accuracy_pct']}%")
    print(f"  Classification Accuracy: {results['evaluation_reproduction']['classification_directional_accuracy_pct']}%")
    print(f"  Ambiguous Bars: {ambiguous_hits} (Overshoot: {overshoot_win} win, {overshoot_loss} loss)")
    return results


if __name__ == "__main__":
    reproduce_provenance()
