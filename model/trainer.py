"""
model/trainer.py — Training loop with early stopping, purge/embargo, and rigorous evaluation.
"""

import logging
import os
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import config
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    TRAIN_SPLIT,
    SIGNAL_THRESHOLD,
    CLASSIFICATION_WEIGHT,
    REGRESSION_WEIGHT,
    TARGET_MULTIPLIER,
    NOISE_INJECTION,
    RECENCY_WEIGHT,
    TARGET_HORIZON,
)
from model.architecture import HybridModel
from model.dataset import ETHDataset
from data.preprocessor import split_train_val_purged

logger = logging.getLogger(__name__)


def directional_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of predictions where sign(pred) == sign(target)."""
    if len(preds) == 0:
        return 0.0
    return float(np.mean((preds > 0) == (targets > 0)))


def compute_comprehensive_metrics(
    reg_preds: np.ndarray,
    clf_logits: np.ndarray,
    targets: np.ndarray,
    signal_threshold: float = SIGNAL_THRESHOLD,
    target_multiplier: float = TARGET_MULTIPLIER,
) -> Dict[str, Any]:
    """
    Rigorously evaluate both regression and classification heads independently.
    
    Ensures:
    1. Trading signal accuracy is evaluated directly on the regression head used by live trading.
    2. Classification head directional accuracy and balanced accuracy are evaluated independently.
    3. Regression MSE and MAE are compared against the trivial zero baseline (MSE=1.0 for +/-1 targets).
    4. Confusion matrix and long/short breakdowns are computed.
    5. Threshold filtering reports passing counts, percentages, and filtered performance.
    """
    N = len(targets)
    assert len(reg_preds) == len(clf_logits) == N, "Mismatched evaluation lengths"

    # Actual trading signal used by live/signal_engine.py:
    # pred = raw_reg / TARGET_MULTIPLIER
    trading_signals = reg_preds / target_multiplier

    # Directional Accuracies
    trading_signal_acc = directional_accuracy(trading_signals, targets)
    reg_dir_acc = directional_accuracy(reg_preds, targets)
    clf_dir_acc = directional_accuracy(clf_logits, targets)

    # Regression Losses
    reg_mse = float(np.mean((reg_preds - targets) ** 2))
    reg_mae = float(np.mean(np.abs(reg_preds - targets)))
    trivial_zero_mse = float(np.mean((0.0 - targets) ** 2))
    mse_vs_trivial = reg_mse / trivial_zero_mse if trivial_zero_mse > 0 else 1.0

    # Classification Metrics & Confusion Matrix
    clf_binary = (clf_logits > 0).astype(int)
    target_binary = (targets > 0).astype(int)

    tp = int(np.sum((clf_binary == 1) & (target_binary == 1)))
    fp = int(np.sum((clf_binary == 1) & (target_binary == 0)))
    tn = int(np.sum((clf_binary == 0) & (target_binary == 0)))
    fn = int(np.sum((clf_binary == 0) & (target_binary == 1)))

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Sensitivity / Long Recall
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # Specificity / Short Recall
    balanced_acc = float(0.5 * (tpr + tnr))

    # Long vs Short Breakdown
    long_mask = targets > 0
    short_mask = targets < 0
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())

    long_reg_acc = float(np.mean(reg_preds[long_mask] > 0)) if n_long > 0 else 0.0
    short_reg_acc = float(np.mean(reg_preds[short_mask] < 0)) if n_short > 0 else 0.0

    long_clf_acc = float(np.mean(clf_logits[long_mask] > 0)) if n_long > 0 else 0.0
    short_clf_acc = float(np.mean(clf_logits[short_mask] < 0)) if n_short > 0 else 0.0

    # Threshold-filtered analysis
    scaled_thresh = signal_threshold * target_multiplier
    thresh_mask = np.abs(reg_preds) >= scaled_thresh
    n_thresh = int(thresh_mask.sum())
    pct_thresh = float((n_thresh / N) * 100) if N > 0 else 0.0

    if n_thresh > 0:
        thresh_trading_acc = directional_accuracy(trading_signals[thresh_mask], targets[thresh_mask])
        thresh_clf_acc = directional_accuracy(clf_logits[thresh_mask], targets[thresh_mask])
        thresh_reg_mse = float(np.mean((reg_preds[thresh_mask] - targets[thresh_mask]) ** 2))
    else:
        thresh_trading_acc = 0.0
        thresh_clf_acc = 0.0
        thresh_reg_mse = 0.0

    metrics = {
        "validation_samples": N,
        "trading_signal_directional_accuracy": trading_signal_acc,
        "regression_directional_accuracy": reg_dir_acc,
        "classification_directional_accuracy": clf_dir_acc,
        "regression_mse": reg_mse,
        "regression_mae": reg_mae,
        "trivial_zero_mse": trivial_zero_mse,
        "mse_vs_trivial_ratio": mse_vs_trivial,
        "classification_balanced_accuracy": balanced_acc,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr_long_recall": tpr,
            "tnr_short_recall": tnr,
        },
        "target_distribution": {
            "long_samples": n_long,
            "short_samples": n_short,
            "long_pct": float((n_long / N) * 100) if N > 0 else 0.0,
            "short_pct": float((n_short / N) * 100) if N > 0 else 0.0,
        },
        "directional_accuracy_by_side": {
            "long_regression_acc": long_reg_acc,
            "short_regression_acc": short_reg_acc,
            "long_classification_acc": long_clf_acc,
            "short_classification_acc": short_clf_acc,
        },
        "threshold_filtered": {
            "threshold": signal_threshold,
            "samples_passing": n_thresh,
            "pct_passing": pct_thresh,
            "trading_signal_accuracy": thresh_trading_acc,
            "classification_accuracy": thresh_clf_acc,
            "regression_mse": thresh_reg_mse,
        },
    }
    return metrics


def _build_recency_weights(n_samples: int, recency_factor: float = 1.5) -> np.ndarray:
    """
    Build sample weights that increase linearly toward the most recent data.
    """
    weights = np.linspace(1.0, recency_factor, n_samples)
    return weights / weights.sum() * n_samples


def train_model(
    X_candle: np.ndarray,
    X_market: np.ndarray,
    y: np.ndarray,
    device: str = "auto",
) -> Tuple[HybridModel, Dict[str, Any]]:
    """
    Full training pipeline with temporal leakage protection (purge/embargo)
    and rigorous evaluation reporting.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    # ── Dataset split with temporal purge ──────────────────────────────────
    (X_c_train, X_m_train, y_train), (X_c_val, X_m_val, y_val), split_meta = split_train_val_purged(
        X_candle, X_market, y,
        train_split=TRAIN_SPLIT,
        purge_horizon=TARGET_HORIZON,
    )

    train_ds = ETHDataset(X_c_train, X_m_train, y_train)
    val_ds = ETHDataset(X_c_val, X_m_val, y_val)

    # Recency-weighted sampling
    sample_weights = _build_recency_weights(len(train_ds), RECENCY_WEIGHT)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(
        "Train samples: %d (purged boundary: %d) | Val samples: %d | Recency weight: %.1fx",
        len(train_ds), split_meta["purge_size"], len(val_ds), RECENCY_WEIGHT,
    )

    # ── Model, optimizer, loss ─────────────────────────────────────────────
    model = HybridModel().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    
    mse_criterion = nn.MSELoss()
    label_smoothing = 0.25
    pos_weight = torch.tensor([1.17], device=device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_losses = []

        for x_c, x_m, target in train_loader:
            x_c = x_c.to(device)
            x_m = x_m.to(device)
            target = target.to(device).unsqueeze(-1) * TARGET_MULTIPLIER

            if epoch <= 5:
                warmup_lr = LEARNING_RATE * (epoch / 5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            if NOISE_INJECTION > 0:
                x_c = x_c + torch.randn_like(x_c) * NOISE_INJECTION
                x_m = x_m + torch.randn_like(x_m) * NOISE_INJECTION

            raw_tbm = target / TARGET_MULTIPLIER
            clf_label = (raw_tbm > 0).float()

            optimizer.zero_grad()
            reg_pred, clf_pred = model(x_c, x_m)

            smoothed_label = clf_label * (1.0 - label_smoothing) + 0.5 * label_smoothing
            loss_clf = bce_criterion(clf_pred, smoothed_label)

            loss = CLASSIFICATION_WEIGHT * loss_clf
            if REGRESSION_WEIGHT > 0:
                loss_reg = mse_criterion(reg_pred, target)
                loss += REGRESSION_WEIGHT * loss_reg

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses, val_clf_preds, val_reg_preds, val_targets = [], [], [], []
        with torch.no_grad():
            for x_c, x_m, target in val_loader:
                x_c = x_c.to(device)
                x_m = x_m.to(device)
                scaled_target = target.to(device).unsqueeze(-1) * TARGET_MULTIPLIER

                reg_pred, clf_pred = model(x_c, x_m)
                val_losses.append(mse_criterion(reg_pred, scaled_target).item())

                val_clf_preds.extend(clf_pred.squeeze(-1).cpu().numpy())
                val_reg_preds.extend(reg_pred.squeeze(-1).cpu().numpy())
                val_targets.extend(target.numpy())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        scheduler.step()
        elapsed = time.time() - t0

        val_reg_arr = np.array(val_reg_preds)
        val_clf_arr = np.array(val_clf_preds)
        val_y_arr = np.array(val_targets)

        reg_acc_epoch = directional_accuracy(val_reg_arr, val_y_arr)
        clf_acc_epoch = directional_accuracy(val_clf_arr, val_y_arr)

        logger.info(
            "Epoch %3d/%d | Train Loss: %.4f | Val MSE: %.4f | Reg Acc: %.2f%% | Clf Acc: %.2f%% | LR: %.2e (%.1fs)",
            epoch, MAX_EPOCHS, train_loss, val_loss, reg_acc_epoch * 100, clf_acc_epoch * 100,
            optimizer.param_groups[0]["lr"], elapsed,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), config.MODEL_PATH)
            patience_counter = 0
            logger.info("  [BEST] Model saved (Val MSE: %.4f | Epoch: %d)", best_val_loss, best_epoch)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d (patience=%d exhausted)", epoch, PATIENCE)
                break

    # Load best checkpoint
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    logger.info("Training complete. Restored best model from epoch %d (Val MSE: %.4f)", best_epoch, best_val_loss)

    # ── Final rigorous evaluation on best model ────────────────────────────
    model.eval()
    final_clf_preds, final_reg_preds, final_targets = [], [], []
    with torch.no_grad():
        for x_c, x_m, target in val_loader:
            reg_pred, clf_pred = model(x_c.to(device), x_m.to(device))
            final_clf_preds.extend(clf_pred.squeeze(-1).cpu().numpy())
            final_reg_preds.extend(reg_pred.squeeze(-1).cpu().numpy())
            final_targets.extend(target.numpy())

    metrics = compute_comprehensive_metrics(
        reg_preds=np.array(final_reg_preds, dtype=np.float32),
        clf_logits=np.array(final_clf_preds, dtype=np.float32),
        targets=np.array(final_targets, dtype=np.float32),
        signal_threshold=SIGNAL_THRESHOLD,
        target_multiplier=TARGET_MULTIPLIER,
    )
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss_mse"] = best_val_loss
    metrics["split_metadata"] = split_meta

    logger.info("=" * 60)
    logger.info("RIGOROUS VALIDATION METRICS:")
    logger.info("  Trading Signal Directional Accuracy: %.2f%%", metrics["trading_signal_directional_accuracy"] * 100)
    logger.info("  Classification Directional Accuracy:  %.2f%%", metrics["classification_directional_accuracy"] * 100)
    logger.info("  Regression MSE: %.4f (vs Trivial Zero: %.4f | Ratio: %.3f)",
                metrics["regression_mse"], metrics["trivial_zero_mse"], metrics["mse_vs_trivial_ratio"])
    logger.info("  Classification Balanced Accuracy:    %.2f%% (Long recall: %.2f%%, Short recall: %.2f%%)",
                metrics["classification_balanced_accuracy"] * 100,
                metrics["confusion_matrix"]["tpr_long_recall"] * 100,
                metrics["confusion_matrix"]["tnr_short_recall"] * 100)
    tf = metrics["threshold_filtered"]
    logger.info("  Above Threshold (|pred| >= %.4f):   %d/%d (%.2f%% passing)",
                tf["threshold"], tf["samples_passing"], metrics["validation_samples"], tf["pct_passing"])
    logger.info("  Threshold Trading Signal Accuracy:   %.2f%%", tf["trading_signal_accuracy"] * 100)
    logger.info("=" * 60)

    return model, metrics
