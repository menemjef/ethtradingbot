"""
model/trainer.py — Training loop with early stopping, temporal purge/embargo,
probability calibration tracking, and decoupled evaluation.
"""

import logging
import os
import time
from typing import Dict, Any, Tuple, Optional
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
    Includes probability calibration metrics: Brier score, Log Loss, and ECE.
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

    # Probability Calibration Metrics
    clf_probs = 1.0 / (1.0 + np.exp(-np.clip(clf_logits, -30.0, 30.0)))
    brier_score = float(np.mean((clf_probs - target_binary) ** 2))

    eps = 1e-12
    clipped_probs = np.clip(clf_probs, eps, 1.0 - eps)
    log_loss = float(-np.mean(target_binary * np.log(clipped_probs) + (1.0 - target_binary) * np.log(1.0 - clipped_probs)))

    # Expected Calibration Error (ECE) with 10 bins
    n_bins = 10
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability_curve = []

    for b in range(n_bins):
        bin_lower = float(bin_boundaries[b])
        bin_upper = float(bin_boundaries[b + 1])
        in_bin = (clf_probs >= bin_lower) & (clf_probs < bin_upper if b < n_bins - 1 else clf_probs <= bin_upper)
        bin_count = int(in_bin.sum())

        if bin_count > 0:
            bin_conf = float(np.mean(clf_probs[in_bin]))
            bin_acc = float(np.mean(target_binary[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / N) * bin_error
            reliability_curve.append({
                "bin_idx": b,
                "bin_range": (bin_lower, bin_upper),
                "sample_count": bin_count,
                "mean_confidence": bin_conf,
                "empirical_accuracy": bin_acc,
                "calibration_gap": float(bin_error),
            })
        else:
            reliability_curve.append({
                "bin_idx": b,
                "bin_range": (bin_lower, bin_upper),
                "sample_count": 0,
                "mean_confidence": 0.0,
                "empirical_accuracy": 0.0,
                "calibration_gap": 0.0,
            })
    ece = float(ece)

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
        "calibration": {
            "brier_score": brier_score,
            "log_loss": log_loss,
            "expected_calibration_error_ece": ece,
            "mean_predicted_probability": float(np.mean(clf_probs)),
            "probability_std": float(np.std(clf_probs)),
            "reliability_curve": reliability_curve,
        },
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
    """Build sample weights that increase linearly toward the most recent data."""
    weights = np.linspace(1.0, recency_factor, n_samples)
    return weights / weights.sum() * n_samples


def train_model(
    X_candle: np.ndarray,
    X_market: np.ndarray,
    y: np.ndarray,
    device: str = "auto",
    pos_weight_mode: str = "balanced",
    label_smoothing: float = 0.0,
    checkpoint_metric: str = "regression_mse",
) -> Tuple[HybridModel, Dict[str, Any]]:
    """
    Full training pipeline with temporal leakage protection (purge/embargo),
    calibrated loss weighting, and comprehensive reporting.

    Args:
        pos_weight_mode:
            - "balanced" (default): pos_weight = neg_samples / pos_samples from training split strictly.
            - "unweighted": pos_weight = 1.0.
            - "legacy_1.17": pos_weight = 1.17 (hardcoded legacy).
        label_smoothing:
            - 0.0 (default): No label smoothing for uncompromised probability calibration.
            - 0.25: Legacy label smoothing.
        checkpoint_metric:
            - "regression_mse" (default): Selects checkpoint minimizing validation MSE.
            - "classification_loss": Selects checkpoint minimizing validation BCE.
            - "directional_accuracy": Selects checkpoint maximizing classification directional accuracy.
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

    # Calculate class weighting strictly from training partition
    n_pos_train = int((y_train > 0).sum())
    n_neg_train = int((y_train <= 0).sum())
    data_ratio = float(n_neg_train / max(1, n_pos_train))

    if pos_weight_mode == "balanced":
        pos_weight_val = data_ratio
        rationale = f"Data-driven from training partition: {n_neg_train} neg / {n_pos_train} pos = {data_ratio:.4f}"
    elif pos_weight_mode == "unweighted":
        pos_weight_val = 1.0
        rationale = "Unweighted BCE (pos_weight = 1.0)"
    elif pos_weight_mode == "legacy_1.17":
        pos_weight_val = 1.17
        bias_pct = ((1.17 / data_ratio) - 1.0) * 100.0 if data_ratio > 0 else 0.0
        rationale = f"Legacy hardcoded pos_weight = 1.17 ({bias_pct:+.1f}% positive bias over training ratio)"
    else:
        pos_weight_val = data_ratio
        rationale = f"Defaulted to data-driven balanced ({data_ratio:.4f})"

    logger.info("Classification Loss Weighting: pos_weight=%.4f (%s) | Label smoothing: %.2f",
                pos_weight_val, rationale, label_smoothing)
    logger.info("Checkpoint Selection Metric: '%s'", checkpoint_metric)

    # ── Model, optimizer, loss ─────────────────────────────────────────────
    model = HybridModel().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    mse_criterion = nn.MSELoss()
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # ── Training loop ──────────────────────────────────────────────────────
    best_eval_metric = float("inf") if "loss" in checkpoint_metric or "mse" in checkpoint_metric else float("-inf")
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

            if label_smoothing > 0:
                smoothed_label = clf_label * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                smoothed_label = clf_label
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
        val_mse_losses, val_bce_losses = [], []
        val_clf_preds, val_reg_preds, val_targets = [], [], []
        with torch.no_grad():
            for x_c, x_m, target in val_loader:
                x_c = x_c.to(device)
                x_m = x_m.to(device)
                scaled_target = target.to(device).unsqueeze(-1) * TARGET_MULTIPLIER

                reg_pred, clf_pred = model(x_c, x_m)
                val_mse_losses.append(mse_criterion(reg_pred, scaled_target).item())

                raw_tbm = target.to(device).unsqueeze(-1)
                clf_label = (raw_tbm > 0).float()
                val_bce_losses.append(bce_criterion(clf_pred, clf_label).item())

                val_clf_preds.extend(clf_pred.squeeze(-1).cpu().numpy())
                val_reg_preds.extend(reg_pred.squeeze(-1).cpu().numpy())
                val_targets.extend(target.numpy())

        train_loss = float(np.mean(train_losses))
        val_loss_mse = float(np.mean(val_mse_losses))
        val_loss_bce = float(np.mean(val_bce_losses))

        scheduler.step()
        elapsed = time.time() - t0

        val_reg_arr = np.array(val_reg_preds)
        val_clf_arr = np.array(val_clf_preds)
        val_y_arr = np.array(val_targets)

        reg_acc_epoch = directional_accuracy(val_reg_arr, val_y_arr)
        clf_acc_epoch = directional_accuracy(val_clf_arr, val_y_arr)

        logger.info(
            "Epoch %3d/%d | Train Loss: %.4f | Val MSE: %.4f | Val BCE: %.4f | Reg Acc: %.2f%% | Clf Acc: %.2f%% | LR: %.2e (%.1fs)",
            epoch, MAX_EPOCHS, train_loss, val_loss_mse, val_loss_bce, reg_acc_epoch * 100, clf_acc_epoch * 100,
            optimizer.param_groups[0]["lr"], elapsed,
        )

        # Checkpoint selection evaluation
        if checkpoint_metric == "regression_mse":
            current_metric = val_loss_mse
            is_better = current_metric < best_eval_metric
        elif checkpoint_metric == "classification_loss":
            current_metric = val_loss_bce
            is_better = current_metric < best_eval_metric
        elif checkpoint_metric == "directional_accuracy":
            current_metric = clf_acc_epoch
            is_better = current_metric > best_eval_metric
        else:
            current_metric = val_loss_mse
            is_better = current_metric < best_eval_metric

        if is_better:
            best_eval_metric = current_metric
            best_epoch = epoch
            torch.save(model.state_dict(), config.MODEL_PATH)
            patience_counter = 0
            logger.info("  [BEST] Model saved (Criterion: %s = %.4f | Epoch: %d)",
                        checkpoint_metric, best_eval_metric, best_epoch)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d (patience=%d exhausted)", epoch, PATIENCE)
                break

    # Load best checkpoint
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    logger.info("Training complete. Restored best model from epoch %d (%s: %.4f)",
                best_epoch, checkpoint_metric, best_eval_metric)

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
    metrics["checkpoint_metric"] = checkpoint_metric
    metrics["best_checkpoint_metric_val"] = best_eval_metric
    metrics["pos_weight_applied"] = pos_weight_val
    metrics["pos_weight_mode"] = pos_weight_mode
    metrics["label_smoothing_applied"] = label_smoothing
    metrics["split_metadata"] = split_meta

    logger.info("=" * 60)
    logger.info("RIGOROUS VALIDATION METRICS (Checkpoint: %s):", checkpoint_metric)
    logger.info("  Trading Signal Directional Accuracy: %.2f%%", metrics["trading_signal_directional_accuracy"] * 100)
    logger.info("  Classification Directional Accuracy:  %.2f%%", metrics["classification_directional_accuracy"] * 100)
    logger.info("  Regression MSE: %.4f (vs Trivial Zero: %.4f | Ratio: %.3f)",
                metrics["regression_mse"], metrics["trivial_zero_mse"], metrics["mse_vs_trivial_ratio"])
    logger.info("  Classification Balanced Accuracy:    %.2f%% (Long recall: %.2f%%, Short recall: %.2f%%)",
                metrics["classification_balanced_accuracy"] * 100,
                metrics["confusion_matrix"]["tpr_long_recall"] * 100,
                metrics["confusion_matrix"]["tnr_short_recall"] * 100)
    cal = metrics["calibration"]
    logger.info("  Calibration: Brier Score = %.4f | Log Loss = %.4f | ECE = %.4f",
                cal["brier_score"], cal["log_loss"], cal["expected_calibration_error_ece"])
    tf = metrics["threshold_filtered"]
    logger.info("  Above Threshold (|pred| >= %.4f):   %d/%d (%.2f%% passing)",
                tf["threshold"], tf["samples_passing"], metrics["validation_samples"], tf["pct_passing"])
    logger.info("  Threshold Trading Signal Accuracy:   %.2f%%", tf["trading_signal_accuracy"] * 100)
    logger.info("=" * 60)

    return model, metrics
