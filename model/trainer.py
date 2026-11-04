"""
model/trainer.py — Training loop with 3-way chronological partitioning,
purged boundaries, probability calibration, and statistical uncertainty estimation.
"""

import logging
import os
import time
from typing import Dict, Any, Tuple, Optional, List
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
from data.preprocessor import (
    split_train_val_purged,
    split_train_val_test_purged,
    block_bootstrap_confidence_interval,
)
from model.calibration import (
    evaluate_calibration,
    fit_and_apply_temperature_scaling,
)

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
    cal_eval = evaluate_calibration(clf_probs, target_binary)

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
        "sample_count": N,
        "trading_signal_directional_accuracy": trading_signal_acc,
        "regression_directional_accuracy": reg_dir_acc,
        "classification_directional_accuracy": clf_dir_acc,
        "regression_mse": reg_mse,
        "regression_mae": reg_mae,
        "trivial_zero_mse": trivial_zero_mse,
        "mse_vs_trivial_ratio": mse_vs_trivial,
        "classification_balanced_accuracy": balanced_acc,
        "calibration": cal_eval,
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
    use_three_way_split: bool = True,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    purge_horizon: int = TARGET_HORIZON,
) -> Tuple[HybridModel, Dict[str, Any]]:
    """
    Production-integrated training pipeline with:
    1. Chronological 3-way split: 60% Train, 20% Val/Calibration, 20% Untouched Test.
    2. Temporal purge gaps at both boundaries.
    3. Scalers fit strictly on training data.
    4. Post-hoc probability calibration on designated calibration partition.
    5. Untouched Test evaluated only after model parameters are frozen.
    6. Moving block bootstrap uncertainty estimation across multiple block sizes [12, 24, 48, 168].
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    # ── 1. Chronological Partitioning with Temporal Purging ───────────────
    if use_three_way_split:
        (X_c_tr, X_m_tr, y_tr), (X_c_va, X_m_va, y_va), (X_c_te, X_m_te, y_te), split_meta = split_train_val_test_purged(
            X_candle, X_market, y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            purge_horizon=purge_horizon,
        )
    else:
        (X_c_tr, X_m_tr, y_tr), (X_c_va, X_m_va, y_va), split_meta = split_train_val_purged(
            X_candle, X_market, y,
            train_split=TRAIN_SPLIT,
            purge_horizon=purge_horizon,
        )
        X_c_te, X_m_te, y_te = None, None, None

    train_ds = ETHDataset(X_c_tr, X_m_tr, y_tr)
    val_ds = ETHDataset(X_c_va, X_m_va, y_va)

    sample_weights = _build_recency_weights(len(train_ds), RECENCY_WEIGHT)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Class weighting strictly from training partition
    n_pos_train = int((y_tr > 0).sum())
    n_neg_train = int((y_tr <= 0).sum())
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

    # ── 2. Model, Optimizer, Loss ──────────────────────────────────────────
    model = HybridModel().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    mse_criterion = nn.MSELoss()
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # ── 3. Training Loop (Model Selection on Validation Only) ──────────────
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

        # Validation evaluation for early stopping
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

    # ── 4. Restore Best Checkpoint ─────────────────────────────────────────
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    logger.info("Restored best model from epoch %d (%s: %.4f)", best_epoch, checkpoint_metric, best_eval_metric)

    # ── 5. Validation Evaluation on Best Model ────────────────────────────
    model.eval()
    final_clf_preds, final_reg_preds, final_targets = [], [], []
    with torch.no_grad():
        for x_c, x_m, target in val_loader:
            reg_pred, clf_pred = model(x_c.to(device), x_m.to(device))
            final_clf_preds.extend(clf_pred.squeeze(-1).cpu().numpy())
            final_reg_preds.extend(reg_pred.squeeze(-1).cpu().numpy())
            final_targets.extend(target.numpy())

    val_reg_np = np.array(final_reg_preds, dtype=np.float32)
    val_clf_np = np.array(final_clf_preds, dtype=np.float32)
    val_y_np = np.array(final_targets, dtype=np.float32)

    val_metrics = compute_comprehensive_metrics(
        reg_preds=val_reg_np,
        clf_logits=val_clf_np,
        targets=val_y_np,
        signal_threshold=SIGNAL_THRESHOLD,
        target_multiplier=TARGET_MULTIPLIER,
    )
    val_metrics["best_epoch"] = best_epoch
    val_metrics["checkpoint_metric"] = checkpoint_metric
    val_metrics["best_checkpoint_metric_val"] = best_eval_metric
    val_metrics["split_metadata"] = split_meta

    # ── 6. Probability Calibration (Fit Temperature Scaling) ──────────────
    # Fits temperature strictly on the validation/calibration partition
    cal_val_probs, fitted_t, uncal_val_cal, cal_val_cal = fit_and_apply_temperature_scaling(
        cal_logits=val_clf_np,
        cal_targets=val_y_np,
        eval_logits=val_clf_np,
        eval_targets=val_y_np,
    )

    calibration_results = {
        "fitted_temperature": fitted_t,
        "validation_uncalibrated": uncal_val_cal,
        "validation_calibrated": cal_val_cal,
    }

    # ── 7. Untouched Final Test Partition Evaluation (Frozen Model) ────────
    test_metrics = None
    uncertainty_report = None

    if use_three_way_split and X_c_te is not None:
        logger.info("Evaluating FROZEN model on UNTOUCHED final test partition (%d samples)...", len(y_te))
        test_ds = ETHDataset(X_c_te, X_m_te, y_te)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        test_reg_preds, test_clf_preds, test_targets_list = [], [], []
        with torch.no_grad():
            for x_c, x_m, target in test_loader:
                reg_pred, clf_pred = model(x_c.to(device), x_m.to(device))
                test_reg_preds.extend(reg_pred.squeeze(-1).cpu().numpy())
                test_clf_preds.extend(clf_pred.squeeze(-1).cpu().numpy())
                test_targets_list.extend(target.numpy())

        te_reg_np = np.array(test_reg_preds, dtype=np.float32)
        te_clf_np = np.array(test_clf_preds, dtype=np.float32)
        te_y_np = np.array(test_targets_list, dtype=np.float32)

        test_metrics = compute_comprehensive_metrics(
            reg_preds=te_reg_np,
            clf_logits=te_clf_np,
            targets=te_y_np,
            signal_threshold=SIGNAL_THRESHOLD,
            target_multiplier=TARGET_MULTIPLIER,
        )

        # Apply the frozen temperature parameter fitted on calibration partition
        te_cal_probs = 1.0 / (1.0 + np.exp(-np.clip(te_clf_np / fitted_t, -30.0, 30.0)))
        test_cal_eval = evaluate_calibration(te_cal_probs, (te_y_np > 0).astype(int))
        test_metrics["calibrated_with_val_temperature"] = test_cal_eval
        test_metrics["temperature_applied"] = fitted_t

        # ── 8. Block Bootstrap Uncertainty on Final Test Partition ─────────
        logger.info("Computing Moving Block Bootstrap Uncertainty on Test Partition across blocks [12, 24, 48, 168]...")
        block_sizes = [12, 24, 48, 168]
        bootstrap_results = {}

        def acc_metric(preds, targets):
            return directional_accuracy(preds, targets)

        def mse_metric(preds, targets):
            return float(np.mean((preds - targets) ** 2))

        def brier_metric(probs, targets):
            y_bin = (targets > 0).astype(np.float64)
            return float(np.mean((probs - y_bin) ** 2))

        for b_size in block_sizes:
            if b_size >= len(te_y_np) // 2:
                continue
            ci_acc = block_bootstrap_confidence_interval(te_y_np, te_reg_np, acc_metric, block_size=b_size, n_bootstraps=500)
            ci_mse = block_bootstrap_confidence_interval(te_y_np, te_reg_np, mse_metric, block_size=b_size, n_bootstraps=500)
            ci_brier = block_bootstrap_confidence_interval(te_y_np, te_cal_probs, brier_metric, block_size=b_size, n_bootstraps=500)

            bootstrap_results[f"block_{b_size}_hours"] = {
                "block_size": b_size,
                "directional_accuracy": ci_acc,
                "regression_mse": ci_mse,
                "brier_score": ci_brier,
            }

        uncertainty_report = {
            "test_sample_count": len(te_y_np),
            "bootstrap_replications": 500,
            "confidence_level": 0.95,
            "block_evaluations": bootstrap_results,
        }

    # Consolidated Results Manifest
    results = {
        "validation_metrics": val_metrics,
        "calibration_results": calibration_results,
        "final_test_metrics": test_metrics,
        "test_uncertainty_analysis": uncertainty_report,
        "split_metadata": split_meta,
    }

    # Logging summary
    logger.info("=" * 65)
    logger.info("TRAINING PIPELINE COMPLETE (3-Way Split Integration)")
    logger.info("  Validation Acc (Selection):    %.2f%% | MSE: %.4f",
                val_metrics["trading_signal_directional_accuracy"] * 100, val_metrics["regression_mse"])
    logger.info("  Fitted Calibration Temperature: T = %.4f", fitted_t)
    if test_metrics:
        logger.info("  UNTOUCHED FINAL TEST ACCURACY: %.2f%% | MSE: %.4f",
                    test_metrics["trading_signal_directional_accuracy"] * 100, test_metrics["regression_mse"])
        logger.info("  Test Brier Score:              %.4f (Baseline: %.4f)",
                    test_metrics["calibrated_with_val_temperature"]["brier_score"],
                    test_metrics["calibrated_with_val_temperature"]["brier_baseline_constant_prior"])
        if uncertainty_report and "block_12_hours" in uncertainty_report["block_evaluations"]:
            ci = uncertainty_report["block_evaluations"]["block_12_hours"]["directional_accuracy"]
            logger.info("  Test Acc 95%% CI (Block 12h):   [%.2f%% .. %.2f%%] (StdErr: %.4f)",
                        ci["ci_lower"] * 100, ci["ci_upper"] * 100, ci["std_error"])
    logger.info("=" * 65)

    return model, results
