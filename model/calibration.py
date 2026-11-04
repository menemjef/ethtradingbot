"""
model/calibration.py — Probability calibration and temperature scaling.

Provides post-hoc calibration methods (Temperature Scaling / Platt Scaling)
evaluated strictly on a designated calibration partition to ensure predicted
probabilities correspond to empirical frequencies.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS

logger = logging.getLogger(__name__)


class ModelWithTemperature(nn.Module):
    """
    Temperature scaling post-processing wrapper for classification logits.
    Output: calibrated_probabilities = sigmoid(logits / temperature).
    Temperature > 1 softens overconfident probabilities.
    Temperature < 1 sharpens underconfident probabilities.
    """
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """
        Fit temperature parameter using L-BFGS optimizer on calibration partition.
        val_labels must be binary float {0.0, 1.0}.
        """
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()
        bce_criterion = nn.BCEWithLogitsLoss()

        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = bce_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        with torch.no_grad():
            self.temperature.clamp_(min=0.01, max=10.0)
        fitted_t = float(self.temperature.item())
        logger.info("Temperature scaling fitted: T = %.4f", fitted_t)
        return fitted_t


def evaluate_calibration(
    probs: np.ndarray,
    binary_targets: np.ndarray,
    n_bins: int = 10,
    fixed_prior: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate calibration metrics against a constant prior probability baseline.

    Args:
        probs: 1D array of predicted probabilities in [0, 1].
        binary_targets: 1D array of true binary labels in {0, 1}.
        n_bins: Number of bins for reliability diagram and ECE.
        fixed_prior: Optional baseline prior probability fixed from training/calibration data.
                     If None, defaults to the empirical positive fraction of binary_targets.
    """
    N = len(binary_targets)
    assert len(probs) == N, "Mismatched lengths"
    eps = 1e-12
    p = np.clip(probs, eps, 1.0 - eps)
    y = binary_targets.astype(np.float64)

    # 1. Prior baseline (constant-probability baseline)
    if fixed_prior is not None:
        p_prior = float(np.clip(fixed_prior, eps, 1.0 - eps))
    else:
        p_prior = float(np.clip(np.mean(y), eps, 1.0 - eps))

    # 2. Brier Score
    brier = float(np.mean((p - y) ** 2))
    brier_baseline = float(np.mean((p_prior - y) ** 2))
    brier_skill_score = float(1.0 - (brier / brier_baseline)) if brier_baseline > 0 else 0.0

    # 3. Log Loss (Cross-Entropy)
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    log_loss_baseline = float(-(p_prior * np.log(p_prior) + (1.0 - p_prior) * np.log(1.0 - p_prior)))
    log_loss_skill_score = float(1.0 - (log_loss / log_loss_baseline)) if log_loss_baseline > 0 else 0.0

    # 4. Expected Calibration Error (ECE) and Reliability Curve
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability_curve = []

    for b in range(n_bins):
        b_low = float(bin_boundaries[b])
        b_high = float(bin_boundaries[b + 1])
        mask = (p >= b_low) & (p < b_high if b < n_bins - 1 else p <= b_high)
        count = int(np.sum(mask))

        if count > 0:
            bin_acc = float(np.mean(y[mask]))
            bin_conf = float(np.mean(p[mask]))
            ece += (count / N) * abs(bin_acc - bin_conf)
            reliability_curve.append({
                "bin": b,
                "bin_range": (b_low, b_high),
                "sample_count": count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "calibration_gap": abs(bin_acc - bin_conf),
            })
        else:
            reliability_curve.append({
                "bin": b,
                "bin_range": (b_low, b_high),
                "sample_count": 0,
                "confidence": 0.5 * (b_low + b_high),
                "accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return {
        "sample_count": N,
        "validation_samples": N,
        "brier_score": brier,
        "brier_baseline_constant_prior": brier_baseline,
        "brier_improvement_over_baseline": float(brier_baseline - brier),
        "brier_skill_score": brier_skill_score,
        "log_loss": log_loss,
        "log_loss_baseline_constant_prior": log_loss_baseline,
        "log_loss_skill_score": log_loss_skill_score,
        "expected_calibration_error_ece": float(ece),
        "mean_predicted_probability": float(np.mean(p)),
        "empirical_prior_positive_rate": p_prior,
        "reliability_curve": reliability_curve,
    }


def fit_and_apply_temperature_scaling(
    cal_logits: np.ndarray,
    cal_targets: np.ndarray,
    eval_logits: np.ndarray,
    eval_targets: np.ndarray,
    fixed_prior: Optional[float] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any], Dict[str, Any]]:
    """
    Fits temperature parameter strictly on the calibration split,
    and applies it to the evaluation split.

    Returns:
        calibrated_eval_probs, fitted_T, uncalibrated_eval_metrics, calibrated_eval_metrics
    """
    cal_logits_t = torch.from_numpy(cal_logits.astype(np.float32))
    cal_labels_t = torch.from_numpy((cal_targets > 0).astype(np.float32))

    temp_scaler = ModelWithTemperature()
    fitted_t = temp_scaler.fit(cal_logits_t, cal_labels_t)

    # Uncalibrated evaluation metrics
    raw_eval_probs = 1.0 / (1.0 + np.exp(-np.clip(eval_logits, -30.0, 30.0)))
    eval_binary = (eval_targets > 0).astype(int)
    uncal_metrics = evaluate_calibration(raw_eval_probs, eval_binary, fixed_prior=fixed_prior)

    # Calibrated evaluation metrics
    with torch.no_grad():
        cal_eval_logits_t = temp_scaler(torch.from_numpy(eval_logits.astype(np.float32)))
        cal_eval_probs = torch.sigmoid(cal_eval_logits_t).numpy()

    cal_metrics = evaluate_calibration(cal_eval_probs, eval_binary, fixed_prior=fixed_prior)
    cal_metrics["temperature_fitted"] = fitted_t

    return cal_eval_probs, fitted_t, uncal_metrics, cal_metrics
