"""
model/trainer.py — Training loop with early stopping and model checkpointing.
"""

import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

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
)
from model.architecture import HybridModel
from model.dataset import ETHDataset

logger = logging.getLogger(__name__)


def directional_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of predictions where sign(pred) == sign(target)."""
    return float(np.mean((preds > 0) == (targets > 0)))


def _build_recency_weights(n_samples: int, recency_factor: float = 1.5) -> np.ndarray:
    """
    Build sample weights that increase linearly toward the most recent data.
    Oldest sample gets weight 1.0, newest gets weight `recency_factor`.
    This forces the model to prioritize learning from current market conditions.
    """
    weights = np.linspace(1.0, recency_factor, n_samples)
    return weights / weights.sum() * n_samples  # Normalize so sum = n_samples


def train_model(
    X_candle: np.ndarray,
    X_market: np.ndarray,
    y: np.ndarray,
    device: str = "auto",
) -> HybridModel:
    """
    Full training pipeline.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    # ── Dataset split ──────────────────────────────────────────────────────
    N = len(y)
    split_idx = int(N * TRAIN_SPLIT)

    train_ds = ETHDataset(X_candle[:split_idx], X_market[:split_idx], y[:split_idx])
    val_ds = ETHDataset(X_candle[split_idx:], X_market[split_idx:], y[split_idx:])

    # Recency-weighted sampling: recent training samples get higher probability
    sample_weights = _build_recency_weights(len(train_ds), RECENCY_WEIGHT)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Train samples: %d | Val samples: %d | Recency weight: %.1fx", len(train_ds), len(val_ds), RECENCY_WEIGHT)

    # ── Model, optimizer, loss ─────────────────────────────────────────────
    model = HybridModel().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    
    mse_criterion = nn.MSELoss()
    # Label smoothing: convert hard 0/1 labels to 0.125/0.875
    label_smoothing = 0.25
    # Weighted BCE to counter 54/46 Short/Long imbalance
    pos_weight = torch.tensor([1.17], device=device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_losses = []

        for x_c, x_m, target in train_loader:
            x_c = x_c.to(device)
            x_m = x_m.to(device)
            target = target.to(device).unsqueeze(-1) * TARGET_MULTIPLIER       # (B, 1)
            
            # Linear LR Warmup (first 5 epochs)
            if epoch <= 5:
                warmup_lr = LEARNING_RATE * (epoch / 5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Noise injection for regularization
            if NOISE_INJECTION > 0:
                x_c = x_c + torch.randn_like(x_c) * NOISE_INJECTION
                x_m = x_m + torch.randn_like(x_m) * NOISE_INJECTION
            
            # Binary label for direction: TBM 1 -> 1.0, TBM -1 -> 0.0
            # Target is coming in as scaled by TARGET_MULTIPLIER
            raw_tbm = target / TARGET_MULTIPLIER
            clf_label = (raw_tbm > 0).float()

            optimizer.zero_grad()
            reg_pred, clf_pred = model(x_c, x_m)
            
            # Apply label smoothing: 0 -> 0.05, 1 -> 0.95
            smoothed_label = clf_label * (1.0 - label_smoothing) + 0.5 * label_smoothing
            loss_clf = bce_criterion(clf_pred, smoothed_label)
            
            # Multi-Task Loss - skip regression if weight is 0
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
        val_losses, val_clf_preds, val_targets = [], [], []
        with torch.no_grad():
            for x_c, x_m, target in val_loader:
                x_c = x_c.to(device)
                x_m = x_m.to(device)
                target = target.to(device).unsqueeze(-1) * TARGET_MULTIPLIER
                
                reg_pred, clf_pred = model(x_c, x_m)
                
                # We primarily use regression prediction for accuracy calculation
                val_losses.append(mse_criterion(reg_pred, target).item())
                val_clf_preds.extend(clf_pred.squeeze(-1).cpu().numpy())
                val_targets.extend(target.squeeze(-1).cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        # ─── Top-Confidence Checkpointing (Round 6) ───────────────────────
        # Calculate win rate for the Top 20% most confident predictions
        val_logits = np.array(val_clf_preds)
        val_y = np.array(val_targets)
        
        # Sort by absolute logit (higher = more confident)
        top_k = int(len(val_logits) * 0.20)
        top_indices = np.argsort(np.abs(val_logits))[-top_k:]
        
        top_acc = directional_accuracy(val_logits[top_indices], val_y[top_indices])
        
        scheduler.step()
        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | Train Total Loss: %.4f | Val MSE: %.2f | Top-20%% Acc: %.2f%% | LR: %.2e",
            epoch, MAX_EPOCHS, train_loss, val_loss, top_acc * 100,
            optimizer.param_groups[0]["lr"]
        )

        # Save best model based on VALIDATION LOSS (not accuracy — prevents overfitting to noise)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = top_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            patience_counter = 0
            logger.info("  [BEST] Model saved (Val Loss: %.4f | Top-20%% Acc: %.2f%%)", best_val_loss, best_val_acc * 100)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, PATIENCE)
                break
    
    # Load the best checkpoint if it exists
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    logger.info("Training complete. Best Top-20%% Accuracy: %.2f%%", best_val_acc * 100)

    # Final validation metrics
    model.eval()
    all_clf_preds, all_reg_preds, all_targets = [], [], []
    with torch.no_grad():
        for x_c, x_m, target in val_loader:
            target = target * TARGET_MULTIPLIER
            reg_pred, clf_pred = model(x_c.to(device), x_m.to(device))
            all_clf_preds.extend(clf_pred.squeeze(-1).cpu().numpy())
            all_reg_preds.extend(reg_pred.squeeze(-1).cpu().numpy())
            all_targets.extend(target.numpy())

    all_clf_preds = np.array(all_clf_preds)
    all_reg_preds = np.array(all_reg_preds)
    all_targets = np.array(all_targets)
    final_dir_acc = directional_accuracy(all_clf_preds, all_targets)

    # Threshold-filtered accuracy: Filter by regression magnitude > scaled threshold
    scaled_thresh = SIGNAL_THRESHOLD * TARGET_MULTIPLIER
    mask = np.abs(all_reg_preds) >= scaled_thresh
    thresh_acc = 0.0
    if mask.sum() > 0:
        thresh_acc = directional_accuracy(all_clf_preds[mask], all_targets[mask])
        logger.info(
            "Val directional accuracy (all): %.2f%% | Above thresh (|reg| >= %.4f): %.2f%% on %d samples",
            final_dir_acc * 100, SIGNAL_THRESHOLD, thresh_acc * 100, mask.sum(),
        )
    else:
        logger.info("Val directional accuracy: %.2f%% (No samples above threshold)", final_dir_acc * 100)

    return model, final_dir_acc, thresh_acc
