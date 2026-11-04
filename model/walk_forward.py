"""
model/walk_forward.py — Walk-Forward Cross-Validation Engine.

Executes rigorous out-of-sample walk-forward validation:
1. Iterates chronologically across multiple expanding or rolling folds.
2. In each fold, fits scalers strictly on the fold's training partition.
3. Trains a fresh model with purged boundaries.
4. Evaluates out-of-sample performance on the subsequent unseen test window.
5. Simulates trade execution and calculates financial performance metrics.
6. Aggregates statistical performance across all folds.
"""

import logging
import os
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import config
from config import (
    CANDLE_FEATURES,
    MARKET_FEATURES,
    LOOKBACK,
    TARGET_HORIZON,
    TARGET_MULTIPLIER,
    SIGNAL_THRESHOLD,
    TBM_PT_MULT,
    TBM_SL_MULT,
)
from model.architecture import HybridModel
from model.dataset import ETHDataset
from model.trainer import directional_accuracy, compute_comprehensive_metrics
from model.calibration import evaluate_calibration
from data.indicators import add_tbm_target
from data.preprocessor import (
    build_features,
    build_windows,
    generate_walk_forward_splits,
    transform,
)

logger = logging.getLogger(__name__)


def simulate_trades_from_signals(
    signals: np.ndarray,
    close_prices: np.ndarray,
    atr_values: np.ndarray,
    signal_threshold: float = SIGNAL_THRESHOLD,
    pt_mult: float = TBM_PT_MULT,
    sl_mult: float = TBM_SL_MULT,
    holding_max: int = TARGET_HORIZON,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0005,
    slippage: float = 0.0003,
) -> Dict[str, Any]:
    """
    Simulate trade execution from raw signal predictions using barrier stops.
    Accounts for realistic execution fees (maker/taker) and slippage.
    """
    N = len(signals)
    trades = []
    in_pos = False
    side = 0
    entry_price = 0.0
    entry_bar = 0
    pt_price = 0.0
    sl_price = 0.0

    cost_per_trade = (taker_fee + maker_fee + 2 * slippage)

    for i in range(N):
        current_price = close_prices[i]
        current_atr = atr_values[i]
        sig = signals[i]

        if not in_pos:
            if sig >= signal_threshold:
                in_pos = True
                side = 1  # Long
                entry_price = current_price
                entry_bar = i
                pt_price = entry_price + (pt_mult * current_atr)
                sl_price = entry_price - (sl_mult * current_atr)
            elif sig <= -signal_threshold:
                in_pos = True
                side = -1  # Short
                entry_price = current_price
                entry_bar = i
                pt_price = entry_price - (pt_mult * current_atr)
                sl_price = entry_price + (sl_mult * current_atr)
        else:
            holding_bars = i - entry_bar
            exit_trade = False
            exit_price = current_price
            exit_reason = "horizon"

            if side == 1:
                if current_price >= pt_price:
                    exit_price = pt_price
                    exit_trade = True
                    exit_reason = "take_profit"
                elif current_price <= sl_price:
                    exit_price = sl_price
                    exit_trade = True
                    exit_reason = "stop_loss"
                elif holding_bars >= holding_max:
                    exit_price = current_price
                    exit_trade = True
                    exit_reason = "horizon"
            elif side == -1:
                if current_price <= pt_price:
                    exit_price = pt_price
                    exit_trade = True
                    exit_reason = "take_profit"
                elif current_price >= sl_price:
                    exit_price = sl_price
                    exit_trade = True
                    exit_reason = "stop_loss"
                elif holding_bars >= holding_max:
                    exit_price = current_price
                    exit_trade = True
                    exit_reason = "horizon"

            if exit_trade:
                gross_ret = side * ((exit_price - entry_price) / entry_price)
                net_ret = gross_ret - cost_per_trade
                trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "side": "LONG" if side == 1 else "SHORT",
                    "gross_return": float(gross_ret),
                    "net_return": float(net_ret),
                    "exit_reason": exit_reason,
                    "holding_bars": holding_bars,
                })
                in_pos = False

    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_net_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": [],
        }

    df_tr = pd.DataFrame(trades)
    wins = df_tr[df_tr["net_return"] > 0]
    losses = df_tr[df_tr["net_return"] < 0]

    gross_profit = float(wins["net_return"].sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses["net_return"].sum())) if len(losses) > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    win_rate = float(len(wins) / len(df_tr))

    # Equity curve and maximum drawdown
    equity = (1.0 + df_tr["net_return"]).cumprod()
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    max_dd = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0
    total_net_pnl = float((equity.iloc[-1] - 1.0) * 100.0) if len(equity) > 0 else 0.0

    return {
        "trade_count": len(df_tr),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_net_pnl_pct": total_net_pnl,
        "max_drawdown_pct": max_dd * 100.0,
        "trades": trades,
    }


def run_walk_forward_evaluation(
    df_raw: pd.DataFrame,
    n_splits: int = 4,
    train_ratio: float = 0.50,
    test_ratio: float = 0.125,
    purge_horizon: int = TARGET_HORIZON,
    max_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 5e-4,
    device: str = "auto",
    signal_threshold: float = SIGNAL_THRESHOLD,
) -> Dict[str, Any]:
    """
    Full end-to-end Walk-Forward Evaluation pipeline.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Walk-Forward Evaluation on device: %s (%d folds)", device, n_splits)

    # 1. Feature Engineering
    df_feat = build_features(df_raw, add_targets=False)
    df_feat = add_tbm_target(
        df_feat,
        pt_mult=TBM_PT_MULT,
        sl_mult=TBM_SL_MULT,
        horizon=TARGET_HORIZON,
        ambiguous_resolution="exclude",
    ).dropna(subset=["target"])

    total_samples = len(df_feat) - LOOKBACK
    splits = generate_walk_forward_splits(
        total_samples=total_samples,
        n_splits=n_splits,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        purge_horizon=purge_horizon,
        expanding=True,
    )
    logger.info("Generated %d walk-forward splits from %d total window samples", len(splits), total_samples)

    fold_results = []

    for fold_info in splits:
        f_idx = fold_info["split_idx"]
        tr_start, tr_end = fold_info["train_indices"]
        te_start, te_end = fold_info["test_indices"]

        logger.info(
            "--- Fold %d/%d: Train [%d..%d] (%d samples) | Test [%d..%d] (%d samples) | Purged %d ---",
            f_idx + 1, len(splits), tr_start, tr_end, fold_info["train_count"],
            te_start, te_end, fold_info["test_count"], fold_info["purge_count"],
        )

        # Pre-roll window index alignment:
        # Window k corresponds to df_feat rows [k : k + LOOKBACK]
        # Features for train partition are fit on df_feat up to tr_end + LOOKBACK
        train_df_slice = df_feat.iloc[tr_start : tr_end + LOOKBACK]
        test_df_slice = df_feat.iloc[te_start : te_end + LOOKBACK]

        # 2. Fit Scalers STRICTLY on this fold's training partition
        scaler_c = StandardScaler()
        scaler_m = StandardScaler()
        scaler_c.fit(train_df_slice[CANDLE_FEATURES].values)
        scaler_m.fit(train_df_slice[MARKET_FEATURES].values)

        # 3. Transform train and test partitions independently
        c_tr, m_tr = transform(train_df_slice, scaler_c, scaler_m)
        y_tr = train_df_slice["target"].values.astype(np.float32)
        Xc_tr, Xm_tr, y_tr_win = build_windows(c_tr, m_tr, y_tr, lookback=LOOKBACK)

        c_te, m_te = transform(test_df_slice, scaler_c, scaler_m)
        y_te = test_df_slice["target"].values.astype(np.float32)
        Xc_te, Xm_te, y_te_win = build_windows(c_te, m_te, y_te, lookback=LOOKBACK)

        # Data loaders
        train_ds = ETHDataset(Xc_tr, Xm_tr, y_tr_win)
        test_ds = ETHDataset(Xc_te, Xm_te, y_te_win)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # 4. Train fresh model on fold training partition
        model = HybridModel().to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        mse_loss = nn.MSELoss()

        n_pos = int((y_tr_win > 0).sum())
        n_neg = int((y_tr_win <= 0).sum())
        pos_weight_val = float(n_neg / max(1, n_pos))
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

        model.train()
        for ep in range(1, max_epochs + 1):
            for xc, xm, yt in train_loader:
                xc = xc.to(device)
                xm = xm.to(device)
                target_reg = yt.to(device).unsqueeze(-1) * TARGET_MULTIPLIER
                target_clf = (yt.to(device).unsqueeze(-1) > 0).float()

                optimizer.zero_grad()
                reg_out, clf_out = model(xc, xm)
                loss = mse_loss(reg_out, target_reg) + bce_loss(clf_out, target_clf)
                loss.backward()
                optimizer.step()

        # 5. Out-of-sample evaluation on unseen test partition
        model.eval()
        test_reg_preds, test_clf_preds, test_targets = [], [], []
        with torch.no_grad():
            for xc, xm, yt in test_loader:
                reg_out, clf_out = model(xc.to(device), xm.to(device))
                test_reg_preds.extend(reg_out.squeeze(-1).cpu().numpy())
                test_clf_preds.extend(clf_out.squeeze(-1).cpu().numpy())
                test_targets.extend(yt.numpy())

        reg_p = np.array(test_reg_preds, dtype=np.float32)
        clf_p = np.array(test_clf_preds, dtype=np.float32)
        y_eval = np.array(test_targets, dtype=np.float32)

        fold_metrics = compute_comprehensive_metrics(
            reg_preds=reg_p,
            clf_logits=clf_p,
            targets=y_eval,
            signal_threshold=signal_threshold,
            target_multiplier=TARGET_MULTIPLIER,
        )

        # 6. Simulate trades
        test_close = test_df_slice["close"].iloc[LOOKBACK:].values[:len(y_eval)]
        test_atr = test_df_slice["atr_14"].iloc[LOOKBACK:].values[:len(y_eval)] if "atr_14" in test_df_slice else np.ones(len(y_eval)) * 10.0
        signals_sim = reg_p / TARGET_MULTIPLIER

        sim_res = simulate_trades_from_signals(
            signals=signals_sim,
            close_prices=test_close,
            atr_values=test_atr,
            signal_threshold=signal_threshold,
        )

        record = {
            "fold": f_idx,
            "train_samples": len(y_tr_win),
            "test_samples": len(y_eval),
            "directional_accuracy_reg_pct": float(round(fold_metrics["trading_signal_directional_accuracy"] * 100, 2)),
            "directional_accuracy_clf_pct": float(round(fold_metrics["classification_directional_accuracy"] * 100, 2)),
            "regression_mse": float(round(fold_metrics["regression_mse"], 4)),
            "brier_score": float(round(fold_metrics["calibration"]["brier_score"], 4)),
            "log_loss": float(round(fold_metrics["calibration"]["log_loss"], 4)),
            "trade_count": sim_res["trade_count"],
            "win_rate_pct": float(round(sim_res["win_rate"] * 100, 2)),
            "profit_factor": float(round(sim_res["profit_factor"], 2)),
            "net_pnl_pct": float(round(sim_res["total_net_pnl_pct"], 2)),
            "max_drawdown_pct": float(round(sim_res["max_drawdown_pct"], 2)),
        }
        fold_results.append(record)
        logger.info(
            "Fold %d Result: DirAcc=%.2f%% | MSE=%.4f | Brier=%.4f | Trades=%d | WinRate=%.2f%% | PF=%.2f | NetPnL=%.2f%%",
            f_idx + 1, record["directional_accuracy_reg_pct"], record["regression_mse"],
            record["brier_score"], record["trade_count"], record["win_rate_pct"],
            record["profit_factor"], record["net_pnl_pct"],
        )

    # 7. Aggregate Across All Folds
    acc_vals = [f["directional_accuracy_reg_pct"] for f in fold_results]
    mse_vals = [f["regression_mse"] for f in fold_results]
    brier_vals = [f["brier_score"] for f in fold_results]
    pf_vals = [f["profit_factor"] for f in fold_results]
    pnl_vals = [f["net_pnl_pct"] for f in fold_results]

    aggregate = {
        "total_folds": len(fold_results),
        "mean_directional_accuracy_pct": float(np.mean(acc_vals)),
        "std_directional_accuracy_pct": float(np.std(acc_vals)),
        "mean_regression_mse": float(np.mean(mse_vals)),
        "mean_brier_score": float(np.mean(brier_vals)),
        "mean_profit_factor": float(np.mean(pf_vals)),
        "total_net_pnl_sum_pct": float(np.sum(pnl_vals)),
        "total_trades_all_folds": int(sum(f["trade_count"] for f in fold_results)),
    }

    results = {
        "aggregate_summary": aggregate,
        "fold_by_fold_results": fold_results,
    }
    return results
