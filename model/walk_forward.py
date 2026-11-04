"""
model/walk_forward.py — Walk-Forward Cross-Validation Engine with OHLC-Aware Simulation.

Executes rigorous out-of-sample walk-forward validation:
1. Clear chronological boundaries: all training targets resolve before the first out-of-sample prediction.
2. Scalers fitted strictly on each fold's historical training slice.
3. Preceding historical bars used as continuous lookback context for out-of-sample test windows.
4. Inner chronological validation partition for early stopping and checkpoint selection.
5. Next-bar Open execution with modeled slippage and explicit maker/taker fee accounting.
6. Intrabar High/Low barrier checks with conservative stop-first collision resolution.
7. Forced mark-to-market and closure of end-of-fold open positions.
8. Pooled profit factor and sequential compounded return aggregation.
"""

import logging
import os
import time
from typing import Dict, Any, List, Tuple, Optional
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
    transform,
)

logger = logging.getLogger(__name__)


def simulate_trades_ohlc(
    signals: np.ndarray,
    df_test: pd.DataFrame,
    signal_threshold: float = SIGNAL_THRESHOLD,
    pt_mult: float = TBM_PT_MULT,
    sl_mult: float = TBM_SL_MULT,
    holding_max: int = TARGET_HORIZON,
    taker_fee: float = 0.0005,
    maker_fee: float = 0.0002,
    slippage: float = 0.0003,
) -> Dict[str, Any]:
    """
    OHLC-aware barrier trade simulation:
    - Signal is observed at candle close of bar i.
    - Entry order executes at bar i + 1 OPEN with modeled slippage.
    - Intrabar TP/SL evaluated against subsequent candle High and Low.
    - If both TP and SL are touched in the same candle: resolves conservatively STOP-FIRST.
    - Time-horizon exit at candle close of bar (entry_bar + holding_max).
    - Forced end-of-fold close: any remaining open position at the final test bar
      is marked to market and closed at final candle CLOSE.
    - Itemized tracking of gross PnL, fees, slippage, and net PnL.
    """
    N = len(signals)
    assert len(df_test) >= N, f"df_test length ({len(df_test)}) must cover signals length ({N})"

    # Align arrays with signals
    opens = df_test["open"].values[:N]
    highs = df_test["high"].values[:N]
    lows = df_test["low"].values[:N]
    closes = df_test["close"].values[:N]
    atrs = df_test["atr_14"].values[:N] if "atr_14" in df_test else np.full(N, 10.0)
    timestamps = df_test.index[:N]

    trades = []
    in_pos = False
    side = 0
    entry_price = 0.0
    entry_bar = 0
    entry_time = None
    pt_price = 0.0
    sl_price = 0.0
    pending_signal = 0  # 1 for long, -1 for short
    pending_signal_bar = -1

    trade_id = 0

    for i in range(N):
        # 1. Execute pending entry from previous candle signal at current candle OPEN
        if not in_pos and pending_signal != 0:
            side = pending_signal
            entry_bar = i
            entry_time = timestamps[i]
            current_open = opens[i]

            # Slippage on entry (market taker order)
            if side == 1:
                entry_price = current_open * (1.0 + slippage)
                pt_price = entry_price + (pt_mult * atrs[pending_signal_bar])
                sl_price = entry_price - (sl_mult * atrs[pending_signal_bar])
            else:
                entry_price = current_open * (1.0 - slippage)
                pt_price = entry_price - (pt_mult * atrs[pending_signal_bar])
                sl_price = entry_price + (sl_mult * atrs[pending_signal_bar])

            in_pos = True
            pending_signal = 0
            pending_signal_bar = -1

        # 2. Check position exits during candle i
        if in_pos:
            holding_bars = i - entry_bar
            exit_trade = False
            exit_price = 0.0
            exit_reason = "horizon"
            exit_fee = taker_fee
            is_collision = False

            if side == 1:  # LONG
                hit_sl = bool(lows[i] <= sl_price)
                hit_tp = bool(highs[i] >= pt_price)

                if hit_sl and hit_tp:
                    # Intrabar collision: Conservative STOP-FIRST resolution
                    exit_trade = True
                    exit_price = sl_price * (1.0 - slippage)
                    exit_reason = "stop_loss_collision"
                    exit_fee = taker_fee
                    is_collision = True
                elif hit_sl:
                    exit_trade = True
                    exit_price = sl_price * (1.0 - slippage)
                    exit_reason = "stop_loss"
                    exit_fee = taker_fee
                elif hit_tp:
                    exit_trade = True
                    exit_price = pt_price * (1.0 - slippage)
                    exit_reason = "take_profit"
                    exit_fee = maker_fee  # Limit order fill at profit target
                elif holding_bars >= holding_max:
                    exit_trade = True
                    exit_price = closes[i] * (1.0 - slippage)
                    exit_reason = "horizon"
                    exit_fee = taker_fee

            elif side == -1:  # SHORT
                hit_sl = bool(highs[i] >= sl_price)
                hit_tp = bool(lows[i] <= pt_price)

                if hit_sl and hit_tp:
                    # Intrabar collision: Conservative STOP-FIRST resolution
                    exit_trade = True
                    exit_price = sl_price * (1.0 + slippage)
                    exit_reason = "stop_loss_collision"
                    exit_fee = taker_fee
                    is_collision = True
                elif hit_sl:
                    exit_trade = True
                    exit_price = sl_price * (1.0 + slippage)
                    exit_reason = "stop_loss"
                    exit_fee = taker_fee
                elif hit_tp:
                    exit_trade = True
                    exit_price = pt_price * (1.0 + slippage)
                    exit_reason = "take_profit"
                    exit_fee = maker_fee
                elif holding_bars >= holding_max:
                    exit_trade = True
                    exit_price = closes[i] * (1.0 + slippage)
                    exit_reason = "horizon"
                    exit_fee = taker_fee

            # Forced end-of-fold close at final bar
            if not exit_trade and i == N - 1:
                exit_trade = True
                exit_price = closes[i] * (1.0 - slippage) if side == 1 else closes[i] * (1.0 + slippage)
                exit_reason = "end_of_fold"
                exit_fee = taker_fee

            if exit_trade:
                trade_id += 1
                gross_ret = side * ((exit_price - entry_price) / entry_price)
                total_fees = taker_fee + exit_fee
                # Explicit fee and slippage breakdown
                net_ret = gross_ret - total_fees

                trades.append({
                    "trade_id": trade_id,
                    "entry_bar": entry_bar,
                    "entry_time": str(entry_time),
                    "exit_bar": i,
                    "exit_time": str(timestamps[i]),
                    "side": "LONG" if side == 1 else "SHORT",
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "pt_price": float(pt_price),
                    "sl_price": float(sl_price),
                    "exit_reason": exit_reason,
                    "is_collision": is_collision,
                    "holding_bars": holding_bars,
                    "gross_return": float(gross_ret),
                    "fee_cost": float(total_fees),
                    "slippage_cost": float(2.0 * slippage),
                    "funding_cost": 0.0,  # Explicitly recorded: marked as excluding perpetual funding
                    "net_return": float(net_ret),
                })
                in_pos = False

        # 3. Check for new signal at close of candle i (if not currently in position)
        if not in_pos and i < N - 1:
            sig = signals[i]
            if sig >= signal_threshold:
                pending_signal = 1
                pending_signal_bar = i
            elif sig <= -signal_threshold:
                pending_signal = -1
                pending_signal_bar = i

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

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = 99.0 if gross_profit > 0 else 0.0

    win_rate = float(len(wins) / len(df_tr))
    total_net_pnl = float(df_tr["net_return"].sum()) * 100.0

    # Drawdown calculation
    equity_curve = (1.0 + df_tr["net_return"]).cumprod()
    peak = equity_curve.cummax()
    drawdowns = (peak - equity_curve) / peak
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

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
    pt_mult: float = TBM_PT_MULT,
    sl_mult: float = TBM_SL_MULT,
) -> Dict[str, Any]:
    """
    Full end-to-end Walk-Forward Evaluation pipeline with:
    1. Continuous hourly feature preservation without row compression.
    2. Strict temporal purge boundary: all training labels resolve before test prediction starts.
    3. Scalers fitted strictly on training partition available before test window.
    4. Inner validation partition with early stopping (no blind 300 epochs).
    5. OHLC-aware simulation with next-bar open fill, intrabar TP/SL, and stop-first collision resolution.
    6. Pooled profit factor and sequential compounded return aggregation.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Rigorous Walk-Forward Evaluation on device: %s (%d folds)", device, n_splits)

    # 1. Feature Engineering on unbroken timeline
    df_feat = build_features(df_raw, add_targets=False)
    df_feat = add_tbm_target(
        df_feat,
        pt_mult=TBM_PT_MULT,
        sl_mult=TBM_SL_MULT,
        horizon=purge_horizon,
        ambiguous_resolution="exclude",
    )

    N_total = len(df_feat)
    test_size = int(N_total * test_ratio)
    init_train_size = int(N_total * train_ratio)

    fold_results = []
    all_trades = []

    for f_idx in range(n_splits):
        te_start_idx = init_train_size + f_idx * test_size
        te_end_idx = min(N_total, te_start_idx + test_size)
        if te_start_idx >= N_total or (te_end_idx - te_start_idx) < purge_horizon + 10:
            break

        # Boundary: Training target must fully resolve before the first test prediction timestamp
        # Test starts at te_start_idx. Preceding purge_horizon bars are purged.
        tr_end_idx = max(LOOKBACK + 50, te_start_idx - purge_horizon)
        tr_start_idx = 0

        # Exact timestamps
        t_tr_start = df_feat.index[tr_start_idx]
        t_tr_end = df_feat.index[tr_end_idx - 1]
        t_tr_res_max = t_tr_end + pd.Timedelta(hours=purge_horizon)
        t_te_start = df_feat.index[te_start_idx]
        t_te_end = df_feat.index[te_end_idx - 1]

        assert t_tr_res_max <= t_te_start, (
            f"Temporal leakage in fold {f_idx}: train label resolves at {t_tr_res_max} > test start {t_te_start}"
        )

        logger.info(
            "--- Fold %d/%d Boundaries ---", f_idx + 1, n_splits
        )
        logger.info("  Train Window: [%s -> %s] (%d bars)", t_tr_start, t_tr_end, tr_end_idx - tr_start_idx)
        logger.info("  Purge Gap:    [%s -> %s] (%d bars)", t_tr_end, t_te_start, te_start_idx - tr_end_idx)
        logger.info("  Test Window:  [%s -> %s] (%d bars)", t_te_start, t_te_end, te_end_idx - te_start_idx)

        # 2. Fit Scalers STRICTLY on this fold's available historical training slice
        df_train_history = df_feat.iloc[tr_start_idx:tr_end_idx]
        scaler_c = StandardScaler()
        scaler_m = StandardScaler()
        scaler_c.fit(df_train_history[CANDLE_FEATURES].values)
        scaler_m.fit(df_train_history[MARKET_FEATURES].values)

        # Transform train history
        c_tr_full, m_tr_full = transform(df_train_history, scaler_c, scaler_m)
        y_tr_full = df_train_history["target"].values.astype(np.float32)

        # Build windows on continuous train history
        Xc_tr_all, Xm_tr_all, y_tr_all, meta_tr = build_windows(
            c_tr_full, m_tr_full, y_tr_full,
            timestamps=df_train_history.index,
            lookback=LOOKBACK,
            purge_horizon=purge_horizon,
            drop_nan_targets=True,
            return_metadata=True,
        )

        # 3. Inner chronological validation partition for early stopping (75% train, 25% val with purge)
        n_avail = len(y_tr_all)
        inner_val_split = int(n_avail * 0.75)
        inner_purge = purge_horizon
        inner_tr_end = max(10, inner_val_split - inner_purge)

        Xc_inner_tr, Xm_inner_tr, y_inner_tr = Xc_tr_all[:inner_tr_end], Xm_tr_all[:inner_tr_end], y_tr_all[:inner_tr_end]
        Xc_inner_va, Xm_inner_va, y_inner_va = Xc_tr_all[inner_val_split:], Xm_tr_all[inner_val_split:], y_tr_all[inner_val_split:]

        inner_train_ds = ETHDataset(Xc_inner_tr, Xm_inner_tr, y_inner_tr)
        inner_val_ds = ETHDataset(Xc_inner_va, Xm_inner_va, y_inner_va)

        train_loader = DataLoader(inner_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(inner_val_ds, batch_size=batch_size, shuffle=False)

        # 4. Train Model with Early Stopping
        model = HybridModel().to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        mse_loss = nn.MSELoss()

        n_pos = int((y_inner_tr > 0).sum())
        n_neg = int((y_inner_tr <= 0).sum())
        pos_weight_val = float(n_neg / max(1, n_pos))
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

        best_val_loss = float("inf")
        best_state = None
        patience = 4
        patience_ctr = 0

        for ep in range(1, max_epochs + 1):
            model.train()
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

            # Inner validation evaluation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xc, xm, yt in val_loader:
                    xc = xc.to(device)
                    xm = xm.to(device)
                    target_reg = yt.to(device).unsqueeze(-1) * TARGET_MULTIPLIER
                    reg_out, _ = model(xc, xm)
                    val_losses.append(mse_loss(reg_out, target_reg).item())

            mean_v_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if mean_v_loss < best_val_loss:
                best_val_loss = mean_v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    logger.debug("Fold %d early stopping at epoch %d", f_idx + 1, ep)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # 5. Out-of-Sample Test Window with Historical Lookback Context
        # To evaluate test window [te_start_idx : te_end_idx], we include preceding LOOKBACK bars from history
        context_start_idx = te_start_idx - LOOKBACK
        df_test_with_context = df_feat.iloc[context_start_idx:te_end_idx]

        c_te_ctx, m_te_ctx = transform(df_test_with_context, scaler_c, scaler_m)
        y_te_ctx = df_test_with_context["target"].values.astype(np.float32)

        Xc_te, Xm_te, y_te, meta_te = build_windows(
            c_te_ctx, m_te_ctx, y_te_ctx,
            timestamps=df_test_with_context.index,
            lookback=LOOKBACK,
            purge_horizon=purge_horizon,
            drop_nan_targets=False,  # Keep all test bars for sequential trade execution
            return_metadata=True,
        )

        test_ds = ETHDataset(Xc_te, Xm_te, np.nan_to_num(y_te))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model.eval()
        test_reg_preds, test_clf_preds = [], []
        with torch.no_grad():
            for xc, xm, _ in test_loader:
                xc = xc.to(device)
                xm = xm.to(device)
                reg_out, clf_out = model(xc, xm)
                test_reg_preds.extend(reg_out.squeeze(-1).cpu().numpy())
                test_clf_preds.extend(clf_out.squeeze(-1).cpu().numpy())

        reg_p = np.array(test_reg_preds, dtype=np.float32)
        clf_p = np.array(test_clf_preds, dtype=np.float32)

        # Directional accuracy evaluated only on bars with valid targets
        valid_mask = ~np.isnan(y_te)
        if valid_mask.sum() > 0:
            reg_acc = directional_accuracy(reg_p[valid_mask], y_te[valid_mask])
            clf_acc = directional_accuracy(clf_p[valid_mask], y_te[valid_mask])
            mse_val = float(np.mean((reg_p[valid_mask] - y_te[valid_mask]) ** 2))
            probs = 1.0 / (1.0 + np.exp(-np.clip(clf_p[valid_mask], -30.0, 30.0)))
            cal_res = evaluate_calibration(probs, (y_te[valid_mask] > 0).astype(int))
            brier_val = cal_res["brier_score"]
            log_loss_val = cal_res["log_loss"]
        else:
            reg_acc, clf_acc, mse_val, brier_val, log_loss_val = 0.5, 0.5, 1.0, 0.25, 0.693

        # 6. OHLC-Aware Trade Simulation
        # df_test_slice corresponds to decision candles
        df_test_slice = df_feat.iloc[te_start_idx:te_end_idx]
        signals_sim = reg_p / TARGET_MULTIPLIER

        sim_res = simulate_trades_ohlc(
            signals=signals_sim,
            df_test=df_test_slice,
            signal_threshold=signal_threshold,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            holding_max=purge_horizon,
        )

        record = {
            "fold": f_idx,
            "train_samples": len(y_inner_tr),
            "val_samples": len(y_inner_va),
            "test_samples": len(df_test_slice),
            "train_timestamps": (str(t_tr_start), str(t_tr_end)),
            "test_timestamps": (str(t_te_start), str(t_te_end)),
            "directional_accuracy_reg_pct": float(round(reg_acc * 100, 2)),
            "directional_accuracy_clf_pct": float(round(clf_acc * 100, 2)),
            "regression_mse": float(round(mse_val, 4)),
            "brier_score": float(round(brier_val, 4)),
            "log_loss": float(round(log_loss_val, 4)),
            "trade_count": sim_res["trade_count"],
            "win_rate_pct": float(round(sim_res["win_rate"] * 100, 2)),
            "profit_factor": float(round(sim_res["profit_factor"], 2)),
            "net_pnl_pct": float(round(sim_res["total_net_pnl_pct"], 2)),
            "max_drawdown_pct": float(round(sim_res["max_drawdown_pct"], 2)),
            "trades": sim_res["trades"],
        }
        fold_results.append(record)
        all_trades.extend(sim_res["trades"])

        logger.info(
            "Fold %d Result: DirAcc=%.2f%% | MSE=%.4f | Brier=%.4f | Trades=%d | WinRate=%.2f%% | PF=%.2f | NetPnL=%+.2f%% | MaxDD=%.2f%%",
            f_idx + 1, record["directional_accuracy_reg_pct"], record["regression_mse"],
            record["brier_score"], record["trade_count"], record["win_rate_pct"],
            record["profit_factor"], record["net_pnl_pct"], record["max_drawdown_pct"],
        )

    # 7. Multi-Fold Result Aggregation
    # Pooled profit factor from aggregated trade ledgers
    if all_trades:
        pooled_df = pd.DataFrame(all_trades)
        wins = pooled_df[pooled_df["net_return"] > 0]
        losses = pooled_df[pooled_df["net_return"] < 0]
        gross_p = float(wins["net_return"].sum()) if len(wins) > 0 else 0.0
        gross_l = float(abs(losses["net_return"].sum())) if len(losses) > 0 else 0.0
        pooled_pf = float(gross_p / gross_l) if gross_l > 0 else (99.0 if gross_p > 0 else 0.0)
        pooled_win_rate = float(len(wins) / len(pooled_df)) * 100.0
    else:
        pooled_pf = 0.0
        pooled_win_rate = 0.0

    # Sequential Compounded Net Return across non-overlapping folds
    compounded_ret = 1.0
    for f in fold_results:
        compounded_ret *= (1.0 + f["net_pnl_pct"] / 100.0)
    compounded_net_return_pct = float(round((compounded_ret - 1.0) * 100.0, 2))

    trade_folds = [f for f in fold_results if f["trade_count"] > 0]
    mean_win_rate_pct = float(np.mean([f["win_rate_pct"] for f in trade_folds])) if trade_folds else 0.0
    worst_max_dd = float(max([f["max_drawdown_pct"] for f in fold_results])) if fold_results else 0.0

    aggregate = {
        "total_folds": len(fold_results),
        "mean_directional_accuracy_pct": float(np.mean([f["directional_accuracy_reg_pct"] for f in fold_results])),
        "std_directional_accuracy_pct": float(np.std([f["directional_accuracy_reg_pct"] for f in fold_results])),
        "mean_regression_mse": float(np.mean([f["regression_mse"] for f in fold_results])),
        "mean_brier_score": float(np.mean([f["brier_score"] for f in fold_results])),
        "mean_win_rate_pct": mean_win_rate_pct,
        "pooled_win_rate_pct": pooled_win_rate,
        "pooled_profit_factor": pooled_pf,
        "mean_profit_factor": float(np.mean([f["profit_factor"] for f in fold_results])),
        "compounded_net_return_pct": compounded_net_return_pct,
        "total_net_pnl_sum_pct": float(np.sum([f["net_pnl_pct"] for f in fold_results])),
        "worst_max_drawdown_pct": worst_max_dd,
        "total_trades_all_folds": len(all_trades),
    }

    return {
        "aggregate_summary": aggregate,
        "fold_by_fold_results": fold_results,
        "pooled_trades": all_trades,
    }
