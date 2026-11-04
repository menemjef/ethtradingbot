"""
model/walk_forward.py — Walk-Forward Cross-Validation Engine with OHLC-Aware Simulation.

Pass 5 Corrections:
1. Signal alignment: First test prediction evaluates at candle k-1 close, executes at candle k OPEN.
   Final test candle cannot generate trades outside fold. Every trade records decision_time and entry_time.
2. Scaler isolation: Scalers fit strictly on inner training partition, zero leakage from inner val.
3. Realistic OHLC execution: Stop-loss gaps filled at market open (worse price), not trigger price.
4. Intratrade adverse excursion: Tracks hourly mark-to-market equity for true maximum drawdown.
5. Itemized accounting: Separate gross price return, entry slippage, exit slippage, entry fee, exit fee, and funding.
6. Pooled profit factor and compounded net return across folds.
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
    decision_timestamps: Optional[List[Any]] = None,
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
    - Signal at index i has an explicit decision timestamp (candle close of bar i - 1 or preceding candle).
    - Entry order executes at candle i OPEN with modeled slippage.
    - If candle i opens beyond the stop level (stop-loss gap), fills at candle i OPEN (worse price).
    - Intrabar TP/SL evaluated against subsequent candle High and Low.
    - If both TP and SL are touched in the same candle: resolves conservatively STOP-FIRST.
    - Time-horizon exit at candle close of bar (entry_bar + holding_max).
    - Forced end-of-fold close: any remaining open position at the final test bar
      is marked to market and closed at final candle CLOSE.
    - Intratrade adverse excursion tracked on hourly bars for true maximum drawdown.
    - Itemized tracking of gross price PnL, slippage, entry/exit fees, funding, and net PnL.
    """
    N = len(df_test)
    assert len(signals) == N, f"signals length ({len(signals)}) must match df_test length ({N})"

    opens = df_test["open"].values
    highs = df_test["high"].values
    lows = df_test["low"].values
    closes = df_test["close"].values
    atrs = df_test["atr_14"].values if "atr_14" in df_test else np.full(N, 10.0)
    timestamps = df_test.index

    trades = []
    in_pos = False
    side = 0
    entry_price = 0.0
    raw_entry_price = 0.0
    entry_bar = 0
    entry_time = None
    dec_time = None
    pt_price = 0.0
    sl_price = 0.0
    entry_slip_cost = 0.0

    trade_id = 0
    # Hourly equity series for intratrade drawdown tracking
    hourly_equity = [1.0]
    current_equity = 1.0

    for i in range(N):
        curr_open = opens[i]
        curr_high = highs[i]
        curr_low = lows[i]
        curr_close = closes[i]
        curr_time = timestamps[i]

        if decision_timestamps is not None:
            # Pre-aligned signals: signals[i] was decided at decision_timestamps[i] (candle i-1 close)
            # and executes at candle i OPEN (for i < N - 1 so final test bar starts no new trade).
            can_enter_this_bar = (not in_pos) and (i < N - 1)
            sig = signals[i]
            curr_dec_time = decision_timestamps[i]
        else:
            # Standalone df_test signals: signals[k] is decided at candle k CLOSE (timestamps[k])
            # and executes at candle k+1 OPEN (timestamps[k+1]).
            # Thus at bar i >= 1, we execute signals[i - 1] decided at timestamps[i - 1].
            # signals[N - 1] (at the final test candle close) cannot open a trade outside the fold.
            can_enter_this_bar = (not in_pos) and (i >= 1)
            sig = signals[i - 1] if i >= 1 else 0.0
            curr_dec_time = timestamps[i - 1] if i >= 1 else curr_time

        # 1. Check entry at candle i OPEN if not in position
        if can_enter_this_bar:
            if sig >= signal_threshold:
                side = 1
                raw_entry_price = curr_open
                entry_price = curr_open * (1.0 + slippage)
                entry_slip_cost = slippage
                entry_bar = i
                entry_time = curr_time
                dec_time = curr_dec_time
                current_atr = atrs[i]
                pt_price = entry_price + (pt_mult * current_atr)
                sl_price = entry_price - (sl_mult * current_atr)
                in_pos = True
            elif sig <= -signal_threshold:
                side = -1
                raw_entry_price = curr_open
                entry_price = curr_open * (1.0 - slippage)
                entry_slip_cost = slippage
                entry_bar = i
                entry_time = curr_time
                dec_time = curr_dec_time
                current_atr = atrs[i]
                pt_price = entry_price - (pt_mult * current_atr)
                sl_price = entry_price + (sl_mult * current_atr)
                in_pos = True

        # 2. Check position exits during candle i
        if in_pos:
            holding_bars = i - entry_bar
            exit_trade = False
            exit_price = 0.0
            raw_exit_price = 0.0
            exit_slip_cost = 0.0
            exit_reason = "horizon"
            exit_fee = taker_fee
            is_collision = False

            if side == 1:  # LONG
                # Stop-loss opening gap: market opened below SL level
                if curr_open <= sl_price:
                    exit_trade = True
                    raw_exit_price = curr_open
                    exit_price = curr_open * (1.0 - slippage)  # Fills at opening gap price
                    exit_slip_cost = slippage
                    exit_reason = "stop_loss_gap"
                    exit_fee = taker_fee
                # Take-profit opening gap: market opened above PT level
                elif curr_open >= pt_price:
                    exit_trade = True
                    raw_exit_price = curr_open
                    exit_price = curr_open * (1.0 - slippage)
                    exit_slip_cost = slippage
                    exit_reason = "take_profit_gap"
                    exit_fee = maker_fee
                else:
                    hit_sl = bool(curr_low <= sl_price)
                    hit_tp = bool(curr_high >= pt_price)

                    if hit_sl and hit_tp:
                        # Intrabar collision: Conservative STOP-FIRST resolution
                        exit_trade = True
                        raw_exit_price = sl_price
                        exit_price = sl_price * (1.0 - slippage)
                        exit_slip_cost = slippage
                        exit_reason = "stop_loss_collision"
                        exit_fee = taker_fee
                        is_collision = True
                    elif hit_sl:
                        exit_trade = True
                        raw_exit_price = sl_price
                        exit_price = sl_price * (1.0 - slippage)
                        exit_slip_cost = slippage
                        exit_reason = "stop_loss"
                        exit_fee = taker_fee
                    elif hit_tp:
                        exit_trade = True
                        raw_exit_price = pt_price
                        exit_price = pt_price * (1.0 - slippage)
                        exit_slip_cost = slippage
                        exit_reason = "take_profit"
                        exit_fee = maker_fee
                    elif holding_bars >= holding_max:
                        exit_trade = True
                        raw_exit_price = curr_close
                        exit_price = curr_close * (1.0 - slippage)
                        exit_slip_cost = slippage
                        exit_reason = "horizon"
                        exit_fee = taker_fee

            elif side == -1:  # SHORT
                # Stop-loss opening gap: market opened above SL level
                if curr_open >= sl_price:
                    exit_trade = True
                    raw_exit_price = curr_open
                    exit_price = curr_open * (1.0 + slippage)
                    exit_slip_cost = slippage
                    exit_reason = "stop_loss_gap"
                    exit_fee = taker_fee
                # Take-profit opening gap: market opened below PT level
                elif curr_open <= pt_price:
                    exit_trade = True
                    raw_exit_price = curr_open
                    exit_price = curr_open * (1.0 + slippage)
                    exit_slip_cost = slippage
                    exit_reason = "take_profit_gap"
                    exit_fee = maker_fee
                else:
                    hit_sl = bool(curr_high >= sl_price)
                    hit_tp = bool(curr_low <= pt_price)

                    if hit_sl and hit_tp:
                        # Intrabar collision: Conservative STOP-FIRST resolution
                        exit_trade = True
                        raw_exit_price = sl_price
                        exit_price = sl_price * (1.0 + slippage)
                        exit_slip_cost = slippage
                        exit_reason = "stop_loss_collision"
                        exit_fee = taker_fee
                        is_collision = True
                    elif hit_sl:
                        exit_trade = True
                        raw_exit_price = sl_price
                        exit_price = sl_price * (1.0 + slippage)
                        exit_slip_cost = slippage
                        exit_reason = "stop_loss"
                        exit_fee = taker_fee
                    elif hit_tp:
                        exit_trade = True
                        raw_exit_price = pt_price
                        exit_price = pt_price * (1.0 + slippage)
                        exit_slip_cost = slippage
                        exit_reason = "take_profit"
                        exit_fee = maker_fee
                    elif holding_bars >= holding_max:
                        exit_trade = True
                        raw_exit_price = curr_close
                        exit_price = curr_close * (1.0 + slippage)
                        exit_slip_cost = slippage
                        exit_reason = "horizon"
                        exit_fee = taker_fee

            # Forced end-of-fold close at final test bar
            if not exit_trade and i == N - 1:
                exit_trade = True
                raw_exit_price = curr_close
                exit_price = curr_close * (1.0 - slippage) if side == 1 else curr_close * (1.0 + slippage)
                exit_slip_cost = slippage
                exit_reason = "end_of_fold"
                exit_fee = taker_fee

            if exit_trade:
                trade_id += 1
                # Itemized PnL decomposition:
                # 1. Gross price return
                gross_price_ret = side * ((raw_exit_price - raw_entry_price) / raw_entry_price)
                # 2. Total slippage cost
                tot_slip = entry_slip_cost + exit_slip_cost
                # 3. Fees
                entry_f = taker_fee
                exit_f = exit_fee
                tot_fees = entry_f + exit_f
                # 4. Funding cost (explicitly zero / excluded)
                funding_f = 0.0

                # 5. Net return
                net_ret = float(side * ((exit_price - entry_price) / entry_price) - tot_fees)

                trades.append({
                    "trade_id": trade_id,
                    "decision_time": str(dec_time),
                    "entry_bar": entry_bar,
                    "entry_time": str(entry_time),
                    "exit_bar": i,
                    "exit_time": str(curr_time),
                    "side": "LONG" if side == 1 else "SHORT",
                    "raw_entry_price": float(raw_entry_price),
                    "entry_price": float(entry_price),
                    "raw_exit_price": float(raw_exit_price),
                    "exit_price": float(exit_price),
                    "pt_price": float(pt_price),
                    "sl_price": float(sl_price),
                    "exit_reason": exit_reason,
                    "is_collision": is_collision,
                    "holding_bars": holding_bars,
                    "gross_price_return": float(gross_price_ret),
                    "entry_slippage_cost": float(entry_slip_cost),
                    "exit_slippage_cost": float(exit_slip_cost),
                    "total_slippage_cost": float(tot_slip),
                    "entry_fee_cost": float(entry_f),
                    "exit_fee_cost": float(exit_f),
                    "funding_cost": float(funding_f),
                    "funding_disclosure": "Excluded (historical funding rate data not attached)",
                    "net_return": float(net_ret),
                })

                # Update cumulative equity
                current_equity *= (1.0 + net_ret)
                hourly_equity.append(current_equity)
                in_pos = False
            else:
                # Intratrade mark-to-market adverse excursion for drawdown tracking
                if side == 1:
                    worst_bar_ret = (curr_low - entry_price) / entry_price - taker_fee
                else:
                    worst_bar_ret = (entry_price - curr_high) / entry_price - taker_fee
                hourly_equity.append(current_equity * (1.0 + worst_bar_ret))
        else:
            hourly_equity.append(current_equity)

    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "compounded_net_return_pct": 0.0,
            "sum_net_pnl_pct": 0.0,
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
    sum_net_pnl = float(df_tr["net_return"].sum()) * 100.0

    # Compounded net return across equity curve
    compounded_ret_pct = (current_equity - 1.0) * 100.0

    # Intratrade maximum drawdown calculation
    eq_series = pd.Series(hourly_equity)
    running_peak = eq_series.cummax()
    drawdowns = (running_peak - eq_series) / running_peak
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    return {
        "trade_count": len(df_tr),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "compounded_net_return_pct": compounded_ret_pct,
        "sum_net_pnl_pct": sum_net_pnl,
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
    3. Scalers fitted strictly on INNER training partition (zero validation leakage).
    4. Inner validation partition with early stopping.
    5. Clean test signal alignment: prediction at candle k-1 close executes at candle k OPEN.
    6. Realistic OHLC-aware simulation with stop-loss gaps and intratrade drawdown tracking.
    7. Pooled profit factor and sequential compounded return aggregation.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Rigorous Walk-Forward Evaluation on device: %s (%d folds)", device, n_splits)

    # 1. Feature Engineering on unbroken timeline
    df_feat = build_features(df_raw, add_targets=False)
    df_feat = add_tbm_target(
        df_feat,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
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

        # 1. Outer Train and Test Boundaries:
        # All training labels must resolve before the first out-of-sample prediction timestamp
        tr_end_idx = max(LOOKBACK + 50, te_start_idx - purge_horizon)
        tr_start_idx = 0

        t_tr_start = df_feat.index[tr_start_idx]
        t_tr_end = df_feat.index[tr_end_idx - 1]
        t_tr_res_max = t_tr_end + pd.Timedelta(hours=purge_horizon)
        t_te_start = df_feat.index[te_start_idx]
        t_te_end = df_feat.index[te_end_idx - 1]

        assert t_tr_res_max <= t_te_start, (
            f"Temporal leakage in fold {f_idx}: train label resolves at {t_tr_res_max} > test start {t_te_start}"
        )

        # 2. Define Inner Training and Inner Validation partitions within outer train history
        outer_train_len = tr_end_idx - tr_start_idx
        inner_val_size = int(outer_train_len * 0.25)
        inner_tr_end_idx = max(LOOKBACK + 20, tr_end_idx - inner_val_size - purge_horizon)
        inner_va_start_idx = tr_end_idx - inner_val_size

        df_inner_train = df_feat.iloc[tr_start_idx:inner_tr_end_idx]
        df_inner_val = df_feat.iloc[inner_va_start_idx:tr_end_idx]

        t_in_tr_end = df_inner_train.index[-1]
        t_in_va_start = df_inner_val.index[0]
        assert t_in_tr_end + pd.Timedelta(hours=purge_horizon) <= t_in_va_start, (
            f"Inner purge leakage in fold {f_idx}: inner train label resolves after inner val start"
        )

        logger.info(
            "--- Fold %d/%d Partition Hierarchy ---", f_idx + 1, n_splits
        )
        logger.info("  Inner Train (Fit Scalers Here): [%s -> %s] (%d bars)", df_inner_train.index[0], t_in_tr_end, len(df_inner_train))
        logger.info("  Inner Val (Model Selection):   [%s -> %s] (%d bars)", t_in_va_start, df_inner_val.index[-1], len(df_inner_val))
        logger.info("  Outer Test (Frozen Out-of-Sample): [%s -> %s] (%d bars)", t_te_start, t_te_end, te_end_idx - te_start_idx)

        # 3. Fit Scalers EXCLUSIVELY on Inner Training History (Zero Validation Leakage)
        scaler_c = StandardScaler()
        scaler_m = StandardScaler()
        scaler_c.fit(df_inner_train[CANDLE_FEATURES].values)
        scaler_m.fit(df_inner_train[MARKET_FEATURES].values)

        # 4. Transform inner train and inner val using those frozen scalers
        c_in_tr, m_in_tr = transform(df_inner_train, scaler_c, scaler_m)
        y_in_tr = df_inner_train["target"].values.astype(np.float32)

        Xc_in_tr, Xm_in_tr, y_in_tr_win = build_windows(
            c_in_tr, m_in_tr, y_in_tr,
            lookback=LOOKBACK,
            purge_horizon=purge_horizon,
            drop_nan_targets=True,
            return_metadata=False,
        )

        c_in_va, m_in_va = transform(df_inner_val, scaler_c, scaler_m)
        y_in_va = df_inner_val["target"].values.astype(np.float32)

        Xc_in_va, Xm_in_va, y_in_va_win = build_windows(
            c_in_va, m_in_va, y_in_va,
            lookback=LOOKBACK,
            purge_horizon=purge_horizon,
            drop_nan_targets=True,
            return_metadata=False,
        )

        inner_train_ds = ETHDataset(Xc_in_tr, Xm_in_tr, y_in_tr_win)
        inner_val_ds = ETHDataset(Xc_in_va, Xm_in_va, y_in_va_win)

        train_loader = DataLoader(inner_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(inner_val_ds, batch_size=batch_size, shuffle=False)

        # 5. Train Model on Inner Train with Early Stopping on Inner Val
        model = HybridModel().to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        mse_loss = nn.MSELoss()

        n_pos = int((y_in_tr_win > 0).sum())
        n_neg = int((y_in_tr_win <= 0).sum())
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
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # 6. Test Window Signal Alignment:
        # The first test prediction is made at candle (te_start_idx - 1) CLOSE,
        # using the preceding LOOKBACK historical candles ending at (te_start_idx - 1).
        # Its execution occurs at candle te_start_idx OPEN.
        # We need lookback context ending at candle te_end_idx - 2 (executing at te_end_idx - 1 OPEN).
        # Slicing from (te_start_idx - LOOKBACK) to te_end_idx:
        df_test_window = df_feat.iloc[te_start_idx:te_end_idx].copy()
        N_test = len(df_test_window)

        # Build feature windows for the test predictions:
        # For each candle j in [0 .. N_test - 1], its entry occurs at df_test_window.iloc[j] OPEN.
        # The signal for that entry was generated at candle (te_start_idx + j - 1) CLOSE,
        # using historical lookback bars ending at (te_start_idx + j - 1).
        test_signals = np.zeros(N_test, dtype=np.float32)
        dec_times = []

        model.eval()
        with torch.no_grad():
            for j in range(N_test):
                dec_idx = te_start_idx + j - 1
                dec_ts = df_feat.index[dec_idx]
                dec_times.append(dec_ts)

                # If j == N_test - 1: this is the final candle of the fold. It cannot generate an entry.
                if j == N_test - 1:
                    test_signals[j] = 0.0
                    continue

                # Lookback slice [dec_idx - LOOKBACK + 1 : dec_idx + 1] (length LOOKBACK)
                ctx_df = df_feat.iloc[dec_idx - LOOKBACK + 1 : dec_idx + 1]
                c_w, m_w = transform(ctx_df, scaler_c, scaler_m)
                xc_t = torch.from_numpy(c_w).unsqueeze(0).to(device)
                xm_t = torch.from_numpy(m_w).unsqueeze(0).to(device)

                reg_p, _ = model(xc_t, xm_t)
                pred_signal = float(reg_p.squeeze().cpu().item()) / TARGET_MULTIPLIER
                test_signals[j] = pred_signal

        # Directional accuracy on valid targets in test window
        y_test_raw = df_test_window["target"].values.astype(np.float32)
        valid_m = ~np.isnan(y_test_raw)
        if valid_m.sum() > 0:
            reg_acc = directional_accuracy(test_signals[valid_m], y_test_raw[valid_m])
            mse_val = float(np.mean((test_signals[valid_m] - y_test_raw[valid_m]) ** 2))
        else:
            reg_acc, mse_val = 0.5, 1.0

        # 7. OHLC-Aware Trade Simulation with Next-Bar Execution and Gap Accounting
        sim_res = simulate_trades_ohlc(
            signals=test_signals,
            df_test=df_test_window,
            decision_timestamps=dec_times,
            signal_threshold=signal_threshold,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            holding_max=purge_horizon,
        )

        record = {
            "fold": f_idx,
            "train_samples": len(y_in_tr_win),
            "val_samples": len(y_in_va_win),
            "test_samples": N_test,
            "train_timestamps": (str(df_inner_train.index[0]), str(t_in_tr_end)),
            "test_timestamps": (str(t_te_start), str(t_te_end)),
            "test_start_ts": str(t_te_start),
            "test_end_ts": str(t_te_end),
            "first_decision_ts": str(dec_times[0]) if len(dec_times) > 0 else None,
            "first_execution_open_ts": str(df_test_window.index[0]) if len(df_test_window) > 0 else None,
            "directional_accuracy_reg_pct": float(round(reg_acc * 100, 2)),
            "regression_mse": float(round(mse_val, 4)),
            "trade_count": sim_res["trade_count"],
            "win_rate_pct": float(round(sim_res["win_rate"] * 100, 2)),
            "profit_factor": float(round(sim_res["profit_factor"], 2)),
            "compounded_net_return_pct": float(round(sim_res["compounded_net_return_pct"], 2)),
            "sum_net_pnl_pct": float(round(sim_res["sum_net_pnl_pct"], 2)),
            "max_drawdown_pct": float(round(sim_res["max_drawdown_pct"], 2)),
            "trades": sim_res["trades"],
        }
        fold_results.append(record)
        all_trades.extend(sim_res["trades"])

        logger.info(
            "Fold %d Result: DirAcc=%.2f%% | MSE=%.4f | Trades=%d | WinRate=%.2f%% | PF=%.2f | NetReturn=%+.2f%% | MaxDD=%.2f%%",
            f_idx + 1, record["directional_accuracy_reg_pct"], record["regression_mse"],
            record["trade_count"], record["win_rate_pct"],
            record["profit_factor"], record["compounded_net_return_pct"], record["max_drawdown_pct"],
        )

    # 8. Pooled Profit Factor and Sequential Compounded Return
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
        compounded_ret *= (1.0 + f["compounded_net_return_pct"] / 100.0)
    compounded_net_return_pct = float(round((compounded_ret - 1.0) * 100.0, 2))

    trade_folds = [f for f in fold_results if f["trade_count"] > 0]
    mean_win_rate_pct = float(np.mean([f["win_rate_pct"] for f in trade_folds])) if trade_folds else 0.0
    worst_max_dd = float(max([f["max_drawdown_pct"] for f in fold_results])) if fold_results else 0.0

    aggregate = {
        "total_folds": len(fold_results),
        "mean_directional_accuracy_pct": float(np.mean([f["directional_accuracy_reg_pct"] for f in fold_results])),
        "std_directional_accuracy_pct": float(np.std([f["directional_accuracy_reg_pct"] for f in fold_results])),
        "mean_regression_mse": float(np.mean([f["regression_mse"] for f in fold_results])),
        "mean_win_rate_pct": mean_win_rate_pct,
        "pooled_win_rate_pct": pooled_win_rate,
        "pooled_profit_factor": pooled_pf,
        "mean_profit_factor": float(np.mean([f["profit_factor"] for f in fold_results])),
        "compounded_net_return_pct": compounded_net_return_pct,
        "total_net_pnl_sum_pct": float(np.sum([f["sum_net_pnl_pct"] for f in fold_results])),
        "worst_max_drawdown_pct": worst_max_dd,
        "total_trades_all_folds": len(all_trades),
    }

    return {
        "aggregate_summary": aggregate,
        "fold_by_fold_results": fold_results,
        "pooled_trades": all_trades,
    }
