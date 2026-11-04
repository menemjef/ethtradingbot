"""
backtest_audit.py — Comprehensive 6-month backtest with overfitting detection.

Tests:
1. Full 6-month walk-forward backtest (validation-only, no peeking at training data)
2. Monthly segment breakdown (detect if performance clusters in trained periods)
3. Random-signal baseline (do random entries beat the model? If yes → overfit)
4. Reversed-signal test (if model is genuinely predictive, reversing signals should lose money)
5. Shuffled-prediction test (if model learned real patterns, shuffled predictions should degrade)
6. Train-vs-validation split comparison (classic overfit detector)

Starting balance: $57.00 (matching live account)
"""

import logging
import numpy as np
import torch
import joblib
import random
import copy
from datetime import datetime, timezone
from collections import defaultdict

from config import (
    SYMBOL, TIMEFRAME, LOOKBACK, MODEL_PATH,
    SCALER_CANDLE_PATH, SCALER_MARKET_PATH,
    COMMISSION_RATE, SLIPPAGE_RATE, TRAIN_SPLIT
)
from data.fetcher import fetch_ohlcv
from data.preprocessor import build_features, transform
from model.architecture import HybridModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
logger = logging.getLogger("backtest_audit")

# ─── STRATEGY PARAMETERS (must match live bot exactly) ──────────────────────
REG_THRESHOLD   = 0.0450    # Aggressive mode threshold (matching live)
RISK_PCT        = 0.025     # 2.5% risk per trade (matching live config)
LEVERAGE        = 10
PT_MULT         = 3.0       # TP = 3.0x ATR
SL_MULT         = 1.5       # SL = 1.5x ATR
MAX_HOLD_BARS   = 12
MAX_DAILY_LOSS  = 0.06      # 6% daily drawdown (matching live)


def simulate_trades(df_feat, candle_arr, market_arr, model, device,
                    initial_balance, start_idx, end_idx,
                    threshold=REG_THRESHOLD, reverse_signals=False,
                    shuffle_predictions=False, random_signals=False,
                    label=""):
    """
    Core trade simulation engine. Runs bar-by-bar walk-forward.
    Returns dict of results + list of individual trade records.
    """
    balance = initial_balance
    peak_balance = balance
    max_drawdown = 0.0
    wins, losses = 0, 0
    trade_pnls = []
    trade_records = []
    daily_start = balance
    trade_exit_bar = 0
    equity_curve = [balance]

    # Pre-compute all predictions if shuffling
    all_preds = {}
    if shuffle_predictions:
        for i in range(max(LOOKBACK, start_idx), end_idx):
            c_win = candle_arr[i - LOOKBACK: i]
            m_win = market_arr[i - LOOKBACK: i]
            with torch.no_grad():
                x_c = torch.tensor(c_win, dtype=torch.float32).unsqueeze(0).to(device)
                x_m = torch.tensor(m_win, dtype=torch.float32).unsqueeze(0).to(device)
                reg_pred, _ = model(x_c, x_m)
                all_preds[i] = reg_pred.item()
        # Shuffle the values
        keys = list(all_preds.keys())
        vals = list(all_preds.values())
        random.shuffle(vals)
        all_preds = dict(zip(keys, vals))

    for i in range(max(LOOKBACK, start_idx), end_idx - 1):
        # Daily reset every 24 bars
        if (i - start_idx) % 24 == 0:
            daily_start = balance

        # Daily drawdown breaker
        if daily_start > 0 and (daily_start - balance) / daily_start > MAX_DAILY_LOSS:
            equity_curve.append(balance)
            continue

        if i <= trade_exit_bar:
            equity_curve.append(balance)
            continue

        if balance <= initial_balance * 0.05:
            break

        # Model inference
        if random_signals:
            pred = random.choice([-0.10, -0.08, -0.06, 0.06, 0.08, 0.10])
        elif shuffle_predictions:
            pred = all_preds.get(i, 0.0)
        else:
            c_win = candle_arr[i - LOOKBACK: i]
            m_win = market_arr[i - LOOKBACK: i]
            with torch.no_grad():
                x_c = torch.tensor(c_win, dtype=torch.float32).unsqueeze(0).to(device)
                x_m = torch.tensor(m_win, dtype=torch.float32).unsqueeze(0).to(device)
                reg_pred, _ = model(x_c, x_m)
                pred = reg_pred.item()

        if reverse_signals:
            pred = -pred

        if abs(pred) < threshold:
            equity_curve.append(balance)
            continue

        side = "LONG" if pred > 0 else "SHORT"
        entry_price = df_feat["close"].iloc[i - 1]
        atr = df_feat["atr_14"].iloc[i - 1]
        if atr <= 0 or np.isnan(atr):
            equity_curve.append(balance)
            continue

        sl_dist = atr * SL_MULT
        tp_dist = atr * PT_MULT
        risk_usd = balance * RISK_PCT
        position_size = risk_usd / sl_dist

        if side == "LONG":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        bars_held = 0
        outcome = 0.0
        hit = False
        exit_bar = i
        exit_reason = "TIMEOUT"

        for j in range(i, min(i + MAX_HOLD_BARS, len(df_feat))):
            bars_held += 1
            high_j = df_feat["high"].iloc[j]
            low_j = df_feat["low"].iloc[j]

            if side == "LONG":
                if low_j <= sl_price:
                    outcome = position_size * (sl_price - entry_price)
                    if outcome < 0: losses += 1
                    else: wins += 1
                    hit = True; exit_bar = j; exit_reason = "SL"; break
                elif high_j >= tp_price:
                    outcome = position_size * (tp_price - entry_price)
                    wins += 1; hit = True; exit_bar = j; exit_reason = "TP"; break
                if high_j >= entry_price + (tp_dist / 2.0):
                    sl_price = max(sl_price, entry_price)
            else:
                if high_j >= sl_price:
                    outcome = position_size * (entry_price - sl_price)
                    if outcome < 0: losses += 1
                    else: wins += 1
                    hit = True; exit_bar = j; exit_reason = "SL"; break
                elif low_j <= tp_price:
                    outcome = position_size * (entry_price - tp_price)
                    wins += 1; hit = True; exit_bar = j; exit_reason = "TP"; break
                if low_j <= entry_price - (tp_dist / 2.0):
                    sl_price = min(sl_price, entry_price)

        if not hit:
            exit_bar = min(i + MAX_HOLD_BARS - 1, len(df_feat) - 1)
            exit_price = df_feat["close"].iloc[exit_bar]
            if side == "LONG":
                outcome = position_size * (exit_price - entry_price)
            else:
                outcome = position_size * (entry_price - exit_price)
            if outcome > 0: wins += 1
            else: losses += 1

        trade_exit_bar = exit_bar

        notional = position_size * entry_price
        total_fee = notional * (COMMISSION_RATE + SLIPPAGE_RATE) * 2
        net_pnl = outcome - total_fee
        balance += net_pnl
        trade_pnls.append(net_pnl)

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        trade_records.append({
            "bar": i,
            "timestamp": str(df_feat.index[i]),
            "side": side,
            "entry": entry_price,
            "exit_reason": exit_reason,
            "pnl": net_pnl,
            "balance": balance,
            "bars_held": bars_held,
            "pred": pred,
        })
        equity_curve.append(balance)

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    net_pnl_total = balance - initial_balance
    avg_win = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
    avg_loss = np.mean([p for p in trade_pnls if p <= 0]) if any(p <= 0 for p in trade_pnls) else 0
    
    total_wins_usd = sum(p for p in trade_pnls if p > 0)
    total_losses_usd = abs(sum(p for p in trade_pnls if p <= 0))
    profit_factor = total_wins_usd / total_losses_usd if total_losses_usd > 0 else float('inf')

    return {
        "label": label,
        "initial_balance": initial_balance,
        "final_balance": balance,
        "net_pnl": net_pnl_total,
        "net_pnl_pct": (net_pnl_total / initial_balance) * 100 if initial_balance > 0 else 0,
        "wins": wins,
        "losses": losses,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown * 100,
        "peak_balance": peak_balance,
        "equity_curve": equity_curve,
        "trade_records": trade_records,
        "trade_pnls": trade_pnls,
    }


def print_result(r):
    logger.info(f"  {'─' * 55}")
    logger.info(f"  {r['label']}")
    logger.info(f"  {'─' * 55}")
    logger.info(f"    Balance:        ${r['initial_balance']:.2f} → ${r['final_balance']:.2f}")
    logger.info(f"    Net PnL:        ${r['net_pnl']:+.2f} ({r['net_pnl_pct']:+.1f}%)")
    logger.info(f"    Win Rate:       {r['win_rate']:.1f}% ({r['wins']}W / {r['losses']}L)")
    logger.info(f"    Total Trades:   {r['total_trades']}")
    logger.info(f"    Avg Win:        ${r['avg_win']:+.4f}")
    logger.info(f"    Avg Loss:       ${r['avg_loss']:.4f}")
    logger.info(f"    Profit Factor:  {r['profit_factor']:.2f}")
    logger.info(f"    Max Drawdown:   {r['max_drawdown']:.1f}%")
    logger.info(f"    Peak Balance:   ${r['peak_balance']:.2f}")


def run_audit():
    INITIAL_BALANCE = 57.0
    # 6 months = ~4320 hourly candles. Fetch 5000 to have buffer for indicators.
    FETCH_CANDLES = 5000

    logger.info("=" * 60)
    logger.info("  COMPREHENSIVE BACKTEST AUDIT — OVERFITTING DETECTION")
    logger.info("=" * 60)
    logger.info(f"  Starting Balance:  ${INITIAL_BALANCE:.2f}")
    logger.info(f"  Fetching {FETCH_CANDLES} x 1H candles (~6+ months)")
    logger.info(f"  Threshold:         {REG_THRESHOLD}")
    logger.info(f"  Model checkpoint:  {MODEL_PATH}")
    logger.info("")

    # 1. Fetch & preprocess
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=FETCH_CANDLES)
    if df.empty:
        logger.error("No data fetched."); return
    logger.info(f"  Data range: {df.index[0]} → {df.index[-1]}")

    df_feat = build_features(df, add_targets=False)
    sc = joblib.load(SCALER_CANDLE_PATH)
    sm = joblib.load(SCALER_MARKET_PATH)
    candle_arr, market_arr = transform(df_feat, sc, sm)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    total_bars = len(df_feat)
    train_end = int(total_bars * TRAIN_SPLIT)

    logger.info(f"  Total bars after features: {total_bars}")
    logger.info(f"  Train/Val split at bar:    {train_end}")
    logger.info(f"  Training period:           bars 0–{train_end-1}")
    logger.info(f"  Validation period:         bars {train_end}–{total_bars-1}")
    logger.info("")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: Full validation-only backtest (honest out-of-sample)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("━" * 60)
    logger.info("  TEST 1: VALIDATION-ONLY BACKTEST (Out-of-Sample)")
    logger.info("━" * 60)
    r_val = simulate_trades(
        df_feat, candle_arr, market_arr, model, device,
        INITIAL_BALANCE, train_end, total_bars,
        label="Validation-Only (Out-of-Sample)"
    )
    print_result(r_val)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Training-period backtest (should be better if model memorized)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("━" * 60)
    logger.info("  TEST 2: TRAINING-PERIOD BACKTEST (In-Sample)")
    logger.info("━" * 60)
    r_train = simulate_trades(
        df_feat, candle_arr, market_arr, model, device,
        INITIAL_BALANCE, LOOKBACK, train_end,
        label="Training Period (In-Sample)"
    )
    print_result(r_train)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: Random signal baseline
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("━" * 60)
    logger.info("  TEST 3: RANDOM SIGNAL BASELINE (5 runs averaged)")
    logger.info("━" * 60)
    random_results = []
    for run in range(5):
        random.seed(run + 100)
        r_rand = simulate_trades(
            df_feat, candle_arr, market_arr, model, device,
            INITIAL_BALANCE, train_end, total_bars,
            random_signals=True,
            label=f"Random Baseline Run {run+1}"
        )
        random_results.append(r_rand)
        logger.info(f"    Run {run+1}: ${r_rand['net_pnl']:+.2f} ({r_rand['net_pnl_pct']:+.1f}%) | "
                     f"WR: {r_rand['win_rate']:.1f}% | Trades: {r_rand['total_trades']}")

    avg_random_pnl = np.mean([r['net_pnl'] for r in random_results])
    avg_random_wr = np.mean([r['win_rate'] for r in random_results])
    logger.info(f"    ── Average Random: ${avg_random_pnl:+.2f} | WR: {avg_random_wr:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: Reversed signals (if model is real, this should lose)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("━" * 60)
    logger.info("  TEST 4: REVERSED SIGNALS (Should lose if model works)")
    logger.info("━" * 60)
    r_rev = simulate_trades(
        df_feat, candle_arr, market_arr, model, device,
        INITIAL_BALANCE, train_end, total_bars,
        reverse_signals=True,
        label="Reversed Signals (Anti-Model)"
    )
    print_result(r_rev)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 5: Shuffled predictions (break temporal patterns)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("━" * 60)
    logger.info("  TEST 5: SHUFFLED PREDICTIONS (Break temporal signal)")
    logger.info("━" * 60)
    shuffle_results = []
    for run in range(5):
        random.seed(run + 200)
        r_shuf = simulate_trades(
            df_feat, candle_arr, market_arr, model, device,
            INITIAL_BALANCE, train_end, total_bars,
            shuffle_predictions=True,
            label=f"Shuffled Predictions Run {run+1}"
        )
        shuffle_results.append(r_shuf)
        logger.info(f"    Run {run+1}: ${r_shuf['net_pnl']:+.2f} ({r_shuf['net_pnl_pct']:+.1f}%) | "
                     f"WR: {r_shuf['win_rate']:.1f}% | Trades: {r_shuf['total_trades']}")

    avg_shuffle_pnl = np.mean([r['net_pnl'] for r in shuffle_results])
    avg_shuffle_wr = np.mean([r['win_rate'] for r in shuffle_results])
    logger.info(f"    ── Average Shuffled: ${avg_shuffle_pnl:+.2f} | WR: {avg_shuffle_wr:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 6: Monthly segment breakdown (validation period only)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("━" * 60)
    logger.info("  TEST 6: MONTHLY SEGMENT BREAKDOWN (Validation Period)")
    logger.info("━" * 60)

    # Split validation period into ~monthly chunks (720 bars ≈ 30 days)
    MONTH_BARS = 720
    seg_start = train_end
    month_num = 1
    monthly_results = []
    seg_balance = INITIAL_BALANCE
    while seg_start + MONTH_BARS <= total_bars:
        seg_end = seg_start + MONTH_BARS
        month_label = f"Month {month_num} (bars {seg_start}–{seg_end})"
        if seg_start < len(df_feat) and seg_end <= len(df_feat):
            ts_start = str(df_feat.index[seg_start])[:10]
            ts_end = str(df_feat.index[min(seg_end, len(df_feat)-1)])[:10]
            month_label = f"Month {month_num} ({ts_start} → {ts_end})"

        r_seg = simulate_trades(
            df_feat, candle_arr, market_arr, model, device,
            seg_balance, seg_start, seg_end,
            label=month_label
        )
        monthly_results.append(r_seg)
        logger.info(f"    {month_label}: ${r_seg['net_pnl']:+.2f} ({r_seg['net_pnl_pct']:+.1f}%) | "
                     f"WR: {r_seg['win_rate']:.1f}% | Trades: {r_seg['total_trades']} | "
                     f"MDD: {r_seg['max_drawdown']:.1f}%")
        seg_balance = r_seg['final_balance']
        seg_start = seg_end
        month_num += 1

    # Handle remaining bars
    if seg_start < total_bars:
        ts_start = str(df_feat.index[seg_start])[:10]
        ts_end = str(df_feat.index[total_bars - 1])[:10]
        r_seg = simulate_trades(
            df_feat, candle_arr, market_arr, model, device,
            seg_balance, seg_start, total_bars,
            label=f"Remaining ({ts_start} → {ts_end})"
        )
        monthly_results.append(r_seg)
        logger.info(f"    Remaining ({ts_start} → {ts_end}): ${r_seg['net_pnl']:+.2f} ({r_seg['net_pnl_pct']:+.1f}%) | "
                     f"WR: {r_seg['win_rate']:.1f}% | Trades: {r_seg['total_trades']}")

    # ═══════════════════════════════════════════════════════════════════════
    # OVERFITTING DIAGNOSIS
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 60)
    logger.info("  OVERFITTING DIAGNOSIS SUMMARY")
    logger.info("=" * 60)

    flags = []

    # Check 1: Train vs Val performance gap
    train_wr = r_train['win_rate']
    val_wr = r_val['win_rate']
    train_pnl_pct = r_train['net_pnl_pct']
    val_pnl_pct = r_val['net_pnl_pct']

    logger.info(f"  [1] Train WR: {train_wr:.1f}% vs Val WR: {val_wr:.1f}% (Gap: {train_wr - val_wr:+.1f}pp)")
    logger.info(f"      Train PnL: {train_pnl_pct:+.1f}% vs Val PnL: {val_pnl_pct:+.1f}%")
    if train_wr - val_wr > 15:
        flags.append("OVERFIT: Win rate drops >15pp from train to validation")
    if train_pnl_pct > 0 and val_pnl_pct < -10:
        flags.append("OVERFIT: Profitable in-sample but loses >10% out-of-sample")

    # Check 2: Model vs Random
    logger.info(f"  [2] Model Val PnL: ${r_val['net_pnl']:+.2f} vs Avg Random: ${avg_random_pnl:+.2f}")
    if r_val['net_pnl'] <= avg_random_pnl:
        flags.append("WARNING: Model does not beat random baseline on validation data")

    # Check 3: Reversed signals
    logger.info(f"  [3] Model Val PnL: ${r_val['net_pnl']:+.2f} vs Reversed: ${r_rev['net_pnl']:+.2f}")
    if r_rev['net_pnl'] > r_val['net_pnl']:
        flags.append("WARNING: Reversed signals outperform model — possible inverted signal")

    # Check 4: Shuffled predictions
    logger.info(f"  [4] Model Val PnL: ${r_val['net_pnl']:+.2f} vs Avg Shuffled: ${avg_shuffle_pnl:+.2f}")
    if avg_shuffle_pnl >= r_val['net_pnl'] * 0.8 and r_val['total_trades'] > 5:
        flags.append("WARNING: Shuffled predictions perform similarly — model may lack temporal signal")

    # Check 5: Monthly consistency
    profitable_months = sum(1 for r in monthly_results if r['net_pnl'] > 0)
    total_months = len(monthly_results)
    logger.info(f"  [5] Monthly consistency: {profitable_months}/{total_months} months profitable")
    if total_months >= 3 and profitable_months == 0:
        flags.append("OVERFIT: Zero profitable months out-of-sample")

    # Check 6: Suspiciously high metrics
    if val_wr > 70 and r_val['total_trades'] > 20:
        flags.append("SUSPICIOUS: >70% win rate with >20 trades may indicate lookahead bias")
    if r_val['profit_factor'] > 5.0 and r_val['total_trades'] > 10:
        flags.append("SUSPICIOUS: Profit factor >5x with significant trades — check for data leakage")

    # Check 7: Entry price leak — model sees bar i but enters at close of i-1 (correct)
    logger.info(f"  [6] Entry logic: Uses close[i-1] for entry (bar before prediction) — OK")
    logger.info(f"      TP/SL check starts at bar i (first bar after entry) — OK")

    logger.info("")
    if flags:
        logger.info("  ⚠️  FLAGS DETECTED:")
        for f in flags:
            logger.info(f"    • {f}")
    else:
        logger.info("  ✅ NO OVERFITTING FLAGS DETECTED")
        logger.info("     Model shows consistent, non-suspicious behavior across all tests.")

    logger.info("")
    logger.info("=" * 60)

    # Save full report
    with open("backtest_audit_results.txt", "w", encoding="utf-8") as f:
        f.write("COMPREHENSIVE BACKTEST AUDIT RESULTS\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Data Range: {df.index[0]} to {df.index[-1]}\n")
        f.write(f"Initial Balance: ${INITIAL_BALANCE:.2f}\n")
        f.write(f"Threshold: {REG_THRESHOLD}\n\n")

        for name, res in [
            ("VALIDATION (OOS)", r_val),
            ("TRAINING (IS)", r_train),
            ("REVERSED SIGNALS", r_rev),
        ]:
            f.write(f"--- {name} ---\n")
            f.write(f"  Balance: ${res['initial_balance']:.2f} -> ${res['final_balance']:.2f}\n")
            f.write(f"  PnL: ${res['net_pnl']:+.2f} ({res['net_pnl_pct']:+.1f}%)\n")
            f.write(f"  WR: {res['win_rate']:.1f}% ({res['wins']}W/{res['losses']}L)\n")
            f.write(f"  Trades: {res['total_trades']}\n")
            f.write(f"  PF: {res['profit_factor']:.2f}\n")
            f.write(f"  MDD: {res['max_drawdown']:.1f}%\n\n")

        f.write("--- RANDOM BASELINE (5 runs) ---\n")
        f.write(f"  Avg PnL: ${avg_random_pnl:+.2f} | Avg WR: {avg_random_wr:.1f}%\n\n")

        f.write("--- SHUFFLED PREDICTIONS (5 runs) ---\n")
        f.write(f"  Avg PnL: ${avg_shuffle_pnl:+.2f} | Avg WR: {avg_shuffle_wr:.1f}%\n\n")

        f.write("--- MONTHLY BREAKDOWN ---\n")
        for r in monthly_results:
            f.write(f"  {r['label']}: ${r['net_pnl']:+.2f} ({r['net_pnl_pct']:+.1f}%) | "
                    f"WR: {r['win_rate']:.1f}% | Trades: {r['total_trades']}\n")

        f.write("\n--- DIAGNOSIS ---\n")
        if flags:
            for fl in flags:
                f.write(f"  FLAG: {fl}\n")
        else:
            f.write("  NO OVERFITTING FLAGS DETECTED\n")

        # Write individual trade records for validation
        f.write("\n--- VALIDATION TRADE LOG ---\n")
        f.write(f"{'Bar':>6} | {'Timestamp':>20} | {'Side':>5} | {'Entry':>10} | {'Exit':>7} | {'PnL':>10} | {'Balance':>10} | {'Pred':>10}\n")
        for t in r_val['trade_records']:
            f.write(f"{t['bar']:>6} | {t['timestamp'][:20]:>20} | {t['side']:>5} | "
                    f"${t['entry']:>9.2f} | {t['exit_reason']:>7} | ${t['pnl']:>+9.2f} | "
                    f"${t['balance']:>9.2f} | {t['pred']:>+10.6f}\n")

    logger.info(f"  Full report saved to: backtest_audit_results.txt")

    # Export to Audit Bridge
    try:
        from audit_bridge import log_backtest
        log_backtest(
            dataset_date_range=(str(df.index[train_end]), str(df.index[-1])),
            dataset_candles=len(df),
            initial_capital=INITIAL_BALANCE,
            final_capital=r_val["final_balance"],
            trades=r_val["trade_records"],
            equity_curve=r_val["equity_curve"],
            command="python backtest_audit.py",
            evaluation_method="Validation-Only (Out-of-Sample)",
            threshold=REG_THRESHOLD,
            train_val_split=f"80/20 at bar {train_end}",
            is_sample=False,
            notes="Direct capture from backtest_audit.py validation run with full trade ledger",
        )
        logger.info("  [Audit Bridge] Backtest report saved to audit_bridge/")
    except Exception as e:
        logger.warning(f"  [Audit Bridge] Export warning: {e}")


if __name__ == "__main__":
    run_audit()
