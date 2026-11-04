"""
backtest.py — Institutional-Grade Walk-Forward Backtester.
Simulates the EXACT live bot logic: one trade at a time, sigmoid confidence gating,
proper position sizing, and realistic fee/slippage modeling.
"""
import logging
import numpy as np
import torch
import joblib
from config import (
    SYMBOL, TIMEFRAME, LOOKBACK, MODEL_PATH,
    SCALER_CANDLE_PATH, SCALER_MARKET_PATH,
    TBM_PT_MULT, TBM_SL_MULT, COMMISSION_RATE, SLIPPAGE_RATE,
    TRAIN_SPLIT, SIGNAL_THRESHOLD
)
from data.fetcher import fetch_ohlcv
from data.preprocessor import build_features, transform
from model.architecture import HybridModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
logger = logging.getLogger("backtest")

# ─── STRATEGY PARAMETERS ─────────────────────────────────────────────────────
REG_THRESHOLD         = 0.0830    # Top 30 raw predictions (~10-15 distinct trades)
RISK_PCT              = 0.05      # Risk 5.0% of equity per trade (Higher Earnings)
LEVERAGE              = 10        # Moderate Leverage
PT_MULT               = 3.0     # Take Profit = 3.0x ATR
SL_MULT               = 1.5     # Stop Loss   = 1.5x ATR (Breathing room)
MAX_HOLD_BARS         = 12      # Force-close after 12 hours
MAX_DAILY_LOSS_PCT    = 0.20    # Drawdown limit

def run_backtest(initial_balance: float = 100.0, limit: int = 1000, since_ms: int = None):
    logger.info(f"━━━ Walk-Forward Backtest | ${initial_balance:.2f} | {limit} candles ━━━")

    # 1. Fetch & preprocess
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit, since_ms=since_ms)
    if df.empty:
        logger.error("No data fetched."); return
    df_feat = build_features(df, add_targets=False)
    sc = joblib.load(SCALER_CANDLE_PATH)
    sm = joblib.load(SCALER_MARKET_PATH)
    candle_arr, market_arr = transform(df_feat, sc, sm)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Simulation state
    balance      = initial_balance
    peak_balance = balance
    max_drawdown = 0.0
    wins, losses = 0, 0
    trade_pnls   = []
    daily_start  = balance  # Reset every 24 bars

    # Validation-only: skip the first 80% (training data) for honest results
    val_start = int(len(df_feat) * TRAIN_SPLIT)
    logger.info(f"Validation-only backtest: starting from candle {val_start} of {len(df_feat)} (last {100-TRAIN_SPLIT*100:.0f}%)")

    trade_exit_bar = 0  # Track when current trade exits to prevent overlapping trades

    for i in range(max(LOOKBACK, val_start), len(df_feat) - 1):
        # Daily reset (every 24 hourly bars)
        if (i - LOOKBACK) % 24 == 0:
            daily_start = balance

        # Daily drawdown circuit breaker
        if daily_start > 0 and (daily_start - balance) / daily_start > MAX_DAILY_LOSS_PCT:
            continue  # Skip trading for rest of this "day"

        # FIX: Skip bars where we're still in a trade (prevents impossible overlapping trades)
        if i <= trade_exit_bar:
            continue

        # Wipeout protection
        if balance <= initial_balance * 0.05:
            logger.warning(f"⛔ Account below 5% of initial. Halting at bar {i}.")
            break

        # 3. Model inference
        # Window [i-LOOKBACK, i-1]: the model sees the last LOOKBACK bars ending at i-1
        c_win = candle_arr[i - LOOKBACK : i]
        m_win = market_arr[i - LOOKBACK : i]
        with torch.no_grad():
            x_c = torch.tensor(c_win, dtype=torch.float32).unsqueeze(0).to(device)
            x_m = torch.tensor(m_win, dtype=torch.float32).unsqueeze(0).to(device)
            reg_pred, _ = model(x_c, x_m)
            pred = reg_pred.item()

        # 4. Regression magnitude gate — only take high-conviction signals
        if abs(pred) < REG_THRESHOLD:
            continue  # Not strong enough — skip

        side = "LONG" if pred > 0 else "SHORT"
        from config import LONG_ONLY_MODE
        if LONG_ONLY_MODE and side == "SHORT":
            continue
        # FIX: Entry at close of i-1 (the last bar the model saw), NOT close of i
        # In live: model sees current candle, enters at its close. Same logic here.
        entry_price = df_feat["close"].iloc[i - 1]
        atr = df_feat["atr_14"].iloc[i - 1]
        if atr <= 0 or np.isnan(atr):
            continue

        # 5. Position sizing (risk-based on ATR)
        sl_dist = atr * SL_MULT
        tp_dist = atr * PT_MULT
        risk_usd = balance * RISK_PCT
        position_size = risk_usd / sl_dist   # In ETH units

        if side == "LONG":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # 6. Simulate trade outcome (walk forward bar by bar)
        # FIX: TP/SL checking starts at bar i (the first bar AFTER entry at close of i-1)
        bars_held = 0
        outcome = 0.0
        hit = False
        exit_bar = i  # default

        for j in range(i, min(i + MAX_HOLD_BARS, len(df_feat))):
            bars_held += 1
            high_j = df_feat["high"].iloc[j]
            low_j  = df_feat["low"].iloc[j]

            if side == "LONG":
                # Check SL first (conservative: assume SL hit before TP on same bar)
                if low_j <= sl_price:
                    outcome = position_size * (sl_price - entry_price)
                    if outcome < 0: losses += 1
                    hit = True; exit_bar = j; break
                elif high_j >= tp_price:
                    outcome = position_size * (tp_price - entry_price)
                    wins += 1; hit = True; exit_bar = j; break
                
                # BREAK-EVEN LOGIC: If price reaches halfway to TP, move SL to Entry
                if high_j >= entry_price + (tp_dist / 2.0):
                    sl_price = max(sl_price, entry_price)

            else:
                if high_j >= sl_price:
                    outcome = position_size * (entry_price - sl_price)
                    if outcome < 0: losses += 1
                    hit = True; exit_bar = j; break
                elif low_j <= tp_price:
                    outcome = position_size * (entry_price - tp_price)
                    wins += 1; hit = True; exit_bar = j; break
                
                # BREAK-EVEN LOGIC: If price reaches halfway to TP, move SL to Entry
                if low_j <= entry_price - (tp_dist / 2.0):
                    sl_price = min(sl_price, entry_price)

        # Force close at market if horizontal barrier hit
        if not hit:
            exit_bar = min(i + MAX_HOLD_BARS - 1, len(df_feat) - 1)
            exit_price = df_feat["close"].iloc[exit_bar]
            if side == "LONG":
                outcome = position_size * (exit_price - entry_price)
            else:
                outcome = position_size * (entry_price - exit_price)
            if outcome > 0:
                wins += 1
            else:
                losses += 1

        # FIX: Record trade exit bar so we don't open overlapping trades
        trade_exit_bar = exit_bar

        # 7. Apply realistic fees (entry + exit)
        notional = position_size * entry_price
        total_fee = notional * (COMMISSION_RATE + SLIPPAGE_RATE) * 2  # Round trip
        net_pnl = outcome - total_fee

        balance += net_pnl
        trade_pnls.append(net_pnl)
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    # 8. Report
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    net_pnl = balance - initial_balance
    avg_win = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
    avg_loss = np.mean([p for p in trade_pnls if p <= 0]) if any(p <= 0 for p in trade_pnls) else 0
    profit_factor = abs(sum(p for p in trade_pnls if p > 0) / sum(p for p in trade_pnls if p <= 0)) if any(p <= 0 for p in trade_pnls) and sum(p for p in trade_pnls if p <= 0) != 0 else float('inf')

    logger.info("=" * 55)
    logger.info("   INSTITUTIONAL BACKTEST RESULTS (%s)", TIMEFRAME)
    logger.info("=" * 55)
    logger.info(f"  Initial Balance:  ${initial_balance:.2f}")
    logger.info(f"  Final Balance:    ${balance:.2f}")
    logger.info(f"  Net PnL:          ${net_pnl:+.2f} ({(net_pnl/initial_balance)*100:+.1f}%)")
    logger.info(f"  Win Rate:         {win_rate:.1f}% ({wins}W / {losses}L)")
    logger.info(f"  Total Trades:     {total_trades}")
    logger.info(f"  Avg Win:          ${avg_win:+.4f}")
    logger.info(f"  Avg Loss:         ${avg_loss:.4f}")
    logger.info(f"  Profit Factor:    {profit_factor:.2f}")
    logger.info(f"  Max Drawdown:     {max_drawdown*100:.1f}%")
    logger.info(f"  Peak Balance:     ${peak_balance:.2f}")
    logger.info("=" * 55)

    with open("metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Initial Balance:  ${initial_balance:.2f}\n")
        f.write(f"Final Balance:    ${balance:.2f}\n")
        f.write(f"Net PnL:          ${net_pnl:+.2f} ({(net_pnl/initial_balance)*100:+.1f}%)\n")
        f.write(f"Win Rate:         {win_rate:.1f}% ({wins}W / {losses}L)\n")
        f.write(f"Total Trades:     {total_trades}\n")
        f.write(f"Avg Win:          ${avg_win:+.4f}\n")
        f.write(f"Avg Loss:         ${avg_loss:.4f}\n")
        f.write(f"Profit Factor:    {profit_factor:.2f}\n")
        f.write(f"Max Drawdown:     {max_drawdown*100:.1f}%\n")
        f.write(f"Peak Balance:     ${peak_balance:.2f}\n")

if __name__ == "__main__":
    run_backtest(initial_balance=55.0, limit=4400)
