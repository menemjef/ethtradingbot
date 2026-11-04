"""
live/trader.py — High-level trading logic and risk management.

WHAT THIS FILE DOES:
If `signal_engine.py` is the "Eyes and Brain", this file is the "Hands and Shield".
When a signal is generated, this file decides:
1. Should we take the trade? (Checking Drawdown limits, Order Book Imbalance, BTC Correlation).
2. How big should the trade be? (Calculating position size based on Risk %).
3. Where is our Stop Loss? (Calculating SL using ATR - Average True Range).
4. When should we exit? (Trailing Stop Loss, Signal Flips).
"""

import logging
import csv
import os
import math
from datetime import datetime, timezone
from config import (
    SYMBOL, RISK_PER_TRADE_PERCENT, 
    MAX_POSITION_SIZE_USD, TRADING_ENABLED, TRADES_CSV, ACTIVITY_CSV,
    BTC_CORRELATION_GUARD, DAILY_DRAWDOWN_LIMIT, LEVERAGE,
    MIN_NOTIONAL_USD, get_current_threshold,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT
)
from live.execution_engine import (
    ExecutionEngine, PositionStatus, BalanceStatus, OrderStatus,
    PositionInfo, BalanceInfo, OrdersInfo
)
from utils.notifier import NotificationService
from utils.database import save_trade, save_portfolio_snapshot

logger = logging.getLogger(__name__)


class Trader:
    def __init__(self, execution_engine: ExecutionEngine):
        self.engine = execution_engine
        self.current_position = None  # None, 'LONG', or 'SHORT'
        self.entry_price = 0.0
        self.amount = 0.0
        self.current_sl = 0.0
        self.current_tp = 0.0
        
        # Drawdown Tracking (Explicit UTC Day Policy)
        self.daily_pnl = 0.0
        self.last_pnl_reset = datetime.now(timezone.utc).date()
        self.drawdown_blocked = False
        self.paused = False
        self.last_trade_close_time = None  # Cooldown: skip 2 hours after closing
        
        # Ensure trades file has header
        if not os.path.exists(TRADES_CSV):
            with open(TRADES_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'side', 'entry_price', 'amount', 'status', 'exit_price'])

        # Ensure activity log has header
        if not os.path.exists(ACTIVITY_CSV):
            with open(ACTIVITY_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'eth_price', 'ai_prediction', 'signal_direction',
                    'threshold', 'signal_strong', 'current_position',
                    'decision', 'reason', 'balance_usd', 'atr'
                ])
                
        # Restore state from previous run
        self._restore_state()

    def _restore_state(self):
        """Restore position state with Fail-Closed exchange reconciliation."""
        session = None
        last_trade = None
        try:
            from utils.database import SessionLocal, TradeLog
            session = SessionLocal()
            last_trade = session.query(TradeLog).order_by(TradeLog.timestamp.desc()).first()
        except Exception as db_err:
            logger.warning(f"Could not read last trade from DB (uninitialized or empty): {db_err}")

        try:
            
            mock_mode = getattr(self.engine, 'mock', False)
            
            if not mock_mode:
                # 1. LIVE TRADING: Query Exchange State
                real_pos = self.engine.get_open_position(SYMBOL)
                
                # CRITICAL FAIL-CLOSED BEHAVIOR:
                # If API fails, NEVER assume flat! Do NOT cancel orders. Halt and require inspection.
                if isinstance(real_pos, PositionInfo) and real_pos.status == PositionStatus.UNKNOWN_ERROR:
                    logger.critical(
                        "🛑 STARTUP RECONCILIATION FAILED: Exchange returned UNKNOWN_ERROR. "
                        "Halting bot to protect open positions and protective orders."
                    )
                    self.paused = True
                    self.drawdown_blocked = True
                    return

                if real_pos and getattr(real_pos, 'is_open', False):
                    self.current_position = real_pos['side']
                    self.amount = real_pos['amount']
                    self.entry_price = real_pos['entry_price']
                    
                    # Fetch open orders to check if bracket orders (TP/SL) exist
                    open_orders_info = self.engine.get_open_orders(SYMBOL)
                    if getattr(open_orders_info, 'is_error', False):
                        logger.critical("⚠️ Could not verify open orders on startup. Retaining existing exchange state.")
                        return

                    open_orders = open_orders_info.orders if isinstance(open_orders_info, OrdersInfo) else open_orders_info
                    has_sl = any(o.get('type') == 'STOP_MARKET' for o in open_orders)
                    has_tp = any(o.get('type') == 'TAKE_PROFIT_MARKET' for o in open_orders)
                    
                    if not has_sl or not has_tp:
                        logger.warning(f"⚠️ Position unprotected (Missing {'SL' if not has_sl else ''} {'TP' if not has_tp else ''}). Restoring bracket orders.")
                        if self.current_position == "LONG":
                            tp_price = round(self.entry_price * (1.0 + TAKE_PROFIT_PCT), 2)
                            sl_price = round(self.entry_price * (1.0 - STOP_LOSS_PCT), 2)
                        else:
                            tp_price = round(self.entry_price * (1.0 - TAKE_PROFIT_PCT), 2)
                            sl_price = round(self.entry_price * (1.0 + STOP_LOSS_PCT), 2)
                            
                        self.current_sl = sl_price
                        self.current_tp = tp_price
                        self.engine.place_tp_sl_orders(SYMBOL, self.current_position, self.amount, self.current_tp, self.current_sl)
                        self._log_trade(datetime.now(timezone.utc), self.current_position, self.entry_price, self.amount, "SYNC_UPDATE", tp_sl=self.current_sl)
                    else:
                        sl_orders = [o for o in open_orders if o.get('type') == 'STOP_MARKET']
                        tp_orders = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
                        if sl_orders and sl_orders[0].get('price'):
                            self.current_sl = float(sl_orders[0]['price'])
                        if tp_orders and tp_orders[0].get('price'):
                            self.current_tp = float(tp_orders[0]['price'])
                    
                    logger.info(f"🔄 Synced from BINANCE: {self.current_position} {self.amount} @ {self.entry_price} (SL: {self.current_sl})")
                elif real_pos and getattr(real_pos, 'is_flat', False):
                    # Confirmed flat on exchange
                    open_orders_info = self.engine.get_open_orders(SYMBOL)
                    if getattr(open_orders_info, 'is_verified', True) and len(open_orders_info) > 0:
                        logger.warning(f"⚠️ Position confirmed FLAT on exchange but {len(open_orders_info)} orphan orders found. Cancelling.")
                        self.engine.cancel_all_orders(SYMBOL)
                    
                    if last_trade and last_trade.status == 'OPEN':
                        logger.warning("⚠️ Binance confirmed FLAT, but DB reported OPEN. Syncing local state to FLAT.")
                        self._log_trade(datetime.now(timezone.utc), last_trade.side, float(last_trade.price), float(last_trade.amount), "SYNC_CLOSE")
                        self.last_trade_close_time = datetime.now(timezone.utc)
                    self.current_position = None
                    self.amount = 0.0
                    self.current_sl = 0.0
                    self.current_tp = 0.0
            else:
                # 2. PAPER TRADING: Database is source of truth
                if last_trade and last_trade.status == 'OPEN':
                    self.current_position = last_trade.side
                    self.entry_price = float(last_trade.price)
                    self.amount = float(last_trade.amount)
                    if self.current_sl == 0.0:
                        self.current_sl = float(last_trade.tp_sl) if last_trade.tp_sl else 0.0
                    logger.info(f"🔄 Restored active PAPER position: {self.current_position} {self.amount} @ {self.entry_price}")
                    self.engine.mock_side = self.current_position
                    self.engine.mock_entry_price = self.entry_price
                    self.engine.mock_amount = self.amount

        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
        finally:
            if 'session' in locals() and session:
                session.close()

    def sync_account(self):
        """Background thread sync to ensure exchange state matches DB."""
        self._restore_state()

    def handle_signal(self, timestamp: datetime, prediction: float, close_price: float, atr: float = None, btc_bullish: bool = True):
        """
        THE CORE TRADING LOGIC.
        Risk gate checks are strictly performed at the very beginning before any order sizing or execution.
        """
        if not TRADING_ENABLED:
            return

        # 0. Daily Reset Logic (Explicit UTC Date)
        today = datetime.now(timezone.utc).date()
        if today > self.last_pnl_reset:
            logger.info(f"New day detected ({today} UTC). Resetting daily PnL.")
            self.daily_pnl = 0.0
            self.last_pnl_reset = today
            self.drawdown_blocked = False

        # 1. Immediate Safety Gates
        if self.paused:
            logger.info("Trading PAUSED. Ignoring incoming signal.")
            self._log_activity(timestamp, close_price, prediction, atr, "PAUSED", "Bot is paused")
            return

        # 2. Daily Drawdown Guard: Check and snapshot portfolio BEFORE any entry decision
        bal_resp = self.engine.get_balance()
        bal = float(bal_resp)
        unrealized_pnl = 0.0
        if self.current_position == "LONG":
            unrealized_pnl = (close_price - self.entry_price) * self.amount
        elif self.current_position == "SHORT":
            unrealized_pnl = (self.entry_price - close_price) * self.amount
        
        equity = bal + unrealized_pnl
        save_portfolio_snapshot(balance=bal, equity=equity, pnl=self.daily_pnl + unrealized_pnl)

        # Check Drawdown Limit
        if bal > 0 and (self.daily_pnl + unrealized_pnl) < -(bal * DAILY_DRAWDOWN_LIMIT):
            if not self.drawdown_blocked:
                logger.warning(f"🛑 DAILY DRAWDOWN LIMIT REACHED ({DAILY_DRAWDOWN_LIMIT*100:.1f}%). Trading suspended for the day.")
                self.drawdown_blocked = True
                if self.current_position:
                    logger.info("Closing open position due to drawdown limit.")
                    side_to_send = "sell" if self.current_position == "LONG" else "buy"
                    close_res = self.engine.close_position(SYMBOL, side_to_send, self.amount, price=close_price)
                    if close_res and close_res.get('status') in ['closed', 'verified_flat']:
                        self.engine.cancel_all_orders(SYMBOL)
                        self.current_position = None
                        self.amount = 0.0
                        self.current_sl = 0.0
                        self.current_tp = 0.0
                    else:
                        logger.critical("Drawdown close failed to verify flat. Preserving state and halting.")
                        self.paused = True

        if self.drawdown_blocked:
            self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", "Daily drawdown limit reached")
            logger.info("Entry blocked: Daily drawdown limit reached.")
            return

        current_threshold = get_current_threshold()
        threshold_met = abs(prediction) >= current_threshold
        signal_side = "LONG" if prediction > 0 else "SHORT"

        # 3. Logic to Close Position
        if self.current_position:
            should_close = False
            reason = ""

            # Signal Flip check
            if threshold_met and self.current_position != signal_side:
                should_close, reason = True, "SIGNAL FLIP"

            if should_close:
                logger.info(f"Closing {self.current_position} at {close_price} ({reason}).")
                side_to_send = "sell" if self.current_position == "LONG" else "buy"
                close_result = self.engine.close_position(SYMBOL, side_to_send, self.amount, price=close_price)
                
                # Check verified closure before clearing internal state
                if close_result and close_result.get('status') in ['closed', 'verified_flat']:
                    self.engine.cancel_all_orders(SYMBOL)
                    trade_pnl = (close_price - self.entry_price) * self.amount if self.current_position == "LONG" else (self.entry_price - close_price) * self.amount
                    self.daily_pnl += trade_pnl
                    logger.info(f"Trade PnL: ${trade_pnl:.2f} | Daily PnL: ${self.daily_pnl:.2f}")

                    self._log_trade(timestamp, "CLOSE", close_price, self.amount, reason)
                    self._log_activity(timestamp, close_price, prediction, atr, "TRADE_CLOSED", f"{reason} | PnL=${trade_pnl:.2f}")
                    NotificationService.alert_trade(SYMBOL, "CLOSE", close_price, self.amount, reason)
                    self.current_position = None
                    self.amount = 0.0
                    self.current_sl = 0.0
                    self.current_tp = 0.0
                    self.last_trade_close_time = timestamp
                    logger.info(f"Trade closed ({reason}). Waiting for next candle before allowing new entries.")
                    return
                else:
                    logger.critical(f"🚨 Close position FAILED or UNCONFIRMED! Preserving position state. Halting bot.")
                    self.paused = True
                    return
            else:
                self._log_activity(timestamp, close_price, prediction, atr, "HOLD", f"In {self.current_position} @ {self.entry_price:.2f} | SL={self.current_sl:.2f}")

        # 4. Logic to Open Position
        if threshold_met and self.current_position is None:
            # Cooldown check: Wait at least 2 hours (7200 seconds) after closing a trade
            if self.last_trade_close_time is not None:
                ts1 = timestamp.replace(tzinfo=None)
                ts2 = self.last_trade_close_time.replace(tzinfo=None)
                if (ts1 - ts2).total_seconds() < 7200:
                    logger.info("Entry blocked by 2-hour cooldown.")
                    self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", "Cooldown active (2h)")
                    return

            # SAFETY: Before opening, query exchange position and fail closed
            if not getattr(self.engine, 'mock', False):
                real_pos = self.engine.get_open_position(SYMBOL)
                if isinstance(real_pos, PositionInfo) and real_pos.status == PositionStatus.UNKNOWN_ERROR:
                    logger.error("Cannot verify exchange flat status before entry (UNKNOWN_ERROR). Aborting entry.")
                    self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", "Exchange query failed")
                    return
                if real_pos and getattr(real_pos, 'is_open', False):
                    logger.warning(f"⚠️ Exchange already shows {real_pos['side']} {real_pos['amount']}. Syncing state.")
                    self.current_position = real_pos['side']
                    self.amount = real_pos['amount']
                    self.entry_price = real_pos['entry_price']
                    self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", f"Exchange already in {real_pos['side']}")
                    return

            # BTC Correlation Guard
            if signal_side == "LONG" and BTC_CORRELATION_GUARD and not btc_bullish:
                logger.info("Signal blocked: BTC is bearish.")
                self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", "BTC correlation guard")
                return

            # Long Only Mode
            from config import LONG_ONLY_MODE
            if LONG_ONLY_MODE and signal_side == "SHORT":
                logger.info("Signal blocked: LONG_ONLY_MODE is enabled.")
                self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", "LONG_ONLY_MODE enabled")
                return

            # Order Book Imbalance (Phase 5)
            from config import FETCH_ORDER_BOOK
            from data.fetcher import fetch_order_book_imbalance
            if FETCH_ORDER_BOOK:
                obi = fetch_order_book_imbalance(SYMBOL)
                if (signal_side == "LONG" and obi < -0.25) or (signal_side == "SHORT" and obi > 0.25):
                    logger.info(f"Entry blocked by OBI pressure: {obi:.4f}")
                    self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", f"OBI={obi:.4f} against signal")
                    return

            # Execute Entry
            sized_amount = self._calculate_position_size(close_price, atr, signal_side)
            if sized_amount <= 0:
                logger.info("Entry skipped: Position sizing returned 0 (e.g. min notional exceeds risk budget).")
                self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", "Position size 0 / Risk budget violation")
                return

            self.amount = sized_amount
            side_to_send = "buy" if signal_side == "LONG" else "sell"
            logger.info(f"Opening {signal_side} at {close_price} | TP={self.current_tp:.2f} | SL={self.current_sl:.2f}")

            # Cancel leftover orders before placing new entry
            self.engine.cancel_all_orders(SYMBOL)

            entry_res = self.engine.place_order(SYMBOL, side_to_send, self.amount, price=close_price)
            if not entry_res or entry_res.get('status') == 'failed':
                logger.error("Market entry order placement failed.")
                self.amount = 0.0
                self.current_position = None
                return

            # Obtain actual fill info from exchange
            filled_qty = float(entry_res.get('filled', 0.0) or entry_res.get('amount', self.amount))
            avg_price = float(entry_res.get('average', 0.0) or entry_res.get('price', close_price) or close_price)
            if filled_qty > 0:
                self.amount = filled_qty
            if avg_price > 0:
                self.entry_price = avg_price

            # Recalculate TP/SL based on actual fill execution
            if atr and atr > 0:
                sl_dist = max(1.5 * atr, self.entry_price * 0.015)
                tp_dist = max(3.0 * atr, sl_dist * 2.0)
            else:
                sl_dist = self.entry_price * STOP_LOSS_PCT
                tp_dist = self.entry_price * TAKE_PROFIT_PCT

            if signal_side == "LONG":
                self.current_sl = round(self.entry_price - sl_dist, 2)
                self.current_tp = round(self.entry_price + tp_dist, 2)
            else:
                self.current_sl = round(self.entry_price + sl_dist, 2)
                self.current_tp = round(self.entry_price - tp_dist, 2)

            # Place and verify protective bracket orders
            bracket_res = self.engine.place_tp_sl_orders(SYMBOL, signal_side, self.amount, self.current_tp, self.current_sl)
            
            # Check Stop Loss protection status
            if isinstance(bracket_res, dict) and bracket_res.get('emergency_status') == 'EMERGENCY_HALT':
                logger.critical("EMERGENCY: Bracket Stop Loss failed and emergency close failed. Halting bot.")
                self.paused = True
                return
            elif isinstance(bracket_res, dict) and bracket_res.get('emergency_status') == 'EMERGENCY_CLOSED':
                logger.error("Bracket Stop Loss failed; position was emergency closed. Keeping local state flat.")
                self.current_position = None
                self.amount = 0.0
                return

            self.current_position = signal_side
            self._log_trade(timestamp, signal_side, self.entry_price, self.amount, "OPEN", tp_sl=self.current_sl)
            self._log_activity(timestamp, self.entry_price, prediction, atr, "TRADE_OPENED", f"{signal_side} @ {self.entry_price:.2f} (TP={self.current_tp:.2f}, SL={self.current_sl:.2f})")
            NotificationService.alert_trade(SYMBOL, signal_side, self.entry_price, self.amount, "OPEN")

        elif threshold_met and self.current_position is not None:
            self._log_activity(timestamp, close_price, prediction, atr, "HOLD", f"Already in {self.current_position}")
        elif not threshold_met and self.current_position is None:
            self._log_activity(timestamp, close_price, prediction, atr, "WAITING", f"Signal too weak: |{prediction:.4f}| < {get_current_threshold():.4f}")

    def _calculate_position_size(self, price: float, atr: float = None, side: str = "LONG") -> float:
        """
        Calculates position size strictly respecting the RISK_PER_TRADE_PERCENT (2.5%) budget.
        If exchange minimum notional exceeds the risk budget, the trade is skipped.
        """
        balance = float(self.engine.get_balance())
        if balance <= 0:
            return 0.0

        risk_usd = balance * (RISK_PER_TRADE_PERCENT / 100.0)

        # Dynamic or fixed stop distance
        if atr and atr > 0:
            sl_dist = 1.5 * atr
            min_sl_dist = price * 0.015
            if sl_dist < min_sl_dist:
                sl_dist = min_sl_dist
            tp_dist = sl_dist * 2.0
        else:
            sl_dist = price * STOP_LOSS_PCT
            tp_dist = price * TAKE_PROFIT_PCT

        if side == "LONG":
            self.current_sl = round(price - sl_dist, 2)
            self.current_tp = round(price + tp_dist, 2)
        else:
            self.current_sl = round(price + sl_dist, 2)
            self.current_tp = round(price - tp_dist, 2)

        if sl_dist <= 0:
            return 0.0

        amount = risk_usd / sl_dist
        size_usd = amount * price

        # Institutional Margin Cap: never commit more than 30% margin
        max_margin = balance * 0.30
        max_notional = max_margin * LEVERAGE
        if size_usd > max_notional:
            logger.info(f"Position size ${size_usd:.2f} scaled to max margin cap ${max_notional:.2f}.")
            size_usd = max_notional
            amount = size_usd / price

        # Exchange Minimum Notional ($21.00 USD) Check:
        # Never increase risk beyond the intended risk budget to meet minimum notional.
        if size_usd < MIN_NOTIONAL_USD:
            # Check if MIN_NOTIONAL_USD violates the risk budget
            min_notional_risk = MIN_NOTIONAL_USD * (sl_dist / price)
            if min_notional_risk > risk_usd:
                logger.info(
                    f"Position notional ${size_usd:.2f} below Binance min ${MIN_NOTIONAL_USD:.2f}. "
                    f"Bumping would require ${min_notional_risk:.2f} risk, exceeding budget ${risk_usd:.2f} "
                    f"({RISK_PER_TRADE_PERCENT}%). SKIPPING TRADE to protect capital."
                )
                return 0.0
            
            # Risk is within budget, safe to bump
            logger.info(f"Position size ${size_usd:.2f} bumped to Binance min notional ${MIN_NOTIONAL_USD:.2f} (risk within budget).")
            size_usd = MIN_NOTIONAL_USD
            amount = size_usd / price

        # Institutional Leverage Cap
        max_leverage_size = balance * LEVERAGE
        if size_usd > max_leverage_size:
            logger.warning(f"Position size ${size_usd:.2f} exceeds leverage cap ${max_leverage_size:.2f}. Scaling down.")
            size_usd = max_leverage_size
            amount = size_usd / price

        if MAX_POSITION_SIZE_USD is not None and size_usd > MAX_POSITION_SIZE_USD:
            amount = MAX_POSITION_SIZE_USD / price

        amount = round(amount, 3)
        if amount > 0 and (amount * price) < MIN_NOTIONAL_USD:
            ceil_amount = math.ceil((MIN_NOTIONAL_USD / price) * 1000.0) / 1000.0
            if (ceil_amount * sl_dist) > risk_usd:
                logger.info("Precision adjusted amount violates risk budget. Skipping trade.")
                return 0.0
            amount = ceil_amount

        return amount

    def _log_trade(self, timestamp, side, price, amount, status, tp_sl=None):
        """Append trade to trades.csv and database."""
        with open(TRADES_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp.isoformat(), side, price, amount, status, price if status == 'CLOSE' else ''])
        save_trade(SYMBOL, side, price, amount, status, tp_sl=tp_sl)

    def _log_activity(self, timestamp, price, prediction, atr, decision, reason):
        """Append one row per candle scan to bot_activity.csv."""
        try:
            balance = float(self.engine.get_balance())
        except Exception:
            balance = 0.0
        current_threshold = get_current_threshold()
        threshold_met = abs(prediction) >= current_threshold
        signal_direction = "LONG" if prediction > 0 else "SHORT"
        with open(ACTIVITY_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(),
                round(price, 4),
                round(prediction, 6),
                signal_direction,
                round(current_threshold, 4),
                threshold_met,
                self.current_position or "FLAT",
                decision,
                reason,
                round(balance, 4),
                round(atr, 4) if atr else ""
            ])

    def check_trailing_stops(self):
        """Background thread logic to step Stop Loss with verified replacement."""
        from config import TRAILING_STOP_ENABLED
        if not TRAILING_STOP_ENABLED:
            return
            
        if not self.current_position or not self.entry_price or not self.amount:
            return
            
        mock_mode = getattr(self.engine, 'mock', False)
        if mock_mode:
            return
            
        current_price = self.engine.get_current_price(SYMBOL)
        if not current_price or current_price <= 0:
            return
            
        if self.current_position == "LONG":
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            
        new_sl_price = None
        log_msg = ""
        
        # Step 2: If profit >= 2.0%, move SL to +1.0% profit
        if pnl_pct >= 0.020:
            if self.current_position == "LONG":
                target_sl = round(self.entry_price * 1.01, 2)
                if self.current_sl < target_sl:
                    new_sl_price = target_sl
                    log_msg = f"Step 2: Securing +1.0% profit at {new_sl_price}"
            else:
                target_sl = round(self.entry_price * 0.99, 2)
                if self.current_sl > target_sl or self.current_sl == 0:
                    new_sl_price = target_sl
                    log_msg = f"Step 2: Securing +1.0% profit at {new_sl_price}"
                    
        # Step 1: If profit >= 1.0%, move SL to Break-Even (+0.1% to cover fees)
        elif pnl_pct >= 0.010:
            if self.current_position == "LONG":
                target_sl = round(self.entry_price * 1.001, 2)
                if self.current_sl < target_sl:
                    new_sl_price = target_sl
                    log_msg = f"Step 1: Moving to Break-Even at {new_sl_price}"
            else:
                target_sl = round(self.entry_price * 0.999, 2)
                if self.current_sl > target_sl or self.current_sl == 0:
                    new_sl_price = target_sl
                    log_msg = f"Step 1: Moving to Break-Even at {new_sl_price}"
                    
        if new_sl_price:
            logger.info(f"📈 Trailing Stop Activated! {log_msg} (Current Price: {current_price})")
            # Safe replacement: Place replacement order FIRST and verify before canceling previous
            res = self.engine.place_tp_sl_orders(SYMBOL, self.current_position, self.amount, self.current_tp, new_sl_price)
            if res and res.get('sl'):
                self.current_sl = new_sl_price
                self._log_trade(datetime.now(timezone.utc), self.current_position, self.entry_price, self.amount, "SYNC_UPDATE", tp_sl=self.current_sl)
            else:
                logger.error("Failed to place replacement trailing stop. Retaining previous protective stop.")
