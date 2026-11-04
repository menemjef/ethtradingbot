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
from datetime import datetime
from config import (
    SYMBOL, RISK_PER_TRADE_PERCENT, 
    MAX_POSITION_SIZE_USD, TRADING_ENABLED, TRADES_CSV, ACTIVITY_CSV,
    BTC_CORRELATION_GUARD, DAILY_DRAWDOWN_LIMIT, LEVERAGE,
    MIN_NOTIONAL_USD, get_current_threshold
)
from live.execution_engine import ExecutionEngine
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
        
        # Drawdown Tracking
        self.daily_pnl = 0.0
        self.last_pnl_reset = datetime.now().date()
        self.drawdown_blocked = False
        self.paused = False
        self.last_trade_close_time = None  # Cooldown: skip 2 candles after closing
        
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
        """Restore position state from the Database to survive restarts."""
        try:
            from utils.database import SessionLocal, TradeLog
            session = SessionLocal()
            last_trade = session.query(TradeLog).order_by(TradeLog.timestamp.desc()).first()
            
            mock_mode = getattr(self.engine, 'mock', False)
            
            if not mock_mode:
                # 1. LIVE TRADING: Exchange is Absolute Truth
                real_pos = self.engine.get_open_position(SYMBOL)
                if real_pos:
                    self.current_position = real_pos['side']
                    self.amount = real_pos['amount']
                    self.entry_price = real_pos['entry_price']
                    
                    # Fetch open orders to check if bracket orders (TP/SL) exist
                    open_orders = self.engine.get_open_orders(SYMBOL)
                    has_sl = any(o.get('type') == 'STOP_MARKET' for o in open_orders)
                    has_tp = any(o.get('type') == 'TAKE_PROFIT_MARKET' for o in open_orders)
                    
                    if not has_sl or not has_tp:
                        logger.warning(f"⚠️ Position unprotected (Missing {'SL' if not has_sl else ''} {'TP' if not has_tp else ''}). Restoring bracket orders.")
                        from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT
                        if self.current_position == "LONG":
                            tp_price = round(self.entry_price * (1.0 + TAKE_PROFIT_PCT), 2)
                            sl_price = round(self.entry_price * (1.0 - STOP_LOSS_PCT), 2)
                        else:
                            tp_price = round(self.entry_price * (1.0 - TAKE_PROFIT_PCT), 2)
                            sl_price = round(self.entry_price * (1.0 + STOP_LOSS_PCT), 2)
                            
                        self.current_sl = sl_price
                        self.current_tp = tp_price
                        # Clear any remaining orphan before replacing both
                        self.engine.cancel_all_orders(SYMBOL)
                        self.engine.place_tp_sl_orders(SYMBOL, self.current_position, self.amount, self.current_tp, self.current_sl)
                        
                        # Save the updated tp_sl to DB so it persists
                        self._log_trade(datetime.now(), self.current_position, self.entry_price, self.amount, "SYNC_UPDATE", tp_sl=self.current_sl)
                    elif last_trade and last_trade.status == 'OPEN' and last_trade.side == self.current_position:
                        # We have orders, restore internal state from live orders
                        sl_orders = [o for o in open_orders if o.get('type') == 'STOP_MARKET']
                        tp_orders = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
                        if sl_orders and sl_orders[0].get('price'):
                            self.current_sl = float(sl_orders[0]['price'])
                        elif self.current_sl == 0.0:
                            self.current_sl = float(last_trade.tp_sl) if last_trade.tp_sl else 0.0
                        if tp_orders and tp_orders[0].get('price'):
                            self.current_tp = float(tp_orders[0]['price'])
                    else:
                        # Orders exist but no matching DB trade — extract from orders
                        sl_orders = [o for o in open_orders if o.get('type') == 'STOP_MARKET']
                        tp_orders = [o for o in open_orders if o.get('type') == 'TAKE_PROFIT_MARKET']
                        if sl_orders and sl_orders[0].get('price'):
                            self.current_sl = float(sl_orders[0]['price'])
                        if tp_orders and tp_orders[0].get('price'):
                            self.current_tp = float(tp_orders[0]['price'])
                    
                    logger.info(f"🔄 Synced from BINANCE: {self.current_position} {self.amount} @ {self.entry_price} (SL: {self.current_sl})")
                else:
                    # Always cancel any orphan TP/SL orders when position is flat
                    open_orders = self.engine.get_open_orders(SYMBOL)
                    if open_orders:
                        logger.warning(f"⚠️ Position is FLAT but {len(open_orders)} orphan orders found. Cancelling.")
                        self.engine.cancel_all_orders(SYMBOL)
                    
                    if last_trade and last_trade.status == 'OPEN':
                        logger.warning("⚠️ Binance reports flat, but DB reported OPEN. Syncing to FLAT.")
                        self._log_trade(datetime.now(), last_trade.side, float(last_trade.price), float(last_trade.amount), "SYNC_CLOSE")
                        self.last_trade_close_time = datetime.now()
                        self.current_position = None
                        self.amount = 0.0
                        self.current_sl = 0.0
                        self.current_tp = 0.0
            else:
                # 2. PAPER TRADING: Database is Absolute Truth
                if last_trade and last_trade.status == 'OPEN':
                    self.current_position = last_trade.side
                    self.entry_price = float(last_trade.price)
                    self.amount = float(last_trade.amount)
                    
                    if self.current_sl == 0.0:
                        self.current_sl = float(last_trade.tp_sl) if last_trade.tp_sl else 0.0
                    
                    logger.info(f"🔄 Restored active PAPER position: {self.current_position} {self.amount} @ {self.entry_price} (SL: {self.current_sl})")
                    
                    # Sync DB to Ephemeral Mock Engine
                    self.engine.mock_side = self.current_position
                    self.engine.mock_entry_price = self.entry_price
                    self.engine.mock_amount = self.amount

        except Exception as e:
            logger.error(f"Failed to restore state from DB: {e}")
        finally:
            if 'session' in locals() and session:
                session.close()

    def sync_account(self):
        """Background thread sync to ensure exchange state matches DB."""
        self._restore_state()

    def handle_signal(self, timestamp: datetime, prediction: float, close_price: float, atr: float = None, btc_bullish: bool = True):
        """
        THE CORE TRADING LOGIC.
        Every time the AI predicts something, it goes through this gauntlet:
        
        1. Is trading paused? If yes, ignore.
        2. Are we already in a trade? 
           - Should we close it? (Hit stop loss? AI changed its mind? Trend reversed?)
        3. Should we open a NEW trade?
           - Is BTC crashing while we want to buy ETH? (BTC Correlation Guard)
           - Is the Order Book heavily against us? (OBI filter)
           - Have we lost too much money today? (Daily Drawdown Guard)
        4. If all checks pass, calculate position size and execute the trade!
        """
        if not TRADING_ENABLED:
            return

        # 0. Daily Reset Logic
        today = datetime.now().date()
        if today > self.last_pnl_reset:
            logger.info(f"New day detected ({today}). Resetting daily PnL.")
            self.daily_pnl = 0.0
            self.last_pnl_reset = today
            self.drawdown_blocked = False

        if self.paused:
            logger.info("Trading PAUSED. Ignoring incoming signal.")
            self._log_activity(timestamp, close_price, prediction, atr, "PAUSED", "Bot is paused")
            return

        current_threshold = get_current_threshold()
        threshold_met = abs(prediction) >= current_threshold
        signal_side = "LONG" if prediction > 0 else "SHORT"

        # 1. Logic to Close Position
        if self.current_position:
            should_close = False
            reason = ""

            # Signal Flip check
            if threshold_met and self.current_position != signal_side:
                should_close, reason = True, "SIGNAL FLIP"

            if should_close:
                logger.info(f"Closing {self.current_position} at {close_price} ({reason}). TP={self.current_tp:.2f}, SL={self.current_sl:.2f}")
                side_to_send = "sell" if self.current_position == "LONG" else "buy"
                close_result = self.engine.close_position(SYMBOL, side_to_send, self.amount, price=close_price)
                
                # Cancel open TP/SL bracket orders on Binance
                self.engine.cancel_all_orders(SYMBOL)
                
                # Calculate realized PnL for drawdown tracking
                trade_pnl = 0.0
                if self.current_position == "LONG":
                    trade_pnl = (close_price - self.entry_price) * self.amount
                else:
                    trade_pnl = (self.entry_price - close_price) * self.amount
                
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
                
                # After any close (Signal Flip, Stop Loss, Weak Signal), 
                # do NOT immediately re-open on the same candle. 
                # Wait for the next hourly scan to confirm a fresh entry.
                logger.info(f"Trade closed ({reason}). Waiting for next candle before allowing new entries.")
                return
            else:
                # Trade is fine — log HOLD with trailing stop info
                self._log_activity(timestamp, close_price, prediction, atr, "HOLD", f"In {self.current_position} @ {self.entry_price:.2f} | SL={self.current_sl:.2f}")

        # 3. Logic to Open Position
        if threshold_met and self.current_position is None:
            # Cooldown check: Wait at least 2 hours (7200 seconds) after closing a trade
            if self.last_trade_close_time is not None:
                # Handle timezone-aware/naive mixing gracefully
                ts1 = timestamp.replace(tzinfo=None)
                ts2 = self.last_trade_close_time.replace(tzinfo=None)
                if (ts1 - ts2).total_seconds() < 7200:
                    logger.info("Entry blocked by 2-hour cooldown.")
                    self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", "Cooldown active (2h)")
                    return

            # SAFETY: Before opening, verify we are actually FLAT on Binance
            if not getattr(self.engine, 'mock', False):
                real_pos = self.engine.get_open_position(SYMBOL)
                if real_pos is not None:
                    logger.warning(f"\u26a0\ufe0f Cannot open new trade: Binance still shows {real_pos['side']} {real_pos['amount']}. Syncing.")
                    self.current_position = real_pos['side']
                    self.amount = real_pos['amount']
                    self.entry_price = real_pos['entry_price']
                    self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", f"Binance still has {real_pos['side']} position")
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
                logger.info(f"OBI Pressure: {obi:.4f}")
                # Filter: Don't buy if sellers dominated, don't sell if buyers dominate
                if (signal_side == "LONG" and obi < -0.25) or (signal_side == "SHORT" and obi > 0.25):
                    logger.info(f"Entry blocked by OBI pressure: {obi:.4f}")
                    self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", f"OBI={obi:.4f} against signal")
                    return

            # Execute Entry
            self.amount = self._calculate_position_size(close_price, atr, signal_side)
            if self.amount > 0:
                logger.info(f"Opening {signal_side} at {close_price} | TP={self.current_tp:.2f} | SL={self.current_sl:.2f}")
                side_to_send = "buy" if signal_side == "LONG" else "sell"
                
                # FIX 1: Cancel any existing leftover orders BEFORE opening a new position to prevent landmines
                self.engine.cancel_all_orders(SYMBOL)
                
                # FIX 3: Pass close_price to market order so Mock Engine knows the entry price
                res = self.engine.place_order(SYMBOL, side_to_send, self.amount, price=close_price)
                if res:
                    # Place native Binance TP & SL bracket orders!
                    self.engine.place_tp_sl_orders(SYMBOL, signal_side, self.amount, self.current_tp, self.current_sl)

                    self.current_position = signal_side
                    self.entry_price = close_price
                    self._log_trade(timestamp, signal_side, close_price, self.amount, "OPEN", tp_sl=self.current_sl)
                    self._log_activity(timestamp, close_price, prediction, atr, "TRADE_OPENED", f"{signal_side} @ {close_price:.2f} (TP={self.current_tp:.2f}, SL={self.current_sl:.2f})")
                    NotificationService.alert_trade(SYMBOL, signal_side, close_price, self.amount, "OPEN")
        elif threshold_met and self.current_position is not None:
            # Already in a trade, signal is strong but same direction — just hold
            self._log_activity(timestamp, close_price, prediction, atr, "HOLD", f"Already in {self.current_position}")
        elif not threshold_met and self.current_position is None:
            # Weak signal, no position — bot is scanning and waiting
            self._log_activity(timestamp, close_price, prediction, atr, "WAITING", f"Signal too weak: |{prediction:.4f}| < {get_current_threshold():.4f}")
        elif self.drawdown_blocked:
            self._log_activity(timestamp, close_price, prediction, atr, "BLOCKED", "Daily drawdown limit reached")
        
        # 4. Periodically Snapshot Portfolio & Check Risk (Every candle)
        bal = self.engine.get_balance()
        
        # Calculate Unrealized PnL
        unrealized_pnl = 0.0
        if self.current_position == "LONG":
            unrealized_pnl = (close_price - self.entry_price) * self.amount
        elif self.current_position == "SHORT":
            unrealized_pnl = (self.entry_price - close_price) * self.amount
        
        equity = bal + unrealized_pnl
        
        # Check Drawdown Limit
        if self.daily_pnl + unrealized_pnl < -(bal * DAILY_DRAWDOWN_LIMIT):
            if not self.drawdown_blocked:
                logger.warning(f"🛑 DAILY DRAWDOWN LIMIT REACHED ({DAILY_DRAWDOWN_LIMIT*100}%). Trading suspended for the day.")
                self.drawdown_blocked = True
                # If we have a position, we should probably close it for safety
                if self.current_position:
                    logger.info("Closing open position due to drawdown limit.")
                    side_to_send = "sell" if self.current_position == "LONG" else "buy"
                    self.engine.close_position(SYMBOL, side_to_send, self.amount, price=close_price)
                    # FIX 5: Cancel associated TP/SL orders so they don't become landmines
                    self.engine.cancel_all_orders(SYMBOL)
                    self.current_position = None
                    self.amount = 0.0
                    self.current_sl = 0.0
                    self.current_tp = 0.0
        
        save_portfolio_snapshot(balance=bal, equity=equity, pnl=self.daily_pnl + unrealized_pnl)

        # Drawdown check for entry
        if threshold_met and self.current_position is None:
            if self.drawdown_blocked:
                logger.info("Entry blocked: Daily drawdown limit reached.")
                return

    def _calculate_position_size(self, price: float, atr: float = None, side: str = "LONG") -> float:
        """
        Calculates EXACT position size for fixed percentage TP/SL bracket orders.
        Risks EXACTLY RISK_PER_TRADE_PERCENT (e.g. 1.8%) of account balance on every trade.
        """
        balance = self.engine.get_balance()
        if balance <= 0: return 0.0

        risk_usd = balance * (RISK_PER_TRADE_PERCENT / 100.0)
        
        from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT
        
        # Use ATR-based dynamic stops if ATR is available (better risk/reward)
        if atr and atr > 0:
            sl_dist = 1.5 * atr  # SL is 1.5x ATR
            tp_dist = 3.0 * atr  # TP is 3.0x ATR (1:2 Risk-Reward)
            
            if side == "LONG":
                self.current_sl = round(price - sl_dist, 2)
                self.current_tp = round(price + tp_dist, 2)
            else:
                self.current_sl = round(price + sl_dist, 2)
                self.current_tp = round(price - tp_dist, 2)
        else:
            sl_dist = price * STOP_LOSS_PCT
            tp_dist = price * TAKE_PROFIT_PCT
            
            if side == "LONG":
                self.current_sl = round(price - sl_dist, 2)
                self.current_tp = round(price + tp_dist, 2)
            else:
                self.current_sl = round(price + sl_dist, 2)
                self.current_tp = round(price - tp_dist, 2)

        # Ensure minimum SL distance of 1.5% of price (floor)
        min_sl_dist = price * 0.015
        if sl_dist < min_sl_dist:
            logger.info(f"SL distance ${sl_dist:.2f} too tight, flooring to ${min_sl_dist:.2f} (1.5%)")
            sl_dist = min_sl_dist
            # Recalculate SL/TP with floored distance
            if side == "LONG":
                self.current_sl = round(price - sl_dist, 2)
                self.current_tp = round(price + sl_dist * 2, 2)  # Maintain 1:2 RR
            else:
                self.current_sl = round(price + sl_dist, 2)
                self.current_tp = round(price - sl_dist * 2, 2)

        amount = risk_usd / sl_dist if sl_dist > 0 else 0.0

        size_usd = amount * price
        
        # Margin cap: never commit more than 30% of account balance as margin at 10x leverage
        # (e.g. ~$17 margin / ~$170 notional on $57 balance)
        max_margin = balance * 0.30
        max_notional = max_margin * LEVERAGE
        if size_usd > max_notional:
            logger.info(f"Position size ${size_usd:.2f} scaled to max margin cap ${max_notional:.2f} (30% margin).")
            size_usd = max_notional
            amount = size_usd / price

        # Floor at Binance minimum notional ($21.00 USD)
        if size_usd < MIN_NOTIONAL_USD:
            logger.info(f"Position size ${size_usd:.2f} below Binance min notional ${MIN_NOTIONAL_USD:.2f}. Bumping up.")
            size_usd = MIN_NOTIONAL_USD
            amount = size_usd / price
        
        # Leverage Cap (Institutional Safety)
        max_leverage_size = balance * LEVERAGE
        if size_usd > max_leverage_size:
            logger.warning(f"Position size ${size_usd:.2f} exceeds leverage cap ${max_leverage_size:.2f}. Scaling down.")
            size_usd = max_leverage_size
            amount = size_usd / price

        if MAX_POSITION_SIZE_USD is not None and size_usd > MAX_POSITION_SIZE_USD:
            amount = MAX_POSITION_SIZE_USD / price

        # Ensure precision rounding does not push notional below Binance minimum
        amount = round(amount, 3)
        if amount > 0 and (amount * price) < MIN_NOTIONAL_USD:
            import math
            amount = math.ceil((MIN_NOTIONAL_USD / price) * 1000.0) / 1000.0

        return round(amount, 3)

    def process_command(self, cmd_text: str):
        """Process manual override commands from the dashboard."""
        cmd = cmd_text.upper()
        logger.info(f"🤖 Processing Command: {cmd}")
        
        if cmd == "CLOSE_ALL":
            if self.current_position:
                logger.warning("F-CLOSE: Force-closing open position via manual command.")
                side_to_send = "sell" if self.current_position == "LONG" else "buy"
                self.engine.close_position(SYMBOL, side_to_send, self.amount)
                self.engine.cancel_all_orders(SYMBOL)
                self.current_position = None
                self.amount = 0.0
                self.current_sl = 0.0
                self.current_tp = 0.0
            else:
                logger.info("F-CLOSE: No open position to close.")
        
        elif cmd == "PAUSE":
            self.paused = True
            logger.warning("⏸ BOT PAUSED: No new trades will be opened.")
        
        elif cmd == "RESUME":
            self.paused = False
            logger.info("▶️ BOT RESUMED: Trading back online.")
        
        else:
            logger.warning(f"Unknown command: {cmd}")

    def _log_trade(self, timestamp, side, price, amount, status, tp_sl=None):
        """Append trade details to CSV and Database."""
        # 1. CSV
        with open(TRADES_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp.isoformat(), side, price, amount, status, tp_sl])
        
        # 2. Database
        save_trade(SYMBOL, side, price, amount, status, tp_sl=tp_sl)

    def _log_activity(self, timestamp, price, prediction, atr, decision, reason):
        """Append one row per candle scan to bot_activity.csv."""
        try:
            balance = self.engine.get_balance()
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
        """Background thread logic to step the Stop Loss to Break-Even and beyond."""
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
            self.engine.cancel_all_orders(SYMBOL)
            self.current_sl = new_sl_price
            self.engine.place_tp_sl_orders(SYMBOL, self.current_position, self.amount, self.current_tp, self.current_sl)
            self._log_trade(datetime.now(), self.current_position, self.entry_price, self.amount, "SYNC_UPDATE", tp_sl=self.current_sl)
