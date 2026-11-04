"""
live/execution_engine.py — Interfacing with Binance Futures API via CCXT.

WHAT THIS FILE DOES:
This file is the "Wallet and Broker". It doesn't know anything about AI or Trading Logic. 
All it does is securely talk to Binance using the CCXT library.
It asks Binance: "How much money do I have?", "What trades are currently open?"
And it tells Binance: "Buy 10 ETH at Market Price!"

It also has a "Mock Mode" (Paper Trading) where it fakes the connection to Binance 
so you can test the bot safely without risking real money.
"""

import logging
import ccxt
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, LEVERAGE, 
    USE_BINANCE_TESTNET, INITIAL_PAPER_BALANCE, BASE_DIR
)
import os
import json

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, mock: bool = True):
        self.mock = mock
        if not self.mock:
            if not BINANCE_API_KEY or not BINANCE_API_SECRET:
                raise ValueError("Binance API credentials missing in config.")
            self.exchange = ccxt.binanceusdm({
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_API_SECRET,
                'options': {'defaultType': 'future'},
                'enableRateLimit': True,
            })
            if USE_BINANCE_TESTNET:
                self.exchange.set_sandbox_mode(True)
                logger.info("🌐 Binance Testnet (Sandbox) ENABLED.")
            
            self._setup_exchange()
            self.failure_count = 0  # Circuit breaker counter
        else:
            logger.info("⚡ ExecutionEngine initialized in MOCK mode.")
            self.failure_count = 0
            self.balance_file = os.path.join(BASE_DIR, "logs", "paper_balance.json")
            self._init_mock_balance()
            self.mock_entry_price = 0.0
            self.mock_amount = 0.0
            self.mock_side = None # 'LONG' or 'SHORT'

    def _init_mock_balance(self):
        """Initialize or load persistent mock balance."""
        if not os.path.exists(self.balance_file):
            self._save_mock_balance(INITIAL_PAPER_BALANCE)
        
    def _save_mock_balance(self, balance: float):
        with open(self.balance_file, 'w') as f:
            json.dump({"balance": balance}, f)

    def _load_mock_balance(self) -> float:
        try:
            with open(self.balance_file, 'r') as f:
                return float(json.load(f)["balance"])
        except:
            return INITIAL_PAPER_BALANCE

    def _setup_exchange(self):
        """Set leverage and initial exchange configs."""
        try:
            # We don't need to pass symbol here for leverage in some ccxt versions, 
            # but usually it's per symbol for futures.
            # Using ETH/USDT:USDT as default for this bot.
            symbol = "ETH/USDT:USDT"
            
            # FIX 2: Set Isolated Margin mode to protect account from total liquidation
            try:
                self.exchange.set_margin_mode('isolated', symbol)
                logger.info(f"Margin mode set to ISOLATED for {symbol}")
            except Exception as e:
                # CCXT will sometimes raise if the symbol is already isolated
                logger.info(f"Margin mode notice: {e}")
                
            self.exchange.set_leverage(LEVERAGE, symbol)
            logger.info(f"Leverage set to {LEVERAGE}x for {symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage/margin: {e}")

    def get_balance(self) -> float:
        """Fetch USDT balance directly from Binance Futures (FAPI v2).
        Uses the raw futures endpoint to avoid triggering spot sapi calls.
        """
        if self.mock:
            return self._load_mock_balance()
        
        try:
            # Direct FAPI v2 call — never touches the spot sapi endpoint
            raw = self.exchange.fapiPrivateV2GetBalance()
            for item in raw:
                if item.get('asset') == 'USDT':
                    return float(item.get('balance', 0.0))
            return 0.0
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0

    def get_open_position(self, symbol: str) -> dict:
        """Fetch real open position directly from Binance Futures (FAPI)."""
        if self.mock:
            if self.mock_side:
                return {'side': self.mock_side, 'amount': self.mock_amount, 'entry_price': self.mock_entry_price}
            return None
            
        try:
            # Direct FAPI call — avoids spot sapi endpoint
            ccxt_symbol = symbol  # e.g. 'ETH/USDT:USDT'
            positions = self.exchange.fetch_positions([ccxt_symbol])
            for pos in positions:
                amt = float(pos.get('contracts', 0))
                if amt != 0:
                    # 'side' is returned by CCXT as 'long' or 'short'
                    side = pos.get('side', '').upper()
                    
                    # Fallback to Binance's raw info if side is missing
                    if not side or side not in ['LONG', 'SHORT']:
                        raw_amt = float(pos.get('info', {}).get('positionAmt', 0))
                        side = 'LONG' if raw_amt > 0 else 'SHORT'
                        
                    entry_price = float(pos.get('entryPrice', 0) or pos.get('info', {}).get('entryPrice', 0))
                    return {'side': side, 'amount': abs(amt), 'entry_price': entry_price}
            return None
        except Exception as e:
            logger.error(f"Failed to fetch real position: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        """Fetch the current market price (last price) for a symbol."""
        if self.mock:
            # In mock mode, we don't have a live feed easily accessible here, 
            # but we can return the mock entry price as a fallback if needed.
            # Usually mock mode uses historical backtest data anyway.
            return self.mock_entry_price
            
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get('last', 0.0))
        except Exception as e:
            logger.error(f"Failed to fetch current price for {symbol}: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, amount: float, price: float = None, **kwargs):
        """
        Place a market order on Binance (or simulate one in Mock mode).
        Includes a Circuit Breaker: If Binance returns an error 3 times in a row,
        it shuts down trading to prevent infinite error loops.
        """
        side = side.upper()
        reduce_only = kwargs.get('reduce_only', False)
        if self.mock:
            price_val = price if price else 0.0
            logger.info(f"🚀 [MOCK] Order Placed: {side} {amount:.4f} {symbol} @ {price_val}")
            
            current_balance = self._load_mock_balance()
            commission = amount * price_val * 0.0005
            
            # Simple PnL simulation
            pnl = 0.0
            if self.mock_side is not None:
                # We are closing or flipping a position
                direction = 1 if self.mock_side == 'LONG' else -1
                pnl = direction * (price_val - self.mock_entry_price) * self.mock_amount
                logger.info(f"💰 [MOCK] Closed {self.mock_side} | PnL: ${pnl:.2f}")
                self.mock_side = None
                self.mock_entry_price = 0.0
                self.mock_amount = 0.0
            else:
                # We are opening a position
                self.mock_side = 'LONG' if side == 'BUY' else 'SHORT'
                self.mock_entry_price = price_val
                self.mock_amount = amount
            
            new_balance = current_balance + pnl - commission
            self._save_mock_balance(new_balance)
            return {'id': 'mock_order_id', 'status': 'closed', 'price': price_val}

        if self.failure_count >= 3:
            logger.error("🛑 CIRCUIT BREAKER TRIPPED: Too many execution failures. Manual intervention required.")
            return None

        try:
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            order = self.exchange.create_market_order(symbol, side, amount, params=params)
            logger.info(f"✅ Order Placed: {side} {amount:.4f} {symbol} | reduceOnly={reduce_only} | ID: {order['id']}")
            self.failure_count = 0  # Reset on success
            return order
        except Exception as e:
            self.failure_count += 1
            logger.error(f"❌ Failed to place order ({self.failure_count}/3): {e}")
            return None

    def close_position(self, symbol: str, side: str, amount: float, price: float = None):
        """Close an existing position using the REAL position size from Binance.
        
        Uses reduceOnly=True so Binance will NEVER accidentally open a new position.
        Queries Binance for the actual position size to avoid partial closes.
        """
        if self.mock:
            return self.place_order(symbol, side, amount, price=price, reduce_only=True)
        
        # Step 1: Get the REAL position from Binance (not our internal tracker)
        try:
            real_pos = self.get_open_position(symbol)
            if real_pos is None:
                logger.warning("close_position called but Binance reports NO open position. Skipping.")
                return {'id': 'no_position', 'status': 'skipped'}
            
            real_amount = real_pos['amount']
            logger.info(f"Closing REAL Binance position: {real_pos['side']} {real_amount} (bot tracked: {amount})")
            
            if abs(real_amount - amount) > 0.001:
                logger.warning(f"⚠️ Position size MISMATCH! Binance={real_amount}, Bot={amount}. Using Binance amount.")
            
            # Step 2: Close with reduceOnly=True using the REAL amount
            result = self.place_order(symbol, side, real_amount, reduce_only=True)
            
            # Step 3: Verify the position is actually flat
            if result:
                import time
                time.sleep(0.5)  # Brief pause for Binance to process
                verify = self.get_open_position(symbol)
                if verify is not None:
                    logger.error(f"🚨 CLOSE FAILED! Binance still shows position: {verify}. Will retry.")
                    # One more attempt with the remaining amount
                    retry_amount = verify['amount']
                    self.place_order(symbol, side, retry_amount, reduce_only=True)
                else:
                    logger.info("✅ Position confirmed FLAT on Binance.")
            
            return result
        except Exception as e:
            logger.error(f"❌ close_position failed: {e}")
            return None

    def place_trailing_stop(self, symbol: str, side: str, amount: float, callback_rate: float = 1.0):
        """Place a native Binance TRAILING_STOP_MARKET order."""
        if self.mock:
            logger.info(f"🚀 [MOCK] Trailing Stop Placed: {side} {amount:.4f} {symbol} | Rate: {callback_rate}%")
            return {'id': 'mock_trailing_stop_id'}
        
        try:
            # Binance callback rate must be between 0.1 and 5.0
            callback_rate = max(0.1, min(5.0, callback_rate))
            callback_rate = round(callback_rate, 1)  # 1 decimal place max
            
            params = {
                'reduceOnly': True,
                'callbackRate': callback_rate
            }
            # create_order(symbol, type, side, amount, price, params)
            order = self.exchange.create_order(symbol, 'TRAILING_STOP_MARKET', side, amount, None, params=params)
            logger.info(f"✅ Trailing Stop Placed: {side} {amount:.4f} {symbol} | Rate: {callback_rate}% | ID: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"❌ Failed to place trailing stop: {e}")
            return None

    def place_tp_sl_orders(self, symbol: str, side: str, amount: float, tp_price: float, sl_price: float):
        """Place native Binance TAKE_PROFIT_MARKET and STOP_MARKET orders directly on exchange."""
        if self.mock:
            logger.info(f"🚀 [MOCK] TP/SL Placed: {side} {amount:.4f} {symbol} | TP={tp_price:.2f} | SL={sl_price:.2f}")
            return {'tp_id': 'mock_tp', 'sl_id': 'mock_sl'}
        
        results = {}
        exit_side = "sell" if side.upper() in ["BUY", "LONG"] else "buy"
        
        # 1. Place Stop Loss (STOP_MARKET)
        try:
            sl_params = {
                'reduceOnly': True,
                'stopPrice': round(sl_price, 2)
            }
            sl_order = self.exchange.create_order(symbol, 'STOP_MARKET', exit_side, amount, None, params=sl_params)
            logger.info(f"✅ Stop Loss Order Placed: {exit_side.upper()} {amount:.4f} {symbol} @ trigger {sl_price:.2f} | ID: {sl_order['id']}")
            results['sl'] = sl_order
        except Exception as e:
            logger.error(f"❌ Failed to place Stop Loss order: {e}")

        # 2. Place Take Profit (TAKE_PROFIT_MARKET)
        try:
            tp_params = {
                'reduceOnly': True,
                'stopPrice': round(tp_price, 2)
            }
            tp_order = self.exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', exit_side, amount, None, params=tp_params)
            logger.info(f"✅ Take Profit Order Placed: {exit_side.upper()} {amount:.4f} {symbol} @ trigger {tp_price:.2f} | ID: {tp_order['id']}")
            results['tp'] = tp_order
        except Exception as e:
            logger.error(f"❌ Failed to place Take Profit order: {e}")

        return results

    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol (e.g. leftover trailing stops)."""
        if self.mock:
            return True
        try:
            self.exchange.cancel_all_orders(symbol)
            # Also cancel algo orders (Stop Loss / Take Profit)
            try:
                base_symbol = symbol.split(':')[0].replace('/', '')
                self.exchange.fapiPrivateDeleteAlgoOpenOrders({'symbol': base_symbol})
            except Exception as e:
                logger.warning(f"Could not cancel algo orders: {e}")
                
            logger.info(f"✅ Cancelled all open orders for {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to cancel orders: {e}")
            return False

    def get_open_orders(self, symbol: str) -> list:
        """Fetch all open orders for a symbol from Binance."""
        if self.mock:
            return []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            
            # Binance Futures migrated conditional orders to Algo service
            try:
                base_symbol = symbol.split(':')[0].replace('/', '')
                algo_orders = self.exchange.fapiPrivateGetOpenAlgoOrders({'symbol': base_symbol})
                if algo_orders:
                    for o in algo_orders:
                        # Map to standard CCXT structure expected by trader.py
                        orders.append({
                            'id': o.get('algoId'),
                            'type': o.get('orderType'),
                            'side': o.get('side', '').lower(),
                            'amount': float(o.get('quantity', 0)),
                            'price': float(o.get('triggerPrice', 0))
                        })
            except Exception as e:
                logger.warning(f"Could not fetch algo orders: {e}")
                
            return orders
        except Exception as e:
            logger.error(f"❌ Failed to fetch open orders: {e}")
            return []

