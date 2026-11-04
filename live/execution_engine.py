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
import os
import json
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import ccxt

from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, LEVERAGE, 
    USE_BINANCE_TESTNET, INITIAL_PAPER_BALANCE, BASE_DIR,
    ALLOW_LIVE_TRADING, LIVE_TRADING_ACK_TOKEN
)

logger = logging.getLogger(__name__)


class PositionStatus(str, Enum):
    VERIFIED_FLAT = "VERIFIED_FLAT"
    VERIFIED_OPEN = "VERIFIED_OPEN"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class BalanceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class OrderStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class PositionInfo:
    status: PositionStatus
    side: Optional[str] = None          # 'LONG' or 'SHORT'
    amount: float = 0.0
    entry_price: float = 0.0
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    @property
    def is_flat(self) -> bool:
        return self.status == PositionStatus.VERIFIED_FLAT

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.VERIFIED_OPEN

    @property
    def is_error(self) -> bool:
        return self.status == PositionStatus.UNKNOWN_ERROR

    def __getitem__(self, item):
        if item == 'side': return self.side
        if item == 'amount': return self.amount
        if item == 'entry_price': return self.entry_price
        if item == 'status': return self.status
        if item == 'error': return self.error
        raise KeyError(item)

    def get(self, item, default=None):
        try:
            val = self[item]
            return val if val is not None else default
        except KeyError:
            return default

    def __bool__(self):
        # Truthy only if verified open
        return self.status == PositionStatus.VERIFIED_OPEN


@dataclass
class BalanceInfo:
    status: BalanceStatus
    balance: float = 0.0
    error: Optional[str] = None
    raw: Optional[Any] = None

    @property
    def is_verified(self) -> bool:
        return self.status == BalanceStatus.VERIFIED

    @property
    def is_error(self) -> bool:
        return self.status == BalanceStatus.UNKNOWN_ERROR

    def __float__(self):
        return float(self.balance)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.balance == other
        if isinstance(other, BalanceInfo):
            return self.status == other.status and self.balance == other.balance
        return False

    def __lt__(self, other):
        return self.balance < float(other)

    def __le__(self, other):
        return self.balance <= float(other)

    def __gt__(self, other):
        return self.balance > float(other)

    def __ge__(self, other):
        return self.balance >= float(other)


@dataclass
class OrdersInfo:
    status: OrderStatus
    orders: List[Dict[str, Any]]
    error: Optional[str] = None

    @property
    def is_verified(self) -> bool:
        return self.status == OrderStatus.VERIFIED

    @property
    def is_error(self) -> bool:
        return self.status == OrderStatus.UNKNOWN_ERROR

    def __iter__(self):
        return iter(self.orders)

    def __len__(self):
        return len(self.orders)

    def __getitem__(self, item):
        return self.orders[item]


class ExecutionEngine:
    def __init__(self, mock: bool = True):
        self.mock = mock
        self.emergency_halt = False
        self.failure_count = 0  # Circuit breaker counter

        if not self.mock:
            # Explicit Opt-In Enforcement: Live execution cannot start without token
            if ALLOW_LIVE_TRADING != LIVE_TRADING_ACK_TOKEN:
                raise RuntimeError(
                    f"LIVE TRADING BLOCKED: Environment variable ALLOW_LIVE_TRADING must equal "
                    f"'{LIVE_TRADING_ACK_TOKEN}' to initialize real exchange execution."
                )

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
        else:
            logger.info("⚡ ExecutionEngine initialized in MOCK mode.")
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
            symbol = "ETH/USDT:USDT"
            try:
                self.exchange.set_margin_mode('isolated', symbol)
                logger.info(f"Margin mode set to ISOLATED for {symbol}")
            except Exception as e:
                logger.info(f"Margin mode notice: {e}")
                
            self.exchange.set_leverage(LEVERAGE, symbol)
            logger.info(f"Leverage set to {LEVERAGE}x for {symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage/margin: {e}")

    def get_balance(self) -> BalanceInfo:
        """Fetch USDT balance directly from Binance Futures (FAPI v2).
        Returns BalanceInfo with explicit status. Never silently returns 0.0 on error.
        """
        if self.mock:
            return BalanceInfo(status=BalanceStatus.VERIFIED, balance=self._load_mock_balance())
        
        try:
            raw = self.exchange.fapiPrivateV2GetBalance()
            for item in raw:
                if item.get('asset') == 'USDT':
                    return BalanceInfo(
                        status=BalanceStatus.VERIFIED,
                        balance=float(item.get('balance', 0.0)),
                        raw=item
                    )
            return BalanceInfo(status=BalanceStatus.VERIFIED, balance=0.0)
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return BalanceInfo(status=BalanceStatus.UNKNOWN_ERROR, balance=0.0, error=str(e))

    def get_open_position(self, symbol: str) -> PositionInfo:
        """Fetch real open position directly from Binance Futures (FAPI).
        Returns PositionInfo with explicit status (VERIFIED_FLAT, VERIFIED_OPEN, UNKNOWN_ERROR).
        """
        if self.mock:
            if self.mock_side:
                return PositionInfo(
                    status=PositionStatus.VERIFIED_OPEN,
                    side=self.mock_side,
                    amount=self.mock_amount,
                    entry_price=self.mock_entry_price
                )
            return PositionInfo(status=PositionStatus.VERIFIED_FLAT)
            
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                amt = float(pos.get('contracts', 0))
                if amt != 0:
                    side = pos.get('side', '').upper()
                    if not side or side not in ['LONG', 'SHORT']:
                        raw_amt = float(pos.get('info', {}).get('positionAmt', 0))
                        side = 'LONG' if raw_amt > 0 else 'SHORT'
                        
                    entry_price = float(pos.get('entryPrice', 0) or pos.get('info', {}).get('entryPrice', 0))
                    return PositionInfo(
                        status=PositionStatus.VERIFIED_OPEN,
                        side=side,
                        amount=abs(amt),
                        entry_price=entry_price,
                        raw=pos
                    )
            return PositionInfo(status=PositionStatus.VERIFIED_FLAT)
        except Exception as e:
            logger.error(f"Failed to fetch real position: {e}")
            return PositionInfo(status=PositionStatus.UNKNOWN_ERROR, error=str(e))

    def get_current_price(self, symbol: str) -> float:
        """Fetch current market price (last price) for a symbol."""
        if self.mock:
            return self.mock_entry_price
            
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get('last', 0.0))
        except Exception as e:
            logger.error(f"Failed to fetch current price for {symbol}: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, amount: float, price: float = None, **kwargs):
        """Place market order with Circuit Breaker and Emergency Halt checks."""
        if self.emergency_halt:
            logger.critical("🛑 EMERGENCY HALT ACTIVE: Orders blocked until manual intervention.")
            return None

        side = side.upper()
        reduce_only = kwargs.get('reduce_only', False)
        if self.mock:
            price_val = price if price else 0.0
            logger.info(f"🚀 [MOCK] Order Placed: {side} {amount:.4f} {symbol} @ {price_val}")
            
            current_balance = self._load_mock_balance()
            commission = amount * price_val * 0.0005
            
            pnl = 0.0
            if self.mock_side is not None:
                direction = 1 if self.mock_side == 'LONG' else -1
                pnl = direction * (price_val - self.mock_entry_price) * self.mock_amount
                logger.info(f"💰 [MOCK] Closed {self.mock_side} | PnL: ${pnl:.2f}")
                self.mock_side = None
                self.mock_entry_price = 0.0
                self.mock_amount = 0.0
            else:
                self.mock_side = 'LONG' if side == 'BUY' else 'SHORT'
                self.mock_entry_price = price_val
                self.mock_amount = amount
            
            new_balance = current_balance + pnl - commission
            self._save_mock_balance(new_balance)
            return {
                'id': 'mock_order_id',
                'status': 'closed',
                'price': price_val,
                'amount': amount,
                'filled': amount,
                'average': price_val
            }

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
        """Close an existing position using verified exchange position queries and reduceOnly=True.
        Guarantees that state is only confirmed flat when exchange confirms 0 contracts.
        """
        if self.mock:
            return self.place_order(symbol, side, amount, price=price, reduce_only=True)
        
        # Step 1: Query REAL position from exchange
        real_pos = self.get_open_position(symbol)
        if real_pos.status == PositionStatus.UNKNOWN_ERROR:
            logger.error("close_position ABORTED: Exchange position status is UNKNOWN_ERROR. Preserving state.")
            return {'id': 'error', 'status': 'failed', 'error': 'Exchange position status unknown'}

        if real_pos.status == PositionStatus.VERIFIED_FLAT:
            logger.warning("close_position called but Binance confirms NO open position (VERIFIED_FLAT).")
            return {'id': 'no_position', 'status': 'verified_flat'}
        
        real_amount = real_pos.amount
        logger.info(f"Closing REAL Binance position: {real_pos.side} {real_amount} (requested: {amount})")
        if abs(real_amount - amount) > 0.001:
            logger.warning(f"⚠️ Position size mismatch! Binance={real_amount}, Requested={amount}. Using Binance size.")
        
        # Step 2: Execute reduceOnly order
        result = self.place_order(symbol, side, real_amount, price=price, reduce_only=True)
        if not result:
            logger.error("close_position: reduceOnly order submission failed.")
            return {'id': 'error', 'status': 'failed', 'error': 'Order submission failed'}

        # Step 3: Verify the position is actually flat on exchange
        time.sleep(0.5)
        verify = self.get_open_position(symbol)
        if verify.status == PositionStatus.VERIFIED_FLAT:
            logger.info("✅ Position confirmed FLAT on Binance.")
            return {'id': result.get('id', 'closed'), 'status': 'closed', 'verified': True}
        elif verify.status == PositionStatus.VERIFIED_OPEN:
            logger.error(f"🚨 CLOSE INCOMPLETE: Binance still shows position: {verify.amount}. Retrying once.")
            retry_amount = verify.amount
            result_retry = self.place_order(symbol, side, retry_amount, price=price, reduce_only=True)
            time.sleep(0.5)
            verify_retry = self.get_open_position(symbol)
            if verify_retry.status == PositionStatus.VERIFIED_FLAT:
                logger.info("✅ Position confirmed FLAT on Binance after retry.")
                return {'id': result_retry.get('id', 'closed') if result_retry else 'closed', 'status': 'closed', 'verified': True}
            else:
                logger.critical(f"🚨 RETRY FAILED: Binance position still open: {verify_retry.amount}! Latching EMERGENCY HALT.")
                self.emergency_halt = True
                return {'id': 'retry_failed', 'status': 'failed', 'remaining': verify_retry.amount}
        else:
            logger.critical("🚨 Post-close verification returned UNKNOWN_ERROR. Latching EMERGENCY HALT.")
            self.emergency_halt = True
            return {'id': 'unverified', 'status': 'failed', 'error': 'Verification failed with API error'}

    def place_trailing_stop(self, symbol: str, side: str, amount: float, callback_rate: float = 1.0):
        """Place native Binance TRAILING_STOP_MARKET order."""
        if self.mock:
            logger.info(f"🚀 [MOCK] Trailing Stop Placed: {side} {amount:.4f} {symbol} | Rate: {callback_rate}%")
            return {'id': 'mock_trailing_stop_id'}
        
        try:
            callback_rate = max(0.1, min(5.0, callback_rate))
            callback_rate = round(callback_rate, 1)
            params = {
                'reduceOnly': True,
                'callbackRate': callback_rate
            }
            order = self.exchange.create_order(symbol, 'TRAILING_STOP_MARKET', side, amount, None, params=params)
            logger.info(f"✅ Trailing Stop Placed: {side} {amount:.4f} {symbol} | Rate: {callback_rate}% | ID: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"❌ Failed to place trailing stop: {e}")
            return None

    def place_tp_sl_orders(self, symbol: str, side: str, amount: float, tp_price: float, sl_price: float):
        """Place native Binance TAKE_PROFIT_MARKET and STOP_MARKET orders directly on exchange.
        Enforces verified protection: if stop-loss placement fails, immediately triggers an
        emergency reduce-only close to prevent operating an unprotected position.
        """
        if self.mock:
            logger.info(f"🚀 [MOCK] TP/SL Placed: {side} {amount:.4f} {symbol} | TP={tp_price:.2f} | SL={sl_price:.2f}")
            return {
                'tp': {'id': 'mock_tp', 'status': 'open'},
                'sl': {'id': 'mock_sl', 'status': 'open'}
            }
        
        results = {'tp': None, 'sl': None}
        exit_side = "sell" if side.upper() in ["BUY", "LONG"] else "buy"
        
        # 1. Place Stop Loss (STOP_MARKET) - CRITICAL
        try:
            sl_params = {
                'reduceOnly': True,
                'stopPrice': round(sl_price, 2)
            }
            sl_order = self.exchange.create_order(symbol, 'STOP_MARKET', exit_side, amount, None, params=sl_params)
            logger.info(f"✅ Stop Loss Order Placed: {exit_side.upper()} {amount:.4f} {symbol} @ trigger {sl_price:.2f} | ID: {sl_order['id']}")
            results['sl'] = sl_order
        except Exception as e:
            logger.critical(f"❌ Failed to place Stop Loss order: {e}")
            results['sl'] = None

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
            results['tp'] = None

        # 3. Emergency Policy: If Stop Loss failed, position is unprotected!
        if results['sl'] is None:
            logger.critical(
                "🚨 CRITICAL SAFETY BREACH: Stop-loss placement failed! Position is unprotected. "
                "Executing immediate reduce-only emergency close."
            )
            emer_close = self.close_position(symbol, exit_side, amount)
            if emer_close and emer_close.get('status') in ['closed', 'verified_flat']:
                logger.warning("Emergency close succeeded. Account returned to safe flat state.")
                results['emergency_status'] = 'EMERGENCY_CLOSED'
            else:
                logger.critical("🚨 EMERGENCY CLOSE FAILED! Latching EMERGENCY_HALT.")
                self.emergency_halt = True
                results['emergency_status'] = 'EMERGENCY_HALT'

        return results

    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol (e.g. leftover bracket orders)."""
        if self.mock:
            return True
        try:
            self.exchange.cancel_all_orders(symbol)
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

    def get_open_orders(self, symbol: str) -> OrdersInfo:
        """Fetch all open orders for a symbol from Binance.
        Returns OrdersInfo with explicit status. Never silently returns [] on error.
        """
        if self.mock:
            return OrdersInfo(status=OrderStatus.VERIFIED, orders=[])
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            try:
                base_symbol = symbol.split(':')[0].replace('/', '')
                algo_orders = self.exchange.fapiPrivateGetOpenAlgoOrders({'symbol': base_symbol})
                if algo_orders:
                    for o in algo_orders:
                        orders.append({
                            'id': o.get('algoId'),
                            'type': o.get('orderType'),
                            'side': o.get('side', '').lower(),
                            'amount': float(o.get('quantity', 0)),
                            'price': float(o.get('triggerPrice', 0))
                        })
            except Exception as e:
                logger.warning(f"Could not fetch algo orders: {e}")
                
            return OrdersInfo(status=OrderStatus.VERIFIED, orders=orders)
        except Exception as e:
            logger.error(f"❌ Failed to fetch open orders: {e}")
            return OrdersInfo(status=OrderStatus.UNKNOWN_ERROR, orders=[], error=str(e))
