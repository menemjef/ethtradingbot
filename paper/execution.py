"""
paper/execution.py — Realistic Paper Execution Engine with State Persistence.

Simulates exchange fills, bid/ask spread costs, conservative adverse slippage,
taker fees, and funding rate accrual. Maintains fail-closed state persistence
in SQLite across restarts.
"""

import os
import sys
import uuid
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from paper.config import (
    PAPER_DB_PATH,
    INITIAL_CAPITAL_USD,
    TAKER_FEE_RATE,
    SLIPPAGE_RATE,
    MAX_LEVERAGE_CAP
)

logger = logging.getLogger("paper.execution")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] paper.execution: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PaperExecutionEngine:
    """
    Realistic paper trading execution engine.
    Ensures 100% fail-closed persistence in SQLite.
    """

    def __init__(
        self,
        db_path: str = PAPER_DB_PATH,
        initial_capital: float = INITIAL_CAPITAL_USD,
        taker_fee_rate: float = TAKER_FEE_RATE,
        slippage_rate: float = SLIPPAGE_RATE,
        max_leverage: float = MAX_LEVERAGE_CAP
    ):
        self.db_path = db_path
        self.taker_fee_rate = taker_fee_rate
        self.slippage_rate = slippage_rate
        self.max_leverage = max_leverage

        self._init_db(initial_capital)
        self.portfolio: Dict[str, Any] = {}
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self._load_state_from_db(initial_capital)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self, initial_capital: float):
        """Creates SQLite tables for paper portfolio, positions, orders, and trades."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            # 1. Portfolio State table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cash_balance REAL NOT NULL,
                equity REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                margin_used REAL NOT NULL,
                peak_equity REAL NOT NULL,
                drawdown_pct REAL NOT NULL
            );
            """)

            # 2. Active & Historical Positions table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                notional REAL NOT NULL,
                margin REAL NOT NULL,
                leverage REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                status TEXT NOT NULL,
                accumulated_funding REAL NOT NULL DEFAULT 0.0
            );
            """)

            # 3. Executed Orders table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_orders (
                order_id TEXT PRIMARY KEY,
                position_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                requested_price REAL NOT NULL,
                fill_price REAL NOT NULL,
                quantity REAL NOT NULL,
                notional REAL NOT NULL,
                fee_usd REAL NOT NULL,
                slippage_usd REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                filled_at TEXT NOT NULL
            );
            """)

            # 4. Completed Closed Trades table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                holding_hours REAL NOT NULL,
                gross_pnl REAL NOT NULL,
                net_pnl REAL NOT NULL,
                fee_cost_usd REAL NOT NULL,
                slippage_cost_usd REAL NOT NULL,
                funding_cost_usd REAL NOT NULL,
                exit_reason TEXT NOT NULL
            );
            """)

            # Check if portfolio initialized
            cursor = conn.execute("SELECT COUNT(*) FROM paper_portfolio")
            if cursor.fetchone()[0] == 0:
                now_str = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                INSERT INTO paper_portfolio (
                    timestamp, cash_balance, equity, realized_pnl, unrealized_pnl,
                    margin_used, peak_equity, drawdown_pct
                ) VALUES (?, ?, ?, 0.0, 0.0, 0.0, ?, 0.0);
                """, (now_str, initial_capital, initial_capital, initial_capital))
            conn.commit()

    def _load_state_from_db(self, initial_capital: float):
        """Loads latest portfolio and open positions from SQLite."""
        with self._get_connection() as conn:
            # Load latest portfolio snapshot
            cursor = conn.execute("""
            SELECT cash_balance, equity, realized_pnl, unrealized_pnl, margin_used, peak_equity, drawdown_pct
            FROM paper_portfolio ORDER BY id DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                self.portfolio = {
                    "cash_balance": float(row[0]),
                    "equity": float(row[1]),
                    "realized_pnl": float(row[2]),
                    "unrealized_pnl": float(row[3]),
                    "margin_used": float(row[4]),
                    "peak_equity": float(row[5]),
                    "drawdown_pct": float(row[6])
                }
            else:
                self.portfolio = {
                    "cash_balance": initial_capital,
                    "equity": initial_capital,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "margin_used": 0.0,
                    "peak_equity": initial_capital,
                    "drawdown_pct": 0.0
                }

            # Load any OPEN positions
            cursor = conn.execute("""
            SELECT position_id, symbol, side, entry_price, quantity, notional, margin, leverage,
                   stop_loss, take_profit, entry_time, accumulated_funding
            FROM paper_positions WHERE status = 'OPEN'
            """)
            for p_row in cursor.fetchall():
                p_id = p_row[0]
                self.active_positions[p_row[1]] = {
                    "position_id": p_id,
                    "symbol": p_row[1],
                    "side": p_row[2],
                    "entry_price": float(p_row[3]),
                    "quantity": float(p_row[4]),
                    "notional": float(p_row[5]),
                    "margin": float(p_row[6]),
                    "leverage": float(p_row[7]),
                    "stop_loss": float(p_row[8]),
                    "take_profit": float(p_row[9]),
                    "entry_time": p_row[10],
                    "accumulated_funding": float(p_row[11]),
                    "status": "OPEN"
                }

        logger.info(f"Loaded Paper State | Cash: ${self.portfolio['cash_balance']:.2f} | Equity: ${self.portfolio['equity']:.2f} | Active Positions: {list(self.active_positions.keys())}")

    def save_portfolio_snapshot(self):
        """Persists current portfolio metrics to SQLite."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO paper_portfolio (
                timestamp, cash_balance, equity, realized_pnl, unrealized_pnl,
                margin_used, peak_equity, drawdown_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                now_str,
                round(self.portfolio["cash_balance"], 2),
                round(self.portfolio["equity"], 2),
                round(self.portfolio["realized_pnl"], 2),
                round(self.portfolio["unrealized_pnl"], 2),
                round(self.portfolio["margin_used"], 2),
                round(self.portfolio["peak_equity"], 2),
                round(self.portfolio["drawdown_pct"], 2)
            ))
            conn.commit()

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
        entry_time_str: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a paper market entry order with spread & adverse slippage.
        """
        if symbol in self.active_positions:
            logger.warning(f"Position already open for {symbol}. Rejecting duplicate entry.")
            return None

        side = side.upper()
        # Compute realistic fill price using best_bid / best_ask or conservative slippage
        if side == "LONG":
            base_px = best_ask if (best_ask and best_ask > 0) else price
            fill_price = base_px * (1.0 + self.slippage_rate)
        else:
            base_px = best_bid if (best_bid and best_bid > 0) else price
            fill_price = base_px * (1.0 - self.slippage_rate)

        notional = fill_price * quantity
        fee_usd = notional * self.taker_fee_rate
        slippage_usd = abs(fill_price - base_px) * quantity

        # Leverage and margin check
        eq = self.portfolio["equity"]
        if notional > eq * self.max_leverage:
            # Scale down to hard ceiling
            quantity = (eq * self.max_leverage) / fill_price
            notional = fill_price * quantity
            fee_usd = notional * self.taker_fee_rate

        margin_req = notional / self.max_leverage
        if margin_req > self.portfolio["cash_balance"]:
            logger.warning(f"Insufficient cash balance (${self.portfolio['cash_balance']:.2f}) for margin ${margin_req:.2f}.")
            return None

        # Deduct entry fee
        self.portfolio["cash_balance"] -= fee_usd
        self.portfolio["margin_used"] += margin_req
        self.portfolio["equity"] = self.portfolio["cash_balance"] + self.portfolio["margin_used"]

        now_str = entry_time_str or datetime.now(timezone.utc).isoformat()
        pos_id = f"pos_{uuid.uuid4().hex[:8]}"
        order_id = f"ord_{uuid.uuid4().hex[:8]}"

        pos = {
            "position_id": pos_id,
            "symbol": symbol,
            "side": side,
            "entry_price": fill_price,
            "quantity": quantity,
            "notional": notional,
            "margin": margin_req,
            "leverage": notional / eq if eq > 0 else 1.0,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": now_str,
            "accumulated_funding": 0.0,
            "status": "OPEN"
        }

        with self._get_connection() as conn:
            # Insert position
            conn.execute("""
            INSERT INTO paper_positions (
                position_id, symbol, side, entry_price, quantity, notional, margin, leverage,
                stop_loss, take_profit, entry_time, status, accumulated_funding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 0.0);
            """, (
                pos_id, symbol, side, fill_price, quantity, notional, margin_req,
                pos["leverage"], stop_loss, take_profit, now_str
            ))

            # Insert order
            conn.execute("""
            INSERT INTO paper_orders (
                order_id, position_id, symbol, side, order_type, requested_price,
                fill_price, quantity, notional, fee_usd, slippage_usd, status, created_at, filled_at
            ) VALUES (?, ?, ?, ?, 'MARKET_ENTRY', ?, ?, ?, ?, ?, ?, 'FILLED', ?, ?);
            """, (
                order_id, pos_id, symbol, side, price, fill_price, quantity, notional,
                fee_usd, slippage_usd, now_str, now_str
            ))
            conn.commit()

        self.active_positions[symbol] = pos
        self.save_portfolio_snapshot()
        logger.info(f"Opened {side} Position: {symbol} | Qty: {quantity:.4f} | Entry Px: ${fill_price:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | Fee: ${fee_usd:.2f}")
        return pos

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
        exit_time_str: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Closes an open position with realistic exit fees, slippage, and PnL calculation.
        """
        pos = self.active_positions.get(symbol)
        if not pos:
            logger.warning(f"No active position to close for {symbol}.")
            return None

        side = pos["side"]
        quantity = pos["quantity"]
        entry_price = pos["entry_price"]

        # Realistic fill price with adverse slippage
        if side == "LONG":
            base_px = best_bid if (best_bid and best_bid > 0) else exit_price
            fill_price = base_px * (1.0 - self.slippage_rate)
            gross_pnl = (fill_price - entry_price) * quantity
        else:
            base_px = best_ask if (best_ask and best_ask > 0) else exit_price
            fill_price = base_px * (1.0 + self.slippage_rate)
            gross_pnl = (entry_price - fill_price) * quantity

        notional_exit = fill_price * quantity
        fee_usd = notional_exit * self.taker_fee_rate
        slippage_usd = abs(fill_price - base_px) * quantity
        funding_usd = pos.get("accumulated_funding", 0.0)

        # Net PnL = Gross PnL - Exit Fee - Funding
        net_pnl = gross_pnl - fee_usd - funding_usd

        # Release margin and update cash balance
        margin_req = pos["margin"]
        self.portfolio["cash_balance"] += margin_req + net_pnl
        self.portfolio["margin_used"] = max(0.0, self.portfolio["margin_used"] - margin_req)
        self.portfolio["realized_pnl"] += net_pnl
        self.portfolio["unrealized_pnl"] = 0.0
        self.portfolio["equity"] = self.portfolio["cash_balance"]

        # Track peak equity and drawdown
        if self.portfolio["equity"] > self.portfolio["peak_equity"]:
            self.portfolio["peak_equity"] = self.portfolio["equity"]
        peak = self.portfolio["peak_equity"]
        self.portfolio["drawdown_pct"] = ((peak - self.portfolio["equity"]) / peak * 100.0) if peak > 0 else 0.0

        now_str = exit_time_str or datetime.now(timezone.utc).isoformat()
        trade_id = f"tr_{uuid.uuid4().hex[:8]}"
        order_id = f"ord_{uuid.uuid4().hex[:8]}"

        # Calculate holding hours
        try:
            t_ent = datetime.fromisoformat(pos["entry_time"])
            t_ex = datetime.fromisoformat(now_str)
            holding_hours = max(0.1, (t_ex - t_ent).total_seconds() / 3600.0)
        except Exception:
            holding_hours = 1.0

        with self._get_connection() as conn:
            # Update position status
            conn.execute("""
            UPDATE paper_positions
            SET status = 'CLOSED', exit_time = ?
            WHERE position_id = ?;
            """, (now_str, pos["position_id"]))

            # Insert exit order
            conn.execute("""
            INSERT INTO paper_orders (
                order_id, position_id, symbol, side, order_type, requested_price,
                fill_price, quantity, notional, fee_usd, slippage_usd, status, created_at, filled_at
            ) VALUES (?, ?, ?, ?, 'MARKET_EXIT', ?, ?, ?, ?, ?, ?, 'FILLED', ?, ?);
            """, (
                order_id, pos["position_id"], symbol, "SELL" if side == "LONG" else "BUY",
                exit_price, fill_price, quantity, notional_exit, fee_usd, slippage_usd, now_str, now_str
            ))

            # Insert completed trade record
            conn.execute("""
            INSERT INTO paper_trades (
                trade_id, symbol, side, entry_time, exit_time, entry_price, exit_price,
                quantity, holding_hours, gross_pnl, net_pnl, fee_cost_usd, slippage_cost_usd,
                funding_cost_usd, exit_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                trade_id, symbol, side, pos["entry_time"], now_str, entry_price, fill_price,
                quantity, holding_hours, round(gross_pnl, 2), round(net_pnl, 2),
                round(fee_usd, 4), round(slippage_usd, 4), round(funding_usd, 4), exit_reason
            ))
            conn.commit()

        del self.active_positions[symbol]
        self.save_portfolio_snapshot()
        logger.info(f"Closed {side} Position: {symbol} | Exit Px: ${fill_price:.2f} | Reason: {exit_reason} | Net PnL: ${net_pnl:.2f} | Duration: {holding_hours:.1f}h")
        return {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "net_pnl": net_pnl,
            "exit_reason": exit_reason
        }

    def update_mark_to_market(self, symbol: str, current_price: float, high: Optional[float] = None, low: Optional[float] = None) -> Optional[str]:
        """
        Updates unrealized PnL and tests stop-loss / take-profit conditions.
        Returns exit_reason if a stop or target was triggered, else None.
        """
        pos = self.active_positions.get(symbol)
        if not pos:
            return None

        side = pos["side"]
        qty = pos["quantity"]
        entry_px = pos["entry_price"]
        sl = pos["stop_loss"]
        tp = pos["take_profit"]

        # Test intra-candle price breach
        chk_high = high if high is not None else current_price
        chk_low = low if low is not None else current_price

        if side == "LONG":
            unrealized = (current_price - entry_px) * qty
            if chk_low <= sl:
                return "stop_loss"
            elif chk_high >= tp:
                return "take_profit"
        else:
            unrealized = (entry_px - current_price) * qty
            if chk_high >= sl:
                return "stop_loss"
            elif chk_low <= tp:
                return "take_profit"

        self.portfolio["unrealized_pnl"] = unrealized
        self.portfolio["equity"] = self.portfolio["cash_balance"] + pos["margin"] + unrealized
        if self.portfolio["equity"] > self.portfolio["peak_equity"]:
            self.portfolio["peak_equity"] = self.portfolio["equity"]
        return None

    def apply_funding_rate(self, symbol: str, funding_rate: float):
        """Accrues funding rate settlement against open positions."""
        pos = self.active_positions.get(symbol)
        if not pos or funding_rate == 0.0:
            return

        side = pos["side"]
        notional = pos["notional"]
        # Long pays positive funding, receives negative funding. Short is opposite.
        if side == "LONG":
            fee = notional * funding_rate
        else:
            fee = -notional * funding_rate

        pos["accumulated_funding"] += fee
        self.portfolio["cash_balance"] -= fee
        logger.info(f"Funding accrued for {symbol} ({side}): ${fee:.4f} (rate: {funding_rate:.6f})")
