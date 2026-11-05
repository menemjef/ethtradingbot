"""
paper/execution.py — Realistic Paper Execution Engine with Multi-Strategy Isolation.

Repairs and enforces:
- Removal of silent fallbacks to candle close prices when fresh quotes are required.
- Stale quote rejection (configurable max_quote_age_seconds).
- Separated PnL accounting: directional PnL, realized funding PnL, entry fees, exit fees, slippage.
- Realistic stop execution: stop-first collision resolution, adverse opening gaps.
- Multi-strategy virtual position isolation by (strategy_id, symbol).
- Dynamic exchange tick-size, step-size quantization, min/max notional limits.
- Exact idempotent funding settlement using exchange settlement timestamps.
- Portfolio-level risk manager (aggregate leverage caps, drawdown circuit breaker).
"""

import os
import sys
import time
import uuid
import math
import sqlite3
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

from paper.config import (
    PAPER_DB_PATH,
    MARKET_DB_PATH,
    INITIAL_CAPITAL_USD,
    TAKER_FEE_RATE,
    SLIPPAGE_RATE,
    MAX_LEVERAGE_CAP,
    MAX_PORTFOLIO_LEVERAGE,
    MAX_QUOTE_AGE_SECONDS,
    CIRCUIT_BREAKER_MAX_DRAWDOWN_PCT,
    CONTRACT_SPECS_FALLBACK
)

logger = logging.getLogger("paper.execution")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] paper.execution: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class StaleQuoteError(Exception):
    """Raised when an order is attempted with stale or missing market quotes."""
    pass


class PaperExecutionEngine:
    """
    Production-grade paper execution engine with fail-closed persistence,
    multi-strategy isolation, and rigorous fill mechanics.
    """

    def __init__(
        self,
        db_path: str = PAPER_DB_PATH,
        market_db_path: str = MARKET_DB_PATH,
        initial_capital: float = INITIAL_CAPITAL_USD,
        taker_fee_rate: float = TAKER_FEE_RATE,
        slippage_rate: float = SLIPPAGE_RATE,
        max_leverage: float = MAX_LEVERAGE_CAP,
        max_quote_age_sec: float = MAX_QUOTE_AGE_SECONDS
    ):
        self.db_path = db_path
        self.market_db_path = market_db_path
        self.taker_fee_rate = taker_fee_rate
        self.slippage_rate = slippage_rate
        self.max_leverage = max_leverage
        self.max_quote_age_sec = max_quote_age_sec

        # Contract specifications cache
        self.contract_specs: Dict[str, Dict[str, Any]] = dict(CONTRACT_SPECS_FALLBACK)
        self._load_contract_specs()

        self._init_db(initial_capital)
        self.portfolio: Dict[str, Any] = {}
        # Active virtual positions: {(strategy_id, symbol): position_dict}
        self.active_positions: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._load_state_from_db(initial_capital)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _load_contract_specs(self):
        """Loads contract specs from market_data.db if available."""
        if os.path.exists(self.market_db_path):
            try:
                conn = sqlite3.connect(self.market_db_path, timeout=5.0)
                cur = conn.cursor()
                cur.execute("SELECT symbol, tick_size, step_size, min_qty, min_notional, max_notional, funding_interval_hours FROM exchange_metadata")
                for r in cur.fetchall():
                    self.contract_specs[r[0]] = {
                        "tick_size": float(r[1]),
                        "step_size": float(r[2]),
                        "min_qty": float(r[3]),
                        "min_notional": float(r[4]),
                        "max_notional": float(r[5]),
                        "funding_interval_hours": int(r[6])
                    }
                conn.close()
            except Exception:
                pass

    def _init_db(self, initial_capital: float):
        """Initializes SQLite schema for portfolio, positions, orders, trades, and funding."""
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
                drawdown_pct REAL NOT NULL,
                gross_leverage REAL NOT NULL DEFAULT 0.0,
                net_leverage REAL NOT NULL DEFAULT 0.0
            );
            """)

            # 2. Virtual Positions table (Multi-strategy isolated)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                position_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
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
                strategy_id TEXT NOT NULL,
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

            # 4. Completed Closed Trades table (Separated PnL accounting)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                trade_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                holding_hours REAL NOT NULL,
                directional_pnl REAL NOT NULL,
                funding_pnl REAL NOT NULL,
                entry_fee_usd REAL NOT NULL,
                exit_fee_usd REAL NOT NULL,
                slippage_cost_usd REAL NOT NULL,
                net_pnl REAL NOT NULL,
                exit_reason TEXT NOT NULL
            );
            """)

            # 5. Position Funding Payments table (Guarantees single application)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS position_funding_payments (
                payment_id TEXT PRIMARY KEY,
                position_id TEXT NOT NULL,
                settlement_time INTEGER NOT NULL,
                funding_rate REAL NOT NULL,
                mark_price REAL NOT NULL,
                payment_amount REAL NOT NULL,
                applied_at TEXT NOT NULL,
                UNIQUE(position_id, settlement_time)
            );
            """)

            # 6. Apply schema migrations if pre-existing tables lack Phase 2.2B columns
            migration_statements = [
                "ALTER TABLE paper_portfolio ADD COLUMN gross_leverage REAL NOT NULL DEFAULT 0.0;",
                "ALTER TABLE paper_portfolio ADD COLUMN net_leverage REAL NOT NULL DEFAULT 0.0;",
                "ALTER TABLE paper_positions ADD COLUMN strategy_id TEXT NOT NULL DEFAULT 'CandidateA_Momentum_Continuation';",
                "ALTER TABLE paper_trades ADD COLUMN strategy_id TEXT NOT NULL DEFAULT 'CandidateA_Momentum_Continuation';",
                "ALTER TABLE paper_trades ADD COLUMN directional_pnl REAL DEFAULT 0.0;",
                "ALTER TABLE paper_trades ADD COLUMN funding_pnl REAL DEFAULT 0.0;",
                "ALTER TABLE paper_trades ADD COLUMN entry_fee_usd REAL DEFAULT 0.0;",
                "ALTER TABLE paper_trades ADD COLUMN exit_fee_usd REAL DEFAULT 0.0;"
            ]
            for stmt in migration_statements:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Check if portfolio initialized
            cursor = conn.execute("SELECT COUNT(*) FROM paper_portfolio")
            if cursor.fetchone()[0] == 0:
                now_str = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                INSERT INTO paper_portfolio (
                    timestamp, cash_balance, equity, realized_pnl, unrealized_pnl,
                    margin_used, peak_equity, drawdown_pct, gross_leverage, net_leverage
                ) VALUES (?, ?, ?, 0.0, 0.0, 0.0, ?, 0.0, 0.0, 0.0);
                """, (now_str, initial_capital, initial_capital, initial_capital))
            conn.commit()

    def _load_state_from_db(self, initial_capital: float):
        """Loads latest portfolio and active virtual positions from SQLite."""
        with self._get_connection() as conn:
            # Load portfolio
            cursor = conn.execute("""
            SELECT cash_balance, equity, realized_pnl, unrealized_pnl, margin_used, peak_equity, drawdown_pct, gross_leverage, net_leverage
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
                    "drawdown_pct": float(row[6]),
                    "gross_leverage": float(row[7]) if len(row) > 7 else 0.0,
                    "net_leverage": float(row[8]) if len(row) > 8 else 0.0
                }
            else:
                self.portfolio = {
                    "cash_balance": initial_capital,
                    "equity": initial_capital,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "margin_used": 0.0,
                    "peak_equity": initial_capital,
                    "drawdown_pct": 0.0,
                    "gross_leverage": 0.0,
                    "net_leverage": 0.0
                }

            # Load active virtual positions
            cursor = conn.execute("""
            SELECT position_id, strategy_id, symbol, side, entry_price, quantity, notional, margin, leverage,
                   stop_loss, take_profit, entry_time, accumulated_funding
            FROM paper_positions WHERE status = 'OPEN'
            """)
            for p_row in cursor.fetchall():
                key = (p_row[1], p_row[2])
                self.active_positions[key] = {
                    "position_id": p_row[0],
                    "strategy_id": p_row[1],
                    "symbol": p_row[2],
                    "side": p_row[3],
                    "entry_price": float(p_row[4]),
                    "quantity": float(p_row[5]),
                    "notional": float(p_row[6]),
                    "margin": float(p_row[7]),
                    "leverage": float(p_row[8]),
                    "stop_loss": float(p_row[9]),
                    "take_profit": float(p_row[10]),
                    "entry_time": p_row[11],
                    "accumulated_funding": float(p_row[12]),
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
                margin_used, peak_equity, drawdown_pct, gross_leverage, net_leverage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                now_str,
                round(self.portfolio["cash_balance"], 2),
                round(self.portfolio["equity"], 2),
                round(self.portfolio["realized_pnl"], 2),
                round(self.portfolio["unrealized_pnl"], 2),
                round(self.portfolio["margin_used"], 2),
                round(self.portfolio["peak_equity"], 2),
                round(self.portfolio["drawdown_pct"], 2),
                round(self.portfolio.get("gross_leverage", 0.0), 2),
                round(self.portfolio.get("net_leverage", 0.0), 2)
            ))
            conn.commit()

    # =========================================================================
    # Quantization & Validation Helpers
    # =========================================================================

    def quantize_price(self, symbol: str, price: float) -> float:
        """Rounds price to contract tick size."""
        spec = self.contract_specs.get(symbol, {"tick_size": 0.01})
        tick = spec["tick_size"]
        return round(round(price / tick) * tick, 8)

    def quantize_quantity(self, symbol: str, quantity: float) -> float:
        """Floors quantity to contract step size."""
        spec = self.contract_specs.get(symbol, {"step_size": 0.001})
        step = spec["step_size"]
        precision = max(0, -int(math.floor(math.log10(step))))
        floored = math.floor(quantity / step) * step
        return round(floored, precision)

    def validate_quote_freshness(self, quote: Optional[Dict[str, Any]], now_ms: Optional[int] = None) -> Tuple[float, float]:
        """
        Validates quote existence and freshness.
        Strict requirement: Rejects execution if quote is missing or older than max_quote_age_sec.
        Returns: (best_bid, best_ask)
        """
        if not quote:
            raise StaleQuoteError("No market quote available. Execution rejected.")

        best_bid = float(quote.get("best_bid", 0.0))
        best_ask = float(quote.get("best_ask", 0.0))
        if best_bid <= 0 or best_ask <= 0 or best_bid > best_ask:
            raise StaleQuoteError(f"Invalid quote prices: bid={best_bid}, ask={best_ask}. Execution rejected.")

        curr_ms = now_ms or int(time.time() * 1000)
        receipt_ts = quote.get("receipt_ts", 0)
        age_sec = (curr_ms - receipt_ts) / 1000.0 if receipt_ts > 0 else 999.0

        if age_sec > self.max_quote_age_sec:
            raise StaleQuoteError(f"Market quote is stale ({age_sec:.2f}s > {self.max_quote_age_sec}s threshold). Execution rejected.")

        return best_bid, best_ask

    # =========================================================================
    # Position Lifecycle & Execution Methods
    # =========================================================================

    def open_position(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        quote: Dict[str, Any],
        entry_time_str: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a paper market entry order with strict quote freshness and multi-strategy isolation.
        NOTE: Displayed bookTicker quotes do not guarantee fill or capture market impact.
        Adverse slippage is conservatively modeled on all market orders.
        """
        key = (strategy_id, symbol)
        if key in self.active_positions:
            logger.warning(f"Strategy '{strategy_id}' already has open position for {symbol}. Rejecting duplicate entry.")
            return None

        # Drawdown circuit breaker check
        if self.portfolio["drawdown_pct"] >= CIRCUIT_BREAKER_MAX_DRAWDOWN_PCT:
            logger.error(f"Drawdown Circuit Breaker active ({self.portfolio['drawdown_pct']:.2f}% >= {CIRCUIT_BREAKER_MAX_DRAWDOWN_PCT}%). All new entries blocked.")
            return None

        # Quote Freshness Validation (NO SILENT FALLBACK TO CLOSE PRICE)
        best_bid, best_ask = self.validate_quote_freshness(quote)

        side = side.upper()
        if side == "LONG":
            fill_price = self.quantize_price(symbol, best_ask * (1.0 + self.slippage_rate))
            quote_used = best_ask
        else:
            fill_price = self.quantize_price(symbol, best_bid * (1.0 - self.slippage_rate))
            quote_used = best_bid

        # Quantize quantity
        quantized_qty = self.quantize_quantity(symbol, quantity)
        spec = self.contract_specs.get(symbol, {"min_qty": 0.001, "min_notional": 5.0, "max_notional": 500000.0})

        if quantized_qty < spec["min_qty"]:
            logger.warning(f"Order quantity {quantized_qty} below min_qty {spec['min_qty']} for {symbol}.")
            return None

        notional = fill_price * quantized_qty
        if notional < spec["min_notional"]:
            logger.warning(f"Order notional ${notional:.2f} below min_notional ${spec['min_notional']:.2f} for {symbol}.")
            return None

        # Portfolio Gross Leverage Check
        current_gross_notional = sum(p["notional"] for p in self.active_positions.values())
        if (current_gross_notional + notional) > (self.portfolio["equity"] * MAX_PORTFOLIO_LEVERAGE):
            logger.warning(f"Total notional ${(current_gross_notional + notional):.2f} would exceed max portfolio leverage ({MAX_PORTFOLIO_LEVERAGE}x). Rejecting entry.")
            return None

        margin_req = notional / self.max_leverage
        fee_usd = notional * self.taker_fee_rate
        slippage_usd = abs(fill_price - quote_used) * quantized_qty

        if margin_req + fee_usd > self.portfolio["cash_balance"]:
            logger.warning(f"Insufficient cash balance (${self.portfolio['cash_balance']:.2f}) for margin ${margin_req:.2f} + fee ${fee_usd:.2f}.")
            return None

        # Deduct entry fee and allocate margin
        self.portfolio["cash_balance"] -= fee_usd
        self.portfolio["margin_used"] += margin_req
        self.portfolio["equity"] = self.portfolio["cash_balance"] + self.portfolio["margin_used"]

        now_str = entry_time_str or datetime.now(timezone.utc).isoformat()
        pos_id = f"{strategy_id}_{symbol}_{uuid.uuid4().hex[:8]}"
        order_id = f"ord_entry_{uuid.uuid4().hex[:8]}"

        pos = {
            "position_id": pos_id,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side,
            "entry_price": fill_price,
            "quantity": quantized_qty,
            "notional": notional,
            "margin": margin_req,
            "leverage": notional / self.portfolio["equity"] if self.portfolio["equity"] > 0 else 1.0,
            "stop_loss": self.quantize_price(symbol, stop_loss),
            "take_profit": self.quantize_price(symbol, take_profit),
            "entry_time": now_str,
            "entry_fee_usd": fee_usd,
            "accumulated_funding": 0.0,
            "status": "OPEN"
        }

        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO paper_positions (
                position_id, strategy_id, symbol, side, entry_price, quantity, notional, margin, leverage,
                stop_loss, take_profit, entry_time, status, accumulated_funding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 0.0);
            """, (
                pos_id, strategy_id, symbol, side, fill_price, quantized_qty, notional, margin_req,
                pos["leverage"], pos["stop_loss"], pos["take_profit"], now_str
            ))

            conn.execute("""
            INSERT INTO paper_orders (
                order_id, position_id, strategy_id, symbol, side, order_type, requested_price,
                fill_price, quantity, notional, fee_usd, slippage_usd, status, created_at, filled_at
            ) VALUES (?, ?, ?, ?, ?, 'MARKET_ENTRY', ?, ?, ?, ?, ?, ?, 'FILLED', ?, ?);
            """, (
                order_id, pos_id, strategy_id, symbol, side, quote_used, fill_price,
                quantized_qty, notional, fee_usd, slippage_usd, now_str, now_str
            ))
            conn.commit()

        self.active_positions[key] = pos
        self._update_portfolio_leverage()
        self.save_portfolio_snapshot()
        logger.info(f"[{strategy_id}] Opened {side} Position: {symbol} | Qty: {quantized_qty:.4f} | Entry Px: ${fill_price:.2f} | SL: ${pos['stop_loss']:.2f} | TP: ${pos['take_profit']:.2f} | Fee: ${fee_usd:.2f}")
        return pos

    def close_position(
        self,
        strategy_id: str,
        symbol: str,
        exit_reason: str,
        quote: Optional[Dict[str, Any]] = None,
        adverse_gap_price: Optional[float] = None,
        exit_time_str: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Closes an open position with realistic fee, adverse slippage, and separated PnL accounting.
        Supports adverse opening gap fills when triggered by a gap beyond stop-loss.
        """
        key = (strategy_id, symbol)
        pos = self.active_positions.get(key)
        if not pos:
            logger.warning(f"No active position to close for strategy '{strategy_id}' on {symbol}.")
            return None

        side = pos["side"]
        qty = pos["quantity"]
        entry_price = pos["entry_price"]

        # Determine Fill Price:
        # Case A: Adverse Opening Gap (e.g. market opened gap beyond stop loss)
        if adverse_gap_price is not None and adverse_gap_price > 0:
            if side == "LONG":
                fill_price = self.quantize_price(symbol, adverse_gap_price * (1.0 - self.slippage_rate))
                quote_used = adverse_gap_price
            else:
                fill_price = self.quantize_price(symbol, adverse_gap_price * (1.0 + self.slippage_rate))
                quote_used = adverse_gap_price
        else:
            # Case B: Standard Market Exit with Fresh Quote Validation
            best_bid, best_ask = self.validate_quote_freshness(quote)
            if side == "LONG":
                fill_price = self.quantize_price(symbol, best_bid * (1.0 - self.slippage_rate))
                quote_used = best_bid
            else:
                fill_price = self.quantize_price(symbol, best_ask * (1.0 + self.slippage_rate))
                quote_used = best_ask

        # Directional PnL
        if side == "LONG":
            directional_pnl = (fill_price - entry_price) * qty
        else:
            directional_pnl = (entry_price - fill_price) * qty

        notional_exit = fill_price * qty
        exit_fee_usd = notional_exit * self.taker_fee_rate
        entry_fee_usd = pos.get("entry_fee_usd", entry_price * qty * self.taker_fee_rate)
        slippage_cost = abs(fill_price - quote_used) * qty
        funding_pnl = pos.get("accumulated_funding", 0.0)

        # Net PnL = Directional PnL + Funding PnL - Entry Fee - Exit Fee
        net_pnl = directional_pnl + funding_pnl - exit_fee_usd

        # Release Margin & Update Cash Balance
        margin_req = pos["margin"]
        self.portfolio["cash_balance"] += margin_req + net_pnl
        self.portfolio["margin_used"] = max(0.0, self.portfolio["margin_used"] - margin_req)
        self.portfolio["realized_pnl"] += net_pnl
        self.portfolio["unrealized_pnl"] = 0.0
        self.portfolio["equity"] = self.portfolio["cash_balance"] + self.portfolio["margin_used"]

        # Update Peak & Drawdown
        if self.portfolio["equity"] > self.portfolio["peak_equity"]:
            self.portfolio["peak_equity"] = self.portfolio["equity"]
        peak = self.portfolio["peak_equity"]
        self.portfolio["drawdown_pct"] = ((peak - self.portfolio["equity"]) / peak * 100.0) if peak > 0 else 0.0

        now_str = exit_time_str or datetime.now(timezone.utc).isoformat()
        trade_id = f"tr_{uuid.uuid4().hex[:8]}"
        order_id = f"ord_exit_{uuid.uuid4().hex[:8]}"

        try:
            t_ent = datetime.fromisoformat(pos["entry_time"])
            t_ex = datetime.fromisoformat(now_str)
            holding_hours = max(0.05, (t_ex - t_ent).total_seconds() / 3600.0)
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
                order_id, position_id, strategy_id, symbol, side, order_type, requested_price,
                fill_price, quantity, notional, fee_usd, slippage_usd, status, created_at, filled_at
            ) VALUES (?, ?, ?, ?, ?, 'MARKET_EXIT', ?, ?, ?, ?, ?, ?, 'FILLED', ?, ?);
            """, (
                order_id, pos["position_id"], strategy_id, symbol,
                "SELL" if side == "LONG" else "BUY", quote_used, fill_price,
                qty, notional_exit, exit_fee_usd, slippage_cost, now_str, now_str
            ))

            # Insert completed trade record
            conn.execute("""
            INSERT INTO paper_trades (
                trade_id, strategy_id, symbol, side, entry_time, exit_time, entry_price, exit_price,
                quantity, holding_hours, directional_pnl, funding_pnl, entry_fee_usd, exit_fee_usd,
                slippage_cost_usd, net_pnl, exit_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                trade_id, strategy_id, symbol, side, pos["entry_time"], now_str,
                entry_price, fill_price, qty, holding_hours, round(directional_pnl, 2),
                round(funding_pnl, 2), round(entry_fee_usd, 4), round(exit_fee_usd, 4),
                round(slippage_cost, 4), round(net_pnl, 2), exit_reason
            ))
            conn.commit()

        del self.active_positions[key]
        self._update_portfolio_leverage()
        self.save_portfolio_snapshot()
        logger.info(f"[{strategy_id}] Closed {side} Position: {symbol} | Exit Px: ${fill_price:.2f} | Reason: {exit_reason} | Net PnL: ${net_pnl:.2f} | Dur: {holding_hours:.1f}h")
        return {
            "trade_id": trade_id,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": side,
            "exit_price": fill_price,
            "exit_fee_usd": exit_fee_usd,
            "directional_pnl": round(directional_pnl, 2),
            "funding_pnl": round(funding_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "exit_reason": exit_reason
        }

    # =========================================================================
    # Intra-Bar Stop & Gap Checking (Stop-First Behavior)
    # =========================================================================

    def check_position_stops_and_gaps(
        self,
        strategy_id: str,
        symbol: str,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float
    ) -> Optional[Tuple[str, Optional[float]]]:
        """
        Evaluates intra-candle stop-loss and take-profit breaches with:
        1. Adverse opening gap detection.
        2. Strict stop-first resolution on ambiguous same-bar collisions.
        Returns: (exit_reason, adverse_gap_price_or_none) or None.
        """
        key = (strategy_id, symbol)
        pos = self.active_positions.get(key)
        if not pos:
            return None

        side = pos["side"]
        sl = pos["stop_loss"]
        tp = pos["take_profit"]

        if side == "LONG":
            # 1. Check for adverse opening gap down (bar opened below stop loss)
            if bar_open < sl:
                return ("stop_loss_gap", bar_open)

            # 2. Check for same-bar ambiguous collision
            sl_hit = bar_low <= sl
            tp_hit = bar_high >= tp
            if sl_hit and tp_hit:
                # Stop-first rule: Assume stop loss hit first
                return ("stop_loss_collision", None)
            elif sl_hit:
                return ("stop_loss", None)
            elif tp_hit:
                return ("take_profit", None)

        else:  # SHORT
            # 1. Check for adverse opening gap up (bar opened above stop loss)
            if bar_open > sl:
                return ("stop_loss_gap", bar_open)

            # 2. Check for same-bar ambiguous collision
            sl_hit = bar_high >= sl
            tp_hit = bar_low <= tp
            if sl_hit and tp_hit:
                # Stop-first rule: Assume stop loss hit first
                return ("stop_loss_collision", None)
            elif sl_hit:
                return ("stop_loss", None)
            elif tp_hit:
                return ("take_profit", None)

        return None

    def update_mark_to_market(self, quotes: Dict[str, Dict[str, Any]]):
        """
        Updates portfolio unrealized PnL and leverage using fresh quotes for all active positions.
        """
        total_unrealized = 0.0
        for (strat_id, sym), pos in self.active_positions.items():
            q = quotes.get(sym)
            if not q:
                continue
            mid_px = float(q.get("mid_price", pos["entry_price"]))
            if pos["side"] == "LONG":
                unrealized = (mid_px - pos["entry_price"]) * pos["quantity"]
            else:
                unrealized = (pos["entry_price"] - mid_px) * pos["quantity"]
            total_unrealized += unrealized

        self.portfolio["unrealized_pnl"] = total_unrealized
        self.portfolio["equity"] = self.portfolio["cash_balance"] + self.portfolio["margin_used"] + total_unrealized

        if self.portfolio["equity"] > self.portfolio["peak_equity"]:
            self.portfolio["peak_equity"] = self.portfolio["equity"]
        peak = self.portfolio["peak_equity"]
        self.portfolio["drawdown_pct"] = ((peak - self.portfolio["equity"]) / peak * 100.0) if peak > 0 else 0.0
        self._update_portfolio_leverage(quotes)

    def _update_portfolio_leverage(self, quotes: Optional[Dict[str, Dict[str, Any]]] = None):
        """Calculates current portfolio-level gross and net leverage."""
        eq = max(1.0, self.portfolio["equity"])
        gross_notional = 0.0
        net_notional = 0.0

        for (strat_id, sym), pos in self.active_positions.items():
            px = pos["entry_price"]
            if quotes and sym in quotes:
                px = float(quotes[sym].get("mid_price", px))
            notional = pos["quantity"] * px
            gross_notional += notional
            if pos["side"] == "LONG":
                net_notional += notional
            else:
                net_notional -= notional

        self.portfolio["gross_leverage"] = gross_notional / eq
        self.portfolio["net_leverage"] = abs(net_notional) / eq

    # =========================================================================
    # Idempotent Funding Settlement Accounting
    # =========================================================================

    def apply_settled_funding_rates(self) -> int:
        """
        Queries unapplied funding settlements from market_data.db and applies them
        to active virtual positions holding that symbol at settlement_time.
        Guarantees exact single-application (idempotency) via unique constraints.
        """
        if not os.path.exists(self.market_db_path) or not self.active_positions:
            return 0

        settlements_applied = 0
        now_str = datetime.now(timezone.utc).isoformat()

        try:
            m_conn = sqlite3.connect(self.market_db_path, timeout=5.0)
            m_cur = m_conn.cursor()
            m_cur.execute("SELECT symbol, settlement_time, funding_rate, mark_price FROM funding_settlements ORDER BY settlement_time ASC")
            settlement_rows = m_cur.fetchall()
            m_conn.close()
        except Exception as e:
            logger.error(f"Could not read funding_settlements from market_data.db: {e}")
            return 0

        with self._get_connection() as conn:
            for sym, sett_time, f_rate, mp in settlement_rows:
                sett_time_dt = datetime.fromtimestamp(sett_time / 1000, tz=timezone.utc)

                for (strat_id, pos_sym), pos in list(self.active_positions.items()):
                    if pos_sym != sym:
                        continue

                    # Check if position was active at settlement time
                    try:
                        pos_entry_dt = datetime.fromisoformat(pos["entry_time"])
                    except Exception:
                        pos_entry_dt = sett_time_dt

                    if pos_entry_dt > sett_time_dt:
                        continue  # Position opened after settlement

                    # Check if already applied
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT COUNT(*) FROM position_funding_payments WHERE position_id = ? AND settlement_time = ?",
                        (pos["position_id"], sett_time)
                    )
                    if cur.fetchone()[0] > 0:
                        continue  # Already applied

                    # Calculate payment: -signed_qty * mark_price * funding_rate
                    mark_px = mp if (mp and mp > 0) else pos["entry_price"]
                    signed_qty = pos["quantity"] if pos["side"] == "LONG" else -pos["quantity"]
                    payment = -1.0 * signed_qty * mark_px * f_rate

                    payment_id = f"fp_{uuid.uuid4().hex[:8]}"
                    conn.execute("""
                    INSERT INTO position_funding_payments (
                        payment_id, position_id, settlement_time, funding_rate, mark_price, payment_amount, applied_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """, (payment_id, pos["position_id"], sett_time, f_rate, mark_px, payment, now_str))

                    # Update position and cash
                    pos["accumulated_funding"] += payment
                    self.portfolio["cash_balance"] += payment  # positive payment increases cash, negative decreases
                    settlements_applied += 1
                    logger.info(f"[{strat_id}] Funding Applied to {sym} ({pos['side']}): ${payment:+.4f} (Rate: {f_rate:.6f}, Settlement: {sett_time_dt.isoformat()})")

            conn.commit()

        if settlements_applied > 0:
            self.save_portfolio_snapshot()
        return settlements_applied
