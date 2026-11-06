"""
paper/strategy_runner.py — Strict 4-Hour Candle Aggregation & Forward Strategy Runner.

Enforces:
- Strict 4-Hour Candle Completeness: exactly 240 contiguous 1m bars required;
  rejection of 239-bar candles and missing-minute intervals.
- Independent ETH baseline execution (does not block Candidate A when BTC is unavailable).
- Separation of concerns: position management runs independently on market data ticks,
  while strategy signal generation runs strictly on 4-hour boundaries.
- Database-level uniqueness: symbol + strategy_id + strategy_version + timeframe + candle_close_time.
- Restart recovery: processes un-evaluated completed candles sequentially without duplicates.
"""

import os
import sys
import json
import time
import sqlite3
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from contextlib import closing
import pandas as pd
import numpy as np

from paper.config import (
    MARKET_DB_PATH,
    PAPER_DB_PATH,
    CANDIDATE_A_STRATEGY_ID,
    CANDIDATE_A_STRATEGY_VERSION,
    CANDIDATE_A_TIMEFRAME,
    CANDIDATE_A_ATR_PERIOD,
    CANDIDATE_A_EMA_FAST,
    CANDIDATE_A_EMA_SLOW,
    CANDIDATE_A_VOL_SMA,
    CANDIDATE_A_VOL_RATIO_MIN,
    CANDIDATE_A_STOP_ATR_MULT,
    CANDIDATE_A_TARGET_ATR_MULT,
    CANDIDATE_A_MAX_HOLDING_HOURS,
    CANDLE_4H_REQUIRED_1M_BARS,
    EXPECTED_1M_INTERVAL_MS,
    RISK_PCT_PER_TRADE,
    MAX_LEVERAGE_CAP,
    MAX_SIGNAL_AGE_SECONDS
)
from paper.execution import PaperExecutionEngine, StaleQuoteError

logger = logging.getLogger("paper.strategy_runner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] paper.strategy_runner: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class CandleGapError(Exception):
    """Raised when chronological 1m candle replay detects missing intervals."""
    def __init__(self, message: str, symbol: str, gap_start: int, gap_end: int, missing_bars: int):
        super().__init__(message)
        self.symbol = symbol
        self.gap_start = gap_start
        self.gap_end = gap_end
        self.missing_bars = missing_bars


class ManagedConnection:
    """
    Context manager wrapper around sqlite3.Connection that ensures the underlying
    database connection is ALWAYS closed upon context manager exit, resolving
    resource leakage across long-running processes.
    """
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self):
        self._conn.__enter__()
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._conn.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class CandidateAPaperRunner:
    """
    Forward paper-trading runner for Candidate A (Exp1_Momentum_Continuation_4H).
    Enforces strict 240-bar completeness and multi-strategy isolated execution.
    """

    def __init__(
        self,
        execution_engine: PaperExecutionEngine,
        market_db_path: str = MARKET_DB_PATH,
        paper_db_path: str = PAPER_DB_PATH,
        symbol: str = "ETHUSDT",
        strategy_id: str = CANDIDATE_A_STRATEGY_ID,
        strategy_version: str = CANDIDATE_A_STRATEGY_VERSION
    ):
        self.exec_engine = execution_engine
        self.market_db_path = market_db_path
        self.paper_db_path = paper_db_path
        self.symbol = symbol.upper()
        self.strategy_id = strategy_id
        self.strategy_version = strategy_version
        self.last_processed_1m_open_time: Dict[str, int] = {}
        self.flagged_gap_periods: List[Dict[str, Any]] = []
        self.gap_halted: bool = False
        self.data_integrity_unverified: bool = False

        self._init_runner_tables()
        self._load_processed_1m_open_times()

    def _get_connection(self):
        conn = sqlite3.connect(self.paper_db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return ManagedConnection(conn)

    def is_replay_behind(self) -> bool:
        """Checks if the runner has completed 1m candles in market_db newer than last processed."""
        if self.gap_halted:
            return True
        last_ot = self.last_processed_1m_open_time.get(self.symbol, -1)
        if last_ot <= 0:
            return False
        if os.path.exists(self.market_db_path):
            try:
                with closing(sqlite3.connect(self.market_db_path, timeout=5.0)) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT COUNT(*) FROM completed_klines WHERE symbol = ? AND timeframe = '1m' AND open_time > ?",
                        (self.symbol, last_ot)
                    )
                    row = cur.fetchone()
                    return bool(row and row[0] > 0)
            except Exception:
                pass
        return False

    def _load_processed_1m_open_times(self):
        """Loads latest processed 1m candle open_time from SQLite for persistent deduplication."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                SELECT symbol, MAX(open_time)
                FROM processed_1m_candles
                WHERE strategy_id = ?
                GROUP BY symbol
                """, (self.strategy_id,))
                for s, max_ot in cur.fetchall():
                    if max_ot is not None:
                        self.last_processed_1m_open_time[s] = int(max_ot)
        except Exception as e:
            logger.debug(f"Could not load processed_1m_candles: {e}")

    def _init_runner_tables(self):
        """Initializes tables for strategy evaluations, persistent candle tracking, and continuous benchmark tracking."""
        with self._get_connection() as conn:
            # 1. Strategy Evaluations table with composite primary key for uniqueness
            conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_evaluations (
                symbol TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                candle_close_time INTEGER NOT NULL,
                signal_type TEXT NOT NULL,
                rejection_reason TEXT,
                close_price REAL NOT NULL,
                ema_50 REAL,
                vol_ratio REAL,
                atr REAL,
                evaluated_at TEXT NOT NULL,
                features_json TEXT NOT NULL,
                PRIMARY KEY (symbol, strategy_id, strategy_version, timeframe, candle_close_time)
            );
            """)

            # 2. Continuous Benchmarks table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_benchmarks (
                timestamp TEXT PRIMARY KEY,
                strategy_equity REAL NOT NULL,
                eth_buy_and_hold_equity REAL NOT NULL,
                cash_equity REAL NOT NULL,
                eth_price REAL NOT NULL
            );
            """)

            # 3. Persistent Completed 1m Candles Tracking
            conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_1m_candles (
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                close_time INTEGER NOT NULL,
                processed_at TEXT NOT NULL,
                PRIMARY KEY (strategy_id, symbol, open_time)
            );
            """)
            conn.commit()

    # =========================================================================
    # Strict 4-Hour Candle Completeness Engine
    # =========================================================================

    def validate_and_build_4h_candles(
        self,
        symbol: str,
        now_ms: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Builds 4-hour candles from completed 1-minute bars with strict validation:
        1. A 4-hour candle MUST contain exactly 240 unique 1-minute bars.
        2. Timestamps must be strictly contiguous (60,000 ms intervals).
        3. Completed exchange candles only (now >= window end).
        4. Rejects 239-bar candles and missing-minute intervals with explicit audit logs.
        
        Returns:
            (df_4h_valid, rejected_windows_audit)
        """
        curr_time = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)

        if not os.path.exists(self.market_db_path):
            return pd.DataFrame(), []

        # Load completed 1m klines from SQLite
        conn = sqlite3.connect(self.market_db_path)
        try:
            query = """
            SELECT open_time, close_time, open, high, low, close, volume
            FROM completed_klines
            WHERE symbol = ? AND timeframe = '1m'
            ORDER BY open_time ASC
            """
            df_1m = pd.read_sql_query(query, conn, params=(symbol,))
        finally:
            conn.close()

        if df_1m.empty:
            return pd.DataFrame(), []

        # Also supplement with cached historical 1h bars if database has fewer than 15 days of 1m bars
        # This provides the necessary historical warm-up for 50-period 4H EMAs
        hist_4h_warmup = pd.DataFrame()
        earliest_1m_open = df_1m["open_time"].iloc[0]
        hist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", f"{symbol.lower()[:3]}_usdt_1h_3yr.parquet")
        if os.path.exists(hist_path):
            try:
                df_hist_1h = pd.read_parquet(hist_path)
                if not isinstance(df_hist_1h.index, pd.DatetimeIndex):
                    df_hist_1h.index = pd.to_datetime(df_hist_1h.index, utc=True)
                # Resample historical warm-up to 4H candles strictly preceding 1m live data
                h4 = df_hist_1h.resample("4h", closed="left", label="left").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                }).dropna()
                earliest_live_dt = pd.to_datetime(earliest_1m_open, unit="ms", utc=True)
                hist_4h_warmup = h4[h4.index < earliest_live_dt].tail(150)
            except Exception as e:
                logger.warning(f"Could not load historical warm-up for {symbol}: {e}")

        # Group 1m bars into 4-hour bins: [T, T + 4h) aligned to 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
        # Four hours in milliseconds = 4 * 3600 * 1000 = 14,400,000 ms
        FOUR_HOURS_MS = 14_400_000

        # Map each 1m bar to its 4H window start
        df_1m["window_start"] = (df_1m["open_time"] // FOUR_HOURS_MS) * FOUR_HOURS_MS

        grouped = df_1m.groupby("window_start")
        valid_4h_rows = []
        rejected_windows = []

        for w_start, group in grouped:
            w_end = w_start + FOUR_HOURS_MS
            # 1. Check if window is completed
            if curr_time < w_end:
                # Incomplete / in-progress 4H candle -> Skip
                continue

            # Sort by open_time
            group_sorted = group.sort_values("open_time").reset_index(drop=True)
            n_bars = len(group_sorted)

            # 2. Check bar count: MUST BE EXACTLY 240
            if n_bars != CANDLE_4H_REQUIRED_1M_BARS:
                reason = f"Incomplete candle: contains {n_bars} bars (required exactly {CANDLE_4H_REQUIRED_1M_BARS})"
                rejected_windows.append({
                    "symbol": symbol,
                    "window_start": w_start,
                    "window_end": w_end,
                    "bars_count": n_bars,
                    "reason": reason
                })
                logger.debug(f"[{symbol}] 4H candle at {datetime.fromtimestamp(w_start/1000, tz=timezone.utc)} rejected: {reason}")
                continue

            # 3. Check contiguous 60-second intervals
            open_times = group_sorted["open_time"].values
            diffs = open_times[1:] - open_times[:-1]
            has_gap = np.any(diffs != EXPECTED_1M_INTERVAL_MS)
            if has_gap:
                bad_idx = np.where(diffs != EXPECTED_1M_INTERVAL_MS)[0][0]
                gap_ms = diffs[bad_idx]
                reason = f"Missing minute gap detected: interval is {gap_ms}ms at index {bad_idx}"
                rejected_windows.append({
                    "symbol": symbol,
                    "window_start": w_start,
                    "window_end": w_end,
                    "bars_count": n_bars,
                    "reason": reason
                })
                logger.debug(f"[{symbol}] 4H candle at {datetime.fromtimestamp(w_start/1000, tz=timezone.utc)} rejected: {reason}")
                continue

            # 4. Valid completed 240-bar 4H candle
            candle_dt = pd.to_datetime(w_start, unit="ms", utc=True)
            valid_4h_rows.append({
                "timestamp": candle_dt,
                "open": group_sorted["open"].iloc[0],
                "high": group_sorted["high"].max(),
                "low": group_sorted["low"].min(),
                "close": group_sorted["close"].iloc[-1],
                "volume": group_sorted["volume"].sum(),
                "open_time": w_start,
                "close_time": w_end - 1,
                "bars_count": 240
            })

        df_live_4h = pd.DataFrame(valid_4h_rows)
        if not df_live_4h.empty:
            df_live_4h.set_index("timestamp", inplace=True)

        # Merge with historical warm-up if available
        if not hist_4h_warmup.empty:
            if not df_live_4h.empty:
                # Prepend historical warm-up without duplicate timestamps
                df_combined = pd.concat([hist_4h_warmup[hist_4h_warmup.index < df_live_4h.index[0]], df_live_4h])
            else:
                df_combined = hist_4h_warmup
            return df_combined, rejected_windows

        return df_live_4h, rejected_windows

    # =========================================================================
    # Candidate A Signal Evaluation (Frozen Rules)
    # =========================================================================

    def evaluate_candidate_a_logic(self, df_4h: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates Candidate A (Exp1_Momentum_Continuation_4H) logic on verified 4H candles:
        - Long: 4H Close > EMA50, 3 consecutive red candles, engulfing green candle, vol_ratio >= 1.30
        - Short: 4H Close < EMA50, 3 consecutive green candles, engulfing red candle, vol_ratio >= 1.30
        - Stop Loss: 1.5x ATR14
        - Take Profit: 3.0x ATR14
        """
        if len(df_4h) < 60:
            return {
                "signal": "NO_SIGNAL",
                "reason": f"Insufficient verified 4H bars ({len(df_4h)} < 60 required for EMA50 warm-up)"
            }

        close = df_4h["close"]
        open_px = df_4h["open"]
        high = df_4h["high"]
        low = df_4h["low"]
        vol = df_4h["volume"]

        ema_20 = close.ewm(span=CANDIDATE_A_EMA_FAST, adjust=False).mean()
        ema_50 = close.ewm(span=CANDIDATE_A_EMA_SLOW, adjust=False).mean()

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(CANDIDATE_A_ATR_PERIOD).mean()

        vol_sma = vol.rolling(CANDIDATE_A_VOL_SMA).mean()
        vol_ratio = vol / vol_sma.replace(0, np.nan)

        curr_idx = -1
        c_close = float(close.iloc[curr_idx])
        c_open = float(open_px.iloc[curr_idx])
        c_high = float(high.iloc[curr_idx])
        c_low = float(low.iloc[curr_idx])
        c_ema50 = float(ema_50.iloc[curr_idx])
        c_atr = float(atr.iloc[curr_idx])
        c_vol_ratio = float(vol_ratio.iloc[curr_idx]) if not np.isnan(vol_ratio.iloc[curr_idx]) else 1.0

        # Pullback checks (shift 1, 2, 3)
        three_red = (
            (close.iloc[-2] < open_px.iloc[-2]) and
            (close.iloc[-3] < open_px.iloc[-3]) and
            (close.iloc[-4] < open_px.iloc[-4])
        )
        three_green = (
            (close.iloc[-2] > open_px.iloc[-2]) and
            (close.iloc[-3] > open_px.iloc[-3]) and
            (close.iloc[-4] > open_px.iloc[-4])
        )

        engulf_green = (c_close > c_open) and (c_close > high.iloc[-2]) and (c_vol_ratio >= CANDIDATE_A_VOL_RATIO_MIN)
        engulf_red = (c_close < c_open) and (c_close < low.iloc[-2]) and (c_vol_ratio >= CANDIDATE_A_VOL_RATIO_MIN)

        feat_dict = {
            "close": c_close,
            "ema_50": c_ema50,
            "atr_14": c_atr,
            "vol_ratio": round(c_vol_ratio, 2),
            "three_red": bool(three_red),
            "three_green": bool(three_green),
            "engulf_green": bool(engulf_green),
            "engulf_red": bool(engulf_red)
        }

        # Check Long setup
        if c_close > c_ema50:
            if not three_red:
                return {"signal": "REJECTED", "reason": "3-candle red pullback missing in uptrend", "features": feat_dict}
            if not engulf_green:
                if c_vol_ratio < CANDIDATE_A_VOL_RATIO_MIN:
                    return {"signal": "REJECTED", "reason": f"Volume expansion insufficient ({c_vol_ratio:.2f} < {CANDIDATE_A_VOL_RATIO_MIN})", "features": feat_dict}
                return {"signal": "REJECTED", "reason": "Candle failed to engulf prior 4H high", "features": feat_dict}

            # Valid Long
            stop_loss = c_low - CANDIDATE_A_STOP_ATR_MULT * c_atr
            take_profit = c_close + CANDIDATE_A_TARGET_ATR_MULT * c_atr
            return {
                "signal": "LONG",
                "entry_price": c_close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": c_atr,
                "features": feat_dict
            }

        # Check Short setup
        elif c_close < c_ema50:
            if not three_green:
                return {"signal": "REJECTED", "reason": "3-candle green pullback missing in downtrend", "features": feat_dict}
            if not engulf_red:
                if c_vol_ratio < CANDIDATE_A_VOL_RATIO_MIN:
                    return {"signal": "REJECTED", "reason": f"Volume expansion insufficient ({c_vol_ratio:.2f} < {CANDIDATE_A_VOL_RATIO_MIN})", "features": feat_dict}
                return {"signal": "REJECTED", "reason": "Candle failed to engulf prior 4H low", "features": feat_dict}

            # Valid Short
            stop_loss = c_high + CANDIDATE_A_STOP_ATR_MULT * c_atr
            take_profit = c_close - CANDIDATE_A_TARGET_ATR_MULT * c_atr
            return {
                "signal": "SHORT",
                "entry_price": c_close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": c_atr,
                "features": feat_dict
            }

        return {"signal": "NO_SIGNAL", "reason": "No directional trend setup", "features": feat_dict}

    # =========================================================================
    # Strategy Evaluation & Idempotent Persistence
    # =========================================================================

    def record_evaluation(
        self,
        candle_close_time: int,
        signal_type: str,
        close_price: float,
        rejection_reason: Optional[str] = None,
        ema_50: Optional[float] = None,
        vol_ratio: Optional[float] = None,
        atr: Optional[float] = None,
        features: Optional[dict] = None
    ) -> bool:
        """
        Inserts evaluation record with database-level uniqueness.
        Returns True if inserted, False if already evaluated (idempotent duplicate prevention).
        """
        now_str = datetime.now(timezone.utc).isoformat()
        feat_str = json.dumps(features or {})
        with self._get_connection() as conn:
            try:
                conn.execute("""
                INSERT INTO strategy_evaluations (
                    symbol, strategy_id, strategy_version, timeframe, candle_close_time,
                    signal_type, rejection_reason, close_price, ema_50, vol_ratio, atr,
                    evaluated_at, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    self.symbol, self.strategy_id, self.strategy_version, CANDIDATE_A_TIMEFRAME,
                    candle_close_time, signal_type, rejection_reason, close_price, ema_50,
                    vol_ratio, atr, now_str, feat_str
                ))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Already evaluated for this candle
                return False

    def is_candle_evaluated(self, candle_close_time: int) -> bool:
        """Checks if this specific candle has already been evaluated."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT COUNT(*) FROM strategy_evaluations
            WHERE symbol = ? AND strategy_id = ? AND strategy_version = ?
              AND timeframe = ? AND candle_close_time = ?
            """, (self.symbol, self.strategy_id, self.strategy_version, CANDIDATE_A_TIMEFRAME, candle_close_time))
            return cur.fetchone()[0] > 0

    # =========================================================================
    # Execution & Separation of Concerns
    # =========================================================================

    def process_incoming_market_tick(
        self,
        quotes: Dict[str, Dict[str, Any]],
        current_bar: Optional[Dict[str, Any]] = None
    ):
        """
        Manages existing paper positions on every incoming market tick / 1m bar:
        - Evaluates Stop Loss and Take Profit with stop-first collision resolution.
        - Checks adverse opening gaps if current_bar is provided.
        - Enforces 72-hour maximum holding horizon.
        - Applies any pending realized funding rate settlements without look-ahead.
        - Updates continuous portfolio MTM (using candle close during replay).
        - Atomically persists candle markers and funding in a single transaction.
        NOTE: Operates independently of the 4-hour strategy signal evaluation schedule!
        """
        now_str = datetime.now(timezone.utc).isoformat()
        quote = quotes.get(self.symbol) if isinstance(quotes, dict) else None
        key = (self.strategy_id, self.symbol)
        pos = self.exec_engine.active_positions.get(key)

        # ---------------------------------------------------------------------
        # Case A: Live quote tick (current_bar is None)
        # ---------------------------------------------------------------------
        if current_bar is None:
            # Fix 1: While replay is behind or gap_halted, bound funding to last processed candle
            if self.gap_halted or self.is_replay_behind():
                last_ot = self.last_processed_1m_open_time.get(self.symbol, -1)
                funding_bound_ms = (last_ot + 59999) if last_ot > 0 else 0
            else:
                funding_bound_ms = int(time.time() * 1000)

            try:
                self.exec_engine.apply_settled_funding_rates(max_settlement_time_ms=funding_bound_ms)
            except Exception as e:
                logger.error(f"Error applying funding rates on live tick: {e}")

            self.exec_engine.update_mark_to_market(quotes)

            # Protect existing position using fresh live quotes
            if pos and quote:
                try:
                    best_bid, best_ask = self.exec_engine.validate_quote_freshness(quote)
                    side = pos["side"]
                    sl = pos["stop_loss"]
                    tp = pos["take_profit"]
                    tick_size = float(self.exec_engine.contract_specs.get(self.symbol, {}).get("tick_size", 0.01))

                    trigger_reason = None
                    if side == "LONG":
                        sl_hit = best_bid <= sl
                        tp_hit = best_bid >= (tp + tick_size - 1e-9)
                        if sl_hit and tp_hit:
                            trigger_reason = "stop_loss_collision"
                        elif sl_hit:
                            trigger_reason = "stop_loss"
                        elif tp_hit:
                            trigger_reason = "take_profit"
                    else:  # SHORT
                        sl_hit = best_ask >= sl
                        tp_hit = best_ask <= (tp - tick_size + 1e-9)
                        if sl_hit and tp_hit:
                            trigger_reason = "stop_loss_collision"
                        elif sl_hit:
                            trigger_reason = "stop_loss"
                        elif tp_hit:
                            trigger_reason = "take_profit"

                    if trigger_reason:
                        self.exec_engine.close_position(
                            strategy_id=self.strategy_id,
                            symbol=self.symbol,
                            exit_reason=trigger_reason,
                            quote=quote,
                            exit_time_str=now_str,
                            quotes=quotes
                        )
                    else:
                        # Check live 72-hour maximum holding horizon expiry
                        t_ent = datetime.fromisoformat(pos["entry_time"])
                        if t_ent.tzinfo is None:
                            t_ent = t_ent.replace(tzinfo=timezone.utc)
                        t_now = datetime.now(timezone.utc)
                        hours_held = (t_now - t_ent).total_seconds() / 3600.0
                        if hours_held >= CANDIDATE_A_MAX_HOLDING_HOURS:
                            self.exec_engine.close_position(
                                strategy_id=self.strategy_id,
                                symbol=self.symbol,
                                exit_reason="horizon_expiry",
                                quote=quote,
                                exit_time_str=now_str,
                                quotes=quotes
                            )
                except StaleQuoteError as e:
                    logger.warning(f"Could not manage position with live quote: {e}")
                except Exception as e:
                    logger.error(f"Error managing position with live quote: {e}")
            return

        # ---------------------------------------------------------------------
        # Case B: 1m Completed Candle Processing / Replay
        # ---------------------------------------------------------------------
        # Enforce strict symbol isolation: never process a bar belonging to another symbol
        bar_sym = current_bar.get("symbol")
        if bar_sym and bar_sym != self.symbol:
            logger.debug(f"Ignoring bar for {bar_sym}: runner is configured for {self.symbol}.")
            return

        # Persistent Deduplication: process each completed 1m candle only once
        bar_ot = current_bar.get("open_time")
        if bar_ot is not None:
            last_ot = self.last_processed_1m_open_time.get(self.symbol, -1)
            if bar_ot <= last_ot:
                return

        b_open = float(current_bar.get("open", 0.0))
        b_high = float(current_bar.get("high", 0.0))
        b_low = float(current_bar.get("low", 0.0))
        b_close = float(current_bar.get("close", 0.0))
        b_close_time = current_bar.get("close_time")
        b_open_time = current_bar.get("open_time")

        funding_bound_ms = int(b_close_time) if b_close_time else (int(bar_ot) + 59999 if bar_ot else int(time.time() * 1000))
        candle_dt_str = datetime.fromtimestamp(int(b_close_time) / 1000.0, tz=timezone.utc).isoformat() if b_close_time else now_str

        # Fix 4: Mark to the replayed candle close, not live quotes
        candle_quotes = dict(quotes) if isinstance(quotes, dict) else {}
        candle_quotes[self.symbol] = {
            "mid_price": b_close,
            "best_bid": b_close,
            "best_ask": b_close,
            "receipt_ts": int(b_close_time or time.time() * 1000)
        }
        self.exec_engine.update_mark_to_market(candle_quotes, mark_time_str=candle_dt_str)

        trigger = None
        if pos:
            trigger = self.exec_engine.check_position_stops_and_gaps(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                bar_open=b_open,
                bar_high=b_high,
                bar_low=b_low,
                bar_close=b_close,
                bar_close_time=b_close_time,
                bar_open_time=b_open_time
            )

        position_closed = False
        if trigger:
            reason, gap_px = trigger
            # Apply any pending funding up to candle close time before closing
            self.exec_engine.apply_settled_funding_rates(max_settlement_time_ms=funding_bound_ms)
            candle_exit_time = candle_dt_str
            processed_candle_info = {
                "strategy_id": self.strategy_id,
                "symbol": self.symbol,
                "open_time": int(bar_ot),
                "close_time": int(b_close_time),
                "processed_at": now_str
            } if (bar_ot is not None and b_close_time is not None) else None
            close_res = self.exec_engine.close_position(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                exit_reason=reason,
                quote=None,
                adverse_gap_price=gap_px,
                candle_price=b_close,
                exit_time_str=candle_exit_time,
                processed_candle_info=processed_candle_info,
                quotes=candle_quotes
            )
            if close_res and bar_ot is not None:
                self.last_processed_1m_open_time[self.symbol] = int(bar_ot)
            position_closed = True
        elif pos:
            # Check horizon expiry against current bar
            try:
                t_ent = datetime.fromisoformat(pos["entry_time"])
                if b_close_time:
                    bar_dt = datetime.fromtimestamp(int(b_close_time) / 1000.0, tz=timezone.utc)
                    hours_held = (bar_dt - t_ent).total_seconds() / 3600.0
                    if hours_held >= CANDIDATE_A_MAX_HOLDING_HOURS:
                        self.exec_engine.apply_settled_funding_rates(max_settlement_time_ms=funding_bound_ms)
                        candle_exit_time = bar_dt.isoformat()
                        processed_candle_info = {
                            "strategy_id": self.strategy_id,
                            "symbol": self.symbol,
                            "open_time": int(bar_ot),
                            "close_time": int(b_close_time),
                            "processed_at": now_str
                        } if (bar_ot is not None and b_close_time is not None) else None
                        close_res = self.exec_engine.close_position(
                            strategy_id=self.strategy_id,
                            symbol=self.symbol,
                            exit_reason="horizon_expiry",
                            quote=None,
                            candle_price=b_close,
                            exit_time_str=candle_exit_time,
                            processed_candle_info=processed_candle_info,
                            quotes=candle_quotes
                        )
                        if close_res and bar_ot is not None:
                            self.last_processed_1m_open_time[self.symbol] = int(bar_ot)
                        position_closed = True
            except Exception as e:
                logger.error(f"Error checking horizon expiry on bar: {e}")
                raise

        # Fix 2: If candle did not close a position, apply funding and processed candle marker in ONE transaction
        if not position_closed and bar_ot is not None and b_close_time is not None:
            processed_candle_info = {
                "strategy_id": self.strategy_id,
                "symbol": self.symbol,
                "open_time": int(bar_ot),
                "close_time": int(b_close_time),
                "processed_at": now_str
            }
            self.exec_engine.apply_settled_funding_rates(
                max_settlement_time_ms=funding_bound_ms,
                processed_candle_info=processed_candle_info
            )
            self.last_processed_1m_open_time[self.symbol] = int(bar_ot)

    def replay_missed_completed_1m_bars(
        self,
        quotes: Dict[str, Dict[str, Any]],
        server_time: Optional[int] = None
    ) -> int:
        """
        Replays missed completed 1m bars from completed_klines in chronological order (ASC).
        Ensures open positions receive every eligible missed bar after restart.
        Guarantees duplicate processing does not occur via persistent processed_1m_candles.
        Returns count of newly processed 1m bars.
        """
        if not os.path.exists(self.market_db_path):
            return 0

        has_previous_processed = False
        last_ot = self.last_processed_1m_open_time.get(self.symbol)
        if last_ot is None:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT MAX(open_time) FROM processed_1m_candles WHERE strategy_id = ? AND symbol = ?",
                    (self.strategy_id, self.symbol)
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    last_ot = int(row[0])
                    self.last_processed_1m_open_time[self.symbol] = last_ot
                    has_previous_processed = True
        else:
            has_previous_processed = True

        # Determine exchange-adjusted reference timestamp strictly for self.symbol (no cross-symbol leakage)
        exchange_now = None
        has_server_time = False
        if server_time is not None:
            exchange_now = int(server_time)
            has_server_time = True
        elif isinstance(quotes, dict):
            q = quotes.get(self.symbol, {})
            if isinstance(q, dict):
                if q.get("server_time"):
                    exchange_now = int(q["server_time"])
                    has_server_time = True
                elif q.get("exchange_ts"):
                    exchange_now = int(q["exchange_ts"])
                    has_server_time = True
                else:
                    exchange_now = q.get("timestamp") or q.get("receipt_ts")
        if not exchange_now:
            exchange_now = int(time.time() * 1000)

        pos = self.exec_engine.active_positions.get((self.strategy_id, self.symbol))
        if last_ot is None:
            if pos:
                try:
                    entry_dt = datetime.fromisoformat(pos["entry_time"])
                    last_ot = int(entry_dt.timestamp() * 1000) - 60000
                except Exception:
                    last_ot = 0
            else:
                try:
                    with closing(sqlite3.connect(self.market_db_path, timeout=5.0)) as m_conn:
                        cur = m_conn.cursor()
                        cur.execute(
                            "SELECT MAX(open_time) FROM completed_klines WHERE symbol = ? AND timeframe = '1m' AND close_time < ?",
                            (self.symbol, exchange_now)
                        )
                        max_row = cur.fetchone()
                        last_ot = (int(max_row[0]) - 60000) if (max_row and max_row[0] is not None) else 0
                except Exception:
                    last_ot = 0
            self.last_processed_1m_open_time[self.symbol] = last_ot

        raw_bars = []
        try:
            with closing(sqlite3.connect(self.market_db_path, timeout=10.0)) as m_conn:
                cur = m_conn.cursor()
                cur.execute("""
                SELECT open_time, close_time, open, high, low, close
                FROM completed_klines
                WHERE symbol = ? AND timeframe = '1m' AND open_time > ?
                ORDER BY open_time ASC
                """, (self.symbol, last_ot))
                for r in cur.fetchall():
                    c_close = int(r[1])
                    # Issue 2: Never replay a 1m candle unless its close_time is strictly earlier than current exchange-adjusted time
                    if c_close >= exchange_now:
                        logger.debug(f"Skipping unfinished 1m candle in replay: close_time {c_close} >= exchange_now {exchange_now}")
                        continue
                    raw_bars.append({
                        "symbol": self.symbol,
                        "open_time": int(r[0]),
                        "close_time": c_close,
                        "open": float(r[2]),
                        "high": float(r[3]),
                        "low": float(r[4]),
                        "close": float(r[5])
                    })
        except Exception as e:
            logger.error(f"Error querying missed 1m bars from market_data.db: {e}")
            return 0

        # Deduplicate bars by open_time in case of duplicate entries
        seen_ots = set()
        bars_to_process = []
        for b in raw_bars:
            if b["open_time"] not in seen_ots:
                seen_ots.add(b["open_time"])
                bars_to_process.append(b)

        if not bars_to_process:
            # Issue 4: Detect missing trailing candles by comparing latest completed candle against reliable Binance server time,
            # even when no subsequent candle arrives (empty batch).
            if has_server_time and has_previous_processed and last_ot is not None and last_ot > 0:
                if exchange_now >= last_ot + 2 * EXPECTED_1M_INTERVAL_MS:
                    missing_trailing = int((exchange_now - (last_ot + EXPECTED_1M_INTERVAL_MS)) // EXPECTED_1M_INTERVAL_MS)
                    if missing_trailing >= 1:
                        self.gap_halted = True
                        gap_info = {
                            "symbol": self.symbol,
                            "gap_start": last_ot + EXPECTED_1M_INTERVAL_MS,
                            "gap_end": exchange_now,
                            "missing_bars": missing_trailing,
                            "flagged_at": datetime.now(timezone.utc).isoformat()
                        }
                        self.flagged_gap_periods.append(gap_info)
                        msg = f"Missing trailing candle gap detected for {self.symbol} between latest bar {last_ot} and Binance server time {exchange_now} ({missing_trailing} missing bars)."
                        logger.critical(msg)
                        raise CandleGapError(msg, self.symbol, last_ot + EXPECTED_1M_INTERVAL_MS, exchange_now, missing_trailing)
            return 0

        # Issue 3: Verify chronological contiguous intervals without missing minutes
        # 1. Check gap between last_ot and first bar (if the runner previously processed candles)
        if has_previous_processed and last_ot is not None and last_ot > 0:
            expected_first_ot = last_ot + EXPECTED_1M_INTERVAL_MS
            first_ot = bars_to_process[0]["open_time"]
            if first_ot > expected_first_ot:
                self.gap_halted = True
                missing = int((first_ot - expected_first_ot) // EXPECTED_1M_INTERVAL_MS)
                gap_info = {
                    "symbol": self.symbol,
                    "gap_start": expected_first_ot,
                    "gap_end": first_ot,
                    "missing_bars": missing,
                    "flagged_at": datetime.now(timezone.utc).isoformat()
                }
                self.flagged_gap_periods.append(gap_info)
                msg = f"Candle gap detected for {self.symbol} between last processed {last_ot} and next bar {first_ot} ({missing} missing bars)."
                logger.critical(msg)
                raise CandleGapError(msg, self.symbol, expected_first_ot, first_ot, missing)

        # 2. Check gaps between consecutive bars
        for i in range(len(bars_to_process) - 1):
            curr_ot = bars_to_process[i]["open_time"]
            next_ot = bars_to_process[i + 1]["open_time"]
            expected_next = curr_ot + EXPECTED_1M_INTERVAL_MS
            if next_ot > expected_next:
                self.gap_halted = True
                missing = int((next_ot - expected_next) // EXPECTED_1M_INTERVAL_MS)
                gap_info = {
                    "symbol": self.symbol,
                    "gap_start": expected_next,
                    "gap_end": next_ot,
                    "missing_bars": missing,
                    "flagged_at": datetime.now(timezone.utc).isoformat()
                }
                self.flagged_gap_periods.append(gap_info)
                msg = f"Candle gap detected for {self.symbol} between bar {curr_ot} and {next_ot} ({missing} missing bars)."
                logger.critical(msg)
                raise CandleGapError(msg, self.symbol, expected_next, next_ot, missing)

        # 3. Check trailing gap between last bar in batch and Binance server time
        if has_server_time:
            latest_bar_ot = bars_to_process[-1]["open_time"]
            if exchange_now >= latest_bar_ot + 2 * EXPECTED_1M_INTERVAL_MS:
                missing_trailing = int((exchange_now - (latest_bar_ot + EXPECTED_1M_INTERVAL_MS)) // EXPECTED_1M_INTERVAL_MS)
                if missing_trailing >= 1:
                    self.gap_halted = True
                    gap_info = {
                        "symbol": self.symbol,
                        "gap_start": latest_bar_ot + EXPECTED_1M_INTERVAL_MS,
                        "gap_end": exchange_now,
                        "missing_bars": missing_trailing,
                        "flagged_at": datetime.now(timezone.utc).isoformat()
                    }
                    self.flagged_gap_periods.append(gap_info)
                    msg = f"Missing trailing candle gap detected for {self.symbol} between latest bar {latest_bar_ot} and Binance server time {exchange_now} ({missing_trailing} missing bars)."
                    logger.critical(msg)
                    raise CandleGapError(msg, self.symbol, latest_bar_ot + EXPECTED_1M_INTERVAL_MS, exchange_now, missing_trailing)

        # Verified contiguous intervals without gaps: clear gap halt
        self.gap_halted = False

        processed_count = 0
        for bar in bars_to_process:
            self.process_incoming_market_tick(quotes, current_bar=bar)
            processed_count += 1

        return processed_count

    def evaluate_new_completed_candles(
        self,
        quotes: Dict[str, Dict[str, Any]],
        now_utc: Optional[datetime] = None
    ) -> int:
        """
        Evaluates Candidate A on all newly completed, 240-bar-verified 4H candles.
        Processes missed candles sequentially upon restart and guarantees no duplicates.
        """
        if self.gap_halted or getattr(self, "data_integrity_unverified", False):
            logger.warning(
                f"Candidate A strategy evaluation halted for {self.symbol} due to "
                f"active candle gap or unverified server time. All new entries blocked."
            )
            return 0

        df_4h, rejected = self.validate_and_build_4h_candles(self.symbol)
        if df_4h.empty or len(df_4h) < 60:
            return 0

        evaluations_made = 0
        key = (self.strategy_id, self.symbol)

        # Inspect all completed candles chronologically (do not truncate to tail(10))
        for ts, candle in df_4h.iterrows():
            c_val = candle.get("close_time")
            if pd.isna(c_val) or c_val is None or float(c_val) <= 0:
                # Estimate close time from timestamp: ts + 4h - 1ms
                close_time = int(ts.timestamp() * 1000) + 14_400_000 - 1
            else:
                close_time = int(c_val)

            if self.is_candle_evaluated(close_time):
                continue  # Already evaluated

            # Sub-dataframe up to this candle
            sub_df = df_4h[df_4h.index <= ts]
            eval_res = self.evaluate_candidate_a_logic(sub_df)
            sig = eval_res["signal"]

            recorded = self.record_evaluation(
                candle_close_time=close_time,
                signal_type=sig,
                close_price=float(candle["close"]),
                rejection_reason=eval_res.get("reason"),
                ema_50=eval_res.get("features", {}).get("ema_50"),
                vol_ratio=eval_res.get("features", {}).get("vol_ratio"),
                atr=eval_res.get("features", {}).get("atr_14"),
                features=eval_res.get("features")
            )
            evaluations_made += 1

            # Only a genuinely current, eligible signal may open a new position!
            # Historical missed candles must be recorded for continuity but must NOT trigger retroactive entries at current prices.
            is_current_candle = (ts == df_4h.index[-1])

            # Check maximum signal age: candle must not be older than MAX_SIGNAL_AGE_SECONDS
            now_dt = now_utc or datetime.now(timezone.utc)
            candle_close_dt = datetime.fromtimestamp(close_time / 1000.0, tz=timezone.utc)
            signal_age_sec = (now_dt - candle_close_dt).total_seconds()
            is_fresh_signal = (0 <= signal_age_sec <= MAX_SIGNAL_AGE_SECONDS)

            if is_current_candle and not is_fresh_signal and sig in ["LONG", "SHORT"]:
                logger.warning(
                    f"Signal {sig} on current candle {ts} is stale ({signal_age_sec:.1f}s > {MAX_SIGNAL_AGE_SECONDS}s). "
                    "Entry blocked."
                )

            # If signal is actionable, genuinely current, fresh, and no position is open for this strategy
            if is_current_candle and is_fresh_signal and sig in ["LONG", "SHORT"] and key not in self.exec_engine.active_positions:
                quote = quotes.get(self.symbol)
                if not quote:
                    logger.warning(f"Valid signal {sig} on {self.symbol} but no fresh quote available. Entry deferred.")
                    continue

                # Position Sizing: 2.5% account equity risk based on 1.5x ATR stop distance
                eq = self.exec_engine.portfolio["equity"]
                stop_dist = abs(eval_res["entry_price"] - eval_res["stop_loss"])
                stop_pct = (stop_dist / eval_res["entry_price"]) if eval_res["entry_price"] > 0 else 0.015
                target_notional = (eq * RISK_PCT_PER_TRADE) / stop_pct
                capped_notional = min(target_notional, eq * MAX_LEVERAGE_CAP)
                quantity = capped_notional / eval_res["entry_price"]

                try:
                    self.exec_engine.open_position(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        side=sig,
                        quantity=quantity,
                        stop_loss=eval_res["stop_loss"],
                        take_profit=eval_res["take_profit"],
                        quote=quote
                    )
                except StaleQuoteError as e:
                    logger.error(f"Failed to enter position due to stale quote: {e}")

        # Update Benchmarks
        self._update_benchmarks(quotes)
        return evaluations_made

    def _update_benchmarks(self, quotes: Dict[str, Dict[str, Any]]):
        """Records continuous benchmark curves in SQLite."""
        now_str = datetime.now(timezone.utc).isoformat()
        current_px = quotes.get(self.symbol, {}).get("mid_price", 2500.0)

        with self._get_connection() as conn:
            cursor = conn.execute("SELECT eth_price FROM paper_benchmarks ORDER BY timestamp ASC LIMIT 1")
            first_row = cursor.fetchone()
            first_eth_px = first_row[0] if first_row else current_px
            eth_bh_equity = (current_px / first_eth_px * 10000.0) if first_eth_px > 0 else 10000.0

            conn.execute("""
            INSERT OR REPLACE INTO paper_benchmarks (
                timestamp, strategy_equity, eth_buy_and_hold_equity, cash_equity, eth_price
            ) VALUES (?, ?, ?, 10000.0, ?);
            """, (now_str, self.exec_engine.portfolio["equity"], eth_bh_equity, current_px))
            conn.commit()
