"""
paper/strategy_runner.py — Forward Paper-Trading Runner for Candidate A.

Executes Exp1_Momentum_Continuation_4H in forward paper mode.
Logs ALL evaluated signals (including rejected ones with explicit reasons) to SQLite,
tracks benchmark curves (Buy & Hold ETH, Cash, Candidate A), and manages lifecycle.
"""

import os
import sys
import json
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from paper.config import (
    MARKET_DB_PATH,
    PAPER_DB_PATH,
    CANDIDATE_A_TIMEFRAME,
    CANDIDATE_A_ATR_PERIOD,
    CANDIDATE_A_EMA_FAST,
    CANDIDATE_A_EMA_SLOW,
    CANDIDATE_A_VOL_SMA,
    CANDIDATE_A_VOL_RATIO_MIN,
    CANDIDATE_A_STOP_ATR_MULT,
    CANDIDATE_A_TARGET_ATR_MULT,
    CANDIDATE_A_MAX_HOLDING_HOURS,
    RISK_PCT_PER_TRADE,
    MAX_LEVERAGE_CAP
)
from paper.execution import PaperExecutionEngine

logger = logging.getLogger("paper.strategy_runner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] paper.strategy_runner: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class CandidateAPaperRunner:
    """
    Forward paper-trading orchestrator for Candidate A (Exp1_Momentum_Continuation_4H).
    """

    def __init__(
        self,
        execution_engine: PaperExecutionEngine,
        market_db_path: str = MARKET_DB_PATH,
        paper_db_path: str = PAPER_DB_PATH,
        symbol: str = "ETHUSDT"
    ):
        self.exec_engine = execution_engine
        self.market_db_path = market_db_path
        self.paper_db_path = paper_db_path
        self.symbol = symbol.upper()
        self.strategy_name = "Exp1_Momentum_Continuation_4H"
        self._init_signal_table()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.paper_db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_signal_table(self):
        """Initializes paper_signals table to log ALL evaluations (both valid and rejected)."""
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                rejection_reason TEXT,
                close_price REAL NOT NULL,
                ema_50 REAL,
                vol_ratio REAL,
                atr REAL,
                features_json TEXT NOT NULL
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_benchmarks (
                timestamp TEXT PRIMARY KEY,
                strategy_equity REAL NOT NULL,
                eth_buy_and_hold_equity REAL NOT NULL,
                cash_equity REAL NOT NULL,
                eth_price REAL NOT NULL
            );
            """)
            conn.commit()

    def log_signal(
        self,
        timestamp: str,
        signal_type: str,
        close_price: float,
        rejection_reason: Optional[str] = None,
        ema_50: Optional[float] = None,
        vol_ratio: Optional[float] = None,
        atr: Optional[float] = None,
        features: Optional[dict] = None
    ):
        """Logs a signal evaluation into SQLite."""
        feat_str = json.dumps(features or {})
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO paper_signals (
                timestamp, symbol, strategy, signal_type, rejection_reason,
                close_price, ema_50, vol_ratio, atr, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                timestamp, self.symbol, self.strategy_name, signal_type,
                rejection_reason, close_price, ema_50, vol_ratio, atr, feat_str
            ))
            conn.commit()

    def load_recent_klines_from_db(self, min_bars: int = 150) -> pd.DataFrame:
        """Loads completed 1m klines from market_data.db and resamples to 4H candles, combining with historical cache if needed."""
        df_1m = pd.DataFrame()
        if os.path.exists(self.market_db_path):
            conn = sqlite3.connect(self.market_db_path)
            try:
                query = """
                SELECT open_time, open, high, low, close, volume
                FROM completed_klines
                WHERE symbol = ? AND timeframe = '1m'
                ORDER BY open_time ASC
                """
                df_1m = pd.read_sql_query(query, conn, params=(self.symbol,))
            finally:
                conn.close()

        df_4h_live = pd.DataFrame()
        if not df_1m.empty:
            df_1m["timestamp"] = pd.to_datetime(df_1m["open_time"], unit="ms", utc=True)
            df_1m.set_index("timestamp", inplace=True)
            df_4h_live = df_1m.resample("4h", closed="left", label="left").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()

        # If live bars are insufficient for indicators (need at least 60 4H bars), supplement with historical cache
        if len(df_4h_live) < 60:
            hist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "eth_usdt_1h_3yr.parquet")
            if os.path.exists(hist_path):
                df_hist_1h = pd.read_parquet(hist_path)
                if not isinstance(df_hist_1h.index, pd.DatetimeIndex):
                    df_hist_1h.index = pd.to_datetime(df_hist_1h.index, utc=True)
                df_hist_4h = df_hist_1h.resample("4h", closed="left", label="left").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                }).dropna().tail(200)

                if not df_4h_live.empty:
                    combined = pd.concat([df_hist_4h[df_hist_4h.index < df_4h_live.index[0]], df_4h_live])
                    return combined
                else:
                    return df_hist_4h

        return df_4h_live

    def evaluate_candidate_a(self, df_4h: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates Candidate A (Exp1_Momentum_Continuation_4H) logic on 4H candles:
        - 4H Close > EMA50 (Long) or Close < EMA50 (Short)
        - 3 consecutive pullback candles (three red for Long, three green for Short)
        - Engulfing candle confirmation with vol_ratio > 1.30
        """
        if len(df_4h) < 60:
            return {
                "signal": "NO_SIGNAL",
                "reason": f"Insufficient bars ({len(df_4h)} < 60 required for EMA50 and ATR14)"
            }

        # Calculate indicators
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

        # Candidate A exact conditions on latest completed bar (index -1)
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
            "vol_ratio": c_vol_ratio,
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

        return {"signal": "NO_SIGNAL", "reason": "No directional setup", "features": feat_dict}

    def process_candle_close(self, current_bar: Dict[str, Any]):
        """
        Invoked on each completed 4H or 1H candle:
        1. Checks active position for stop-loss, take-profit, or 72h horizon exit.
        2. Evaluates Candidate A for new entries.
        3. Records benchmarks and signal logs.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        current_px = float(current_bar.get("close", 0.0))
        high_px = float(current_bar.get("high", current_px))
        low_px = float(current_bar.get("low", current_px))

        # 1. Manage Open Position Exits
        pos = self.exec_engine.active_positions.get(self.symbol)
        if pos:
            # Check Stop Loss / Take Profit
            trigger = self.exec_engine.update_mark_to_market(self.symbol, current_px, high=high_px, low=low_px)
            if trigger:
                self.exec_engine.close_position(self.symbol, current_px, exit_reason=trigger)
            else:
                # Check 72-hour horizon expiry
                try:
                    t_ent = datetime.fromisoformat(pos["entry_time"])
                    t_now = datetime.fromisoformat(now_str)
                    hours_held = (t_now - t_ent).total_seconds() / 3600.0
                    if hours_held >= CANDIDATE_A_MAX_HOLDING_HOURS:
                        self.exec_engine.close_position(self.symbol, current_px, exit_reason="horizon_expiry")
                except Exception as e:
                    logger.error(f"Error checking horizon expiry: {e}")

        # 2. Evaluate Signals (if FLAT)
        if self.symbol not in self.exec_engine.active_positions:
            df_4h = self.load_recent_klines_from_db()
            eval_res = self.evaluate_candidate_a(df_4h)
            sig = eval_res["signal"]

            self.log_signal(
                timestamp=now_str,
                signal_type=sig,
                close_price=current_px,
                rejection_reason=eval_res.get("reason"),
                ema_50=eval_res.get("features", {}).get("ema_50"),
                vol_ratio=eval_res.get("features", {}).get("vol_ratio"),
                atr=eval_res.get("features", {}).get("atr_14"),
                features=eval_res.get("features")
            )

            if sig in ["LONG", "SHORT"]:
                # Position Sizing: 2.5% account equity risk based on 1.5x ATR stop distance
                eq = self.exec_engine.portfolio["equity"]
                stop_dist = abs(eval_res["entry_price"] - eval_res["stop_loss"])
                stop_pct = (stop_dist / eval_res["entry_price"]) if eval_res["entry_price"] > 0 else 0.015
                target_notional = (eq * RISK_PCT_PER_TRADE) / stop_pct
                capped_notional = min(target_notional, eq * MAX_LEVERAGE_CAP)
                quantity = capped_notional / eval_res["entry_price"]

                self.exec_engine.open_position(
                    symbol=self.symbol,
                    side=sig,
                    price=eval_res["entry_price"],
                    quantity=quantity,
                    stop_loss=eval_res["stop_loss"],
                    take_profit=eval_res["take_profit"],
                    entry_time_str=now_str
                )
        else:
            # Position already open -> log as REJECTED due to active position
            self.log_signal(
                timestamp=now_str,
                signal_type="REJECTED",
                close_price=current_px,
                rejection_reason="Position already open"
            )

        # 3. Record Benchmarks
        with self._get_connection() as conn:
            # Buy and hold ETH benchmark: normalized to initial capital
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
