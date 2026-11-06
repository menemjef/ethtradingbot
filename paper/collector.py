"""
paper/collector.py — Public Market Data WebSocket & REST Collector.

Implements dual-channel Binance USD-M WebSocket streams:
- MARKET: wss://fstream.binance.com/market/stream (klines, markPrice)
- PUBLIC: wss://fstream.binance.com/public/stream (bookTicker)

Features:
- Independent connection supervision and auto-reconnect with exponential backoff.
- Stale-feed detection and per-stream health reporting (verifying actual event freshness).
- Dual timestamps (exchange_ts, receipt_ts) on all ingested records.
- 1-second sampled bookTicker snapshots with queue imbalance and in-memory L1 OFI proxy.
- Open interest polling (/fapi/v1/openInterest and /futures/data/openInterestHist rolling 30d).
- Realized funding settlements (/fapi/v1/fundingRate).
- Dynamic exchange contract metadata (/fapi/v1/exchangeInfo and /fapi/v1/fundingInfo).
- 1m kline gap detection and REST backfilling.
"""

import os
import sys
import time
import json
import sqlite3
import logging
import asyncio
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from paper.config import (
    MARKET_DB_PATH,
    DEFAULT_SYMBOLS,
    WS_MARKET_URL,
    WS_PUBLIC_URL,
    REST_BASE_URL,
    PING_INTERVAL_SEC,
    PING_TIMEOUT_SEC,
    RECONNECT_DELAY_SEC,
    MAX_RECONNECT_DELAY_SEC,
    MAX_EVENT_STALE_SECONDS,
    MAX_KLINE_STALE_SECONDS,
    MAX_QUOTE_AGE_SECONDS,
    OPEN_INTEREST_POLL_INTERVAL_SEC,
    EXCHANGE_INFO_POLL_INTERVAL_SEC,
    CONTRACT_SPECS_FALLBACK
)

logger = logging.getLogger("paper.collector")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] paper.collector: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MarketDataCollector:
    """
    Public market data collector for Binance USD-M perpetual futures.
    Requires ZERO API keys. Uses only public REST and WebSocket endpoints.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        db_path: str = MARKET_DB_PATH,
        market_ws_url: str = WS_MARKET_URL,
        public_ws_url: str = WS_PUBLIC_URL,
        rest_url: str = REST_BASE_URL
    ):
        self.symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        self.db_path = db_path
        self.market_ws_url = market_ws_url
        self.public_ws_url = public_ws_url
        self.rest_url = rest_url
        self.running = False

        # In-memory latest state cache
        self.latest_prices: Dict[str, Dict[str, Any]] = {}
        self.latest_mark_prices: Dict[str, Dict[str, Any]] = {}
        self.last_kline_open_time: Dict[str, int] = {}
        self.contract_specs: Dict[str, Dict[str, Any]] = {}

        # 1-second snapshot throttle cache: {symbol: last_snapshot_sec}
        self.last_book_snapshot_sec: Dict[str, int] = {}
        self.last_mark_snapshot_sec: Dict[str, int] = {}

        # Order Flow Imbalance (OFI) in-memory state:
        # {symbol: {"prev_bid_px": float, "prev_bid_qty": float, "prev_ask_px": float, "prev_ask_qty": float, "ofi": float}}
        self.ofi_state: Dict[str, Dict[str, Any]] = {}

        # Connection health and supervision metrics
        self.stream_health = {
            "market": {
                "connected": False,
                "reconnect_count": 0,
                "last_receipt_ts": 0,
                "last_exchange_ts": 0,
                "last_error": None
            },
            "public": {
                "connected": False,
                "reconnect_count": 0,
                "last_receipt_ts": 0,
                "last_exchange_ts": 0,
                "last_error": None
            }
        }
        self.symbol_health: Dict[str, Dict[str, Any]] = {
            s: {
                "last_kline_receipt_ts": 0,
                "last_kline_open_time": 0,
                "last_kline_exchange_ts": 0,
                "last_market_exchange_ts": 0,
                "last_book_receipt_ts": 0,
                "last_book_exchange_ts": 0,
                "last_mark_receipt_ts": 0,
                "last_mark_exchange_ts": 0,
                "last_oi_poll_ts": 0
            } for s in self.symbols
        }

        self._init_db()
        self._load_cached_contract_specs()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        """Initializes SQLite schema for market data persistence."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            # 1. Completed Klines table (1m bars)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS completed_klines (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                close_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                taker_buy_volume REAL NOT NULL,
                trades_count INTEGER NOT NULL,
                exchange_ts INTEGER NOT NULL,
                receipt_ts INTEGER NOT NULL,
                PRIMARY KEY (symbol, timeframe, open_time)
            );
            """)

            # 2. One-second sampled BookTicker Snapshots with queue imbalance & L1 OFI
            conn.execute("""
            CREATE TABLE IF NOT EXISTS book_ticker_snapshots (
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                best_bid REAL NOT NULL,
                best_bid_qty REAL NOT NULL,
                best_ask REAL NOT NULL,
                best_ask_qty REAL NOT NULL,
                mid_price REAL NOT NULL,
                spread REAL NOT NULL,
                queue_imbalance REAL NOT NULL,
                ofi_accumulated REAL NOT NULL,
                exchange_ts INTEGER NOT NULL,
                receipt_ts INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );
            """)

            # 3. Mark Price & Funding Rate Observations
            conn.execute("""
            CREATE TABLE IF NOT EXISTS funding_and_mark_price (
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                mark_price REAL NOT NULL,
                index_price REAL NOT NULL,
                funding_rate REAL NOT NULL,
                next_funding_time INTEGER NOT NULL,
                exchange_ts INTEGER NOT NULL,
                receipt_ts INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );
            """)

            # 4. Open Interest table (Polled every 5 minutes)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS open_interest (
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open_interest REAL NOT NULL,
                open_interest_usd REAL,
                exchange_ts INTEGER NOT NULL,
                receipt_ts INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );
            """)

            # Migrate open_interest table if legacy schema has NOT NULL constraint on open_interest_usd
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(open_interest);")
            oi_info = {row[1]: row for row in cur.fetchall()}
            if "open_interest_usd" in oi_info and oi_info["open_interest_usd"][3] == 1:  # notnull == 1
                conn.execute("ALTER TABLE open_interest RENAME TO _legacy_open_interest;")
                conn.execute("""
                CREATE TABLE open_interest (
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open_interest REAL NOT NULL,
                    open_interest_usd REAL,
                    exchange_ts INTEGER NOT NULL,
                    receipt_ts INTEGER NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                );
                """)
                conn.execute("""
                INSERT INTO open_interest (
                    symbol, timestamp, open_interest, open_interest_usd, exchange_ts, receipt_ts
                )
                SELECT
                    symbol,
                    timestamp,
                    open_interest,
                    CASE WHEN open_interest_usd <= 0.0 THEN NULL ELSE open_interest_usd END,
                    exchange_ts,
                    receipt_ts
                FROM _legacy_open_interest;
                """)
                conn.execute("DROP TABLE _legacy_open_interest;")

            # 5. Realized Funding Settlements table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS funding_settlements (
                symbol TEXT NOT NULL,
                settlement_time INTEGER NOT NULL,
                funding_rate REAL NOT NULL,
                mark_price REAL,
                PRIMARY KEY (symbol, settlement_time)
            );
            """)

            # 6. Exchange Contract Specifications & Metadata table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS exchange_metadata (
                symbol TEXT PRIMARY KEY,
                tick_size REAL NOT NULL,
                step_size REAL NOT NULL,
                min_qty REAL NOT NULL,
                min_notional REAL NOT NULL,
                max_notional REAL NOT NULL,
                funding_interval_hours INTEGER NOT NULL,
                funding_cap REAL,
                funding_floor REAL,
                updated_at INTEGER NOT NULL
            );
            """)
            conn.commit()

    def _load_cached_contract_specs(self):
        """Loads contract specifications from SQLite or falls back to defaults."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT symbol, tick_size, step_size, min_qty, min_notional, max_notional, funding_interval_hours FROM exchange_metadata")
            rows = cur.fetchall()
            for r in rows:
                self.contract_specs[r[0]] = {
                    "tick_size": float(r[1]),
                    "step_size": float(r[2]),
                    "min_qty": float(r[3]),
                    "min_notional": float(r[4]),
                    "max_notional": float(r[5]),
                    "funding_interval_hours": int(r[6])
                }

        # Apply fallback for any missing symbol
        for s in self.symbols:
            if s not in self.contract_specs:
                self.contract_specs[s] = CONTRACT_SPECS_FALLBACK.get(s, {
                    "tick_size": 0.01,
                    "step_size": 0.001,
                    "min_qty": 0.001,
                    "min_notional": 5.0,
                    "max_notional": 500000.0,
                    "funding_interval_hours": 8
                })

    # =========================================================================
    # REST API Ingestion Methods (Public Endpoints Only)
    # =========================================================================

    def _http_get_json(self, endpoint: str, timeout: float = 10.0) -> Optional[Any]:
        """Performs public GET request to Binance REST endpoint."""
        url = f"{self.rest_url}{endpoint}" if endpoint.startswith("/") else endpoint
        req = urllib.request.Request(url, headers={"User-Agent": "ethtradingbot/paper-collector"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"HTTP GET error for {url}: {e}")
            return None

    def get_server_time(self) -> Optional[int]:
        """
        Retrieves official Binance server time via /fapi/v1/time REST endpoint.
        Returns None if the REST endpoint is unreachable or fails, strictly avoiding
        silent fallback to local time or stale market stream timestamps.
        """
        data = self._http_get_json("/fapi/v1/time")
        if data and isinstance(data, dict) and "serverTime" in data:
            try:
                return int(data["serverTime"])
            except (ValueError, TypeError):
                pass
        logger.warning("Could not obtain verified Binance server time from /fapi/v1/time REST endpoint.")
        return None

    def fetch_exchange_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Polls /fapi/v1/exchangeInfo and /fapi/v1/fundingInfo to refresh contract specs.
        Handles unavailable or stale metadata safely by retaining cached fallbacks.
        """
        ex_info = self._http_get_json("/fapi/v1/exchangeInfo")
        if not ex_info or "symbols" not in ex_info:
            logger.warning("Could not refresh exchangeInfo; using cached contract specs.")
            return self.contract_specs

        funding_info = self._http_get_json("/fapi/v1/fundingInfo") or []
        funding_map = {item["symbol"]: item for item in funding_info if "symbol" in item}

        now_ms = int(time.time() * 1000)
        updated_specs = {}

        for sym_data in ex_info.get("symbols", []):
            s = sym_data.get("symbol")
            if s in self.symbols:
                tick_size = 0.01
                step_size = 0.001
                min_qty = 0.001
                min_notional = 5.0
                max_notional = 1000000.0

                for f in sym_data.get("filters", []):
                    ft = f.get("filterType")
                    if ft == "PRICE_FILTER":
                        tick_size = float(f.get("tickSize", tick_size))
                    elif ft == "LOT_SIZE":
                        step_size = float(f.get("stepSize", step_size))
                        min_qty = float(f.get("minQty", min_qty))
                    elif ft == "MIN_NOTIONAL":
                        min_notional = float(f.get("notional", min_notional))
                    elif ft == "MARKET_LOT_SIZE":
                        max_notional = float(f.get("maxQty", max_notional)) * 1000.0

                f_info = funding_map.get(s, {})
                interval_hours = int(f_info.get("fundingIntervalHours", 8))
                cap = float(f_info.get("adjustedFundingRateCap", 0.75)) if f_info.get("adjustedFundingRateCap") else None
                floor = float(f_info.get("adjustedFundingRateFloor", -0.75)) if f_info.get("adjustedFundingRateFloor") else None

                spec = {
                    "symbol": s,
                    "tick_size": tick_size,
                    "step_size": step_size,
                    "min_qty": min_qty,
                    "min_notional": min_notional,
                    "max_notional": max_notional,
                    "funding_interval_hours": interval_hours,
                    "funding_cap": cap,
                    "funding_floor": floor,
                    "updated_at": now_ms
                }
                updated_specs[s] = spec
                self.contract_specs[s] = spec

        # Persist to SQLite
        if updated_specs:
            with self._get_connection() as conn:
                for s, sp in updated_specs.items():
                    conn.execute("""
                    INSERT OR REPLACE INTO exchange_metadata (
                        symbol, tick_size, step_size, min_qty, min_notional, max_notional,
                        funding_interval_hours, funding_cap, funding_floor, updated_at
                    ) VALUES (:symbol, :tick_size, :step_size, :min_qty, :min_notional, :max_notional,
                              :funding_interval_hours, :funding_cap, :funding_floor, :updated_at);
                    """, sp)
                conn.commit()
            logger.info(f"Updated contract specifications for {list(updated_specs.keys())}.")
        return self.contract_specs

    def fetch_open_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Polls public /fapi/v1/openInterest for base asset open interest and notional.
        """
        data = self._http_get_json(f"/fapi/v1/openInterest?symbol={symbol}")
        if not data or "openInterest" not in data:
            return None

        receipt_ts = int(time.time() * 1000)
        oi_val = float(data.get("openInterest", 0.0))
        oi_record = {
            "symbol": symbol,
            "timestamp": int(data.get("time", receipt_ts)),
            "open_interest": oi_val,
            "open_interest_usd": None,  # Populated only with fresh valid mark price
            "exchange_ts": int(data.get("time", receipt_ts)),
            "receipt_ts": receipt_ts
        }

        # Resolve fresh, valid mark price for USD conversion
        mp = None
        # 1. Check in-memory cached mark price
        cached_mp = self.latest_mark_prices.get(symbol, {})
        cached_px = float(cached_mp.get("mark_price", 0.0))
        cached_ts = int(cached_mp.get("receipt_ts", 0))
        if cached_px > 0 and (receipt_ts - cached_ts) <= (MAX_EVENT_STALE_SECONDS * 1000):
            mp = cached_px
        else:
            # 2. Check funding_and_mark_price table in SQLite for recent fresh mark price
            try:
                with self._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT mark_price, receipt_ts FROM funding_and_mark_price WHERE symbol = ? ORDER BY receipt_ts DESC LIMIT 1",
                        (symbol,)
                    )
                    row = cur.fetchone()
                    if row and row[0] and float(row[0]) > 0:
                        db_px, db_ts = float(row[0]), int(row[1])
                        if (receipt_ts - db_ts) <= (MAX_EVENT_STALE_SECONDS * 1000):
                            mp = db_px
            except Exception as e:
                logger.debug(f"Could not read mark price from funding_and_mark_price for {symbol}: {e}")

        # Preserve base open interest; calculate USD only if fresh mark price is valid
        if mp is not None and mp > 0:
            oi_record["open_interest_usd"] = oi_val * mp
        else:
            # Never persist 0.0 notional when mark price is unavailable; persist None (NULL in DB)
            oi_record["open_interest_usd"] = None

        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO open_interest (
                symbol, timestamp, open_interest, open_interest_usd, exchange_ts, receipt_ts
            ) VALUES (:symbol, :timestamp, :open_interest, :open_interest_usd, :exchange_ts, :receipt_ts);
            """, oi_record)
            conn.commit()

        self.symbol_health[symbol]["last_oi_poll_ts"] = receipt_ts
        return oi_record

    def backfill_open_interest_history(self, symbol: str, limit_days: int = 30) -> int:
        """
        Backfills available open interest history from /futures/data/openInterestHist.
        NOTE: This endpoint strictly enforces a rolling 30-day historical limit.
        It does NOT provide multi-year history.
        """
        clamped_days = min(30, max(1, limit_days))
        url = f"/futures/data/openInterestHist?symbol={symbol}&period=15m&limit=500"
        data = self._http_get_json(url)
        if not data or not isinstance(data, list):
            logger.warning(f"Could not backfill openInterestHist for {symbol}.")
            return 0

        receipt_ts = int(time.time() * 1000)
        records = []
        for row in data:
            usd_val = float(row.get("sumOpenInterestValue", 0.0))
            records.append({
                "symbol": symbol,
                "timestamp": int(row.get("timestamp", 0)),
                "open_interest": float(row.get("sumOpenInterest", 0.0)),
                "open_interest_usd": usd_val if usd_val > 0.0 else None,
                "exchange_ts": int(row.get("timestamp", 0)),
                "receipt_ts": receipt_ts
            })

        if records:
            with self._get_connection() as conn:
                conn.executemany("""
                INSERT OR REPLACE INTO open_interest (
                    symbol, timestamp, open_interest, open_interest_usd, exchange_ts, receipt_ts
                ) VALUES (:symbol, :timestamp, :open_interest, :open_interest_usd, :exchange_ts, :receipt_ts);
                """, records)
                conn.commit()
            logger.info(f"Backfilled {len(records)} rolling 30d open interest snapshots for {symbol}.")
        return len(records)

    def fetch_recent_funding_settlements(self, symbol: str, limit: int = 100) -> int:
        """
        Retrieves actual realized settlement rates from /fapi/v1/fundingRate.
        Uses exchange settlement timestamps to ensure exact single-application accounting.
        """
        url = f"/fapi/v1/fundingRate?symbol={symbol}&limit={limit}"
        data = self._http_get_json(url)
        if not data or not isinstance(data, list):
            return 0

        records = []
        for row in data:
            records.append({
                "symbol": symbol,
                "settlement_time": int(row.get("fundingTime", 0)),
                "funding_rate": float(row.get("fundingRate", 0.0)),
                "mark_price": float(row.get("markPrice", 0.0)) if row.get("markPrice") else None
            })

        if records:
            with self._get_connection() as conn:
                conn.executemany("""
                INSERT OR REPLACE INTO funding_settlements (
                    symbol, settlement_time, funding_rate, mark_price
                ) VALUES (:symbol, :settlement_time, :funding_rate, :mark_price);
                """, records)
                conn.commit()
            logger.info(f"Ingested {len(records)} funding settlements for {symbol}.")
        return len(records)

    def fetch_rest_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Fetches historical klines via public REST API.
        Used for gap recovery and historical backfills.
        """
        url = f"/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if start_time:
            url += f"&startTime={int(start_time)}"
        if end_time:
            url += f"&endTime={int(end_time)}"

        data = self._http_get_json(url)
        if not data or not isinstance(data, list):
            return []

        receipt_ts = int(time.time() * 1000)
        # Determine reliable symbol-specific exchange timing (do NOT use unrelated streams or symbols)
        sym_health = self.symbol_health.get(symbol, {})
        sym_market_ex = max(
            sym_health.get("last_market_exchange_ts", 0),
            sym_health.get("last_kline_exchange_ts", 0)
        )
        if sym_market_ex == 0 and len(self.symbols) == 1:
            sym_market_ex = self.stream_health.get("market", {}).get("last_exchange_ts", 0)
        ref_time_ms = sym_market_ex if sym_market_ex > 0 else receipt_ts

        klines = []
        n_rows = len(data)
        for i, row in enumerate(data):
            c_close_time = int(row[6])
            # Symbol-specific exchange timing and completed-kline confirmation:
            # 1. When symbol-specific market exchange time is available, candle MUST close before that exchange time
            if sym_market_ex > 0 and c_close_time >= sym_market_ex:
                logger.debug(f"[{symbol}] Skipping unfinished REST candle: close_time {c_close_time} >= symbol exchange_ts {sym_market_ex}")
                continue

            # 2. When symbol exchange time is not available, fallback to receipt_ts
            if sym_market_ex == 0 and c_close_time >= receipt_ts:
                logger.debug(f"[{symbol}] Skipping unfinished REST candle: close_time {c_close_time} >= receipt_ts {receipt_ts}")
                continue

            # 3. For the last candle in response, verify completion against reference time
            if i == n_rows - 1 and c_close_time >= ref_time_ms:
                logger.debug(f"[{symbol}] Skipping last unfinished REST candle: close_time {c_close_time} >= ref_time {ref_time_ms}")
                continue

            klines.append({
                "symbol": symbol,
                "timeframe": interval,
                "open_time": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": c_close_time,
                "quote_volume": float(row[7]),
                "trades_count": int(row[8]),
                "taker_buy_volume": float(row[9]),
                "exchange_ts": c_close_time,
                "receipt_ts": receipt_ts
            })
        return klines

    def save_klines(self, klines: List[Dict[str, Any]]) -> int:
        """Saves a batch of completed klines to SQLite with duplicate prevention."""
        if not klines:
            return 0
        with self._get_connection() as conn:
            conn.executemany("""
            INSERT OR REPLACE INTO completed_klines (
                symbol, timeframe, open_time, close_time, open, high, low, close,
                volume, quote_volume, taker_buy_volume, trades_count,
                exchange_ts, receipt_ts
            ) VALUES (
                :symbol, :timeframe, :open_time, :close_time, :open, :high, :low, :close,
                :volume, :quote_volume, :taker_buy_volume, :trades_count,
                :exchange_ts, :receipt_ts
            );
            """, klines)
            conn.commit()
        return len(klines)

    def recover_gaps_on_startup(self, lookback_minutes: int = 240):
        """
        Checks SQLite for the most recent completed kline per symbol.
        Backfills any gap between the database state and the current time.
        """
        now_ms = int(time.time() * 1000)
        with self._get_connection() as conn:
            for sym in self.symbols:
                cursor = conn.execute(
                    "SELECT MAX(close_time) FROM completed_klines WHERE symbol = ? AND timeframe = '1m'",
                    (sym,)
                )
                row = cursor.fetchone()
                last_close_ms = row[0] if (row and row[0] is not None) else None

                if last_close_ms is None:
                    # Initial bootstrap window
                    start_ms = now_ms - (lookback_minutes * 60 * 1000)
                    logger.info(f"Bootstrapping initial {lookback_minutes}m of {sym} klines via REST...")
                    kl = self.fetch_rest_klines(sym, interval="1m", start_time=start_ms, end_time=now_ms)
                    self.save_klines(kl)
                    logger.info(f"Bootstrapped {len(kl)} klines for {sym}.")
                elif now_ms - last_close_ms > 120_000:
                    gap_mins = (now_ms - last_close_ms) // 60_000
                    logger.warning(f"Startup gap detected for {sym}: missing ~{gap_mins} minutes. Backfilling...")
                    kl = self.fetch_rest_klines(sym, interval="1m", start_time=last_close_ms + 1, end_time=now_ms)
                    self.save_klines(kl)
                    logger.info(f"Backfilled {len(kl)} missing klines for {sym}.")

    # =========================================================================
    # WebSocket Event Processing (Deterministic & Pure)
    # =========================================================================

    def process_kline_event(self, sym: str, k: dict, receipt_ts: int) -> Optional[Dict[str, Any]]:
        """Processes a raw 1m kline WebSocket payload."""
        is_closed = k.get("x", False)
        open_time = int(k.get("t", 0))
        exchange_ts = int(k.get("T", 0))

        # Update stream and symbol health
        self.stream_health["market"]["last_receipt_ts"] = receipt_ts
        self.stream_health["market"]["last_exchange_ts"] = exchange_ts
        if sym in self.symbol_health:
            self.symbol_health[sym]["last_kline_receipt_ts"] = receipt_ts
            self.symbol_health[sym]["last_kline_open_time"] = open_time
            self.symbol_health[sym]["last_kline_exchange_ts"] = exchange_ts
            self.symbol_health[sym]["last_market_exchange_ts"] = max(
                self.symbol_health[sym].get("last_market_exchange_ts", 0),
                exchange_ts
            )

        # Gap detection
        last_t = self.last_kline_open_time.get(sym)
        if last_t is not None and is_closed:
            expected_next = last_t + 60_000
            if open_time > expected_next:
                missed_count = (open_time - expected_next) // 60_000
                logger.warning(f"Kline gap detected on stream for {sym}: missed {missed_count} bars. Backfilling...")
                missing = self.fetch_rest_klines(sym, interval="1m", start_time=expected_next, end_time=open_time - 1)
                self.save_klines(missing)

        if is_closed:
            self.last_kline_open_time[sym] = open_time
            record = {
                "symbol": sym,
                "timeframe": "1m",
                "open_time": open_time,
                "close_time": int(k.get("T", 0)),
                "open": float(k.get("o", 0.0)),
                "high": float(k.get("h", 0.0)),
                "low": float(k.get("l", 0.0)),
                "close": float(k.get("c", 0.0)),
                "volume": float(k.get("v", 0.0)),
                "quote_volume": float(k.get("q", 0.0)),
                "taker_buy_volume": float(k.get("V", 0.0)),
                "trades_count": int(k.get("n", 0)),
                "exchange_ts": exchange_ts,
                "receipt_ts": receipt_ts
            }
            self.save_klines([record])
            return record
        return None

    def process_book_ticker_event(self, sym: str, data: dict, receipt_ts: int) -> Optional[Dict[str, Any]]:
        """
        Processes a raw bookTicker WebSocket payload:
        - Updates in-memory latest quote and computes spread in bps.
        - Accumulates in-memory L1 Order Flow Imbalance (OFI).
        - Samples 1-second snapshots into SQLite `book_ticker_snapshots`.
        """
        b_bid = float(data.get("b", 0.0))
        b_bid_q = float(data.get("B", 0.0))
        b_ask = float(data.get("a", 0.0))
        b_ask_q = float(data.get("A", 0.0))
        exchange_ts = int(data.get("E", receipt_ts))

        # Update stream and symbol health
        self.stream_health["public"]["last_receipt_ts"] = receipt_ts
        self.stream_health["public"]["last_exchange_ts"] = exchange_ts
        if sym in self.symbol_health:
            self.symbol_health[sym]["last_book_receipt_ts"] = receipt_ts
            self.symbol_health[sym]["last_book_exchange_ts"] = exchange_ts

        spread = max(0.0, b_ask - b_bid)
        spread_bps = ((spread / b_bid) * 10000.0) if b_bid > 0 else 0.0
        mid_px = (b_bid + b_ask) / 2.0

        # Queue Imbalance = (B - A) / (B + A)
        tot_qty = b_bid_q + b_ask_q
        queue_imb = ((b_bid_q - b_ask_q) / tot_qty) if tot_qty > 0 else 0.0

        # In-memory L1 OFI Accumulation (Cont et al. 2014 framework on best bid/ask)
        if sym not in self.ofi_state:
            self.ofi_state[sym] = {
                "prev_bid_px": b_bid, "prev_bid_qty": b_bid_q,
                "prev_ask_px": b_ask, "prev_ask_qty": b_ask_q,
                "ofi": 0.0
            }
        else:
            prev = self.ofi_state[sym]
            # Bid side contribution
            if b_bid > prev["prev_bid_px"]:
                delta_bid = b_bid_q
            elif b_bid == prev["prev_bid_px"]:
                delta_bid = b_bid_q - prev["prev_bid_qty"]
            else:
                delta_bid = -prev["prev_bid_qty"]

            # Ask side contribution
            if b_ask < prev["prev_ask_px"]:
                delta_ask = -b_ask_q
            elif b_ask == prev["prev_ask_px"]:
                delta_ask = b_ask_q - prev["prev_ask_qty"]
            else:
                delta_ask = prev["prev_ask_qty"]

            event_ofi = delta_bid - delta_ask
            prev["ofi"] += event_ofi
            prev["prev_bid_px"] = b_bid
            prev["prev_bid_qty"] = b_bid_q
            prev["prev_ask_px"] = b_ask
            prev["prev_ask_qty"] = b_ask_q

        current_ofi = self.ofi_state[sym]["ofi"]

        ticker_info = {
            "symbol": sym,
            "best_bid": b_bid,
            "best_bid_qty": b_bid_q,
            "best_ask": b_ask,
            "best_ask_qty": b_ask_q,
            "mid_price": mid_px,
            "spread": round(spread, 4),
            "spread_bps": round(spread_bps, 2),
            "queue_imbalance": round(queue_imb, 4),
            "ofi_accumulated": round(current_ofi, 4),
            "exchange_ts": exchange_ts,
            "receipt_ts": receipt_ts
        }
        self.latest_prices[sym] = ticker_info

        # 1-Second Sampling into SQLite `book_ticker_snapshots`
        curr_sec = int(receipt_ts / 1000)
        last_sec = self.last_book_snapshot_sec.get(sym, 0)
        if curr_sec > last_sec:
            self.last_book_snapshot_sec[sym] = curr_sec
            with self._get_connection() as conn:
                conn.execute("""
                INSERT OR REPLACE INTO book_ticker_snapshots (
                    symbol, timestamp, best_bid, best_bid_qty, best_ask, best_ask_qty,
                    mid_price, spread, queue_imbalance, ofi_accumulated, exchange_ts, receipt_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    sym, curr_sec * 1000, b_bid, b_bid_q, b_ask, b_ask_q,
                    mid_px, spread, queue_imb, current_ofi, exchange_ts, receipt_ts
                ))
                conn.commit()
            return ticker_info
        return None

    def process_mark_price_event(self, sym: str, data: dict, receipt_ts: int) -> Optional[Dict[str, Any]]:
        """
        Processes a markPrice WebSocket payload:
        - Updates latest mark price cache.
        - Samples snapshots at 10-second intervals into `funding_and_mark_price`.
        """
        mark_px = float(data.get("p", 0.0))
        index_px = float(data.get("i", 0.0))
        funding_rate = float(data.get("r", 0.0))
        next_funding_time = int(data.get("T", 0))
        exchange_ts = int(data.get("E", receipt_ts))

        # Update stream and symbol health
        self.stream_health["market"]["last_receipt_ts"] = receipt_ts
        self.stream_health["market"]["last_exchange_ts"] = exchange_ts
        if sym in self.symbol_health:
            self.symbol_health[sym]["last_mark_receipt_ts"] = receipt_ts
            self.symbol_health[sym]["last_mark_exchange_ts"] = exchange_ts
            self.symbol_health[sym]["last_market_exchange_ts"] = max(
                self.symbol_health[sym].get("last_market_exchange_ts", 0),
                exchange_ts
            )

        mark_info = {
            "symbol": sym,
            "mark_price": mark_px,
            "index_price": index_px,
            "funding_rate": funding_rate,
            "next_funding_time": next_funding_time,
            "exchange_ts": exchange_ts,
            "receipt_ts": receipt_ts
        }
        self.latest_mark_prices[sym] = mark_info

        # Sample at 10-second intervals
        curr_10s = int(receipt_ts / 10000)
        last_10s = self.last_mark_snapshot_sec.get(sym, 0)
        if curr_10s > last_10s:
            self.last_mark_snapshot_sec[sym] = curr_10s
            with self._get_connection() as conn:
                conn.execute("""
                INSERT OR REPLACE INTO funding_and_mark_price (
                    symbol, timestamp, mark_price, index_price, funding_rate,
                    next_funding_time, exchange_ts, receipt_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    sym, curr_10s * 10000, mark_px, index_px, funding_rate,
                    next_funding_time, exchange_ts, receipt_ts
                ))
                conn.commit()
            return mark_info
        return None

    # =========================================================================
    # Feed Health & Supervision Reporting
    # =========================================================================

    def get_health_status(self) -> Dict[str, Any]:
        """
        Evaluates health of both WebSocket streams and all symbols.
        Strict requirement: A successful handshake alone is NOT proof of health;
        recent events must be verified against staleness thresholds.
        """
        now_ms = int(time.time() * 1000)

        # Evaluate Market Stream
        m_last = self.stream_health["market"]["last_receipt_ts"]
        m_age = (now_ms - m_last) / 1000.0 if m_last > 0 else 999.0
        m_connected = self.stream_health["market"]["connected"]
        if not m_connected:
            m_status = "DISCONNECTED"
        elif m_age > MAX_EVENT_STALE_SECONDS:
            m_status = "DEGRADED (STALE)"
        else:
            m_status = "HEALTHY"

        # Evaluate Public Stream
        p_last = self.stream_health["public"]["last_receipt_ts"]
        p_age = (now_ms - p_last) / 1000.0 if p_last > 0 else 999.0
        p_connected = self.stream_health["public"]["connected"]
        if not p_connected:
            p_status = "DISCONNECTED"
        elif p_age > MAX_EVENT_STALE_SECONDS:
            p_status = "DEGRADED (STALE)"
        else:
            p_status = "HEALTHY"

        overall_healthy = (m_status == "HEALTHY") and (p_status == "HEALTHY")

        sym_status = {}
        for s in self.symbols:
            sh = self.symbol_health.get(s, {})
            k_age = (now_ms - sh.get("last_kline_receipt_ts", 0)) / 1000.0 if sh.get("last_kline_receipt_ts", 0) > 0 else 999.0
            b_age = (now_ms - sh.get("last_book_receipt_ts", 0)) / 1000.0 if sh.get("last_book_receipt_ts", 0) > 0 else 999.0
            m_age_s = (now_ms - sh.get("last_mark_receipt_ts", 0)) / 1000.0 if sh.get("last_mark_receipt_ts", 0) > 0 else 999.0

            sym_status[s] = {
                "latest_bid": self.latest_prices.get(s, {}).get("best_bid"),
                "latest_ask": self.latest_prices.get(s, {}).get("best_ask"),
                "latest_mark": self.latest_mark_prices.get(s, {}).get("mark_price"),
                "quote_age_sec": round(b_age, 2),
                "kline_age_sec": round(k_age, 2),
                "is_quote_fresh": b_age <= MAX_QUOTE_AGE_SECONDS,
                "is_kline_fresh": k_age <= MAX_KLINE_STALE_SECONDS
            }

        return {
            "overall_healthy": overall_healthy,
            "market_stream": {
                "status": m_status,
                "connected": m_connected,
                "reconnect_count": self.stream_health["market"]["reconnect_count"],
                "last_event_age_sec": round(m_age, 1),
                "last_error": self.stream_health["market"]["last_error"]
            },
            "public_stream": {
                "status": p_status,
                "connected": p_connected,
                "reconnect_count": self.stream_health["public"]["reconnect_count"],
                "last_event_age_sec": round(p_age, 1),
                "last_error": self.stream_health["public"]["last_error"]
            },
            "symbols": sym_status
        }

    # =========================================================================
    # Asynchronous Dual-Stream Runner Loops
    # =========================================================================

    def _build_market_stream_url(self) -> str:
        """Constructs MARKET stream URL: kline_1m and markPrice@1s for all symbols."""
        streams = []
        for s in self.symbols:
            low = s.lower()
            streams.append(f"{low}@kline_1m")
            streams.append(f"{low}@markPrice@1s")
        return f"{self.market_ws_url}?streams={'/'.join(streams)}"

    def _build_public_stream_url(self) -> str:
        """Constructs PUBLIC stream URL: bookTicker for all symbols."""
        streams = []
        for s in self.symbols:
            low = s.lower()
            streams.append(f"{low}@bookTicker")
        return f"{self.public_ws_url}?streams={'/'.join(streams)}"

    async def _run_market_stream(self):
        """Supervised independent loop for Binance USD-M MARKET WebSocket."""
        import websockets

        backoff = RECONNECT_DELAY_SEC
        while self.running:
            url = self._build_market_stream_url()
            try:
                logger.info(f"Connecting MARKET stream: {url}")
                async with websockets.connect(
                    url,
                    ping_interval=PING_INTERVAL_SEC,
                    ping_timeout=PING_TIMEOUT_SEC,
                    close_timeout=5.0
                ) as ws:
                    self.stream_health["market"]["connected"] = True
                    self.stream_health["market"]["last_error"] = None
                    backoff = RECONNECT_DELAY_SEC
                    logger.info("MARKET WebSocket stream connected successfully.")

                    while self.running:
                        msg = await ws.recv()
                        receipt_ts = int(time.time() * 1000)
                        payload = json.loads(msg)
                        stream_name = payload.get("stream", "")
                        data = payload.get("data", {})

                        if "@kline_1m" in stream_name:
                            sym = stream_name.split("@")[0].upper()
                            self.process_kline_event(sym, data.get("k", {}), receipt_ts)
                        elif "@markPrice" in stream_name:
                            sym = stream_name.split("@")[0].upper()
                            self.process_mark_price_event(sym, data, receipt_ts)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.stream_health["market"]["connected"] = False
                self.stream_health["market"]["reconnect_count"] += 1
                self.stream_health["market"]["last_error"] = str(e)
                logger.warning(f"MARKET stream dropped: {e}. Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, MAX_RECONNECT_DELAY_SEC)

    async def _run_public_stream(self):
        """Supervised independent loop for Binance USD-M PUBLIC WebSocket."""
        import websockets

        backoff = RECONNECT_DELAY_SEC
        while self.running:
            url = self._build_public_stream_url()
            try:
                logger.info(f"Connecting PUBLIC stream: {url}")
                async with websockets.connect(
                    url,
                    ping_interval=PING_INTERVAL_SEC,
                    ping_timeout=PING_TIMEOUT_SEC,
                    close_timeout=5.0
                ) as ws:
                    self.stream_health["public"]["connected"] = True
                    self.stream_health["public"]["last_error"] = None
                    backoff = RECONNECT_DELAY_SEC
                    # Reset OFI state baseline on reconnect
                    for s in self.symbols:
                        if s in self.ofi_state:
                            self.ofi_state[s]["ofi"] = 0.0
                    logger.info("PUBLIC WebSocket stream connected successfully.")

                    while self.running:
                        msg = await ws.recv()
                        receipt_ts = int(time.time() * 1000)
                        payload = json.loads(msg)
                        stream_name = payload.get("stream", "")
                        data = payload.get("data", {})

                        if "@bookTicker" in stream_name:
                            sym = stream_name.split("@")[0].upper()
                            self.process_book_ticker_event(sym, data, receipt_ts)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.stream_health["public"]["connected"] = False
                self.stream_health["public"]["reconnect_count"] += 1
                self.stream_health["public"]["last_error"] = str(e)
                logger.warning(f"PUBLIC stream dropped: {e}. Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, MAX_RECONNECT_DELAY_SEC)

    async def _run_periodic_rest_tasks(self):
        """Background scheduled task for polling open interest, metadata, and funding settlements."""
        last_oi_poll = 0
        last_meta_poll = 0
        last_funding_poll = 0

        while self.running:
            try:
                now = time.time()

                # Poll Open Interest every 5 minutes
                if now - last_oi_poll >= OPEN_INTEREST_POLL_INTERVAL_SEC:
                    for s in self.symbols:
                        self.fetch_open_interest(s)
                    last_oi_poll = now

                # Refresh Exchange Metadata every hour
                if now - last_meta_poll >= EXCHANGE_INFO_POLL_INTERVAL_SEC:
                    self.fetch_exchange_metadata()
                    last_meta_poll = now

                # Ingest Funding Settlements every hour
                if now - last_funding_poll >= 3600:
                    for s in self.symbols:
                        self.fetch_recent_funding_settlements(s)
                    last_funding_poll = now

            except Exception as e:
                logger.error(f"Error in periodic REST poll loop: {e}")

            await asyncio.sleep(15.0)

    async def run_async_collector(self):
        """Runs the complete collection suite concurrently."""
        self.running = True
        logger.info(f"Starting MarketDataCollector for symbols: {self.symbols}...")

        # Startup bootstrap
        self.fetch_exchange_metadata()
        self.recover_gaps_on_startup()
        for s in self.symbols:
            self.fetch_recent_funding_settlements(s)
            self.fetch_open_interest(s)

        # Launch concurrent supervised tasks
        tasks = [
            asyncio.create_task(self._run_market_stream()),
            asyncio.create_task(self._run_public_stream()),
            asyncio.create_task(self._run_periodic_rest_tasks())
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Collector tasks cancelled. Shutting down cleanly.")
            self.running = False

    def stop(self):
        """Stops collector loops cleanly."""
        self.running = False
        self.stream_health["market"]["connected"] = False
        self.stream_health["public"]["connected"] = False
        logger.info("MarketDataCollector stopped.")
