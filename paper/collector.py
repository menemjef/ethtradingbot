"""
paper/collector.py — Public Market Data WebSocket & REST Collector.

Ingests Binance USD-M public split-streams:
- <symbol>@kline_1m
- <symbol>@bookTicker
- <symbol>@markPrice@1s

Persists to SQLite with dual timestamps (exchange_ts, receipt_ts), performs
automatic gap detection, REST backfill recovery, and reconnection management.
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
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from paper.config import (
    MARKET_DB_PATH,
    DEFAULT_SYMBOLS,
    WS_STREAM_URL,
    REST_BASE_URL,
    PING_INTERVAL_SEC,
    PING_TIMEOUT_SEC,
    RECONNECT_DELAY_SEC,
    MAX_RECONNECT_DELAY_SEC
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
    Public market data collector for Binance USD-M futures.
    Requires ZERO API keys. Uses only public endpoints.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        db_path: str = MARKET_DB_PATH,
        ws_url: str = WS_STREAM_URL,
        rest_url: str = REST_BASE_URL
    ):
        self.symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        self.db_path = db_path
        self.ws_url = ws_url
        self.rest_url = rest_url
        self.running = False
        self._init_db()

        # In-memory latest state cache for zero-latency queries
        self.latest_prices: Dict[str, Dict[str, Any]] = {}
        self.latest_mark_prices: Dict[str, Dict[str, Any]] = {}
        self.last_kline_open_time: Dict[str, int] = {}

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        """Initializes tables for market data storage."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            # 1. Completed Klines table
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

            # 2. Orderbook & Ticker Snapshots table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                best_bid REAL NOT NULL,
                best_bid_qty REAL NOT NULL,
                best_ask REAL NOT NULL,
                best_ask_qty REAL NOT NULL,
                spread_bps REAL NOT NULL,
                exchange_ts INTEGER NOT NULL,
                receipt_ts INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );
            """)

            # 3. Mark Price & Funding Rate table
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
            conn.commit()

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
        url = f"{self.rest_url}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if start_time:
            url += f"&startTime={int(start_time)}"
        if end_time:
            url += f"&endTime={int(end_time)}"

        req = urllib.request.Request(url, headers={"User-Agent": "ethtradingbot/paper-collector"})
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to fetch REST klines for {symbol} ({url}): {e}")
            return []

        receipt_ts = int(time.time() * 1000)
        klines = []
        for row in data:
            klines.append({
                "symbol": symbol,
                "timeframe": interval,
                "open_time": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": int(row[6]),
                "quote_volume": float(row[7]),
                "trades_count": int(row[8]),
                "taker_buy_volume": float(row[9]),
                "exchange_ts": int(row[6]),
                "receipt_ts": receipt_ts
            })
        return klines

    def save_klines(self, klines: List[Dict[str, Any]]) -> int:
        """Saves a batch of klines to SQLite with conflict resolution."""
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

    def recover_gaps_on_startup(self, lookback_minutes: int = 120):
        """
        Checks SQLite for the most recent completed kline.
        If data is missing between the latest stored bar and now, backfills via REST.
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
                    # No local data, fetch initial bootstrap window
                    start_ms = now_ms - (lookback_minutes * 60 * 1000)
                    logger.info(f"Bootstrapping initial {lookback_minutes}m of {sym} klines via REST...")
                    kl = self.fetch_rest_klines(sym, interval="1m", start_time=start_ms, end_time=now_ms)
                    self.save_klines(kl)
                    logger.info(f"Bootstrapped {len(kl)} klines for {sym}.")
                elif now_ms - last_close_ms > 120_000:
                    # Gap detected greater than 2 minutes
                    gap_mins = (now_ms - last_close_ms) // 60_000
                    logger.warning(f"Startup gap detected for {sym}: missing ~{gap_mins} minutes. Backfilling...")
                    kl = self.fetch_rest_klines(sym, interval="1m", start_time=last_close_ms + 1, end_time=now_ms)
                    self.save_klines(kl)
                    logger.info(f"Backfilled {len(kl)} missing klines for {sym}.")

    def process_kline_event(self, sym: str, k: dict, receipt_ts: int) -> Optional[Dict[str, Any]]:
        """Processes a raw 1m kline WebSocket payload."""
        is_closed = k.get("x", False)
        open_time = int(k.get("t", 0))
        exchange_ts = int(k.get("T", 0))

        # Check for missed candle gap in stream
        last_t = self.last_kline_open_time.get(sym)
        if last_t is not None and is_closed:
            expected_next = last_t + 60_000
            if open_time > expected_next:
                missed_count = (open_time - expected_next) // 60_000
                logger.warning(f"Kline gap detected on stream for {sym}: missed {missed_count} bars between {last_t} and {open_time}. Triggering REST fill...")
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

    def process_book_ticker_event(self, sym: str, data: dict, receipt_ts: int):
        """Processes a raw bookTicker WebSocket payload."""
        b_bid = float(data.get("b", 0.0))
        b_bid_q = float(data.get("B", 0.0))
        b_ask = float(data.get("a", 0.0))
        b_ask_q = float(data.get("A", 0.0))
        exchange_ts = int(data.get("E", receipt_ts))

        spread_bps = ((b_ask - b_bid) / b_bid * 10000.0) if b_bid > 0 else 0.0

        ticker_info = {
            "symbol": sym,
            "best_bid": b_bid,
            "best_bid_qty": b_bid_q,
            "best_ask": b_ask,
            "best_ask_qty": b_ask_q,
            "mid_price": (b_bid + b_ask) / 2.0,
            "spread_bps": round(spread_bps, 2),
            "exchange_ts": exchange_ts,
            "receipt_ts": receipt_ts
        }
        self.latest_prices[sym] = ticker_info

    def process_mark_price_event(self, sym: str, data: dict, receipt_ts: int):
        """Processes a markPrice WebSocket payload."""
        mark_px = float(data.get("p", 0.0))
        index_px = float(data.get("i", 0.0))
        funding_rate = float(data.get("r", 0.0))
        next_funding_time = int(data.get("T", 0))
        exchange_ts = int(data.get("E", receipt_ts))

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

    def build_stream_url(self) -> str:
        """Constructs the multiplexed WebSocket stream URL."""
        streams = []
        for s in self.symbols:
            low = s.lower()
            streams.append(f"{low}@kline_1m")
            streams.append(f"{low}@bookTicker")
            streams.append(f"{low}@markPrice@1s")
        query = "/".join(streams)
        return f"{self.ws_url}?streams={query}"

    async def run_async_collector(self):
        """
        Asynchronous WebSocket loop with heartbeat, exponential backoff,
        and gap recovery.
        """
        import websockets

        self.running = True
        logger.info(f"Starting MarketDataCollector for {self.symbols}...")
        self.recover_gaps_on_startup()

        stream_url = self.build_stream_url()
        backoff = RECONNECT_DELAY_SEC

        while self.running:
            try:
                logger.info(f"Connecting to Binance Futures WS: {stream_url}")
                async with websockets.connect(
                    stream_url,
                    ping_interval=PING_INTERVAL_SEC,
                    ping_timeout=PING_TIMEOUT_SEC,
                    close_timeout=5.0
                ) as ws:
                    logger.info("Connected to Binance USD-M WebSocket split stream.")
                    backoff = RECONNECT_DELAY_SEC  # Reset backoff on successful connect

                    while self.running:
                        msg = await ws.recv()
                        receipt_ts = int(time.time() * 1000)
                        payload = json.loads(msg)

                        stream_name = payload.get("stream", "")
                        data = payload.get("data", {})

                        # Parse stream type and symbol
                        if "@kline_1m" in stream_name:
                            sym = stream_name.split("@")[0].upper()
                            self.process_kline_event(sym, data.get("k", {}), receipt_ts)
                        elif "@bookTicker" in stream_name:
                            sym = stream_name.split("@")[0].upper()
                            self.process_book_ticker_event(sym, data, receipt_ts)
                        elif "@markPrice" in stream_name:
                            sym = stream_name.split("@")[0].upper()
                            self.process_mark_price_event(sym, data, receipt_ts)

            except asyncio.CancelledError:
                logger.info("Collector cancelled. Shutting down gracefully.")
                self.running = False
                break
            except Exception as e:
                logger.warning(f"WebSocket connection dropped: {e}. Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, MAX_RECONNECT_DELAY_SEC)

    def stop(self):
        self.running = False
        logger.info("MarketDataCollector stopped.")
