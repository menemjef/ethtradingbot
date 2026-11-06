# Prospective Public Market Data Collection Plan

**Repository**: `menemjef/ethtradingbot`  
**Working Branch**: `fix/execution-safety-and-backtest-integrity`  
**Scope**: Public Market Data Ingestion, Storage, Validation, and Maintenance  
**Target Instruments**: ETH/USDT and BTC/USDT Perpetual Futures (`ETHUSDT`, `BTCUSDT`)  

---

## 1. Motivation and Economic Rationale

### 1.1 The Microstructural Limitation of Historical REST APIs
While historical OHLCV klines and funding settlement rates can be retrieved retroactively over multi-year horizons, critical microstructural market features cannot:
- **Open Interest (OI)**: Binance public REST endpoints only provide rolling 30-day historical aggregates at 5m/15m granularity. Historical open interest from 2023 or 2024 is inaccessible without commercial data archives.
- **Taker Buy/Sell Volume**: Public endpoints restrict taker flow ratios to rolling 30-day lookback windows.
- **Order-Book Depth (L2 Bid/Ask Skew & Liquidity)**: Historical order book depth snapshots are not provided retrospectively via public REST.
- **Network Latency & Clock Drift**: Retroactive backtests cannot reconstruct the true delay between exchange matching engine event creation and client receipt.

Prospective, continuous, local collection of public market data is therefore strictly required to build an un-compromised, lookahead-free dataset for future microstructure and regime-detection research.

---

## 2. Architecture Overview

```
                          [ Binance USD-M Public WebSocket Streams ]
                                       |
                 +---------------------+---------------------+
                 |                                           |
         <symbol>@kline_1m                           <symbol>@bookTicker
         <symbol>@markPrice@1s                       <symbol>@depth5@100ms
                 |                                           |
                 +---------------------+---------------------+
                                       |
                     [ paper/collector.py: MarketDataCollector ]
                         - Dual-timestamp tagging (exchange_ts, receipt_ts)
                         - Heartbeat monitoring & auto-reconnect
                         - Sliding in-memory buffer (1,000 events)
                                       |
                 +---------------------+---------------------+
                 |                                           |
        [ SQLite Live State ]                       [ Parquet Cold Archive ]
     data/live_market/market.db                    data/historical_archive/
     - Current top-of-book                         - Daily partitioned Parquet
     - Completed 1m/15m/1h/4h bars                 - Compressed with zstd
     - Unsettled funding rates                     - Validated against REST
```

---

## 3. Data Streams and Ingestion Protocols

### 3.1 Real-Time WebSocket Streams (Binance USD-M Futures)
All streaming data is consumed exclusively via public, unauthenticated WebSocket endpoints:
- Base WebSocket URL: `wss://fstream.binance.com/ws` or `wss://fstream.binance.com/stream?streams=`

| Stream Name | Frequency | Target Fields | Research Purpose |
| :--- | :--- | :--- | :--- |
| `<symbol>@kline_1m` | Real-time (bar close) | Open, High, Low, Close, Volume, Taker Buy Vol, Trades Count | Primary OHLCV feed; bar completion signal for multi-timeframe aggregations. |
| `<symbol>@bookTicker` | Real-time (tick-level) | Best Bid Price, Best Bid Qty, Best Ask Price, Best Ask Qty | Exact realistic execution fill simulation (spread and top-of-book liquidity). |
| `<symbol>@markPrice@1s` | 1 second | Mark Price, Index Price, Funding Rate, Next Funding Time | Mark-to-market continuous equity tracking and liquidation distance calculation. |
| `<symbol>@depth5@100ms` | 100 milliseconds | Top 5 bids and asks (prices and quantities) | Liquidity imbalance, bid-ask depth ratio, and microstructural pressure. |

### 3.2 Scheduled REST Polling (Enrichment Streams)
Polled at regular intervals via public REST (`https://fapi.binance.com`):
- **Open Interest (`/fapi/v1/openInterest`)**: Polled every 5 minutes.
- **Long/Short Account Ratio (`/futures/data/topLongShortAccountRatio`)**: Polled every 15 minutes.
- **Taker Buy/Sell Volume Ratio (`/futures/data/takerlongshortRatio`)**: Polled every 15 minutes.
- **Funding Rate History (`/fapi/v1/fundingRate`)**: Polled every 8 hours following settlement (00:00, 08:00, 16:00 UTC).

---

## 4. Latency Tagging and Timestamp Hygiene

To eliminate lookahead bias and record network delays, **every ingested event is recorded with two distinct timestamps**:
1. `exchange_ts`: The millisecond timestamp assigned by the Binance matching engine (`E` or `T` in event JSON).
2. `receipt_ts`: The high-precision local UTC millisecond timestamp recorded immediately upon WebSocket frame decoding.

**Network Latency Metric**:
$$\text{Latency}_{\text{network}} = \text{receipt\_ts} - \text{exchange\_ts}$$
If $\text{Latency}_{\text{network}} > 500\text{ms}$ or $\text{Latency}_{\text{network}} < -100\text{ms}$ (indicating local clock drift), an alert is logged to `logs/market_collector.log`.

---

## 5. Storage Schema and Retention Strategy

### 5.1 Hot Tier: SQLite Database (`data/live_market/market_data.db`)
Optimized for high-speed local queries, restart recovery, and real-time strategy decision-making.

#### Schema 1: `completed_klines`
```sql
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
```

#### Schema 2: `orderbook_snapshots`
```sql
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    best_bid REAL NOT NULL,
    best_bid_qty REAL NOT NULL,
    best_ask REAL NOT NULL,
    best_ask_qty REAL NOT NULL,
    spread_bps REAL NOT NULL,
    bid_depth_usd REAL NOT NULL,
    ask_depth_usd REAL NOT NULL,
    exchange_ts INTEGER NOT NULL,
    receipt_ts INTEGER NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);
```

#### Schema 3: `funding_and_interest`
```sql
CREATE TABLE IF NOT EXISTS funding_and_interest (
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open_interest REAL NOT NULL,
    open_interest_usd REAL NOT NULL,
    funding_rate REAL NOT NULL,
    mark_price REAL NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);
```

### 5.2 Cold Tier: Parquet Daily Archive (`data/historical_archive/`)
At 00:05 UTC each day, the collector exports the prior day's data into columnar Parquet files compressed with `zstandard`:
- Path: `data/historical_archive/{year}/{month}/{symbol}_{timeframe}_{date}.parquet`
- Older raw tick data in SQLite is pruned after 30 days to keep the database size under 2 GB.

---

## 6. Fault Tolerance, Gap Detection, and Recovery

1. **Heartbeat & Reconnect**:
   - The WebSocket collector sends ping frames every 30 seconds.
   - If no message is received for 60 seconds, the connection is aborted and re-established within 2 seconds.
2. **Missing Bar Detection**:
   - On every completed kline received, the collector checks:
     $$\Delta t = \text{open\_time}_{\text{current}} - \text{open\_time}_{\text{previous}}$$
   - For 1m bars, if $\Delta t > 60,000\text{ms}$, a gap is flagged.
3. **REST Gap Backfilling**:
   - When a gap is detected, the collector initiates an immediate asynchronous REST call to `/fapi/v1/klines` requesting the missing range $[t_{\text{prev}} + 60\text{s}, t_{\text{curr}} - 60\text{s}]$, writes the missing bars to SQLite, and logs the reconciliation.
4. **Restart Safety**:
   - Upon system restart, the collector queries `MAX(close_time)` from `completed_klines` and queries REST klines from that timestamp forward to the present before resuming the WebSocket stream.
