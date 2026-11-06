"""
paper/config.py — Configuration for Live Paper-Trading & Market Data Ingestion.

Strictly enforces public-only endpoints, zero credentials, isolated stream routing,
and configurable quote/event freshness thresholds.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Non-negotiable Safety Invariants (Strictly Enforced)
TRADING_ENABLED = False
MOCK_TRADING = True
PRODUCTION_READY = False
USE_BINANCE_TESTNET = True

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "live_market").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "paper_trading").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "backups").mkdir(parents=True, exist_ok=True)

# Database & Lock Paths
MARKET_DB_PATH = str(DATA_DIR / "live_market" / "market_data.db")
PAPER_DB_PATH = str(DATA_DIR / "paper_trading" / "paper_state.db")
DAEMON_PID_PATH = str(DATA_DIR / "paper_trading" / "paper_daemon.pid")

# Public Binance USD-M WebSocket Stream URLs (Separate routing)
WS_MARKET_URL = "wss://fstream.binance.com/market/stream"
WS_PUBLIC_URL = "wss://fstream.binance.com/public/stream"
WS_STREAM_URL = "wss://fstream.binance.com/stream"  # Fallback multiplexed URL
REST_BASE_URL = "https://fapi.binance.com"

# Ingestion Symbols
DEFAULT_SYMBOLS = ["ETHUSDT", "BTCUSDT"]

# WebSocket & Connection Supervision Settings
PING_INTERVAL_SEC = 20
PING_TIMEOUT_SEC = 10
RECONNECT_DELAY_SEC = 2.0
MAX_RECONNECT_DELAY_SEC = 60.0

# Quote & Feed Freshness Thresholds (V4 Specification)
MAX_QUOTE_AGE_SECONDS = 5.0        # Quotes older than 5.0s are rejected as stale
MAX_EVENT_STALE_SECONDS = 15.0     # Feed flagged degraded if no event within 15s
MAX_KLINE_STALE_SECONDS = 120.0    # 1m kline feed flagged degraded if no bar within 120s
MAX_SIGNAL_AGE_SECONDS = 900.0     # Signals older than 15 minutes (900s) rejected as stale

# Polling Intervals for Public REST Metadata & Aggregates
OPEN_INTEREST_POLL_INTERVAL_SEC = 300    # 5 minutes
FUNDING_RATE_POLL_INTERVAL_SEC = 3600    # 1 hour (and checked at settlement)
EXCHANGE_INFO_POLL_INTERVAL_SEC = 3600   # 1 hour metadata cache refresh

# Paper Trading Capital, Fees & Risk Parameters
INITIAL_CAPITAL_USD = 10000.0
RISK_PCT_PER_TRADE = 0.025         # 2.5% account equity risk per trade
MAX_LEVERAGE_CAP = 3.0             # Hard ceiling: 3x account equity notional
TAKER_FEE_RATE = 0.0005            # 0.05% (5 bps) taker fee
SLIPPAGE_RATE = 0.0003             # 0.03% (3 bps) conservative adverse slippage
MAX_PORTFOLIO_LEVERAGE = 3.0       # Portfolio-level gross leverage ceiling
CIRCUIT_BREAKER_MAX_DRAWDOWN_PCT = 15.0  # Halt new entries if drawdown >= 15%

# Contract Specifications Fallbacks (Overridden dynamically by /fapi/v1/exchangeInfo)
CONTRACT_SPECS_FALLBACK = {
    "ETHUSDT": {
        "tick_size": 0.01,
        "step_size": 0.001,
        "min_qty": 0.001,
        "min_notional": 5.0,
        "max_notional": 500000.0,
        "funding_interval_hours": 8
    },
    "BTCUSDT": {
        "tick_size": 0.10,
        "step_size": 0.001,
        "min_qty": 0.001,
        "min_notional": 5.0,
        "max_notional": 1000000.0,
        "funding_interval_hours": 8
    }
}

# Candidate A Strategy Parameters (Frozen)
CANDIDATE_A_STRATEGY_ID = "CandidateA_Momentum_Continuation"
CANDIDATE_A_STRATEGY_VERSION = "1.0.0"
CANDIDATE_A_TIMEFRAME = "4h"
CANDIDATE_A_ATR_PERIOD = 14
CANDIDATE_A_EMA_FAST = 20
CANDIDATE_A_EMA_SLOW = 50
CANDIDATE_A_VOL_SMA = 20
CANDIDATE_A_VOL_RATIO_MIN = 1.30
CANDIDATE_A_STOP_ATR_MULT = 1.50
CANDIDATE_A_TARGET_ATR_MULT = 3.00
CANDIDATE_A_MAX_HOLDING_HOURS = 72

# 4-Hour Candle Strict Integrity Constants
CANDLE_4H_REQUIRED_1M_BARS = 240
EXPECTED_1M_INTERVAL_MS = 60000

# SAFETY INVARIANTS (STRICT ENFORCEMENT)
# Live real orders are structurally impossible in this module.
REAL_TRADING_DISABLED = True
NO_API_KEYS_REQUIRED = True
PUBLIC_ENDPOINTS_ONLY = True
