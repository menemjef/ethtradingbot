"""
paper/config.py — Configuration for Live Paper-Trading & Market Data Ingestion.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "live_market").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "paper_trading").mkdir(parents=True, exist_ok=True)

# Database Paths
MARKET_DB_PATH = str(DATA_DIR / "live_market" / "market_data.db")
PAPER_DB_PATH = str(DATA_DIR / "paper_trading" / "paper_state.db")

# Ingestion Settings
DEFAULT_SYMBOLS = ["ETHUSDT", "BTCUSDT"]
WS_STREAM_URL = "wss://fstream.binance.com/stream"
REST_BASE_URL = "https://fapi.binance.com"
PING_INTERVAL_SEC = 20
PING_TIMEOUT_SEC = 10
RECONNECT_DELAY_SEC = 2.0
MAX_RECONNECT_DELAY_SEC = 60.0

# Paper Trading Capital and Risk Parameters
INITIAL_CAPITAL_USD = 10000.0
RISK_PCT_PER_TRADE = 0.025  # 2.5% account equity risk per trade
MAX_LEVERAGE_CAP = 3.0      # Hard ceiling: 3x account equity notional
TAKER_FEE_RATE = 0.0005     # 0.05% (5 bps) taker fee on entry and exit
SLIPPAGE_RATE = 0.0003      # 0.03% (3 bps) conservative adverse slippage on entry and exit

# Candidate A Strategy Parameters (Frozen)
CANDIDATE_A_TIMEFRAME = "4h"
CANDIDATE_A_ATR_PERIOD = 14
CANDIDATE_A_EMA_FAST = 20
CANDIDATE_A_EMA_SLOW = 50
CANDIDATE_A_VOL_SMA = 20
CANDIDATE_A_VOL_RATIO_MIN = 1.30
CANDIDATE_A_STOP_ATR_MULT = 1.50
CANDIDATE_A_TARGET_ATR_MULT = 3.00
CANDIDATE_A_MAX_HOLDING_HOURS = 72

# SAFETY INVARIANTS (STRICT ENFORCEMENT)
# Live real orders are structurally impossible in this module.
REAL_TRADING_DISABLED = True
NO_API_KEYS_REQUIRED = True
PUBLIC_ENDPOINTS_ONLY = True
