"""
config.py — Central configuration for the ETH trading signal system.
All hyperparameters, paths, and constants live here.
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed locally, env vars set externally

# ─────────────────────────────────────────────────────────────────────────────
# Market Settings
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL = "ETH/USDT:USDT"          # CCXT perpetual futures symbol
WS_SYMBOL = "ethusdt"              # Binance WebSocket symbol (lowercase)
TIMEFRAME = "1h"
WS_BASE = "wss://fstream.binance.com"
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Data & Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK = 24                      # 1 day of 1H context (reduced to prevent overfitting)
FETCH_LIMIT = 20000                # ~2.3 years of 1H data
LIVE_FETCH_LIMIT = 3000            # Enough for indicator warm-up
TARGET_HORIZON = 12                # Horizontal barrier (12 hours max hold)
TARGET_TYPE = "TBM"                # Triple Barrier Method
TBM_PT_MULT = 2.0                  # Profit Taking multiplier (Symmetric for unbiased training)
TBM_SL_MULT = 2.0                  # Stop Loss multiplier (Symmetric for unbiased training)

# Candle-structure branch features (Branch 1)
CANDLE_FEATURES = [
    "open", "high", "low", "close",
    "body", "upper_wick", "lower_wick", "candle_range",
]

# Market-state branch features (Branch 2)
MARKET_FEATURES = [
    "log_return",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "ema_ratio_20_50", "ema_ratio_50_200",
    "atr_14",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "stoch_k", "stoch_d",
    "adx_14", "di_plus", "di_minus",
    "trend_flag",
    "dist_from_high_20", "dist_from_low_20",
    "log_volume", "volume_ema_ratio",
    "fisher_9", "roc_12",
    # Advanced S/R and Fibonacci
    "sup_24", "res_24",
    "sup_168", "res_168",
    "dist_sup_168", "dist_res_168",
    "fib_618_dist_180d", "range_pos_180d",
    # Advanced Quant Features
    "hmm_regime", "garch_vol", "arima_resid",
    # New: Volume/Price/Cloud/Order-Flow features
    "vwap_dist", "obv_momentum",
    "ichi_tk_diff", "ichi_cloud_dist", "ichi_cloud_width",
    "bbp", "vol_momentum",
    "cmf", "hist_vol_20"
]

CANDLE_DIM = len(CANDLE_FEATURES)
MARKET_DIM = len(MARKET_FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
DROPOUT = 0.5
ATTENTION_HEADS = 4
ATTENTION_DROPOUT = 0.3
TRANSFORMER_LAYERS = 1
TRANSFORMER_DIM = 64

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
MAX_EPOCHS = 300
PATIENCE = 30
TRAIN_SPLIT = 0.80

# Loss weights for multi-task objective
CLASSIFICATION_WEIGHT = 1.0
REGRESSION_WEIGHT = 1.0
TARGET_MULTIPLIER = 1.0
NOISE_INJECTION = 0.02
RECENCY_WEIGHT = 1.0               # Uniform sampling (no recency bias to prevent overfitting)

# ─────────────────────────────────────────────────────────────────────────────
# Signal Generation (Dynamic Modes)
# ─────────────────────────────────────────────────────────────────────────────
from utils.settings_manager import get_active_mode

TRADING_MODES = {
    "Expert": {
        "threshold": 0.0750,
        "description": "Sniper Mode (Top ~5% high-conviction signals)",
        "color": "#00f0ff"  # Cyan
    },
    "Professional": {
        "threshold": 0.0600,
        "description": "Balanced Stability (Top ~15% signals)",
        "color": "#10B981"  # Green
    },
    "Aggressive": {
        "threshold": 0.0450,
        "description": "Active Combat Mode (Top ~30% signals)",
        "color": "#EF4444"  # Red
    }
}

def get_current_threshold():
    """Get the threshold based on the selected mode in settings.json."""
    try:
        mode_name = get_active_mode()
        return TRADING_MODES.get(mode_name, TRADING_MODES["Professional"])["threshold"]
    except Exception:
        return 0.0010  # Fallback to Professional

SIGNAL_THRESHOLD = get_current_threshold()

# ─────────────────────────────────────────────────────────────────────────────
# Trading & Execution Settings (Fail-Safe Defaults)
# ─────────────────────────────────────────────────────────────────────────────
TRADING_ENABLED = False            # Fail-safe: Disabled by default
MOCK_TRADING = True               # Fail-safe: Paper trading by default
PRODUCTION_READY = False          # Fail-safe: Guard against unverified live runs
USE_BINANCE_TESTNET = True        # Fail-safe: Testnet by default
INITIAL_PAPER_BALANCE = 10000.0   # Persistent mock balance
LEVERAGE = 3                      # Conservative institutional default leverage (3x)
RISK_PER_TRADE_PERCENT = 2.5      # 2.5% risk per trade
POSITION_SIZE_FRACTION = 0.05     # 5% of account balance per trade
MAX_POSITION_SIZE_USD = None      # No hard cap — scales with account
DAILY_DRAWDOWN_LIMIT = 0.060       # 6.0% Daily Drawdown Hard Cap
MIN_NOTIONAL_USD = 21.0           # Minimum required by Binance ($20) + buffer

# Live Trading Explicit Opt-In Mechanism
LIVE_TRADING_ACK_TOKEN = "I_UNDERSTAND_REAL_MONEY_IS_AT_RISK"
ALLOW_LIVE_TRADING = os.getenv("ALLOW_LIVE_TRADING", "0")

# Risk Management Upgrades (Fixed Bracket Orders on Binance)
USE_ATR_SIZING = False
USE_FIXED_PERCENTAGE_TPSL = True
STOP_LOSS_PCT = 0.015                   # 1.5% fixed Stop Loss
TAKE_PROFIT_PCT = 0.030                  # 3.0% fixed Take Profit (2:1 Reward:Risk Ratio)
TRAILING_STOP_ENABLED = False            # Native Fixed Bracket Orders used instead of Trailing Stop
LONG_ONLY_MODE = False                   # Both LONG and SHORT trades enabled
BTC_CORRELATION_GUARD = False         # Disabled: Model handles direction
BTC_SYMBOL = "BTC/USDT:USDT"

# Phase 4: Model & Backtest Refinement
SECONDARY_TIMEFRAMES = ["15m", "4h"]
# Promote expanded features to default (Dynamic extension)
# Removed: These features cannot be fetched > 30 days historically via Binance API.
# Training the model on zeros and predicting on live massive values causes saturation.
# MARKET_FEATURES = MARKET_FEATURES + [
#     "fundingRate", "openInterestAmount", "oi_pct_change",
#     "rsi_15m", "rsi_4h"
# ]
MARKET_DIM = len(MARKET_FEATURES)   # Should be 42
CANDLE_DIM = len(CANDLE_FEATURES)   # 8
FETCH_ORDER_BOOK = False           # Disabled: Order Book Imbalance (reverts uncalibrated gate to match backtest & March version)
COMMISSION_RATE = 0.0005           # 0.05% per side
SLIPPAGE_RATE = 0.0001             # 0.01% average slippage per side

# Phase 3: Infrastructure Settings
DB_URL = os.getenv("DATABASE_URL", "sqlite:///bot_data.db")

# Telegram Alerts (Phase 1)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# API Credentials (Use environment variables in production)
# IMPORTANT: Never hardcode credentials - always use environment variables
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ─────────────────────────────────────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
SCALER_CANDLE_PATH = os.path.join(CHECKPOINT_DIR, "scaler_candle.pkl")
SCALER_MARKET_PATH = os.path.join(CHECKPOINT_DIR, "scaler_market.pkl")
SIGNALS_CSV   = os.path.join(LOG_DIR, "signals.csv")
TRADES_CSV    = os.path.join(LOG_DIR, "trades.csv")
ACTIVITY_CSV  = os.path.join(LOG_DIR, "bot_activity.csv")

# Create dirs if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_config():
    """Validate critical configuration values."""
    errors = []
    warnings = []
    
    # Validate numeric ranges
    if not 0 < LEVERAGE <= 125:
        errors.append(f"Invalid LEVERAGE: {LEVERAGE} (must be 1-125)")
    
    if not 0 < RISK_PER_TRADE_PERCENT <= 100:
        errors.append(f"Invalid RISK_PER_TRADE_PERCENT: {RISK_PER_TRADE_PERCENT}")
    
    if not LOOKBACK > 0:
        errors.append(f"Invalid LOOKBACK: {LOOKBACK}")
    
    if SIGNAL_THRESHOLD < 0:
        errors.append(f"Invalid SIGNAL_THRESHOLD: {SIGNAL_THRESHOLD}")
    
    # Enforce strict safety guardrails for live trading
    if not MOCK_TRADING:
        if ALLOW_LIVE_TRADING != LIVE_TRADING_ACK_TOKEN:
            errors.append(
                f"LIVE TRADING BLOCKED: Environment variable ALLOW_LIVE_TRADING must be set to "
                f"'{LIVE_TRADING_ACK_TOKEN}'. Live execution cannot start accidentally."
            )
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            errors.append("Missing Binance API credentials for live trading")
    
    # Warnings for potentially risky settings
    if LEVERAGE > 20:
        warnings.append(f"High leverage ({LEVERAGE}x) - ensure proper risk management")
    
    if TRADING_ENABLED and MOCK_TRADING:
        warnings.append("Paper trading enabled - no real money at risk")
    
    return errors, warnings


def get_model_version() -> str:
    """Get model version from checkpoint file."""
    import torch
    import hashlib
    
    if not os.path.exists(MODEL_PATH):
        return "no_model"
    
    try:
        # Load checkpoint to get metadata if available
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        if isinstance(checkpoint, dict) and 'version' in checkpoint:
            return checkpoint['version']
        
        # Fallback: compute hash of the file
        with open(MODEL_PATH, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        return f"hash_{file_hash}"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    # Run validation when module is executed directly
    errors, warnings = validate_config()
    
    if warnings:
        print("⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print("❌ Configuration Errors:")
        for e in errors:
            print(f"  - {e}")
        exit(1)
    else:
        print("✅ Configuration validated successfully")
        exit(0)
