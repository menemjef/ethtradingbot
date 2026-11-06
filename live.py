"""
live.py — Entry point for the live ETH trading signal engine.

WHAT THIS FILE DOES:
This is the main script that starts the live trading bot. When you run this file, it:
1. Performs safety checks (pre-flight check).
2. Connects to the database to track trades and portfolio balance.
3. Starts the Execution Engine (to talk to Binance).
4. Starts the Trader (to manage risk and decide position sizes).
5. Starts the Signal Engine (to listen to WebSockets and run the AI model).
6. Runs in an infinite loop until you shut it down.

Run with:
    python live.py

Prerequisites:
    1. Run python train.py first to generate model + scalers.
    2. Requires checkpoints/best_model.pt, scaler_candle.pkl, scaler_market.pkl
"""

import logging
import os
import sys
import signal
import threading
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging
setup_logging()

from live.signal_engine import SignalEngine
from live.accuracy_tracker import AccuracyTracker
from live.execution_engine import ExecutionEngine
from live.trader import Trader
from config import MOCK_TRADING, TRADING_ENABLED, SYMBOL
from utils.database import init_db, save_signal, save_trade

logger = logging.getLogger("live")

# Global state for graceful shutdown 
_shutdown_event = None


def pre_flight_check():
    """
    Security and connectivity checks before real trading starts.
    Guarantees fail-safe defaults and prevents accidental live execution.
    """
    from config import (
        MOCK_TRADING, PRODUCTION_READY, BINANCE_API_KEY, BINANCE_API_SECRET,
        ALLOW_LIVE_TRADING, LIVE_TRADING_ACK_TOKEN, validate_config
    )
    
    # 0. Strict Config Validation
    cfg_errors, cfg_warnings = validate_config()
    for w in cfg_warnings:
        logger.warning(f"Config Warning: {w}")
    if cfg_errors:
        for err in cfg_errors:
            logger.error(f"Config Error: {err}")
        return False
    
    # If we are actually trading real money (not mock trading)
    if not MOCK_TRADING:
        logger.info("RUNNING PRODUCTION PRE-FLIGHT CHECKS...")
        
        if ALLOW_LIVE_TRADING != LIVE_TRADING_ACK_TOKEN:
            logger.error(
                "CRITICAL: Live execution blocked. Explicit token ALLOW_LIVE_TRADING='%s' required.",
                LIVE_TRADING_ACK_TOKEN
            )
            return False
            
        # Check if we forgot to put our API keys in the .env file
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            logger.error("CRITICAL: API Credentials missing. Trading aborted.")
            return False
            
        # Verify connectivity and balance by asking Binance for our wallet balance
        try:
            temp_engine = ExecutionEngine(mock=False)
            bal_val = temp_engine.get_balance()
            balance = getattr(bal_val, 'balance', bal_val)
            if balance <= 0:
                logger.warning(f"WARNING: Wallet balance is {balance} USDT.")
            else:
                logger.info(f"Pre-flight passed: Balance: ${balance:,.2f} USDT")
        except Exception as e:
            logger.error(f"PRE-FLIGHT FAILED: {e}")
            return False

    # Check scalers exist 
    from config import SCALER_CANDLE_PATH, SCALER_MARKET_PATH
    
    if not os.path.exists(SCALER_CANDLE_PATH) or not os.path.exists(SCALER_MARKET_PATH):
         logger.error("CRITICAL: Scalers (.pkl) missing. Run train.py first.")         
         return False 

    return True 


def main():
    global _shutdown_event 
    
    # ─── SINGLE INSTANCE ENFORCEMENT (GHOST-PROOF) ───
    lock_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_engine.lock")
    try:
        lock_file = open(lock_file_path, "w")
        if sys.platform != "win32":
            import fcntl
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                print("🚨 PREVENTED DUPLICATE PROCESS: Another instance of live.py is already running! Exiting immediately.")
                sys.exit(1)
        else:
            import msvcrt
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except (IOError, OSError):
                print("🚨 PREVENTED DUPLICATE PROCESS: Another instance of live.py is already running! Exiting immediately.")
                sys.exit(1)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
    except Exception as e:
        pass

    # Create shutdown event for graceful termination    
    _shutdown_event = threading.Event()  

    # Set up signal handlers for graceful shutdown    
    def handle_signal(signum, frame):        
        # 1. Set the event IMMEDIATELY before doing ANYTHING else
        _shutdown_event.set()      

        # 2. Try to log (safely)
        try:
            logger.info(f"Received signal {signum} - initiating graceful shutdown...")
        except Exception:
            # If logging fails (e.g. file lock on Windows), just print to stderr
            import sys
            sys.stderr.write(f"⚠️  Shutting down (Signal {signum})...\n")

        # 3. Explicitly stop engine if it exists
        try:
            if 'engine' in locals() and engine:
                engine.stop()
        except Exception:
            pass

    signal.signal(signal.SIGINT, handle_signal)   
    signal.signal(signal.SIGTERM, handle_signal)

    # Validate configuration at startup     
    from config import validate_config       
    errors, warnings = validate_config()       
    if warnings:           
        for w in warnings:               
            logger.warning(w)       
    if errors:          
        for e in errors:
            logger.critical(e)
        sys.exit(1)

    # Dynamic Heartbeat — Start this first!  
    from config import LOG_DIR
    
    def heartbeat():
        hb_path = os.path.join(LOG_DIR, "heartbeat.txt")        
        while not _shutdown_event.is_set():             
            try:
                with open(hb_path, "w") as f:
                    f.write(str(time.time()))
            except Exception:
                pass            
            time.sleep(30)
    
    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()

    from config import TRADING_ENABLED, MOCK_TRADING
    
    logger.info("=" * 62)
    logger.info("  ETH/USDT Perpetual Futures — Live Signal Engine")
    if TRADING_ENABLED:
        mode = "MOCK" if MOCK_TRADING else "LIVE"
        logger.info(f"  EXECUTION ENABLED: {mode} MODE")
    else:
        logger.info("  SOLO SIGNAL MODE (No trades)")
    logger.info("=" * 62)

    if not pre_flight_check():
        logger.error("Pre-flight checks failed. Exiting.")
        sys.exit(1)

    # Initialize Persistence
    init_db()

    tracker = AccuracyTracker()
    
    execution = ExecutionEngine(mock=MOCK_TRADING)
    trader = Trader(execution)

    # High-Frequency Background Sync (60s checks for quick SL/TP cleanup)
    # This thread runs in the background and constantly syncs our local database
    # with Binance to make sure our internal state matches reality.
    def account_sync_loop(t: Trader):
        while not _shutdown_event.is_set():
            try:
                t.sync_account()
                t.check_trailing_stops()
            except Exception as e:
                logger.error(f"Error in background sync: {e}")
            time.sleep(60)

    sync_thread = threading.Thread(target=account_sync_loop, args=(trader,), daemon=True)
    sync_thread.start()

    def on_signal(close_time, prediction, close_price, atr=None, btc_bullish=True):
        """
        Called by SignalEngine on every closed candle (e.g. every hour).
        This is the glue between the AI and the Trading logic.
        """
        # 1. Track accuracy and persist the AI prediction to the database
        tracker.record_signal(close_time, prediction, close_price)
        save_signal(SYMBOL, prediction, close_price)
        
        # 2. Handle trading logic (Trader decides whether to Buy, Sell, or Hold)
        trader.handle_signal(close_time, prediction, close_price, atr=atr, btc_bullish=btc_bullish)

    engine = SignalEngine(on_signal=on_signal, shutdown_event=_shutdown_event)

    try:
        while not _shutdown_event.is_set():
            try:
                engine.run()
            except KeyboardInterrupt:
                logger.info("\nShutdown requested via keyboard")
                break 
            except Exception as e:
                logger.exception("Error in main loop:")
                if _shutdown_event.wait(timeout=60):
                    break
            
            # 🛡️ Safeguard: If engine returns but we aren't shutting down, wait 10s before restart
            if not _shutdown_event.is_set():
                logger.warning("Engine returned unexpectedly. Re-starting in 10s to prevent spam loop...")
                time.sleep(10)
    finally:
        # Final stats output when shutting down 
        try:
            logger.info(
                "Final stats — Win Rate: %.1f%% (%d/%d) | "
                "Threshold Win Rate: %.1f%% (%d/%d)",
                tracker.win_rate * 100, tracker.wins, tracker.total,
                tracker.threshold_win_rate * 100, tracker.threshold_wins, tracker.threshold_total,
            )
        except Exception:
            pass
        
    # Force exit for Windows stability
    sys.exit(0)


if __name__ == "__main__":
    main()
