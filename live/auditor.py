import os
import sys

# Ensure project root is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from utils.database import SessionLocal, SignalLog
from data.fetcher import fetch_ohlcv
from config import SYMBOL, TARGET_HORIZON, TIMEFRAME, SIGNALS_CSV

logger = logging.getLogger(__name__)

def audit_signals_csv():
    """Fallback: Audit signals from the signals.csv file."""
    if not os.path.exists(SIGNALS_CSV):
        return

    try:
        df = pd.read_csv(SIGNALS_CSV)
        if df.empty:
            return

        # 1. Standardize schema if needed (for older CSVs or AccuracyTracker format)
        rename_map = {"candle_close_utc": "timestamp"}
        df = df.rename(columns=rename_map)

        # Ensure necessary columns exist and cast to appropriate dtypes
        # is_verified / was_correct -> bool, actual_return -> float, close_price/prediction -> float
        if 'is_verified' not in df.columns:
            df['is_verified'] = False
        else:
            df['is_verified'] = df['is_verified'].astype('boolean').fillna(False).astype(bool)

        if 'was_correct' not in df.columns:
            df['was_correct'] = False
        else:
            df['was_correct'] = df['was_correct'].astype('boolean').fillna(False).astype(bool)

        if 'actual_return' not in df.columns:
            df['actual_return'] = np.nan
        else:
            df['actual_return'] = pd.to_numeric(df['actual_return'], errors='coerce')

        if 'close_price' not in df.columns:
            df['close_price'] = pd.NA
        else:
            df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')

        if 'prediction' not in df.columns:
            df['prediction'] = 0.0
        else:
            df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce').fillna(0.0)

        if df['timestamp'].isnull().all():
            logger.warning("Auditor CSV has no valid timestamps. Skipping.")
            return

        # 2. Find unverified signals older than horizon
        # Ensure timestamps are timezone-aware (UTC) so comparisons work on Windows
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=TARGET_HORIZON + 1)
        
        unverified = df[(df['is_verified'] == False) & (df['timestamp'] <= cutoff)]
        if unverified.empty:
            return

        logger.info(f"Auditing {len(unverified)} unverified signals from CSV...")

        # 2. Fetch recent historical data
        df_hist = None
        for attempt in range(2):
            try:
                df_hist = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1000)
                if not df_hist.empty: break
            except:
                time.sleep(2)

        if df_hist is None or df_hist.empty: return

        # 3. Process
        for idx_row in unverified.index:
            try:
                sig_time = df.loc[idx_row, 'timestamp']
                target_time = sig_time + timedelta(hours=TARGET_HORIZON)

                # Build a timezone-aware pandas Timestamp robustly
                target_ts = pd.Timestamp(target_time)
                if target_ts.tzinfo is None:
                    target_ts = target_ts.tz_localize('UTC')
                else:
                    try:
                        target_ts = target_ts.tz_convert('UTC')
                    except TypeError:
                        pass

                hist_idx = df_hist.index.get_indexer([target_ts], method='nearest')[0]
                actual_close = df_hist.iloc[hist_idx]['close']
                
                # Use current 'close_price' from CSV
                sig_close = df.loc[idx_row, 'close_price']
                actual_ret = np.log(actual_close / sig_close)
                was_correct = (np.sign(df.loc[idx_row, 'prediction']) == np.sign(actual_ret))
                
                df.at[idx_row, 'actual_return'] = float(actual_ret)
                df.at[idx_row, 'was_correct'] = bool(was_correct)
                df.at[idx_row, 'is_verified'] = True
            except Exception as e:
                logger.error(f"Failed to audit CSV signal at index {idx_row}: {e}")

        # 4. Save and restore schema for AccuracyTracker compatibility
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "candle_close_utc"})

        df.to_csv(SIGNALS_CSV, index=False)
        logger.info("CSV Audit complete.")
    except Exception as e:
        logger.error(f"CSV Auditor error: {e}")

def audit_signals():
    """
    Find unverified signals older than TARGET_HORIZON, 
    fetch actual price at that time, and calculate accuracy.
    """
    try:
        session = SessionLocal()
        # Test connection
        session.execute(text("SELECT 1"))
    except Exception:
        logger.warning("PostgreSQL unavailable. Falling back to CSV Audit.")
        audit_signals_csv()
        return

    try:
        # 1. Get unverified signals older than horizon
        # Use timezone-aware UTC cutoff to match DB / CSV timestamps with tzinfo
        cutoff = datetime.now(timezone.utc) - timedelta(hours=TARGET_HORIZON + 1)
        signals = session.query(SignalLog).filter(
            SignalLog.is_verified == False,
            SignalLog.timestamp <= cutoff
        ).all()

        if not signals:
            return

        logger.info(f"Auditing {len(signals)} unverified signals...")

        # 2. Fetch recent historical data for verification
        # Retry up to 3 times for robustness
        df_hist = None
        for attempt in range(3):
            try:
                df_hist = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1000)
                if not df_hist.empty:
                    break
            except Exception as e:
                logger.warning(f"Auditor fetch attempt {attempt+1} failed: {e}")
                time.sleep(5)

        if df_hist is None or df_hist.empty:
            logger.error("Auditor failed to fetch historical data after retries.")
            return
        
        for sig in signals:
            try:
                # Find the close price at signal_time + horizon
                target_time = sig.timestamp + timedelta(hours=TARGET_HORIZON)

                # Build a timezone-aware pandas Timestamp robustly
                target_ts = pd.Timestamp(target_time)
                if target_ts.tzinfo is None:
                    target_ts = target_ts.tz_localize('UTC')
                else:
                    try:
                        target_ts = target_ts.tz_convert('UTC')
                    except TypeError:
                        pass

                # Find closest index in historical data
                idx = df_hist.index.get_indexer([target_ts], method='nearest')[0]
                actual_close_at_horizon = df_hist.iloc[idx]['close']
                
                # Calculate metrics
                actual_ret = np.log(actual_close_at_horizon / sig.close_price)
                was_correct = (np.sign(sig.prediction) == np.sign(actual_ret))
                
                # Update record
                sig.actual_return = float(actual_ret)
                sig.was_correct = bool(was_correct)
                sig.is_verified = True
                
                logger.debug(f"Signal {sig.id} audited: Pred={sig.prediction:.4f}, Act={actual_ret:.4f}, Correct={was_correct}")
            except Exception as e:
                logger.error(f"Failed to audit signal {sig.id}: {e}")

        session.commit()
    except Exception as e:
        logger.error(f"Auditor loop error: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Signal Auditor...")
    while True:
        audit_signals()
        time.sleep(300) # Run every 5 minutes
