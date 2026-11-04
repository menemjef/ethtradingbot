"""
data/fetcher.py — Historical OHLCV data fetching via CCXT Binance Futures.
Includes retry logic with exponential backoff and data quality validation.
"""

import time
import logging
import pandas as pd
import numpy as np
import ccxt
from typing import Optional

logger = logging.getLogger(__name__)


_EXCHANGE = None

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 30.0  # seconds


def retry_with_backoff(func, max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF):
    """
    Decorator/wrapper for retrying functions with exponential backoff.
    """
    def wrapper(*args, **kwargs):
        backoff = initial_backoff
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                else:
                    logger.error(f"All {max_retries} attempts failed: {e}")
        
        if last_exception is not None:
            raise last_exception
        raise RuntimeError(f"Function {func.__name__} failed after {max_retries} attempts without an exception.")
    
    return wrapper


def get_exchange() -> ccxt.binanceusdm:
    """Return a cached Binance USDM Futures exchange instance."""
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        # Pre-load markets once to ensure implicit API methods are populated
        try:
            _EXCHANGE.load_markets()
        except Exception as e:
            logger.warning(f"Initial market load failed: {e}")
    return _EXCHANGE


def validate_data_quality(df: pd.DataFrame, min_rows: int = 10) -> bool:
    """
    Validate data quality after fetching.
    Returns True if data passes validation, False otherwise.
    """
    if df is None or df.empty:
        logger.error("DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        logger.warning(f"Insufficient data: {len(df)} rows (minimum: {min_rows})")
        return False
    
    # Check for NaN values in critical columns
    critical_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in critical_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return False
        
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column '{col}' has {nan_count} NaN values")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            logger.warning(f"Column '{col}' has {inf_count} infinite values")
    
    # Check for price sanity (all prices should be positive)
    for col in ['open', 'high', 'low', 'close']:
        if (df[col] <= 0).any():
            logger.error(f"Invalid prices found in column: {col}")
            return False
    
    # High should be >= low, open, close
    if (df['high'] < df['low']).any():
        logger.error("Invalid OHLC: high < low")
        return False
    
    return True


def fetch_ohlcv(
    symbol: str = "ETH/USDT:USDT",
    timeframe: str = "4h",
    limit: int = 2500,
    since_ms: int | None = None,
    validate: bool = True,
    warmup_bars: int = 0,
    exclude_unclosed: bool = True,
    raise_on_gap: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles for the given symbol/timeframe.

    Guarantees:
    - Returns the most recent requested number of completed candles up to the latest closed bar.
    - Excludes currently forming (unclosed) candles.
    - Correctly paginates across Binance's maximum page size (1000 per call).
    - Avoids duplicate timestamps, ensures strict chronological ordering, and verifies continuous intervals.
    - Safely detects non-progressing exchange cursors to prevent infinite loops.
    - Explicitly reports any shortfall if the exchange cannot supply the requested count.

    Args:
        symbol:           CCXT symbol, e.g. 'ETH/USDT:USDT'
        timeframe:        Candle size, e.g. '1h', '4h'
        limit:            Number of completed candles to fetch
        since_ms:         Optional explicit start time in milliseconds (UTC epoch)
        validate:         Whether to validate data quality
        warmup_bars:      Extra pre-roll candles to fetch for indicator warm-up
        exclude_unclosed: If True, excludes any candle that has not yet closed
        raise_on_gap:     If True, raises a ValueError on detected missing intervals

    Returns:
        pd.DataFrame with columns [open, high, low, close, volume]
        indexed by open_time (UTC-aware datetime).
    """
    exchange = get_exchange()
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()

    # Total completed candles requested (including warm-up pre-roll)
    total_needed = limit + max(0, warmup_bars)
    per_page = 1000
    all_candles = []

    if since_ms is None:
        # Determine latest completed candle timestamp
        current_candle_open = (now_ms // tf_ms) * tf_ms
        latest_completed_open = current_candle_open - tf_ms
        target_start_ms = latest_completed_open - (total_needed - 1) * tf_ms
        since = target_start_ms
        target_end_ms = latest_completed_open

        logger.info(
            "Fetching %d completed %s candles for %s (target range: %s to %s)",
            total_needed, timeframe, symbol,
            pd.to_datetime(target_start_ms, unit="ms", utc=True),
            pd.to_datetime(latest_completed_open, unit="ms", utc=True),
        )
    else:
        since = since_ms
        target_end_ms = None
        logger.info(
            "Fetching up to %d %s candles for %s starting from %s",
            total_needed, timeframe, symbol,
            pd.to_datetime(since_ms, unit="ms", utc=True),
        )

    prev_since = None
    consecutive_empty = 0
    max_empty_retries = 3

    while True:
        fetch_n = per_page
        if target_end_ms is not None:
            remaining_ms = target_end_ms - since
            if remaining_ms < 0:
                break
            candles_left = int(remaining_ms // tf_ms) + 1
            fetch_n = min(per_page, max(1, candles_left))
        else:
            if len(all_candles) >= total_needed:
                break
            fetch_n = min(per_page, total_needed - len(all_candles))

        @retry_with_backoff
        def _fetch_with_retry():
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=fetch_n)

        try:
            raw = _fetch_with_retry()
        except Exception as exc:
            logger.error("Error fetching OHLCV: %s - giving up after retries", exc)
            raise

        if not raw:
            consecutive_empty += 1
            if target_end_ms is not None and since < target_end_ms and consecutive_empty <= max_empty_retries:
                # Intermediate data gap suspected: advance cursor past empty window toward target_end
                advance_jump = fetch_n * tf_ms
                logger.warning(
                    "Empty page received at since=%d. Intermediate data gap suspected. "
                    "Advancing cursor by %d ms toward target_end=%d (attempt %d/%d).",
                    since, advance_jump, target_end_ms, consecutive_empty, max_empty_retries,
                )
                since += advance_jump
                continue
            else:
                break

        consecutive_empty = 0
        all_candles.extend(raw)

        # If we reached or exceeded target end timestamp, stop pagination
        if target_end_ms is not None and raw[-1][0] >= target_end_ms:
            break

        # Safety: Cursor progression validation to prevent infinite loops
        last_candle_ms = raw[-1][0]
        next_since = last_candle_ms + tf_ms
        if next_since <= since:
            logger.warning(
                "Exchange pagination cursor did not advance (current since: %d, last candle: %d). "
                "Terminating pagination to prevent infinite loop.",
                since, last_candle_ms,
            )
            break
        prev_since = since
        since = next_since

        # In explicit since_ms mode without target_end: break on short page
        if target_end_ms is None and len(raw) < fetch_n:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        raise RuntimeError(f"No OHLCV data returned for {symbol} {timeframe}")

    df = pd.DataFrame(all_candles, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    # 1. Exclude currently forming (unfinished) candles
    if exclude_unclosed:
        cutoff_ms = now_ms - tf_ms
        df = df[df["open_time"].astype("int64") // 10**6 <= cutoff_ms]

    # 2. Check and remove duplicate timestamps
    dup_count = df.duplicated(subset="open_time").sum()
    if dup_count > 0:
        logger.warning("Detected %d duplicate timestamps in fetched OHLCV. Deduplicating.", dup_count)
        df = df.drop_duplicates(subset="open_time", keep="last")

    # 3. Ensure strict chronological order
    df = df.set_index("open_time").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # 4. Check for missing intervals
    if len(df) > 1:
        freq_str = f"{int(tf_ms / 1000)}s"
        expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_str)
        missing_intervals = expected_range.difference(df.index)
        if len(missing_intervals) > 0:
            msg = (
                f"Detected {len(missing_intervals)} missing intervals in fetched OHLCV "
                f"between {df.index.min()} and {df.index.max()}"
            )
            logger.warning(msg)
            if raise_on_gap:
                raise ValueError(f"Data integrity failure: {msg}")

    # 5. Check for shortfall against requested completed candles
    if len(df) < limit:
        shortfall = limit - len(df)
        logger.warning(
            "Shortfall detected in OHLCV fetch for %s %s: requested %d completed candles, "
            "but only %d were available from exchange (shortfall of %d candles).",
            symbol, timeframe, limit, len(df), shortfall,
        )
    elif len(df) > total_needed:
        # Trim to the exact requested slice (keeping the most recent completed candles)
        df = df.iloc[-total_needed:]

    # 6. Validate data quality
    if validate and not validate_data_quality(df, min_rows=min(10, limit)):
        raise ValueError(f"Data quality validation failed for {symbol} {timeframe}")

    logger.info(
        "Successfully fetched %d completed candles for %s %s (Latest completed bar: %s)",
        len(df), symbol, timeframe, df.index[-1] if not df.empty else "N/A",
    )
    return df


def fetch_latest_candles(
    symbol: str = "ETH/USDT:USDT",
    timeframe: str = "4h",
    limit: int = 300,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Fetch the most recent `limit` candles for live inference.
    Lightweight single-call version of fetch_ohlcv.
    """
    exchange = get_exchange()
    
    @retry_with_backoff
    def _fetch_with_retry():
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    try:
        raw = _fetch_with_retry()
    except Exception as exc:
        logger.error("Failed to fetch latest candles: %s", exc)
        raise

    df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").set_index("open_time").sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Validate data quality
    if validate and not validate_data_quality(df, min_rows=limit//2):
        logger.warning("Data quality validation failed for latest candles")

    return df


def fetch_funding_rates(symbol: str, limit: int = 100) -> pd.DataFrame:
    """Fetch recent funding rates for a symbol."""
    exchange = get_exchange()
    
    @retry_with_backoff
    def _fetch_with_retry():
        return exchange.fetch_funding_rate_history(symbol, limit=limit)
    
    try:
        raw = _fetch_with_retry()
        df = pd.DataFrame(raw)
        if df.empty: 
            return pd.DataFrame()
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.rename(columns={"fundingRate": "fundingRate"})
        df = df.set_index("timestamp")[["fundingRate"]].sort_index()
        return df
    except Exception as exc:
        logger.error(f"Failed to fetch funding rates: {exc}")
        return pd.DataFrame()


def fetch_open_interest(symbol: str, limit: int = 100) -> pd.DataFrame:
    """Fetch recent Open Interest data for a symbol."""
    exchange = get_exchange()
    
    # For Binance Futures, ETHUSDT is the required format
    api_symbol = symbol.replace("/", "").split(":")[0]
    
    @retry_with_backoff
    def _fetch_with_retry():
        # Using unified method for better compatibility
        return exchange.fetch_open_interest_history(symbol, timeframe="1h", limit=limit)
    
    try:
        raw = _fetch_with_retry()
        if not raw or len(raw) == 0:
            logger.warning(f"No OI data returned for {symbol}")
            return pd.DataFrame()
        
        # Parse the response - handle both dict and list formats
        records = []
        for item in raw:
            if isinstance(item, dict):
                records.append({
                    "timestamp": item.get("timestamp"),
                    "openInterestAmount": item.get("openInterestAmount")
                })
        
        if not records:
            logger.warning(f"No OI records parsed for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["openInterestAmount"] = df["openInterestAmount"].astype(float)
        df = df.set_index("timestamp")[["openInterestAmount"]].sort_index()
        return df
    except Exception as exc:
        logger.error(f"Failed to fetch open interest: {exc}")
        return pd.DataFrame()


def fetch_order_book_imbalance(symbol: str, depth: int = 10) -> float:
    """
    Calculate Order Book Imbalance (pressure).
    Returns value between -1 (extreme sell pressure) and 1 (extreme buy pressure).
    
    Args:
        symbol: The trading symbol.
        depth: Number of levels to consider for imbalance calculation.
    """
    exchange = get_exchange()
    
    @retry_with_backoff
    def _fetch_with_retry():
        return exchange.fetch_order_book(symbol, limit=depth)
    
    try:
        ob = _fetch_with_retry()
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        if not bids or not asks:
            logger.warning(f"Empty order book for {symbol}")
            return 0.0

        # Weighted volume calculation (optional, here we use simple sum of top `depth` levels)
        total_bid_vol = sum(level[1] for level in bids[:depth])
        total_ask_vol = sum(level[1] for level in asks[:depth])

        if (total_bid_vol + total_ask_vol) == 0:
            return 0.0

        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        return float(imbalance)
    except Exception as exc:
        logger.error(f"Failed to fetch order book imbalance: {exc}")
        return 0.0


def check_data_freshness(df: pd.DataFrame, max_age_seconds: int = 3600) -> bool:
    """
    Check if the data is fresh enough (within max_age_seconds).
    Returns True if data is fresh, False otherwise.
    """
    if df is None or df.empty:
        return False
    
    latest_time = df.index[-1]
    age_seconds = (pd.Timestamp.utcnow() - latest_time).total_seconds()
    
    if age_seconds > max_age_seconds:
        logger.warning(f"Data is stale: {age_seconds:.0f}s old (max: {max_age_seconds}s)")
        return False
    
    return True
