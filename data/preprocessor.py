"""
data/preprocessor.py — Feature engineering, normalization, and window construction.

Provides:
  - build_features(): assembles the two feature branches from indicator DataFrame
  - fit_and_save_scalers(): fits StandardScalers and persists them with joblib
  - load_scalers(): loads persisted scalers from disk
  - transform(): applies saved scalers to live data
  - build_windows(): creates sliding lookback windows for the Dataset
"""

import logging
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

import config
from config import (
    CANDLE_FEATURES,
    MARKET_FEATURES,
    LOOKBACK,
    TARGET_HORIZON,
)
from data.indicators import (
    compute_indicators, 
    add_target, 
    integrate_external_data, 
    compute_mtf_features
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame, 
    add_targets: bool = True,
    funding_df: pd.DataFrame = None,
    oi_df: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    df_4h: pd.DataFrame = None
) -> pd.DataFrame:
    """
    1. Compute indicators on raw OHLCV DataFrame.
    2. Integrate External Data (Funding, OI).
    3. Integrate MTF Features (15m, 4h RSI).
    4. Optionally add regression target.
    5. Drop NaN rows.
    """
    # 1. Standard Indicators
    df = compute_indicators(df)
    
    # 2. External Data Integration (Phase 5)
    df = integrate_external_data(df, funding_df, oi_df)
    
    # 3. MTF Features (Phase 5)
    df = compute_mtf_features(df, df_15m, df_4h)

    if add_targets:
        if config.TARGET_TYPE == "TBM":
            from data.indicators import add_tbm_target
            df = add_tbm_target(df, pt_mult=config.TBM_PT_MULT, sl_mult=config.TBM_SL_MULT, horizon=TARGET_HORIZON)
        else:
            df = add_target(df, horizon=TARGET_HORIZON)

    # Columns we need
    required = CANDLE_FEATURES + MARKET_FEATURES
    if add_targets:
        required = required + ["target"]

    # Safety check: if some advanced columns are missing, fill with 0
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Feature matrix is missing columns: {missing}. Filling with zeros.")
        for c in missing:
            df[c] = 0.0

    df = df[required]
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Core features (Branch 1 & basic indicators) need warmup (approx 200 rows)
    # We drop only if these core features are missing.
    core_features = CANDLE_FEATURES + [
        "log_return", "rsi_14", "macd", "macd_signal", "macd_hist",
        "ema_ratio_20_50", "ema_ratio_50_200", "atr_14",
        "log_volume", "volume_ema_ratio"
    ]
    df = df.dropna(subset=[c for c in core_features if c in df.columns])
    
    # Fill remaining NaNs (External/MTF data gaps or failures in HMM/GARCH) with 0.0
    # Note: 0.0 effectively uses the mean of the StandardScaler for that feature
    # if it was fit on standard returns.
    df = df.fillna(0.0)

    # Final sanity check: ensuring no infs or nans remain before tensor conversion
    if df.isna().any().any():
        logger.warning("Feature matrix still contains NaNs - forcing zeros.")
        df = df.fillna(0.0)

    logger.info("Feature matrix prepared. Final shape: %s", df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Scalers
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_save_scalers(df: pd.DataFrame) -> tuple[StandardScaler, StandardScaler]:
    """
    Fit one StandardScaler per branch on training data and persist to disk.
    Returns (scaler_candle, scaler_market).
    """
    scaler_candle = StandardScaler()
    scaler_market = StandardScaler()

    scaler_candle.fit(df[CANDLE_FEATURES].values)
    scaler_market.fit(df[MARKET_FEATURES].values)

    joblib.dump(scaler_candle, config.SCALER_CANDLE_PATH)
    joblib.dump(scaler_market, config.SCALER_MARKET_PATH)

    logger.info("Scalers saved -> %s, %s", config.SCALER_CANDLE_PATH, config.SCALER_MARKET_PATH)
    return scaler_candle, scaler_market


def load_scalers() -> tuple[StandardScaler, StandardScaler]:
    """Load persisted scalers from disk. Raises if files not found."""
    if not os.path.exists(config.SCALER_CANDLE_PATH):
        raise FileNotFoundError(f"Candle scaler not found: {config.SCALER_CANDLE_PATH}")
    if not os.path.exists(config.SCALER_MARKET_PATH):
        raise FileNotFoundError(f"Market scaler not found: {config.SCALER_MARKET_PATH}")

    scaler_candle = joblib.load(config.SCALER_CANDLE_PATH)
    scaler_market = joblib.load(config.SCALER_MARKET_PATH)
    logger.info("Scalers loaded from disk.")
    return scaler_candle, scaler_market


def transform(
    df: pd.DataFrame,
    scaler_candle: StandardScaler,
    scaler_market: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply fitted scalers to a feature DataFrame.

    Returns:
        candle_arr: (N, len(CANDLE_FEATURES)) float32 array
        market_arr: (N, len(MARKET_FEATURES)) float32 array
    """
    candle_arr = scaler_candle.transform(df[CANDLE_FEATURES].values).astype(np.float32)
    market_arr = scaler_market.transform(df[MARKET_FEATURES].values).astype(np.float32)
    return candle_arr, market_arr


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(
    candle_arr: np.ndarray,
    market_arr: np.ndarray,
    targets: np.ndarray,
    lookback: int = LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct sliding lookback windows.

    Returns:
        X_candle: (N_windows, lookback, candle_dim)
        X_market: (N_windows, lookback, market_dim)
        y:        (N_windows,)
    """
    N = len(candle_arr)
    assert N == len(market_arr) == len(targets), "Array length mismatch"

    X_candle, X_market, y = [], [], []
    for i in range(lookback, N):
        X_candle.append(candle_arr[i - lookback:i])
        X_market.append(market_arr[i - lookback:i])
        # FIX: The feature window ends at index `i - 1`. 
        # The target for a trade taken at the close of `i - 1` is `targets[i - 1]`.
        y.append(targets[i - 1])

    return (
        np.array(X_candle, dtype=np.float32),
        np.array(X_market, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )


def split_train_val_purged(
    X_candle: np.ndarray,
    X_market: np.ndarray,
    y: np.ndarray,
    train_split: float = 0.8,
    purge_horizon: int = 12,
) -> tuple:
    """
    Split time-series sliding windows into training and validation sets
    with strict temporal leakage prevention (purge/embargo).

    Purge Logic:
    A sample at index `k` has a target derived from price action extending up to `k + purge_horizon`.
    If the validation set begins at index `split_idx`, any training sample within the window
    `[split_idx - purge_horizon, split_idx)` has a target that overlaps with prices observed
    during the validation set.
    Purging removes these `purge_horizon` boundary samples from the training set,
    guaranteeing that NO training target incorporates future information from the validation timeline.

    Returns:
        (X_candle_train, X_market_train, y_train),
        (X_candle_val, X_market_val, y_val),
        metadata: {
            "total_windows": int,
            "split_idx": int,
            "purge_size": int,
            "train_samples": int,
            "val_samples": int,
        }
    """
    total = len(y)
    split_idx = int(total * train_split)
    purge_size = min(max(0, purge_horizon), split_idx)
    purged_train_end = split_idx - purge_size

    X_c_train = X_candle[:purged_train_end]
    X_m_train = X_market[:purged_train_end]
    y_train = y[:purged_train_end]

    X_c_val = X_candle[split_idx:]
    X_m_val = X_market[split_idx:]
    y_val = y[split_idx:]

    meta = {
        "total_windows": total,
        "split_idx": split_idx,
        "purge_size": purge_size,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
    }
    logger.info(
        "Temporal Split with Purge: %d total windows | Train: %d (purged %d at boundary) | Val: %d",
        total, len(y_train), purge_size, len(y_val),
    )
    return (X_c_train, X_m_train, y_train), (X_c_val, X_m_val, y_val), meta


def prepare_live_window(
    df_raw: pd.DataFrame,
    scaler_candle: StandardScaler,
    scaler_market: StandardScaler,
    lookback: int = LOOKBACK,
    funding_df: pd.DataFrame = None,
    oi_df: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    df_4h: pd.DataFrame = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Full live preprocessing pipeline with expanded features.
    """
    df = build_features(
        df_raw, 
        add_targets=False,
        funding_df=funding_df,
        oi_df=oi_df,
        df_15m=df_15m,
        df_4h=df_4h
    )

    if len(df) < lookback:
        logger.warning("Not enough rows after indicator computation: %d < %d", len(df), lookback)
        return None

    candle_arr, market_arr = transform(df, scaler_candle, scaler_market)

    # Take the most recent `lookback` rows
    candle_window = candle_arr[-lookback:][np.newaxis, :]   # (1, lookback, dim)
    market_window = market_arr[-lookback:][np.newaxis, :]   # (1, lookback, dim)
    return candle_window, market_window
