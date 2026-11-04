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




def split_train_val_test_purged(
    X_candle: np.ndarray,
    X_market: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    purge_horizon: int = 12,
) -> tuple:
    """
    Split time-series sliding windows into three strictly chronological partitions:
    1. Training Partition: [0, train_end)
       - Purged Boundary 1: [train_end, val_start) discarded
    2. Validation/Calibration Partition: [val_start, val_end)
       - Purged Boundary 2: [val_end, test_start) discarded
    3. Untouched Final Test Partition: [test_start, N)
       - Evaluated strictly out-of-sample once; no hyperparameter tuning or threshold optimization.
    """
    total = len(y)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    val_start = int(total * train_ratio)
    test_start = int(total * (train_ratio + val_ratio))

    train_end = max(0, val_start - purge_horizon)
    val_end = max(val_start, test_start - purge_horizon)

    train_slice = (X_candle[:train_end], X_market[:train_end], y[:train_end])
    val_slice = (X_candle[val_start:val_end], X_market[val_start:val_end], y[val_start:val_end])
    test_slice = (X_candle[test_start:], X_market[test_start:], y[test_start:])

    meta = {
        "total_windows": total,
        "train_samples": len(train_slice[2]),
        "val_samples": len(val_slice[2]),
        "test_samples": len(test_slice[2]),
        "train_purge_size": val_start - train_end,
        "val_purge_size": test_start - val_end,
        "train_indices": (0, train_end),
        "val_indices": (val_start, val_end),
        "test_indices": (test_start, total),
        "purge_horizon": purge_horizon,
    }
    logger.info(
        "3-Way Temporal Split with Purge: Total %d | Train: %d (purged %d) | Val: %d (purged %d) | Test: %d (untouched)",
        total, len(train_slice[2]), meta["train_purge_size"], len(val_slice[2]), meta["val_purge_size"], len(test_slice[2]),
    )
    return train_slice, val_slice, test_slice, meta


def generate_walk_forward_splits(
    total_samples: int,
    n_splits: int = 4,
    train_ratio: float = 0.50,
    test_ratio: float = 0.125,
    purge_horizon: int = 12,
    expanding: bool = True,
) -> list:
    """
    Generate walk-forward cross-validation splits with purged boundaries.
    Guarantees that every out-of-sample test partition is evaluated strictly on future data
    separated by a purged horizon.
    """
    splits = []
    test_size = int(total_samples * test_ratio)
    init_train_size = int(total_samples * train_ratio)

    for i in range(n_splits):
        test_start = init_train_size + i * test_size
        test_end = min(total_samples, test_start + test_size)
        if test_start >= total_samples or (test_end - test_start) < purge_horizon:
            break

        train_start = 0 if expanding else (i * test_size)
        train_end = max(train_start, test_start - purge_horizon)

        splits.append({
            "split_idx": i,
            "train_indices": (train_start, train_end),
            "test_indices": (test_start, test_end),
            "train_count": train_end - train_start,
            "test_count": test_end - test_start,
            "purge_count": test_start - train_end,
        })
    return splits


def block_bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func,
    block_size: int = 12,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Moving block bootstrap for time series with overlapping targets (horizon = block_size).
    Preserves autocorrelation structure when computing standard errors and confidence intervals.
    """
    rng = np.random.default_rng(seed)
    N = len(y_true)
    num_blocks = int(np.ceil(N / block_size))
    bootstrap_metrics = []

    for _ in range(n_bootstraps):
        start_indices = rng.integers(0, max(1, N - block_size + 1), size=num_blocks)
        indices = np.concatenate([np.arange(s, min(s + block_size, N)) for s in start_indices])[:N]
        metric_val = metric_func(y_pred[indices], y_true[indices])
        bootstrap_metrics.append(metric_val)

    bootstrap_metrics = np.array(bootstrap_metrics)
    point_est = metric_func(y_pred, y_true)
    ci_lower = float(np.percentile(bootstrap_metrics, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2)))
    std_err = float(np.std(bootstrap_metrics))

    return {
        "point_estimate": float(point_est),
        "std_error": std_err,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
        "block_size": block_size,
    }


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
