"""
data/preprocessor.py — Feature engineering, normalization, and window construction.

Provides:
  - validate_fibonacci_warmup(): verifies 720-bar Fibonacci indicator warm-up before fillna(0.0)
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
from typing import Optional, Any, Tuple, List, Dict

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

def validate_fibonacci_warmup(
    df_raw: pd.DataFrame,
    df_indicators_pre_fillna: pd.DataFrame,
    requested_start_ts: pd.Timestamp,
    min_history_bars: int = 720,
) -> Dict[str, Any]:
    """
    Validate that the 720-period Fibonacci indicator ('fib_618_dist_180d') has
    sufficient valid historical observations BEFORE requested_start_ts and is
    non-NaN BEFORE fillna(0.0) is applied.
    """
    if requested_start_ts not in df_raw.index:
        raise ValueError(
            f"requested_start_ts ({requested_start_ts}) not found in raw OHLCV index."
        )

    # 1. Verify at least min_history_bars (720) valid historical observations exist BEFORE requested_start_ts
    df_prior = df_raw.loc[df_raw.index < requested_start_ts]
    ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in df_prior.columns]
    valid_prior_bars = int(len(df_prior[ohlc_cols].dropna())) if ohlc_cols else int(len(df_prior))

    if valid_prior_bars < min_history_bars:
        raise ValueError(
            f"Fibonacci warm-up validation failed: only {valid_prior_bars} valid historical "
            f"observations exist before requested_start_ts ({requested_start_ts}); "
            f"at least {min_history_bars} valid pre-roll bars are required."
        )

    # 2. Verify raw indicator validity BEFORE fillna(0.0)
    if requested_start_ts not in df_indicators_pre_fillna.index:
        raise ValueError(
            f"requested_start_ts ({requested_start_ts}) was dropped during core indicator warm-up."
        )

    if "fib_618_dist_180d" not in df_indicators_pre_fillna.columns:
        raise ValueError("Column 'fib_618_dist_180d' missing from indicator matrix before fillna.")

    retained_fib_raw = df_indicators_pre_fillna.loc[requested_start_ts:, "fib_618_dist_180d"]
    first_fib_raw = retained_fib_raw.iloc[0]
    nan_count_retained = int(retained_fib_raw.isna().sum())

    if pd.isna(first_fib_raw) or nan_count_retained > 0:
        raise ValueError(
            f"Fibonacci warm-up validation failed before fillna(0.0): "
            f"fib_618_dist_180d at requested_start_ts ({requested_start_ts}) is {first_fib_raw} "
            f"(retained slice has {nan_count_retained} raw NaNs)."
        )

    return {
        "requested_start_ts": str(requested_start_ts),
        "valid_prior_bars": valid_prior_bars,
        "first_fib_raw": float(first_fib_raw),
        "nan_count_retained_pre_fillna": nan_count_retained,
    }


def build_features(
    df: pd.DataFrame, 
    add_targets: bool = True,
    funding_df: pd.DataFrame = None,
    oi_df: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    df_4h: pd.DataFrame = None,
    requested_start_ts: Optional[pd.Timestamp] = None,
    min_warmup_bars: int = 720,
) -> pd.DataFrame:
    """
    1. Compute indicators on raw OHLCV DataFrame.
    2. Integrate External Data (Funding, OI).
    3. Integrate MTF Features (15m, 4h RSI).
    4. Optionally add regression target.
    5. Validate 720-bar Fibonacci warm-up BEFORE fillna(0.0) if requested_start_ts is given.
    6. Drop core NaN rows and fill remaining auxiliary NaNs.
    """
    df_raw_input = df
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

    # Record raw pre-fillna validity of 720-period Fibonacci before any missing-column fill or fillna(0.0)
    raw_fib_valid_mask = (
        df["fib_618_dist_180d"].replace([np.inf, -np.inf], np.nan).notna().copy()
        if "fib_618_dist_180d" in df.columns
        else pd.Series(False, index=df.index)
    )

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

    # Validate 720-bar Fibonacci warm-up BEFORE fillna(0.0) masks any NaNs
    fib_warmup_info = None
    if requested_start_ts is not None:
        fib_warmup_info = validate_fibonacci_warmup(
            df_raw=df_raw_input,
            df_indicators_pre_fillna=df,
            requested_start_ts=requested_start_ts,
            min_history_bars=min_warmup_bars,
        )
    
    # Fill remaining NaNs (External/MTF data gaps or failures in HMM/GARCH) with 0.0
    # Note: 0.0 effectively uses the mean of the StandardScaler for that feature
    # if it was fit on standard returns.
    df = df.fillna(0.0)

    # Final sanity check: ensuring no infs or nans remain before tensor conversion
    if df.isna().any().any():
        logger.warning("Feature matrix still contains NaNs - forcing zeros.")
        df = df.fillna(0.0)

    df.attrs["raw_fib_valid_mask"] = raw_fib_valid_mask.reindex(df.index, fill_value=False)
    if fib_warmup_info is not None:
        df.attrs["fib_warmup_validation"] = fib_warmup_info

    logger.info("Feature matrix prepared. Final shape: %s", df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Scalers
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_save_scalers(
    df: pd.DataFrame,
    scaler_candle_path: Optional[str] = None,
    scaler_market_path: Optional[str] = None,
) -> tuple[StandardScaler, StandardScaler]:
    """
    Fit one StandardScaler per branch on training data and persist to disk.
    Returns (scaler_candle, scaler_market).
    """
    scaler_candle = StandardScaler()
    scaler_market = StandardScaler()

    scaler_candle.fit(df[CANDLE_FEATURES].values)
    scaler_market.fit(df[MARKET_FEATURES].values)

    c_path = scaler_candle_path if scaler_candle_path is not None else config.SCALER_CANDLE_PATH
    m_path = scaler_market_path if scaler_market_path is not None else config.SCALER_MARKET_PATH

    joblib.dump(scaler_candle, c_path)
    joblib.dump(scaler_market, m_path)

    logger.info("Scalers saved -> %s, %s", c_path, m_path)
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
    timestamps: Optional[Any] = None,
    purge_horizon: int = TARGET_HORIZON,
    drop_nan_targets: bool = True,
    return_metadata: bool = False,
) -> tuple:
    """
    Construct continuous sliding lookback windows across unbroken hourly features.

    Preserves continuous hourly feature timeline:
    Lookback windows [i - lookback : i] are built on continuous rows.
    If drop_nan_targets=True, windows where targets[i - 1] is NaN are filtered OUT
    after window construction, ensuring that internal lookback windows NEVER skip an hour.

    Returns:
        If return_metadata is False:
            (X_candle, X_market, y)
        If return_metadata is True:
            (X_candle, X_market, y, window_meta)
            where window_meta contains decision_timestamps, decision_indices, resolution_timestamps.
    """
    N = len(candle_arr)
    assert N == len(market_arr) == len(targets), "Array length mismatch"

    X_candle, X_market, y = [], [], []
    dec_indices = []
    dec_timestamps = []
    res_timestamps = []

    has_ts = timestamps is not None and len(timestamps) == N

    for i in range(lookback, N):
        dec_idx = i - 1
        target_val = targets[dec_idx]

        if drop_nan_targets and np.isnan(target_val):
            continue

        X_candle.append(candle_arr[i - lookback:i])
        X_market.append(market_arr[i - lookback:i])
        y.append(target_val)
        dec_indices.append(dec_idx)

        if has_ts:
            t = timestamps[dec_idx]
            dec_timestamps.append(t)
            if hasattr(t, "__add__"):
                res_timestamps.append(t + pd.Timedelta(hours=purge_horizon))
            else:
                res_timestamps.append(t)

    X_c_arr = np.array(X_candle, dtype=np.float32)
    X_m_arr = np.array(X_market, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)

    if return_metadata:
        meta = {
            "decision_indices": np.array(dec_indices, dtype=int),
            "decision_timestamps": pd.DatetimeIndex(dec_timestamps) if has_ts else np.array(dec_timestamps),
            "resolution_timestamps": pd.DatetimeIndex(res_timestamps) if has_ts else np.array(res_timestamps),
        }
        return X_c_arr, X_m_arr, y_arr, meta

    return X_c_arr, X_m_arr, y_arr


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
    timestamps: Optional[Any] = None,
    resolution_timestamps: Optional[Any] = None,
) -> tuple:
    """
    Split time-series sliding windows into three strictly chronological partitions:
    1. Training Partition: [0, train_end)
       - Purged Boundary 1: [train_end, val_start) discarded
    2. Validation/Calibration Partition: [val_start, val_end)
       - Purged Boundary 2: [val_end, test_start) discarded
    3. Untouched Final Test Partition: [test_start, N)
       - Evaluated strictly out-of-sample once; no hyperparameter tuning or threshold optimization.

    Timestamp-based purging:
    If timestamps is provided, purges any training window whose target resolution timestamp
    exceeds val_start_ts, and any validation window whose target resolution exceeds test_start_ts.
    """
    total = len(y)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    val_start = int(total * train_ratio)
    test_start = int(total * (train_ratio + val_ratio))

    if timestamps is not None and len(timestamps) == total:
        ts_arr = pd.DatetimeIndex(timestamps) if not isinstance(timestamps, pd.DatetimeIndex) else timestamps
        if resolution_timestamps is not None and len(resolution_timestamps) == total:
            res_ts = pd.DatetimeIndex(resolution_timestamps) if not isinstance(resolution_timestamps, pd.DatetimeIndex) else resolution_timestamps
        else:
            res_ts = ts_arr + pd.Timedelta(hours=purge_horizon)

        val_start_ts = ts_arr[val_start]
        test_start_ts = ts_arr[test_start]

        # Purge Train: all train decision windows must have resolution <= val_start_ts
        train_mask = (res_ts[:val_start] <= val_start_ts)
        valid_train_idx = np.where(train_mask)[0]
        train_end = int(valid_train_idx[-1] + 1) if len(valid_train_idx) > 0 else 0

        # Purge Val: all val decision windows must have resolution <= test_start_ts
        val_mask = (res_ts[val_start:test_start] <= test_start_ts)
        valid_val_idx = np.where(val_mask)[0]
        val_end = int(val_start + valid_val_idx[-1] + 1) if len(valid_val_idx) > 0 else val_start

        # Verification assertion: zero target leakage across boundaries
        if train_end > 0:
            max_train_res = res_ts[:train_end].max()
            assert max_train_res <= val_start_ts, f"Temporal leakage: train label resolves at {max_train_res} > val start {val_start_ts}"
        if val_end > val_start:
            max_val_res = res_ts[val_start:val_end].max()
            assert max_val_res <= test_start_ts, f"Temporal leakage: val label resolves at {max_val_res} > test start {test_start_ts}"

        ts_meta = {
            "train_start_ts": ts_arr[0] if train_end > 0 else None,
            "train_end_ts": ts_arr[train_end - 1] if train_end > 0 else None,
            "val_start_ts": val_start_ts,
            "val_end_ts": ts_arr[val_end - 1] if val_end > val_start else None,
            "test_start_ts": test_start_ts,
            "test_end_ts": ts_arr[-1] if total > test_start else None,
            "train_max_resolution_ts": res_ts[:train_end].max() if train_end > 0 else None,
            "val_max_resolution_ts": res_ts[val_start:val_end].max() if val_end > val_start else None,
        }
    else:
        train_end = max(0, val_start - purge_horizon)
        val_end = max(val_start, test_start - purge_horizon)
        ts_meta = {}

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
        **ts_meta,
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
