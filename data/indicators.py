"""
data/indicators.py — Technical indicator computation for both feature branches.

WHAT THIS FILE DOES:
This file is the "Math Engine" for the AI. 
Raw prices (like $3000, $3010) are hard for AI to understand. This script converts 
raw prices into "Technical Indicators" (momentum, volatility, trend). 
Think of it as turning a raw temperature reading into "It's heating up quickly".

All indicators are computed using pandas and pandas_ta (no TA-Lib dependency).
The output matches EXACTLY what the live engine recomputes at inference time.
"""

import logging
import numpy as np
import pandas as pd
from data.quant_models import compute_hmm_regimes, compute_garch_volatility, compute_arima_forecast

logger = logging.getLogger(__name__)




# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(close: pd.Series, period=20, std_dev=2):
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, width


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100 * (close - lowest_low) / denom
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    """Compute ADX, +DI, -DI."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_pos = (high - prev_high).clip(lower=0)
    dm_neg = (prev_low - low).clip(lower=0)

    # When +DM > -DM use +DM, else 0; vice versa
    dm_pos = dm_pos.where(dm_pos > dm_neg, 0.0)
    dm_neg = dm_neg.where(dm_neg > dm_pos.shift(0), 0.0)  # re-derive cleanly
    # Re-derive correctly
    raw_pos = (high - prev_high).clip(lower=0)
    raw_neg = (prev_low - low).clip(lower=0)
    dm_pos = raw_pos.where(raw_pos > raw_neg, 0.0)
    dm_neg = raw_neg.where(raw_neg >= raw_pos, 0.0)

    smoothed_tr = tr.ewm(com=period - 1, adjust=False).mean()
    smoothed_pos = dm_pos.ewm(com=period - 1, adjust=False).mean()
    smoothed_neg = dm_neg.ewm(com=period - 1, adjust=False).mean()

    di_pos = 100 * smoothed_pos / smoothed_tr.replace(0, np.nan)
    di_neg = 100 * smoothed_neg / smoothed_tr.replace(0, np.nan)

    dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, adjust=False).mean()

    return adx, di_pos, di_neg


def _fisher_transform(close: pd.Series, period: int = 9) -> pd.Series:
    """Fisher Transform of the price based on range extremes."""
    high = close.rolling(period).max()
    low = close.rolling(period).min()
    # Normalize price to [-1, 1]
    val = 0.66 * ((close - low) / (high - low).replace(0, np.nan) - 0.5)
    # Smooth it
    val = val.ewm(span=period, adjust=False).mean().clip(-0.999, 0.999)
    # Fisher transform formula
    fisher = 0.5 * np.log((1 + val) / (1 - val))
    return fisher


def _roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change."""
    return close.pct_change(period)


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow (Institutional buying/selling pressure)."""
    mfv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def _sr_levels(high: pd.Series, low: pd.Series, period: int = 72) -> tuple[pd.Series, pd.Series]:
    """Identify rolling support (min low) and resistance (max high)."""
    res = high.rolling(period).max()
    sup = low.rolling(period).min()
    return sup, res


def _fib_retracement(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 720):
    """
    Calculate Fibonacci retracement levels based on the rolling High/Low.
    Period = 720 hours = 30 days of 1H candles.
    """
    max_h = high.rolling(period).max()
    min_l = low.rolling(period).min()
    diff = max_h - min_l
    
    fib_382 = max_h - 0.382 * diff
    fib_500 = max_h - 0.500 * diff
    fib_618 = max_h - 0.618 * diff
    
    # Distance to the key 61.8% (Golden Pocket) level as fraction of price
    dist_fib_618 = (close - fib_618) / close.replace(0, np.nan)
    
    return dist_fib_618, max_h, min_l


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all feature-engineering columns on an OHLCV DataFrame.
    
    This is the main function. It takes Open, High, Low, Close, Volume data 
    and appends dozens of new columns (features).
    
    The AI model looks at all these columns to make its decision.
    """
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # ── Branch 1: Candle Structure ──────────────────────────────────────────
    df["body"] = (c - o).abs()
    df["upper_wick"] = h - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - l
    df["candle_range"] = h - l

    # ── Branch 2: Market State ───────────────────────────────────────────────
    df["log_return"] = np.log(c / c.shift(1))

    # RSI
    df["rsi_14"] = _rsi(c, 14)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(c)

    # EMA ratios
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ema200 = _ema(c, 200)
    df["ema_ratio_20_50"] = ema20 / ema50.replace(0, np.nan)
    df["ema_ratio_50_200"] = ema50 / ema200.replace(0, np.nan)

    # ATR
    df["atr_14"] = _atr(h, l, c, 14)

    # Bollinger Bands
    df["bb_upper"], df["bb_middle"], df["bb_lower"], df["bb_width"] = _bollinger(c)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = _stochastic(h, l, c)

    # ADX
    df["adx_14"], df["di_plus"], df["di_minus"] = _adx(h, l, c, 14)

    # Trend flag: +1 if close > EMA200, -1 otherwise
    df["trend_flag"] = (c > ema200).astype(float) * 2 - 1

    # Distance from rolling 20-period high/low (as fraction of price)
    roll_high = h.rolling(20).max()
    roll_low = l.rolling(20).min()
    df["dist_from_high_20"] = (roll_high - c) / c.replace(0, np.nan)
    df["dist_from_low_20"] = (c - roll_low) / c.replace(0, np.nan)

    # ── Volume Features ─────────────────────────────────────────────────────
    v = df["volume"]
    df["log_volume"] = np.log(v + 1e-6)
    v_ema20 = _ema(v, 20)
    df["volume_ema_ratio"] = v / v_ema20.replace(0, np.nan)

    # ── Advanced Features ───────────────────────────────────────────────────
    df["fisher_9"] = _fisher_transform(c, 9)
    df["roc_12"] = _roc(c, 12)

    # ── S/R & Fibonacci (Hybrid Context) ───────────────────────────────────
    # We use multiple periods for S/R to give the model multi-scale perspective
    df["sup_24"], df["res_24"] = _sr_levels(h, l, 24)   # 1 day S/R
    df["sup_168"], df["res_168"] = _sr_levels(h, l, 168) # 1 week S/R
    
    df["dist_sup_168"] = (c - df["sup_168"]) / c.replace(0, np.nan)
    df["dist_res_168"] = (df["res_168"] - c) / c.replace(0, np.nan)

    # Fibonacci Golden Pocket distance from a 30-day window (720 1H candles)
    df["fib_618_dist_180d"], df["high_180d"], df["low_180d"] = _fib_retracement(h, l, c, 720)
    
    # Position in 30-day range (0 = Low, 1 = High)
    df["range_pos_180d"] = (c - df["low_180d"]) / (df["high_180d"] - df["low_180d"]).replace(0, np.nan)

    # ── NEW: VWAP Distance ──────────────────────────────────────────────────
    typical_price = (h + l + c) / 3.0
    cum_tp_vol = (typical_price * df["volume"]).rolling(24).sum()
    cum_vol = df["volume"].rolling(24).sum()
    vwap_24 = cum_tp_vol / cum_vol.replace(0, np.nan)
    df["vwap_dist"] = (c - vwap_24) / c.replace(0, np.nan)

    # ── NEW: OBV Momentum ───────────────────────────────────────────────────
    obv_dir = np.where(c > c.shift(1), 1, np.where(c < c.shift(1), -1, 0))
    obv = (df["volume"] * obv_dir).cumsum()
    obv_ema = _ema(pd.Series(obv, index=df.index), 20)
    df["obv_momentum"] = (pd.Series(obv, index=df.index) - obv_ema) / obv_ema.replace(0, np.nan)

    # ── NEW: Ichimoku Cloud Signals ─────────────────────────────────────────
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    df["ichi_tk_diff"] = (tenkan - kijun) / c.replace(0, np.nan)
    df["ichi_cloud_dist"] = (c - senkou_a) / c.replace(0, np.nan)
    df["ichi_cloud_width"] = (senkou_a - senkou_b) / c.replace(0, np.nan)

    # ── Advanced Quant Features (Renaissance Technologies Style) ──────────────
    # HMM Regimes (Bull=0, Bear=1, Sideways=2)
    df["hmm_regime"] = compute_hmm_regimes(df["log_return"])
    
    # GARCH Volatility Prediction
    df["garch_vol"] = compute_garch_volatility(df["log_return"])
    
    # ── ARIMA Trend Residual (Mean Reversion vs Momentum) ─────────────────────
    df["arima_resid"] = compute_arima_forecast(df["log_return"])

    # ── NEW: Order Flow Features ──────────────────────────────────────────
    df["bbp"] = (c - l) / (h - l).replace(0, np.nan)
    df["vol_momentum"] = v / v_ema20.replace(0, np.nan)
    
    # ── Institutional Accuracy Features ─────────────────────────────────────
    df["cmf"] = _cmf(h, l, c, v, 20)
    df["hist_vol_20"] = df["log_return"].rolling(20).std() * np.sqrt(24 * 365)

    return df


def add_tbm_target(
    df: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
    horizon: int,
    ambiguous_resolution: str = "exclude",
    df_ltf: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Triple Barrier Method (TBM) labeling.
    This creates the "Answer Key" for the AI during training.
    
    Instead of just predicting if price goes up or down in exactly 5 hours, 
    this simulates a real trade:
    - Upper Barrier = Profit Target (PT)
    - Lower Barrier = Stop Loss (SL)
    - Time Barrier = Horizon (e.g. 12 hours max)
    
    If the price hits the upper barrier first, the answer is 1 (BUY).
    If it hits the lower barrier first, the answer is -1 (SELL).

    Ambiguous Resolution:
    When both PT and SL barriers are breached inside the same OHLC candle,
    the intra-candle sequence cannot be known from single-timeframe OHLC data alone.
    - "exclude" (default / recommended for directional training): Sets target to NaN.
      Discarding ambiguous samples prevents inducing systematic bearish label bias.
    - "stop_first": Assumes stop loss triggered first (-1.0). Reserved strictly for
      conservative trade simulation / backtest PnL accounting, NOT for directional labeling.
    - "overshoot" (legacy): Chooses barrier with larger overshoot beyond barrier price.
    - If `df_ltf` (lower-timeframe OHLC, e.g. 1m or 5m) is provided, sequential tick/candle
      order is evaluated directly to identify whether PT or SL was hit first chronologically.
    """
    df = df.copy()
    atr = df["atr_14"]
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    labels = []
    ambiguous_count = 0
    resolved_via_ltf = 0

    for i in range(len(df)):
        if i >= len(df) - horizon:
            labels.append(np.nan)
            continue
            
        p_price = close.iloc[i]
        # Use relative ATR to define percentage barriers
        vol_adj = atr.iloc[i] / p_price
        p_pt = p_price * (1 + pt_mult * vol_adj)
        p_sl = p_price * (1 - sl_mult * vol_adj)
        
        # Look ahead
        future_h = high.iloc[i+1 : i+1+horizon]
        future_l = low.iloc[i+1 : i+1+horizon]
        
        label = 0
        for j in range(len(future_h)):
            hit_pt = future_h.iloc[j] >= p_pt
            hit_sl = future_l.iloc[j] <= p_sl
            
            if hit_pt and not hit_sl:
                label = 1
                break
            elif hit_sl and not hit_pt:
                label = -1
                break
            elif hit_pt and hit_sl:
                # Both hit in same candle
                ambiguous_count += 1
                
                # Check if lower-timeframe resolution is possible
                if df_ltf is not None and not df_ltf.empty:
                    bar_time = future_h.index[j]
                    bar_delta = future_h.index[1] - future_h.index[0] if len(future_h) > 1 else pd.Timedelta("1h")
                    sub = df_ltf.loc[(df_ltf.index >= bar_time) & (df_ltf.index < bar_time + bar_delta)]
                    ltf_resolved = False
                    for _, sub_row in sub.iterrows():
                        sub_pt = sub_row["high"] >= p_pt
                        sub_sl = sub_row["low"] <= p_sl
                        if sub_pt and not sub_sl:
                            label = 1
                            ltf_resolved = True
                            resolved_via_ltf += 1
                            break
                        elif sub_sl and not sub_pt:
                            label = -1
                            ltf_resolved = True
                            resolved_via_ltf += 1
                            break
                    if ltf_resolved:
                        break

                if ambiguous_resolution == "exclude":
                    label = np.nan  # Unbiased default: discard from training
                elif ambiguous_resolution == "stop_first":
                    label = -1  # Stop-first (for trade simulation only)
                elif ambiguous_resolution == "overshoot":
                    label = 1 if (future_h.iloc[j] - p_pt) > (p_sl - future_l.iloc[j]) else -1
                else:
                    label = np.nan
                break
        
        # If no barrier hit, use sign of horizon return
        if label == 0:
            label = 1 if close.iloc[i+horizon] > p_price else -1
            
        labels.append(float(label) if not np.isnan(label) else np.nan)
        
    df["target"] = labels
    if ambiguous_count > 0:
        logger.info(
            "TBM labeling: encountered %d ambiguous bars (both PT & SL breached in same candle). "
            "Resolution: '%s' (discarded: %d, resolved via LTF: %d).",
            ambiguous_count, ambiguous_resolution,
            ambiguous_count - resolved_via_ltf if ambiguous_resolution == "exclude" else 0,
            resolved_via_ltf,
        )
    return df


def integrate_external_data(df: pd.DataFrame, funding_df: pd.DataFrame = None, oi_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge funding rate and Open Interest data into the main DataFrame."""
    df = df.copy()

    if funding_df is not None and not funding_df.empty:
        # Forward fill funding rate (it changes every 8h on Binance)
        df = pd.merge_asof(
            df, funding_df,
            left_index=True, right_index=True,
            direction="backward"
        )

    if oi_df is not None and not oi_df.empty:
        df = pd.merge_asof(
            df, oi_df,
            left_index=True, right_index=True,
            direction="backward"
        )
        # Calculate OI change %
        df["oi_pct_change"] = df["openInterestAmount"].pct_change()

    return df


def compute_mtf_features(df: pd.DataFrame, df_15m: pd.DataFrame = None, df_4h: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add features from other timeframes (15m, 4h).
    Ensures indicators are computed on the sub-timeframes before merging.
    """
    df = df.copy()
    
    # FIX: Prevent lookahead bias by merging on candle CLOSE times instead of OPEN times.
    # Otherwise a 1H candle at 13:00 would see the 4H candle (12:00-16:00) before it actually closed!
    orig_idx_col = df.index.name if df.index.name else "open_time"
    df.index.name = "open_time"
    df["_close_time"] = df.index + pd.Timedelta("1h")
    df = df.reset_index().set_index("_close_time")

    if df_15m is not None and not df_15m.empty:
        # Compute subset of indicators for 15m
        rsi_15m = _rsi(df_15m["close"], 14).to_frame("rsi_15m")
        ema20_15m = _ema(df_15m["close"], 20)
        ema50_15m = _ema(df_15m["close"], 50)
        ema_ratio_15m = (ema20_15m / ema50_15m.replace(0, np.nan)).to_frame("ema_ratio_15m")
        
        rsi_15m.index = rsi_15m.index + pd.Timedelta("15m")
        ema_ratio_15m.index = ema_ratio_15m.index + pd.Timedelta("15m")
        
        # Merge
        df = pd.merge_asof(df, rsi_15m, left_index=True, right_index=True, direction="backward")
        df = pd.merge_asof(df, ema_ratio_15m, left_index=True, right_index=True, direction="backward")

    if df_4h is not None and not df_4h.empty:
        # Compute subset of indicators for 4h
        rsi_4h = _rsi(df_4h["close"], 14).to_frame("rsi_4h")
        ema20_4h = _ema(df_4h["close"], 20)
        ema50_4h = _ema(df_4h["close"], 50)
        ema_ratio_4h = (ema20_4h / ema50_4h.replace(0, np.nan)).to_frame("ema_ratio_4h")
        
        rsi_4h.index = rsi_4h.index + pd.Timedelta("4h")
        ema_ratio_4h.index = ema_ratio_4h.index + pd.Timedelta("4h")
        
        df = pd.merge_asof(df, rsi_4h, left_index=True, right_index=True, direction="backward")
        df = pd.merge_asof(df, ema_ratio_4h, left_index=True, right_index=True, direction="backward")

    # Restore original index
    df = df.reset_index().set_index("open_time")
    if orig_idx_col and orig_idx_col != "open_time":
        df.index.name = orig_idx_col
    if "_close_time" in df.columns:
        df = df.drop(columns=["_close_time"])

    return df


def add_target(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    Add the regression target: forward log return over `horizon` candles.
    target = log(close[t+horizon] / close[t])
    The last `horizon` rows will have NaN targets and should be dropped.
    """
    df = df.copy()
    df["target"] = np.log(df["close"].shift(-horizon) / df["close"])
    return df
