"""
research/scripts/run_candidate_a_attribution.py

Rigorous Empirical Edge Attribution Analysis for Candidate A (Exp1_Momentum_Continuation_4H).
Analyzes:
1. Long vs Short performance
2. Trend vs Sideways regimes
3. Relative volume at entry
4. BTC trend condition at entry
5. ETH relative strength against BTC at entry
6. Funding rate conditions
7. Holding duration and exit reasons
8. MFE, MAE, and surrendered profit
9. First 12-hour profit trajectory
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.quant_evaluation import (
    compute_reconciled_trade_metrics,
    compute_deflated_sharpe_ratio
)


def wilson_score_interval(n_success: int, n_total: int, confidence: float = 0.95) -> tuple:
    """Computes Wilson score 95% confidence interval for a proportion."""
    if n_total == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% normal quantile
    p_hat = n_success / n_total
    denom = 1.0 + (z**2) / n_total
    center = (p_hat + (z**2) / (2.0 * n_total)) / denom
    half = (z * math.sqrt((p_hat * (1.0 - p_hat) / n_total) + ((z**2) / (4.0 * (n_total**2))))) / denom
    lower = max(0.0, (center - half) * 100.0)
    upper = min(100.0, (center + half) * 100.0)
    return (round(lower, 2), round(upper, 2))


def bootstrap_expectancy_ci(returns: np.ndarray, n_bootstraps: int = 2000, seed: int = 42) -> tuple:
    """Computes 95% bootstrap confidence interval for trade expectancy in bps."""
    if len(returns) == 0:
        return (0.0, 0.0)
    if len(returns) == 1:
        val = round(float(returns[0]) * 10000.0, 1)
        return (val, val)
    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(returns)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        boot_means.append(np.mean(returns[idx]) * 10000.0)
    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)
    return (round(float(lower), 1), round(float(upper), 1))


def compute_slice_metrics(trade_records: list) -> dict:
    """Computes detailed quantitative metrics for a slice of trade records."""
    n = len(trade_records)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate_pct": 0.0,
            "ci_95_win_rate": [0.0, 0.0],
            "profit_factor": 0.0,
            "expectancy_bps": 0.0,
            "ci_95_expectancy_bps": [0.0, 0.0],
            "avg_trade_pnl_usd": 0.0,
            "total_pnl_usd": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
            "avg_surrendered_profit_pct": 0.0,
            "mfe_mae_ratio": 0.0,
            "avg_holding_hours": 0.0
        }

    rets = np.array([t["net_return"] for t in trade_records], dtype=np.float64)
    pnls = np.array([t.get("trade_pnl_usd", 0.0) for t in trade_records], dtype=np.float64)
    mfes = np.array([t.get("mfe", 0.0) for t in trade_records], dtype=np.float64) * 100.0
    maes = np.array([t.get("mae", 0.0) for t in trade_records], dtype=np.float64) * 100.0
    surrs = np.array([t.get("surrendered_profit", 0.0) for t in trade_records], dtype=np.float64) * 100.0
    holds = np.array([t.get("holding_bars", 0) for t in trade_records], dtype=np.float64)

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    n_wins = len(wins)
    gross_win = float(wins.sum()) if n_wins > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    win_rate = (n_wins / n) * 100.0
    ci_wr = wilson_score_interval(n_wins, n)
    exp_bps = float(np.mean(rets)) * 10000.0
    ci_exp = bootstrap_expectancy_ci(rets)

    avg_mfe = float(np.mean(mfes))
    avg_mae = float(np.mean(maes))
    mfe_mae_ratio = (avg_mfe / avg_mae) if avg_mae > 0 else 99.0

    return {
        "n_trades": n,
        "win_rate_pct": round(win_rate, 2),
        "ci_95_win_rate": list(ci_wr),
        "profit_factor": round(pf, 2),
        "expectancy_bps": round(exp_bps, 1),
        "ci_95_expectancy_bps": list(ci_exp),
        "avg_trade_pnl_usd": round(float(np.mean(pnls)), 2),
        "total_pnl_usd": round(float(np.sum(pnls)), 2),
        "avg_mfe_pct": round(avg_mfe, 2),
        "avg_mae_pct": round(avg_mae, 2),
        "avg_surrendered_profit_pct": round(float(np.mean(surrs)), 2),
        "mfe_mae_ratio": round(mfe_mae_ratio, 2),
        "avg_holding_hours": round(float(np.mean(holds)), 1)
    }


def main():
    print("=" * 70)
    print("CANDIDATE A (Exp1_Momentum_Continuation_4H) EDGE ATTRIBUTION ANALYSIS")
    print("=" * 70)

    # 1. Load Data
    cache_dir = os.path.join(BASE_DIR, "data", "cache")
    eth_path = os.path.join(cache_dir, "eth_usdt_1h_3yr.parquet")
    btc_path = os.path.join(cache_dir, "btc_usdt_1h_3yr.parquet")
    funding_path = os.path.join(cache_dir, "eth_funding_3yr.parquet")

    df_eth_1h = pd.read_parquet(eth_path)
    df_btc_1h = pd.read_parquet(btc_path)
    df_funding = pd.read_parquet(funding_path)

    # Ensure index is datetime
    if not isinstance(df_eth_1h.index, pd.DatetimeIndex):
        df_eth_1h.index = pd.to_datetime(df_eth_1h.index, utc=True)
    if not isinstance(df_btc_1h.index, pd.DatetimeIndex):
        df_btc_1h.index = pd.to_datetime(df_btc_1h.index, utc=True)
    if not isinstance(df_funding.index, pd.DatetimeIndex):
        df_funding.index = pd.to_datetime(df_funding.index, utc=True)

    print(f"Loaded ETH 1H: {len(df_eth_1h)} bars ({df_eth_1h.index[0]} to {df_eth_1h.index[-1]})")
    print(f"Loaded BTC 1H: {len(df_btc_1h)} bars ({df_btc_1h.index[0]} to {df_btc_1h.index[-1]})")
    print(f"Loaded Funding: {len(df_funding)} records")

    # 2. Resample and calculate multi-timeframe indicators
    # ETH 4H
    df_eth_4h = df_eth_1h.resample("4h", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    df_eth_4h["ema_20"] = df_eth_4h["close"].ewm(span=20, adjust=False).mean()
    df_eth_4h["ema_50"] = df_eth_4h["close"].ewm(span=50, adjust=False).mean()
    df_eth_4h["vol_sma_20"] = df_eth_4h["volume"].rolling(20).mean()
    df_eth_4h["vol_ratio"] = df_eth_4h["volume"] / df_eth_4h["vol_sma_20"].replace(0, np.nan)
    
    # ATR 14 on 4H
    tr1 = df_eth_4h["high"] - df_eth_4h["low"]
    tr2 = (df_eth_4h["high"] - df_eth_4h["close"].shift(1)).abs()
    tr3 = (df_eth_4h["low"] - df_eth_4h["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_eth_4h["atr_14"] = tr.rolling(14).mean()

    # ADX 14 on 4H
    up_move = df_eth_4h["high"] - df_eth_4h["high"].shift(1)
    down_move = df_eth_4h["low"].shift(1) - df_eth_4h["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100.0 * (pd.Series(plus_dm, index=df_eth_4h.index).rolling(14).mean() / df_eth_4h["atr_14"].replace(0, 1e-6))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df_eth_4h.index).rolling(14).mean() / df_eth_4h["atr_14"].replace(0, 1e-6))
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-6))
    df_eth_4h["adx_14"] = dx.rolling(14).mean()

    # Bollinger Band Width for Regime detection
    bb_mid = df_eth_4h["close"].rolling(20).mean()
    bb_std = df_eth_4h["close"].rolling(20).std()
    bb_width = (2.0 * bb_std) / bb_mid.replace(0, np.nan)
    df_eth_4h["bb_width"] = bb_width

    # Regime definition:
    # 1: Compression (bb_width < 20th percentile)
    # 3: Strong Trend (adx_14 > 25)
    # 5: Sideways/Chop (adx_14 < 20)
    # 2: Breakout/Expansion (other)
    bb_p20 = bb_width.quantile(0.20)
    regimes = np.where(bb_width <= bb_p20, 1,
              np.where(df_eth_4h["adx_14"] >= 25, 3,
              np.where(df_eth_4h["adx_14"] <= 20, 5, 2)))
    df_eth_4h["regime"] = regimes

    # BTC 4H and 1H indicators
    df_btc_4h = df_btc_1h.resample("4h", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    df_btc_4h["ema_50"] = df_btc_4h["close"].ewm(span=50, adjust=False).mean()
    df_btc_4h["ema_200"] = df_btc_4h["close"].ewm(span=200, adjust=False).mean()

    # BTC ADX on 4H
    b_tr1 = df_btc_4h["high"] - df_btc_4h["low"]
    b_tr2 = (df_btc_4h["high"] - df_btc_4h["close"].shift(1)).abs()
    b_tr3 = (df_btc_4h["low"] - df_btc_4h["close"].shift(1)).abs()
    b_tr = pd.concat([b_tr1, b_tr2, b_tr3], axis=1).max(axis=1)
    b_atr = b_tr.rolling(14).mean()
    b_up = df_btc_4h["high"] - df_btc_4h["high"].shift(1)
    b_down = df_btc_4h["low"].shift(1) - df_btc_4h["low"]
    b_pdm = np.where((b_up > b_down) & (b_up > 0), b_up, 0.0)
    b_mdm = np.where((b_down > b_up) & (b_down > 0), b_down, 0.0)
    b_pdi = 100.0 * (pd.Series(b_pdm, index=df_btc_4h.index).rolling(14).mean() / b_atr.replace(0, 1e-6))
    b_mdi = 100.0 * (pd.Series(b_mdm, index=df_btc_4h.index).rolling(14).mean() / b_atr.replace(0, 1e-6))
    b_dx = 100.0 * ((b_pdi - b_mdi).abs() / (b_pdi + b_mdi).replace(0, 1e-6))
    df_btc_4h["adx_14"] = b_dx.rolling(14).mean()

    # Trailing 24H Return on 1H
    df_eth_1h["ret_24h"] = (df_eth_1h["close"] - df_eth_1h["close"].shift(24)) / df_eth_1h["close"].shift(24)
    df_btc_1h["ret_24h"] = (df_btc_1h["close"] - df_btc_1h["close"].shift(24)) / df_btc_1h["close"].shift(24)

    # Synchronize funding rates
    df_funding_1h = df_funding.reindex(df_eth_1h.index, method="ffill")

    # 3. Load Phase 2.1 Walk-Forward Trades for Candidate A
    trade_log_path = os.path.join(BASE_DIR, "research", "phase_2_1_fold_trade_log.csv")
    df_trades_all = pd.read_csv(trade_log_path)
    df_trades = df_trades_all[df_trades_all["strategy"] == "Exp1_Momentum_Continuation_4H"].copy()
    print(f"Loaded {len(df_trades)} Walk-Forward Trades for Exp1_Momentum_Continuation_4H")

    # 4. Enrich Each Trade with Causal Entry Features and Intratrade Path Trajectory
    attributed_trades = []

    for _, row in df_trades.iterrows():
        t_id = row["trade_id"]
        fold = row["fold"]
        dec_t = pd.to_datetime(row["decision_time"])
        entry_t = pd.to_datetime(row["entry_time"])
        exit_t = pd.to_datetime(row["exit_time"])
        side = row["side"].upper()
        entry_px = float(row["entry_price"])
        exit_px = float(row["exit_price"])
        stop_px = float(row["stop_price"])
        holding_bars = int(row["holding_bars"])
        exit_reason = row["exit_reason"]
        net_ret = float(row["net_return"])
        trade_pnl = float(row["trade_pnl_usd"])

        # Locate decision bar in 4H dataframe
        # Decision time corresponds to the close of that 4H candle (i.e. timestamp = dec_t - 4h if left-labeled)
        # Check matching 4H row
        cand_4h = df_eth_4h[df_eth_4h.index <= dec_t]
        last_4h_idx = cand_4h.index[-1] if len(cand_4h) > 0 else None
        
        regime_val = int(cand_4h.loc[last_4h_idx, "regime"]) if last_4h_idx is not None else 0
        adx_val = float(cand_4h.loc[last_4h_idx, "adx_14"]) if last_4h_idx is not None else 0.0
        vol_ratio_val = float(cand_4h.loc[last_4h_idx, "vol_ratio"]) if last_4h_idx is not None else 1.0

        # BTC Trend at decision time
        cand_btc_4h = df_btc_4h[df_btc_4h.index <= dec_t]
        last_b_idx = cand_btc_4h.index[-1] if len(cand_btc_4h) > 0 else None
        btc_close = float(cand_btc_4h.loc[last_b_idx, "close"]) if last_b_idx is not None else 0.0
        btc_ema50 = float(cand_btc_4h.loc[last_b_idx, "ema_50"]) if last_b_idx is not None else 0.0
        btc_adx = float(cand_btc_4h.loc[last_b_idx, "adx_14"]) if last_b_idx is not None else 0.0

        btc_trend_bullish = btc_close > btc_ema50
        btc_trend_bearish = btc_close < btc_ema50
        btc_trending = btc_adx >= 20.0

        btc_trend_status = (
            "BULLISH_TRENDING" if (btc_trend_bullish and btc_trending) else
            "BULLISH_RANGING" if btc_trend_bullish else
            "BEARISH_TRENDING" if (btc_trend_bearish and btc_trending) else
            "BEARISH_RANGING"
        )
        btc_trend_aligned = (
            (side == "LONG" and btc_trend_bullish) or
            (side == "SHORT" and btc_trend_bearish)
        )

        # ETH Relative Strength against BTC over preceding 24h
        cand_eth_1h = df_eth_1h[df_eth_1h.index <= entry_t]
        cand_b_1h = df_btc_1h[df_btc_1h.index <= entry_t]
        eth_ret_24h = float(cand_eth_1h.iloc[-1]["ret_24h"]) if len(cand_eth_1h) > 0 else 0.0
        btc_ret_24h = float(cand_b_1h.iloc[-1]["ret_24h"]) if len(cand_b_1h) > 0 else 0.0
        rs_24h = (eth_ret_24h - btc_ret_24h) * 100.0  # in percentage points
        rs_status = "ETH_OUTPERFORMING" if rs_24h > 0 else "ETH_UNDERPERFORMING"

        # Funding rate condition at entry
        cand_f = df_funding_1h[df_funding_1h.index <= entry_t]
        entry_funding_rate = float(cand_f.iloc[-1]["fundingRate"]) if len(cand_f) > 0 and "fundingRate" in cand_f.columns else 0.0
        if entry_funding_rate > 0.0001:  # > +0.01%
            funding_condition = "POSITIVE_HIGH"
        elif entry_funding_rate > 0.00003: # > +0.003%
            funding_condition = "POSITIVE_MILD"
        elif entry_funding_rate < -0.00003:
            funding_condition = "NEGATIVE"
        else:
            funding_condition = "NEUTRAL"

        # Intratrade Hourly Path Analysis
        bars_held = df_eth_1h[(df_eth_1h.index >= entry_t) & (df_eth_1h.index <= exit_t)]
        n_held = len(bars_held)

        pnl_1h = pnl_2h = pnl_4h = pnl_8h = pnl_12h = 0.0
        mfe_val = 0.0
        mae_val = 0.0

        if n_held > 0:
            highs = bars_held["high"].values
            lows = bars_held["low"].values
            closes = bars_held["close"].values

            if side == "LONG":
                mfe_val = float(np.max(highs) - entry_px) / entry_px
                mae_val = float(entry_px - np.min(lows)) / entry_px
                # Path returns at specific bar offsets
                pnl_1h = (closes[min(0, n_held - 1)] - entry_px) / entry_px
                pnl_2h = (closes[min(1, n_held - 1)] - entry_px) / entry_px
                pnl_4h = (closes[min(3, n_held - 1)] - entry_px) / entry_px
                pnl_8h = (closes[min(7, n_held - 1)] - entry_px) / entry_px
                pnl_12h = (closes[min(11, n_held - 1)] - entry_px) / entry_px
            else:
                mfe_val = float(entry_px - np.min(lows)) / entry_px
                mae_val = float(np.max(highs) - entry_px) / entry_px
                pnl_1h = (entry_px - closes[min(0, n_held - 1)]) / entry_px
                pnl_2h = (entry_px - closes[min(1, n_held - 1)]) / entry_px
                pnl_4h = (entry_px - closes[min(3, n_held - 1)]) / entry_px
                pnl_8h = (entry_px - closes[min(7, n_held - 1)]) / entry_px
                pnl_12h = (entry_px - closes[min(11, n_held - 1)]) / entry_px

        surrendered = max(0.0, mfe_val - net_ret)
        first_12h_status = "POSITIVE_AT_12H" if pnl_12h > 0 else "NEGATIVE_AT_12H"

        # Categorize relative volume
        if vol_ratio_val >= 2.5:
            vol_tier = "EXTREME_VOL (>2.5x)"
        elif vol_ratio_val >= 1.8:
            vol_tier = "HIGH_VOL (1.8-2.5x)"
        elif vol_ratio_val >= 1.4:
            vol_tier = "MODERATE_VOL (1.4-1.8x)"
        else:
            vol_tier = "BASELINE_VOL (<1.4x)"

        # Regime name
        regime_names = {
            1: "Regime 1: Compression",
            2: "Regime 2: Expansion",
            3: "Regime 3: Strong Trend",
            5: "Regime 5: Chop/Sideways"
        }
        regime_label = regime_names.get(regime_val, f"Regime {regime_val}")

        record = {
            "trade_id": t_id,
            "fold": fold,
            "side": side,
            "decision_time": str(dec_t),
            "entry_time": str(entry_t),
            "exit_time": str(exit_t),
            "holding_bars": holding_bars,
            "exit_reason": exit_reason,
            "net_return": net_ret,
            "net_return_pct": round(net_ret * 100.0, 3),
            "trade_pnl_usd": round(trade_pnl, 2),
            "mfe": round(mfe_val, 4),
            "mae": round(mae_val, 4),
            "surrendered_profit": round(surrendered, 4),
            "regime": regime_label,
            "adx_4h": round(adx_val, 1),
            "vol_ratio": round(vol_ratio_val, 2),
            "vol_tier": vol_tier,
            "btc_trend_status": btc_trend_status,
            "btc_trend_aligned": btc_trend_aligned,
            "rs_24h_pct": round(rs_24h, 2),
            "rs_status": rs_status,
            "funding_rate": round(entry_funding_rate, 6),
            "funding_condition": funding_condition,
            "pnl_1h_pct": round(pnl_1h * 100.0, 3),
            "pnl_2h_pct": round(pnl_2h * 100.0, 3),
            "pnl_4h_pct": round(pnl_4h * 100.0, 3),
            "pnl_8h_pct": round(pnl_8h * 100.0, 3),
            "pnl_12h_pct": round(pnl_12h * 100.0, 3),
            "first_12h_status": first_12h_status
        }
        attributed_trades.append(record)

    # Convert to DataFrame
    df_attr = pd.DataFrame(attributed_trades)
    csv_out = os.path.join(BASE_DIR, "research", "candidate_a_trade_attribution.csv")
    df_attr.to_csv(csv_out, index=False)
    print(f"Saved trade attribution CSV: {csv_out}")

    # 5. Compute Quantitative Slices
    slices = {}

    # Slice 1: Long vs Short
    slices["side"] = {
        "LONG": compute_slice_metrics([t for t in attributed_trades if t["side"] == "LONG"]),
        "SHORT": compute_slice_metrics([t for t in attributed_trades if t["side"] == "SHORT"])
    }

    # Slice 2: Market Regime
    reg_keys = sorted(list(set(t["regime"] for t in attributed_trades)))
    slices["regime"] = {
        rk: compute_slice_metrics([t for t in attributed_trades if t["regime"] == rk])
        for rk in reg_keys
    }

    # Slice 3: Relative Volume Tiers
    vt_keys = ["EXTREME_VOL (>2.5x)", "HIGH_VOL (1.8-2.5x)", "MODERATE_VOL (1.4-1.8x)", "BASELINE_VOL (<1.4x)"]
    slices["volume_tier"] = {
        vk: compute_slice_metrics([t for t in attributed_trades if t["vol_tier"] == vk])
        for vk in vt_keys if any(t["vol_tier"] == vk for t in attributed_trades)
    }

    # Slice 4: BTC Trend Alignment
    slices["btc_trend_alignment"] = {
        "ALIGNED_WITH_BTC": compute_slice_metrics([t for t in attributed_trades if t["btc_trend_aligned"]]),
        "COUNTER_BTC_TREND": compute_slice_metrics([t for t in attributed_trades if not t["btc_trend_aligned"]])
    }

    # Slice 5: ETH Relative Strength against BTC
    slices["eth_relative_strength"] = {
        "ETH_OUTPERFORMING_BTC": compute_slice_metrics([t for t in attributed_trades if t["rs_status"] == "ETH_OUTPERFORMING"]),
        "ETH_UNDERPERFORMING_BTC": compute_slice_metrics([t for t in attributed_trades if t["rs_status"] == "ETH_UNDERPERFORMING"])
    }

    # Slice 6: Funding Rate Condition
    fund_keys = ["POSITIVE_HIGH", "POSITIVE_MILD", "NEUTRAL", "NEGATIVE"]
    slices["funding_condition"] = {
        fk: compute_slice_metrics([t for t in attributed_trades if t["funding_condition"] == fk])
        for fk in fund_keys if any(t["funding_condition"] == fk for t in attributed_trades)
    }

    # Slice 7: Exit Reason
    exit_keys = sorted(list(set(t["exit_reason"] for t in attributed_trades)))
    slices["exit_reason"] = {
        ek: compute_slice_metrics([t for t in attributed_trades if t["exit_reason"] == ek])
        for ek in exit_keys
    }

    # Slice 8: First 12-Hour Trajectory
    slices["first_12h_trajectory"] = {
        "POSITIVE_AT_12H": compute_slice_metrics([t for t in attributed_trades if t["first_12h_status"] == "POSITIVE_AT_12H"]),
        "NEGATIVE_AT_12H": compute_slice_metrics([t for t in attributed_trades if t["first_12h_status"] == "NEGATIVE_AT_12H"])
    }

    # Overall Baseline
    slices["overall_pooled"] = compute_slice_metrics(attributed_trades)

    # Save to JSON
    json_out = os.path.join(BASE_DIR, "research", "candidate_a_attribution.json")
    with open(json_out, "w") as f:
        json.dump(slices, f, indent=2)
    print(f"Saved trade attribution JSON: {json_out}")

    print("\n--- ATTRIBUTION SUMMARY ---")
    print(f"Total Trades: {slices['overall_pooled']['n_trades']}")
    print(f"Win Rate: {slices['overall_pooled']['win_rate_pct']}% (95% CI: {slices['overall_pooled']['ci_95_win_rate']})")
    print(f"Profit Factor: {slices['overall_pooled']['profit_factor']}")
    print(f"Expectancy: {slices['overall_pooled']['expectancy_bps']} bps (95% CI: {slices['overall_pooled']['ci_95_expectancy_bps']})")
    print(f"Total PnL: ${slices['overall_pooled']['total_pnl_usd']}")
    print("\nSide Split:")
    print(f"  LONG:  N={slices['side']['LONG']['n_trades']} | Win={slices['side']['LONG']['win_rate_pct']}% | Exp={slices['side']['LONG']['expectancy_bps']} bps | PnL=${slices['side']['LONG']['total_pnl_usd']}")
    print(f"  SHORT: N={slices['side']['SHORT']['n_trades']} | Win={slices['side']['SHORT']['win_rate_pct']}% | Exp={slices['side']['SHORT']['expectancy_bps']} bps | PnL=${slices['side']['SHORT']['total_pnl_usd']}")
    print("\nBTC Alignment:")
    print(f"  ALIGNED: N={slices['btc_trend_alignment']['ALIGNED_WITH_BTC']['n_trades']} | Win={slices['btc_trend_alignment']['ALIGNED_WITH_BTC']['win_rate_pct']}% | Exp={slices['btc_trend_alignment']['ALIGNED_WITH_BTC']['expectancy_bps']} bps | PnL=${slices['btc_trend_alignment']['ALIGNED_WITH_BTC']['total_pnl_usd']}")
    print(f"  COUNTER: N={slices['btc_trend_alignment']['COUNTER_BTC_TREND']['n_trades']} | Win={slices['btc_trend_alignment']['COUNTER_BTC_TREND']['win_rate_pct']}% | Exp={slices['btc_trend_alignment']['COUNTER_BTC_TREND']['expectancy_bps']} bps | PnL=${slices['btc_trend_alignment']['COUNTER_BTC_TREND']['total_pnl_usd']}")
    print("\nFirst 12-Hour Trajectory:")
    print(f"  POSITIVE AT 12H: N={slices['first_12h_trajectory']['POSITIVE_AT_12H']['n_trades']} | Win={slices['first_12h_trajectory']['POSITIVE_AT_12H']['win_rate_pct']}% | Exp={slices['first_12h_trajectory']['POSITIVE_AT_12H']['expectancy_bps']} bps | PnL=${slices['first_12h_trajectory']['POSITIVE_AT_12H']['total_pnl_usd']}")
    print(f"  NEGATIVE AT 12H: N={slices['first_12h_trajectory']['NEGATIVE_AT_12H']['n_trades']} | Win={slices['first_12h_trajectory']['NEGATIVE_AT_12H']['win_rate_pct']}% | Exp={slices['first_12h_trajectory']['NEGATIVE_AT_12H']['expectancy_bps']} bps | PnL=${slices['first_12h_trajectory']['NEGATIVE_AT_12H']['total_pnl_usd']}")


if __name__ == "__main__":
    main()
