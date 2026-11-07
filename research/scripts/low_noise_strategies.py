"""
research/scripts/low_noise_strategies.py

PRE-REGISTERED low-noise strategies F0 / G0 / G1 / G2 plus a small event-driven 1H simulator.

The validated Phase 2.1 engine (simulate_swing_trades_conservative) is hard-wired to 4H signals and
fixed-ATR take-profit exits, so it cannot express 6H trailing stops, cooldowns or continuous
vol-targeted rebalancing. The engine is NOT modified. This module reuses the validated utilities
(`resample_completed_ohlcv`, `calculate_actual_stop_position_sizing`) and follows the same
conventions: signal at bar close -> fill at NEXT bar open, intrabar high/low stop checks on 1H bars,
opening gaps fill at the open, stop-first, funding applied only while a position is open.

ALL RULES / PARAMETERS BELOW ARE FIXED BEFORE ANY RESULT IS SEEN. NO TUNING.
"""
import os
import sys
import math
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.quant_evaluation import resample_completed_ohlcv, calculate_actual_stop_position_sizing

# ───────────────────────────── pre-registered constants ─────────────────────────────
FEE_BASE = 0.0010          # 10 bps per side (fee + slippage)
FEE_DOUBLE = 0.0020        # 20 bps per side
MAX_LEVERAGE = 2.0
RISK_PCT = 0.01
INITIAL_EQUITY = 10000.0
N_TRIALS = 9               # 5 earlier candidates + these 4

DONCHIAN_N = 20
ATR_MULT = 3.0
SHORT_FRAC = 0.30
COOLDOWN_BARS = 4
EMA_TREND = 100
ER_LEN = 30
ER_MEDIAN_BARS = 365 * 4   # 365 days of 6H bars
F0_LOOKBACKS = (20, 50, 100, 200)
F0_VOL_TARGET = 0.25
F0_BAND = 0.25
F0_VOL_WINDOW = 20

DEV_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
HOLDOUT_ONLY_SYMBOLS = ["ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT"]
ALL_SYMBOLS = DEV_SYMBOLS + HOLDOUT_ONLY_SYMBOLS
DEV_END = pd.Timestamp("2024-12-31 23:00:00+00:00")
HOLDOUT_B_START = pd.Timestamp("2025-01-01 00:00:00+00:00")
ETH_SEEN_START = pd.Timestamp("2026-04-01 00:00:00+00:00")

CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

# ───────────────────────────── guarded data loading ─────────────────────────────
_HOLDOUT_UNLOCKED = False


def unlock_holdout() -> None:
    """Called exactly once, by the single final holdout run."""
    global _HOLDOUT_UNLOCKED
    _HOLDOUT_UNLOCKED = True


def lock_holdout() -> None:
    global _HOLDOUT_UNLOCKED
    _HOLDOUT_UNLOCKED = False


def holdout_unlocked() -> bool:
    return _HOLDOUT_UNLOCKED


def load_symbol(symbol: str, cache_dir: Optional[str] = None):
    """Returns (df_1h, funding_series). While locked: holdout-only symbols are refused and
    development symbols are truncated at DEV_END (so post-2024 data is never even in memory)."""
    cache_dir = cache_dir or CACHE_DIR
    sym = symbol.upper()
    if sym not in ALL_SYMBOLS:
        raise ValueError(f"Unknown symbol {sym}")
    if sym in HOLDOUT_ONLY_SYMBOLS and not _HOLDOUT_UNLOCKED:
        raise PermissionError(f"{sym} is holdout-only; holdout is locked until the final run.")
    df = pd.read_parquet(os.path.join(cache_dir, f"{sym.lower()}_1h_full.parquet"))
    fd = pd.read_parquet(os.path.join(cache_dir, f"{sym.lower()}_funding_full.parquet"))
    fser = normalize_funding(fd)
    if not _HOLDOUT_UNLOCKED:
        df = df.loc[:DEV_END]
        fser = fser.loc[:DEV_END]
    return df, fser


def normalize_funding(fd: pd.DataFrame) -> pd.Series:
    """Round settlement timestamps (sub-second jitter) to the hour; dedupe; rate series."""
    s = fd["funding_rate"].copy()
    s.index = s.index.round("h")
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s.astype("float64")


# ───────────────────────────── causal indicators / signals ─────────────────────────────
def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def efficiency_ratio(c: pd.Series, n: int = ER_LEN) -> pd.Series:
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()


def g_features(df_1h: pd.DataFrame, variant: str = "G0", donchian_n: int = DONCHIAN_N) -> pd.DataFrame:
    """6H feature frame indexed by bar OPEN time (bar closes at index+6h). Every value at row t
    uses data <= bar t only. Columns: close, atr, long_sig, short_sig."""
    d6 = resample_completed_ohlcv(df_1h, target_freq="6h", lower_freq="1h")
    c = d6["close"]
    atr = _atr(d6["high"], d6["low"], c)
    s = c.ewm(span=5, adjust=False).mean() if variant == "G2" else c
    ema = s.ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean()
    hi = s.shift(1).rolling(donchian_n).max()
    lo = s.shift(1).rolling(donchian_n).min()
    long_sig = (s > hi) & (s > ema)
    short_sig = (s < lo) & (s < ema)
    if variant == "G1":
        er = efficiency_ratio(c, ER_LEN)
        med = er.rolling(ER_MEDIAN_BARS, min_periods=ER_MEDIAN_BARS).median()
        gate = er > med
        long_sig = long_sig & gate
        short_sig = short_sig & gate
    out = pd.DataFrame({"close": c, "atr": atr,
                        "long_sig": long_sig.fillna(False).astype(bool),
                        "short_sig": short_sig.fillna(False).astype(bool)})
    return out


def f0_features(df_1h: pd.DataFrame, vol_target: float = F0_VOL_TARGET, long_only: bool = False) -> pd.DataFrame:
    """Daily feature frame indexed by day OPEN (closes at +24h). target = signed leverage target."""
    d1 = resample_completed_ohlcv(df_1h, target_freq="1D", lower_freq="1h")
    c = d1["close"]
    signs = pd.concat([np.sign(c - c.shift(n)).where(c.shift(n).notna()) for n in F0_LOOKBACKS], axis=1)
    score = signs.mean(axis=1, skipna=False)
    sma200 = c.rolling(200).mean()
    rv = np.log(c).diff().rolling(F0_VOL_WINDOW).std() * math.sqrt(365.0)
    if long_only:
        pos_score = score.clip(lower=0.0).where(c >= sma200, 0.0)
        pos_score = pos_score.where(sma200.notna())
    else:
        pos_score = score
    target = (pos_score * (vol_target / rv)).clip(-MAX_LEVERAGE, MAX_LEVERAGE)
    return pd.DataFrame({"close": c, "score": score, "rv": rv, "target": target})


# ───────────────────────────── simulator helpers ─────────────────────────────
def _funding_array(funding: Optional[pd.Series], idx: pd.DatetimeIndex) -> np.ndarray:
    if funding is None or len(funding) == 0:
        return np.zeros(len(idx))
    return funding.reindex(idx).fillna(0.0).values.astype("float64")


def _event_map(idx: pd.DatetimeIndex, feat: pd.DataFrame, bar_hours: int) -> np.ndarray:
    """ev[i] = k if 1H bar i is the LAST hour of feature bar k (bar k closes at idx[i]+1h)."""
    ev = np.full(len(idx), -1, dtype=np.int64)
    last_1h = feat.index + pd.Timedelta(hours=bar_hours) - pd.Timedelta(hours=1)
    pos = idx.get_indexer(last_1h)
    ks = np.arange(len(feat))
    m = pos >= 0
    ev[pos[m]] = ks[m]
    return ev


def _result(symbol, strategy, idx, i0, eq_c, eq_l, trades, fees_total, fund_total, expo_bars, lev_sum, fee):
    index = idx[i0:]
    eq = pd.Series(eq_c, index=index)
    eql = pd.Series(np.minimum(eq_l, eq_c), index=index)
    n_bars = max(1, len(index))
    return {
        "symbol": symbol, "strategy": strategy,
        "equity": eq, "equity_low": eql, "trades": trades,
        "fees_usd": float(fees_total), "funding_usd": float(fund_total),
        "fee_rate_per_side": fee,
        "exposure_frac": expo_bars / n_bars,
        "avg_gross_leverage": lev_sum / n_bars,
        "avg_gross_leverage_in_market": (lev_sum / expo_bars) if expo_bars else 0.0,
    }


def simulate_g(df_1h: pd.DataFrame, funding: Optional[pd.Series], feat6: pd.DataFrame, *,
               trade_start: Optional[pd.Timestamp] = None, atr_mult: float = ATR_MULT,
               short_frac: float = SHORT_FRAC, cooldown_bars: int = COOLDOWN_BARS,
               fee: float = FEE_BASE, symbol: str = "SYM", strategy: str = "G0",
               risk_pct: float = RISK_PCT, max_leverage: float = MAX_LEVERAGE,
               initial: float = INITIAL_EQUITY) -> Dict[str, Any]:
    idx = df_1h.index
    n = len(idx)
    o = df_1h["open"].values.astype("float64")
    h = df_1h["high"].values.astype("float64")
    l = df_1h["low"].values.astype("float64")
    c = df_1h["close"].values.astype("float64")
    f_arr = _funding_array(funding, idx)
    i0 = 0 if trade_start is None else int(idx.searchsorted(trade_start))
    ev = _event_map(idx, feat6, 6)
    cl6 = feat6["close"].values
    atr6 = feat6["atr"].values
    ls = feat6["long_sig"].values
    ss = feat6["short_sig"].values

    cash = float(initial)
    units = 0.0
    side = 0
    stop = float("nan")
    best = float("nan")
    pend_side = 0
    pend_stop = float("nan")
    exit_bar = -10 ** 9
    last_k = -1
    fees_total = 0.0
    fund_total = 0.0
    expo_bars = 0
    lev_sum = 0.0
    ep = {}
    trades: List[Dict[str, Any]] = []
    m = n - i0
    eq_c = np.empty(m)
    eq_l = np.empty(m)

    def close_position(px: float, i: int, reason: str):
        nonlocal cash, units, side, fees_total, exit_bar
        fee_usd = abs(units) * px * fee
        cash += units * px - fee_usd
        fees_total += fee_usd
        ep["fee"] += fee_usd
        pnl = cash - ep["start_eq"]
        peak = ep["peak"]
        trades.append({
            "trade_id": len(trades) + 1, "symbol": symbol, "strategy": strategy,
            "side": "LONG" if side == 1 else "SHORT",
            "entry_time": str(idx[ep["i"]]), "exit_time": str(idx[i]),
            "holding_bars": float(i - ep["i"]),
            "entry_price": ep["px"], "exit_price": float(px), "notional": peak,
            "effective_leverage": peak / ep["start_eq"] if ep["start_eq"] else 0.0,
            "exit_reason": reason,
            "net_return": pnl / peak if peak else 0.0,
            "gross_return": (pnl + ep["fee"] + ep["fund"]) / peak if peak else 0.0,
            "trade_pnl_usd": pnl,
            "entry_fee_cost": ep["fee"] / peak if peak else 0.0,
            "exit_fee_cost": 0.0, "total_slippage_cost": 0.0,
            "funding_cost": ep["fund"] / peak if peak else 0.0,
            "fee_usd": ep["fee"], "funding_usd": ep["fund"],
        })
        units = 0.0
        side = 0
        exit_bar = last_k + 1

    for i in range(i0, n):
        oi = o[i]
        hi_ = h[i]
        li_ = l[i]
        ci = c[i]
        r = f_arr[i]
        if units != 0.0 and r != 0.0:
            fc = units * oi * r          # positive = paid (long pays when rate > 0)
            cash -= fc
            ep["fund"] += fc
            fund_total += fc
        if pend_side != 0:
            s_new = pend_side
            pend_side = 0
            if cash > 0:
                sz = calculate_actual_stop_position_sizing(cash, oi, pend_stop, risk_pct, max_leverage)
                notional = sz["notional"] * (short_frac if s_new == -1 else 1.0)
                if notional > 0:
                    units = s_new * notional / oi
                    fee_usd = notional * fee
                    fees_total += fee_usd
                    ep = {"start_eq": cash, "i": i, "px": float(oi), "peak": notional,
                          "fee": fee_usd, "fund": 0.0}
                    cash -= units * oi + fee_usd
                    side = s_new
                    stop = pend_stop
                    best = float("nan")
        if units != 0.0:
            px = 0.0
            reason = ""
            if side == 1:
                if oi <= stop:
                    px, reason = oi, "stop_gap"
                elif li_ <= stop:
                    px, reason = stop, "trailing_stop"
            else:
                if oi >= stop:
                    px, reason = oi, "stop_gap"
                elif hi_ >= stop:
                    px, reason = stop, "trailing_stop"
            if px != 0.0:
                close_position(px, i, reason)
        j = i - i0
        if units != 0.0:
            expo_bars += 1
            lev_sum += abs(units) * ci / max(cash + units * ci, 1e-9)
            eq_c[j] = cash + units * ci
            eq_l[j] = cash + units * (li_ if units > 0 else hi_)
        else:
            eq_c[j] = cash
            eq_l[j] = cash
        k = ev[i]
        if k >= 0:
            last_k = k
            ck = cl6[k]
            ak = atr6[k]
            if units != 0.0:
                if not math.isnan(ak):
                    if side == 1:
                        best = ck if math.isnan(best) else max(best, ck)
                        stop = max(stop, best - atr_mult * ak)
                    else:
                        best = ck if math.isnan(best) else min(best, ck)
                        stop = min(stop, best + atr_mult * ak)
            elif i + 1 < n and (k - exit_bar) >= cooldown_bars and not math.isnan(ak):
                if ls[k]:
                    pend_side, pend_stop = 1, ck - atr_mult * ak
                elif ss[k]:
                    pend_side, pend_stop = -1, ck + atr_mult * ak
    if units != 0.0:
        close_position(c[n - 1], n - 1, "end_of_window")
        eq_c[m - 1] = cash
        eq_l[m - 1] = min(eq_l[m - 1], cash)
    return _result(symbol, strategy, idx, i0, eq_c, eq_l, trades, fees_total, fund_total,
                   expo_bars, lev_sum, fee)


def simulate_f0(df_1h: pd.DataFrame, funding: Optional[pd.Series], feat1d: pd.DataFrame, *,
                trade_start: Optional[pd.Timestamp] = None, band: float = F0_BAND,
                fee: float = FEE_BASE, symbol: str = "SYM", strategy: str = "F0",
                initial: float = INITIAL_EQUITY) -> Dict[str, Any]:
    idx = df_1h.index
    n = len(idx)
    o = df_1h["open"].values.astype("float64")
    h = df_1h["high"].values.astype("float64")
    l = df_1h["low"].values.astype("float64")
    c = df_1h["close"].values.astype("float64")
    f_arr = _funding_array(funding, idx)
    i0 = 0 if trade_start is None else int(idx.searchsorted(trade_start))
    ev = _event_map(idx, feat1d, 24)
    tgt_arr = feat1d["target"].values

    cash = float(initial)
    units = 0.0
    pend = False
    pend_tgt = 0.0
    fees_total = 0.0
    fund_total = 0.0
    expo_bars = 0
    lev_sum = 0.0
    ep = {}
    trades: List[Dict[str, Any]] = []
    m = n - i0
    eq_c = np.empty(m)
    eq_l = np.empty(m)

    def end_episode(px: float, i: int, reason: str, side_str: str):
        nonlocal cash, units, fees_total
        fee_usd = abs(units) * px * fee
        cash += units * px - fee_usd
        fees_total += fee_usd
        ep["fee"] += fee_usd
        pnl = cash - ep["start_eq"]
        peak = ep["peak"]
        trades.append({
            "trade_id": len(trades) + 1, "symbol": symbol, "strategy": strategy, "side": side_str,
            "entry_time": str(idx[ep["i"]]), "exit_time": str(idx[i]),
            "holding_bars": float(i - ep["i"]), "entry_price": ep["px"], "exit_price": float(px),
            "notional": peak, "effective_leverage": peak / ep["start_eq"] if ep["start_eq"] else 0.0,
            "exit_reason": reason,
            "net_return": pnl / peak if peak else 0.0,
            "gross_return": (pnl + ep["fee"] + ep["fund"]) / peak if peak else 0.0,
            "trade_pnl_usd": pnl,
            "entry_fee_cost": ep["fee"] / peak if peak else 0.0,
            "exit_fee_cost": 0.0, "total_slippage_cost": 0.0,
            "funding_cost": ep["fund"] / peak if peak else 0.0,
            "fee_usd": ep["fee"], "funding_usd": ep["fund"],
        })
        units = 0.0

    for i in range(i0, n):
        oi = o[i]
        hi_ = h[i]
        li_ = l[i]
        ci = c[i]
        r = f_arr[i]
        if units != 0.0 and r != 0.0:
            fc = units * oi * r
            cash -= fc
            ep["fund"] += fc
            fund_total += fc
        if pend:
            pend = False
            eq_now = cash + units * oi
            if eq_now > 0:
                new_units = pend_tgt * eq_now / oi
                cur_sign = 0 if units == 0 else (1 if units > 0 else -1)
                new_sign = 0 if new_units == 0 else (1 if new_units > 0 else -1)
                if units != 0.0 and new_sign != cur_sign:
                    end_episode(oi, i, "flat_or_flip", "LONG" if cur_sign == 1 else "SHORT")
                    cur_sign = 0
                if new_sign != 0:
                    if units == 0.0:
                        fee_usd = abs(new_units) * oi * fee
                        fees_total += fee_usd
                        ep = {"start_eq": cash, "i": i, "px": float(oi), "peak": abs(new_units) * oi,
                              "fee": fee_usd, "fund": 0.0}
                        cash -= new_units * oi + fee_usd
                        units = new_units
                    else:
                        delta = new_units - units
                        fee_usd = abs(delta) * oi * fee
                        fees_total += fee_usd
                        ep["fee"] += fee_usd
                        cash -= delta * oi + fee_usd
                        units = new_units
                        ep["peak"] = max(ep["peak"], abs(units) * oi)
        j = i - i0
        if units != 0.0:
            expo_bars += 1
            lev_sum += abs(units) * ci / max(cash + units * ci, 1e-9)
            eq_c[j] = cash + units * ci
            eq_l[j] = cash + units * (li_ if units > 0 else hi_)
        else:
            eq_c[j] = cash
            eq_l[j] = cash
        k = ev[i]
        if k >= 0 and i + 1 < n:
            tgt = tgt_arr[k]
            if math.isnan(tgt):
                tgt = 0.0
            eq_cl = eq_c[j]
            cur = units * ci / eq_cl if eq_cl > 0 else 0.0
            if cur == 0.0:
                if tgt != 0.0:
                    pend, pend_tgt = True, tgt
            elif abs(tgt - cur) > band * abs(cur):
                pend, pend_tgt = True, tgt
    if units != 0.0:
        end_episode(c[n - 1], n - 1, "end_of_window", "LONG" if units > 0 else "SHORT")
        eq_c[m - 1] = cash
        eq_l[m - 1] = min(eq_l[m - 1], cash)
    return _result(symbol, strategy, idx, i0, eq_c, eq_l, trades, fees_total, fund_total,
                   expo_bars, lev_sum, fee)


# ───────────────────────────── strategy registry ─────────────────────────────
STRATEGIES = ["F0", "F0_LO", "G0", "G1", "G2"]


def run_strategy(strategy: str, df_1h: pd.DataFrame, funding: Optional[pd.Series], *, symbol: str,
                 fee: float = FEE_BASE, trade_start: Optional[pd.Timestamp] = None,
                 donchian_n: int = DONCHIAN_N, atr_mult: float = ATR_MULT,
                 vol_target: float = F0_VOL_TARGET) -> Dict[str, Any]:
    if strategy in ("F0", "F0_LO"):
        feat = f0_features(df_1h, vol_target=vol_target, long_only=(strategy == "F0_LO"))
        return simulate_f0(df_1h, funding, feat, trade_start=trade_start, fee=fee,
                           symbol=symbol, strategy=strategy)
    if strategy in ("G0", "G1", "G2"):
        feat = g_features(df_1h, variant=strategy, donchian_n=donchian_n)
        return simulate_g(df_1h, funding, feat, trade_start=trade_start, atr_mult=atr_mult,
                          fee=fee, symbol=symbol, strategy=strategy)
    raise ValueError(strategy)
