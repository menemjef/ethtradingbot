"""
research/scripts/market_neutral_strategies.py

Pre-registered Market-Neutral Strategies:
  N1: Cross-sectional momentum, dollar-neutral (Sunday 00:00 UTC rebalance, top 3 / bottom 3,
      inverse 30-day volatility weights, 2x gross leverage cap, 7-day hold, return excluding last day).
  N2: Delta-neutral funding carry (8h settlement rebalance, trailing 21-day annualized mean funding > 10%,
      long spot + short perp, 1x fully margined, equal weight across qualifying coins, max 20% capital/coin).

Point-in-Time Universe:
  A coin enters the universe ONLY 180 days after its first bar.

Splits:
  Development: 2019-09-01 00:00:00 UTC to 2022-12-31 23:00:00 UTC
  Validation:  2023-01-01 00:00:00 UTC to 2024-12-31 23:00:00 UTC
  Final Test:  2025-01-01 00:00:00 UTC to end (November 2026)

Locked Guard:
  _FINAL_UNLOCKED = False. Final test data (2025+) cannot be loaded before final run.
"""
import os
import sys
import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.quant_evaluation import compute_continuous_portfolio_metrics

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
           "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT"]

FEE_BASE = 0.0010        # 10 bps per side per leg (fee + slippage)
FEE_DOUBLE = 0.0020      # 20 bps per side per leg
INITIAL_EQUITY = 10000.0
DSR_TRIALS = 14          # 10 prior trials + 3 N1 lookbacks + N2
RISK_FREE_ANNUAL = 0.04  # 4% benchmark

DEV_START = pd.Timestamp("2019-09-01 00:00:00+00:00")
DEV_END = pd.Timestamp("2022-12-31 23:00:00+00:00")
VAL_START = pd.Timestamp("2023-01-01 00:00:00+00:00")
VAL_END = pd.Timestamp("2024-12-31 23:00:00+00:00")
FINAL_START = pd.Timestamp("2025-01-01 00:00:00+00:00")

CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

_FINAL_UNLOCKED = False


def unlock_final() -> None:
    """Called strictly by the single final run stage."""
    global _FINAL_UNLOCKED
    _FINAL_UNLOCKED = True


def lock_final() -> None:
    global _FINAL_UNLOCKED
    _FINAL_UNLOCKED = False


def final_unlocked() -> bool:
    return _FINAL_UNLOCKED


def normalize_funding(fd: pd.DataFrame) -> pd.Series:
    """Rounds timestamps to the nearest hour, dedupes, returns funding rate series."""
    s = fd["funding_rate"].copy()
    s.index = s.index.round("h")
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s.astype("float64")


def get_universe_eligibility(cache_dir: Optional[str] = None) -> Dict[str, pd.Timestamp]:
    """Returns the point-in-time eligibility timestamp (first bar + 180 days) for each symbol."""
    cache_dir = cache_dir or CACHE_DIR
    eligibility = {}
    for sym in SYMBOLS:
        p_path = os.path.join(cache_dir, f"{sym.lower()}_1h_full.parquet")
        s_path = os.path.join(cache_dir, f"{sym.lower()}_spot_1h_full.parquet")
        df_p = pd.read_parquet(p_path)
        first_bar = df_p.index[0]
        if os.path.exists(s_path):
            df_s = pd.read_parquet(s_path)
            first_bar = max(first_bar, df_s.index[0])
        eligibility[sym] = first_bar + pd.Timedelta(days=180)
    return eligibility


def load_all_market_data(cache_dir: Optional[str] = None,
                         truncate_at: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    """Loads all perp 1h klines, spot 1h klines, and funding rates.
    When locked, data is strictly truncated at VAL_END (2024-12-31 23:00 UTC)."""
    cache_dir = cache_dir or CACHE_DIR
    limit = truncate_at or (None if _FINAL_UNLOCKED else VAL_END)
    if not _FINAL_UNLOCKED and limit is not None and limit > VAL_END:
        raise PermissionError(f"Final test data beyond {VAL_END} is locked until the final run.")

    data = {"perp": {}, "spot": {}, "funding": {}, "eligibility": get_universe_eligibility(cache_dir)}
    for sym in SYMBOLS:
        p_path = os.path.join(cache_dir, f"{sym.lower()}_1h_full.parquet")
        s_path = os.path.join(cache_dir, f"{sym.lower()}_spot_1h_full.parquet")
        f_path = os.path.join(cache_dir, f"{sym.lower()}_funding_full.parquet")

        df_p = pd.read_parquet(p_path)
        df_s = pd.read_parquet(s_path)
        df_f = pd.read_parquet(f_path)
        s_fund = normalize_funding(df_f)

        if limit is not None:
            df_p = df_p.loc[:limit]
            df_s = df_s.loc[:limit]
            s_fund = s_fund.loc[:limit]

        data["perp"][sym] = df_p
        data["spot"][sym] = df_s
        data["funding"][sym] = s_fund
    return data


# ==============================================================================
# N1: Cross-Sectional Momentum, Dollar-Neutral
# ==============================================================================
def simulate_n1(market_data: Dict[str, Any], *,
                start_time: pd.Timestamp,
                end_time: pd.Timestamp,
                lookback_days: int = 28,
                fee: float = FEE_BASE,
                initial_equity: float = INITIAL_EQUITY) -> Dict[str, Any]:
    """Simulates N1 Cross-Sectional Momentum strategy between start_time and end_time.
    Rebalances every Sunday 00:00 UTC.
    Momentum return = close(t - 24h) / close(t - lookback_days*24h) - 1.0 (excluding last day).
    Long top 3, short bottom 3, inverse 30-day realized volatility weights, net exposure zero, 2x gross leverage.
    """
    perp = market_data["perp"]
    funding = market_data["funding"]
    eligibility = market_data["eligibility"]

    # Common hourly timeline
    all_indices = [df.index for df in perp.values()]
    t_start = max(start_time, min(idx[0] for idx in all_indices))
    t_end = min(end_time, max(idx[-1] for idx in all_indices))
    timeline = pd.date_range(t_start, t_end, freq="1h")
    n_hours = len(timeline)

    # Pre-extract numpy matrices for speed: open, high, low, close, funding
    opens = {s: perp[s]["open"].reindex(timeline).ffill().values for s in SYMBOLS}
    highs = {s: perp[s]["high"].reindex(timeline).ffill().values for s in SYMBOLS}
    lows = {s: perp[s]["low"].reindex(timeline).ffill().values for s in SYMBOLS}
    closes = {s: perp[s]["close"].reindex(timeline).ffill().values for s in SYMBOLS}
    fund_rates = {s: funding[s].reindex(timeline).fillna(0.0).values for s in SYMBOLS}

    # Precompute causal signals on full historical series to eliminate warmup edge-effects
    h_look = lookback_days * 24
    h_excl = 24
    h_vol = 30 * 24
    score_series: Dict[str, np.ndarray] = {}
    vol_series: Dict[str, np.ndarray] = {}

    for s in SYMBOLS:
        c_full = perp[s]["close"]
        s_score = (c_full.shift(h_excl) / c_full.shift(h_look)) - 1.0
        r_1h = np.log(c_full / c_full.shift(1))
        s_vol = r_1h.rolling(h_vol, min_periods=24 * 7).std() * math.sqrt(8760.0)
        score_series[s] = s_score.reindex(timeline).ffill().values
        vol_series[s] = s_vol.reindex(timeline).ffill().values

    # Identify Sunday 00:00 UTC rebalance bar indices
    rebalance_indices = [i for i, dt in enumerate(timeline)
                         if dt.weekday() == 6 and dt.hour == 0]

    equity = np.full(n_hours, initial_equity, dtype=np.float64)
    equity_low = np.full(n_hours, initial_equity, dtype=np.float64)

    # Positions: {sym: units} (positive = long, negative = short)
    positions: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
    entry_prices: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
    cash = float(initial_equity)
    fees_total = 0.0
    funding_total = 0.0
    turnover_usd = 0.0
    per_coin_pnl: Dict[str, float] = {s: 0.0 for s in SYMBOLS}

    rebal_set = set(rebalance_indices)
    weekly_returns: List[float] = []
    last_week_eq = float(initial_equity)

    for i in range(n_hours):
        t_now = timeline[i]

        # 1. Apply funding rate settlements at settlement hours (if any open position)
        for s in SYMBOLS:
            u = positions[s]
            r = fund_rates[s][i]
            if u != 0.0 and r != 0.0:
                px = opens[s][i]
                # Long (u > 0) pays when r > 0: cash -= u * px * r
                # Short (u < 0) receives when r > 0: cash -= u * px * r = + |u| * px * r
                fc = - u * px * r
                cash += fc
                funding_total += fc
                per_coin_pnl[s] += fc

        # 2. Rebalance check at Sunday 00:00 UTC
        if i in rebal_set:
            # Close existing positions at current bar open
            for s in SYMBOLS:
                u = positions[s]
                if u != 0.0:
                    px = opens[s][i]
                    fee_usd = abs(u) * px * fee
                    trade_pnl = u * (px - entry_prices[s]) - fee_usd
                    cash += u * (px - entry_prices[s]) - fee_usd
                    fees_total += fee_usd
                    turnover_usd += abs(u) * px
                    per_coin_pnl[s] += trade_pnl
                    positions[s] = 0.0
                    entry_prices[s] = 0.0

            cur_eq = cash
            if len(weekly_returns) > 0 or cur_eq != last_week_eq:
                w_ret = (cur_eq / last_week_eq) - 1.0 if last_week_eq > 0 else 0.0
                weekly_returns.append(w_ret)
                last_week_eq = cur_eq

            # Candidate evaluation for new positions
            if cur_eq > 0:
                scores: Dict[str, float] = {}
                vols: Dict[str, float] = {}

                for s in SYMBOLS:
                    # Check 180-day point-in-time eligibility
                    if t_now < eligibility[s]:
                        continue
                    sc = score_series[s][i]
                    vl = vol_series[s][i]
                    if not math.isnan(sc) and not math.isnan(vl) and not math.isinf(sc) and not math.isinf(vl):
                        scores[s] = float(sc)
                        vols[s] = max(float(vl), 0.10)

                # Need at least 6 eligible coins to form 3 long / 3 short
                if len(scores) >= 6:
                    sorted_coins = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
                    long_coins = sorted_coins[:3]
                    short_coins = sorted_coins[-3:]

                    # Inverse volatility weighting
                    inv_vols_long = {s: 1.0 / vols[s] for s in long_coins}
                    inv_vols_short = {s: 1.0 / vols[s] for s in short_coins}
                    sum_inv_long = sum(inv_vols_long.values())
                    sum_inv_short = sum(inv_vols_short.values())

                    # Target notional: 1.0x Equity long, 1.0x Equity short (total 2x gross leverage)
                    # Long leg
                    for s in long_coins:
                        w = inv_vols_long[s] / sum_inv_long
                        target_notional = w * cur_eq
                        px = opens[s][i]
                        u = target_notional / px
                        fee_usd = target_notional * fee
                        cash -= fee_usd
                        fees_total += fee_usd
                        turnover_usd += target_notional
                        per_coin_pnl[s] -= fee_usd
                        positions[s] = u
                        entry_prices[s] = px

                    # Short leg
                    for s in short_coins:
                        w = inv_vols_short[s] / sum_inv_short
                        target_notional = w * cur_eq
                        px = opens[s][i]
                        u = - target_notional / px
                        fee_usd = target_notional * fee
                        cash -= fee_usd
                        fees_total += fee_usd
                        turnover_usd += target_notional
                        per_coin_pnl[s] -= fee_usd
                        positions[s] = u
                        entry_prices[s] = px

        # 3. Mark to market at current bar close and low
        cur_close_eq = cash
        cur_low_eq = cash
        for s in SYMBOLS:
            u = positions[s]
            if u > 0:
                cur_close_eq += u * (closes[s][i] - entry_prices[s])
                cur_low_eq += u * (lows[s][i] - entry_prices[s])
            elif u < 0:
                cur_close_eq += u * (closes[s][i] - entry_prices[s])
                cur_low_eq += u * (highs[s][i] - entry_prices[s])

        if cur_close_eq <= 0.0:
            # Account bankruptcy / liquidation
            cur_close_eq = 0.0
            cur_low_eq = 0.0
            cash = 0.0
            for s in SYMBOLS:
                positions[s] = 0.0

        equity[i] = max(0.0, cur_close_eq)
        equity_low[i] = max(0.0, min(cur_close_eq, cur_low_eq))

    # Close remaining positions at end of simulation
    last_i = n_hours - 1
    for s in SYMBOLS:
        u = positions[s]
        if u != 0.0:
            px = closes[s][last_i]
            fee_usd = abs(u) * px * fee
            trade_pnl = u * (px - entry_prices[s]) - fee_usd
            cash += u * (px - entry_prices[s]) - fee_usd
            fees_total += fee_usd
            turnover_usd += abs(u) * px
            per_coin_pnl[s] += trade_pnl
            positions[s] = 0.0

    eq_series = pd.Series(equity, index=timeline)
    eq_low_series = pd.Series(equity_low, index=timeline)

    return {
        "strategy": f"N1_{lookback_days}d",
        "lookback_days": lookback_days,
        "fee_per_side": fee,
        "start": str(timeline[0]),
        "end": str(timeline[-1]),
        "equity": eq_series,
        "equity_low": eq_low_series,
        "fees_usd": fees_total,
        "funding_usd": funding_total,
        "turnover_usd": turnover_usd,
        "per_coin_pnl": per_coin_pnl,
        "weekly_returns": weekly_returns,
        "initial_equity": initial_equity,
        "final_equity": float(eq_series.iloc[-1]),
    }


# ==============================================================================
# N2: Delta-Neutral Funding Carry
# ==============================================================================
def simulate_n2(market_data: Dict[str, Any], *,
                start_time: pd.Timestamp,
                end_time: pd.Timestamp,
                min_annual_funding: float = 0.10,
                exit_annual_funding: float = 0.10,
                max_capital_per_coin: float = 0.20,
                fee: float = FEE_BASE,
                initial_equity: float = INITIAL_EQUITY) -> Dict[str, Any]:
    """Simulates N2 Delta-Neutral Funding Carry strategy between start_time and end_time.
    Evaluates every 8 hours at settlement times (00:00, 08:00, 16:00 UTC).
    Condition: trailing 21-day mean funding rate, annualized, > 10% (0.10).
    Hold long spot + short perp (same notional, 1x, fully margined).
    Max 20% capital per coin. Return measured on total capital employed (spot notional + perp margin).
    Includes basis PnL, funding PnL on perp, and two-leg fees.
    """
    perp = market_data["perp"]
    spot = market_data["spot"]
    funding = market_data["funding"]
    eligibility = market_data["eligibility"]

    # Common hourly timeline
    all_indices = [df.index for df in perp.values()]
    t_start = max(start_time, min(idx[0] for idx in all_indices))
    t_end = min(end_time, max(idx[-1] for idx in all_indices))
    timeline = pd.date_range(t_start, t_end, freq="1h")
    n_hours = len(timeline)

    # Numpy matrices
    perp_opens = {s: perp[s]["open"].reindex(timeline).ffill().values for s in SYMBOLS}
    perp_highs = {s: perp[s]["high"].reindex(timeline).ffill().values for s in SYMBOLS}
    perp_lows = {s: perp[s]["low"].reindex(timeline).ffill().values for s in SYMBOLS}
    perp_closes = {s: perp[s]["close"].reindex(timeline).ffill().values for s in SYMBOLS}

    spot_opens = {s: spot[s]["open"].reindex(timeline).ffill().values for s in SYMBOLS}
    spot_highs = {s: spot[s]["high"].reindex(timeline).ffill().values for s in SYMBOLS}
    spot_lows = {s: spot[s]["low"].reindex(timeline).ffill().values for s in SYMBOLS}
    spot_closes = {s: spot[s]["close"].reindex(timeline).ffill().values for s in SYMBOLS}

    fund_rates = {s: funding[s].reindex(timeline).fillna(0.0).values for s in SYMBOLS}

    # Precalculate rolling 21-day annualized funding rate series for each symbol
    # 21 days * 3 settlements/day = 63 settlements
    annualized_funding_series: Dict[str, np.ndarray] = {}
    for s in SYMBOLS:
        f_s = funding[s]
        f_8h = f_s[f_s.index.hour.isin([0, 8, 16])]
        f_roll = f_8h.rolling(63, min_periods=63).mean() * (365.0 * 3.0)
        annualized_funding_series[s] = f_roll.reindex(timeline).ffill().fillna(0.0).values

    # 8h settlement evaluation indices (00:00, 08:00, 16:00 UTC)
    eval_indices = [i for i, dt in enumerate(timeline) if dt.hour in (0, 8, 16)]
    eval_set = set(eval_indices)

    equity = np.full(n_hours, initial_equity, dtype=np.float64)
    equity_low = np.full(n_hours, initial_equity, dtype=np.float64)

    # Position tracking per coin:
    # spot_units > 0, perp_units < 0
    spot_units: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
    perp_units: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
    spot_entry_px: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
    perp_entry_px: Dict[str, float] = {s: 0.0 for s in SYMBOLS}

    cash = float(initial_equity)
    fees_total = 0.0
    funding_total = 0.0
    basis_pnl_total = 0.0
    turnover_usd = 0.0
    per_coin_pnl: Dict[str, float] = {s: 0.0 for s in SYMBOLS}
    eight_hour_returns: List[float] = []
    last_8h_eq = float(initial_equity)
    max_basis_stress_usd = 0.0

    for i in range(n_hours):
        t_now = timeline[i]

        # 1. Funding cash flows applied on perp leg at settlement hours
        if i in eval_set:
            for s in SYMBOLS:
                u_p = perp_units[s]
                r = fund_rates[s][i]
                if u_p != 0.0 and r != 0.0:
                    px_p = perp_opens[s][i]
                    # Short perp: u_p < 0. When r > 0, short receives cash flow:
                    # cash_flow = - u_p * px_p * r = + |u_p| * px_p * r
                    fc = - u_p * px_p * r
                    cash += fc
                    funding_total += fc
                    per_coin_pnl[s] += fc

        # 2. Evaluation / Rebalance every 8 hours at settlement time
        if i in eval_set:
            cur_eq = cash
            for s in SYMBOLS:
                if spot_units[s] > 0:
                    cur_eq += spot_units[s] * spot_opens[s][i]
                if perp_units[s] < 0:
                    cur_eq += perp_units[s] * (perp_opens[s][i] - perp_entry_px[s])

            # Record 8h return
            if len(eight_hour_returns) > 0 or cur_eq != last_8h_eq:
                r_8h = (cur_eq / last_8h_eq) - 1.0 if last_8h_eq > 0 else 0.0
                eight_hour_returns.append(r_8h)
                last_8h_eq = cur_eq

            # Check qualifying coins
            qualifying = []
            for s in SYMBOLS:
                if t_now < eligibility[s]:
                    continue
                f_ann = annualized_funding_series[s][i]
                if f_ann > min_annual_funding:
                    qualifying.append(s)

            # Close coins that fall below exit threshold or are no longer qualifying
            for s in SYMBOLS:
                if spot_units[s] > 0:
                    f_ann = annualized_funding_series[s][i]
                    if f_ann < exit_annual_funding or s not in qualifying:
                        px_s = spot_opens[s][i]
                        px_p = perp_opens[s][i]
                        u_s = spot_units[s]
                        u_p = perp_units[s]
                        # Exit fees on both legs
                        fee_s = u_s * px_s * fee
                        fee_p = abs(u_p) * px_p * fee
                        # PnL
                        spot_trade_pnl = u_s * (px_s - spot_entry_px[s]) - fee_s
                        perp_trade_pnl = u_p * (px_p - perp_entry_px[s]) - fee_p
                        basis_trade_pnl = spot_trade_pnl + perp_trade_pnl

                        # Cash flows on close: spot proceeds + perp realized PnL - fees
                        cash += (u_s * px_s - fee_s) + (u_p * (px_p - perp_entry_px[s]) - fee_p)
                        fees_total += (fee_s + fee_p)
                        turnover_usd += (u_s * px_s + abs(u_p) * px_p)
                        basis_pnl_total += basis_trade_pnl
                        per_coin_pnl[s] += basis_trade_pnl

                        spot_units[s] = 0.0
                        perp_units[s] = 0.0
                        spot_entry_px[s] = 0.0
                        perp_entry_px[s] = 0.0

            # Open new positions for qualifying coins that are not yet open
            K = len(qualifying)
            if K > 0 and cur_eq > 0:
                capital_per_coin = min(max_capital_per_coin, 1.0 / K) * cur_eq
                for s in qualifying:
                    if spot_units[s] == 0.0:
                        px_s = spot_opens[s][i]
                        px_p = perp_opens[s][i]
                        if px_s > 0 and px_p > 0:
                            # 1x spot, 1x perp fully margined -> total capital = 2 * notional = capital_per_coin
                            notional = 0.5 * capital_per_coin
                            u_s = notional / px_s
                            u_p = - notional / px_p
                            fee_s = notional * fee
                            fee_p = notional * fee

                            cash -= (notional + fee_s + fee_p) # spot cost + fees; perp margin stays in cash
                            fees_total += (fee_s + fee_p)
                            turnover_usd += 2.0 * notional
                            per_coin_pnl[s] -= (fee_s + fee_p)

                            spot_units[s] = u_s
                            perp_units[s] = u_p
                            spot_entry_px[s] = px_s
                            perp_entry_px[s] = px_p

        # 3. Mark to market at bar close
        cur_close_eq = cash
        for s in SYMBOLS:
            u_s = spot_units[s]
            u_p = perp_units[s]
            if u_s > 0:
                cur_close_eq += u_s * spot_closes[s][i]
            if u_p < 0:
                cur_close_eq += u_p * (perp_closes[s][i] - perp_entry_px[s])

        # For a hedged market-neutral position, net exposure drawdown is evaluated on close-to-close equity
        equity[i] = cur_close_eq
        equity_low[i] = cur_close_eq

        # Hourly basis stress tracking: largest hourly change in (perp_close / spot_close - 1) * notional
        if i > 0:
            for s in SYMBOLS:
                u_s = spot_units[s]
                if u_s > 0:
                    p_c_curr = perp_closes[s][i]
                    s_c_curr = spot_closes[s][i]
                    p_c_prev = perp_closes[s][i - 1]
                    s_c_prev = spot_closes[s][i - 1]
                    if s_c_curr > 0 and s_c_prev > 0 and p_c_curr > 0 and p_c_prev > 0:
                        b_curr = (p_c_curr / s_c_curr) - 1.0
                        b_prev = (p_c_prev / s_c_prev) - 1.0
                        delta_b = abs(b_curr - b_prev)
                        notional = u_s * spot_entry_px[s]
                        stress = delta_b * notional
                        if stress > max_basis_stress_usd:
                            max_basis_stress_usd = stress

    # Close any open positions at the end of the simulation
    last_i = n_hours - 1
    for s in SYMBOLS:
        u_s = spot_units[s]
        u_p = perp_units[s]
        if u_s > 0:
            px_s = spot_closes[s][last_i]
            px_p = perp_closes[s][last_i]
            fee_s = u_s * px_s * fee
            fee_p = abs(u_p) * px_p * fee
            spot_pnl = u_s * (px_s - spot_entry_px[s]) - fee_s
            perp_pnl = u_p * (px_p - perp_entry_px[s]) - fee_p
            basis_pnl_total += (spot_pnl + perp_pnl)
            per_coin_pnl[s] += (spot_pnl + perp_pnl)
            fees_total += (fee_s + fee_p)
            turnover_usd += (u_s * px_s + abs(u_p) * px_p)
            cash += (u_s * px_s - fee_s) + (u_p * (px_p - perp_entry_px[s]) - fee_p)
            spot_units[s] = 0.0
            perp_units[s] = 0.0

    eq_series = pd.Series(equity, index=timeline)
    eq_low_series = pd.Series(equity_low, index=timeline)

    return {
        "strategy": "N2_Funding_Carry",
        "fee_per_side": fee,
        "start": str(timeline[0]),
        "end": str(timeline[-1]),
        "equity": eq_series,
        "equity_low": eq_low_series,
        "fees_usd": fees_total,
        "funding_usd": funding_total,
        "basis_pnl_usd": basis_pnl_total,
        "turnover_usd": turnover_usd,
        "per_coin_pnl": per_coin_pnl,
        "eight_hour_returns": eight_hour_returns,
        "max_basis_stress_usd": max_basis_stress_usd,
        "initial_equity": initial_equity,
        "final_equity": float(eq_series.iloc[-1]),
    }


# ==============================================================================
# Metrics & Benchmark Helpers
# ==============================================================================
def compute_strategy_metrics(res: Dict[str, Any], btc_equity: pd.Series,
                             n_bootstraps: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Computes comprehensive metrics including CAGR, Sharpe (vs 4% and 0%), Sortino,
    MaxDD (intrabar lows), worst month, win rate, turnover, fee drag, funding PnL,
    correlation with BTC, exposure, per-coin contribution, and DSR."""
    eq = res["equity"]
    eql = res["equity_low"]
    initial = res["initial_equity"]
    final = res["final_equity"]

    # Basic duration in years
    years = max(0.01, (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400.0))
    if initial <= 0.0 or final <= 0.0:
        cagr = -100.0
    else:
        cagr = float(np.real(((final / initial) ** (1.0 / years) - 1.0) * 100.0))

    # Daily returns
    d_eq = eq.resample("1D").last().dropna()
    d_vals = d_eq.values
    if len(d_vals) > 1:
        d_rets_arr = (d_vals[1:] - d_vals[:-1]) / np.where(d_vals[:-1] == 0, 1e-6, d_vals[:-1])
        valid_d = ~np.isnan(d_rets_arr) & ~np.isinf(d_rets_arr)
        d_rets = pd.Series(d_rets_arr[valid_d], index=d_eq.index[1:][valid_d])
    else:
        d_rets = pd.Series([], dtype=float)

    # Intratrade Max Drawdown (using hourly lows)
    peak = eql.cummax()
    dd_series = (peak - eql) / np.where(peak == 0, 1e-6, peak)
    max_dd = float(dd_series.max() * 100.0) if len(dd_series) > 0 else 0.0

    # Sharpe (vs risk-free 4% and vs 0%)
    rf_daily = RISK_FREE_ANNUAL / 365.0
    excess_d_rets = d_rets - rf_daily
    d_std = float(np.std(d_rets)) if len(d_rets) > 1 else 0.0
    near_constant = bool(d_std < 1e-4 or len(d_rets) <= 1)

    if near_constant:
        sharpe_rf4 = float("nan")
        sharpe_zero = float("nan")
        sortino = float("nan")
    else:
        sharpe_rf4 = float(np.mean(excess_d_rets) / d_std * math.sqrt(365.0))
        sharpe_zero = float(np.mean(d_rets) / d_std * math.sqrt(365.0))
        # Sortino (downside deviation below rf_daily)
        downside = np.minimum(excess_d_rets, 0.0)
        downside_std = math.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 0.0
        sortino = float(np.mean(excess_d_rets) / downside_std * math.sqrt(365.0)) if downside_std > 0 else 0.0

    # Worst Month
    m_eq = eq.resample("ME").last().dropna()
    m_vals = m_eq.values
    if len(m_vals) > 1:
        m_rets_arr = (m_vals[1:] - m_vals[:-1]) / np.where(m_vals[:-1] == 0, 1e-6, m_vals[:-1])
        valid_m = ~np.isnan(m_rets_arr) & ~np.isinf(m_rets_arr)
        worst_month = float(np.min(m_rets_arr[valid_m]) * 100.0) if np.any(valid_m) else 0.0
    else:
        worst_month = 0.0

    # Win rate
    if "weekly_returns" in res:
        w_rets = res["weekly_returns"]
        win_rate = float(sum(1 for r in w_rets if r > 0) / len(w_rets) * 100.0) if len(w_rets) > 0 else 0.0
    elif "eight_hour_returns" in res:
        h_rets = res["eight_hour_returns"]
        win_rate = float(sum(1 for r in h_rets if r > 0) / len(h_rets) * 100.0) if len(h_rets) > 0 else 0.0
    else:
        win_rate = 0.0

    # Turnover
    avg_eq = float(eq.mean())
    turnover = float(res["turnover_usd"] / (2.0 * avg_eq * years)) if (avg_eq * years) > 0 else 0.0

    # Fee and Funding drags
    fee_drag_pct = float(res["fees_usd"] / initial * 100.0)
    funding_usd = float(res["funding_usd"])
    funding_drag_pct = float(funding_usd / initial * 100.0)

    # Correlation of daily returns with BTC
    d_btc = btc_equity.reindex(d_rets.index).ffill()
    d_btc_rets = (d_btc / d_btc.shift(1) - 1.0).dropna()
    common_idx = d_rets.index.intersection(d_btc_rets.index)
    if len(common_idx) > 10 and np.std(d_btc_rets.loc[common_idx]) > 0 and np.std(d_rets.loc[common_idx]) > 0:
        corr_btc = float(np.corrcoef(d_rets.loc[common_idx], d_btc_rets.loc[common_idx])[0, 1])
    else:
        corr_btc = 0.0

    # Exposure (% of time equity != cash or active positions)
    # Market neutral strategies are continuously invested when eligible
    exposure_pct = 100.0

    # DSR using Bailey & López de Prado (2014) via quant_evaluation
    if near_constant:
        dsr = float("nan")
    else:
        p_metrics = compute_continuous_portfolio_metrics(eq, [], hourly_lowest_equity=eql,
                                                         n_trials=DSR_TRIALS, n_bootstraps=n_bootstraps,
                                                         seed=seed)
        dsr = p_metrics.get("deflated_sharpe_ratio", 0.0)

    # Block-bootstrap for weekly returns (N1)
    weekly_ci = None
    if "weekly_returns" in res and len(res["weekly_returns"]) >= 10:
        w_arr = np.array(res["weekly_returns"])
        rng = np.random.default_rng(seed)
        boot_means = []
        block_size = 4
        n_blocks = max(1, len(w_arr) // block_size)
        for _ in range(n_bootstraps):
            starts = rng.integers(0, len(w_arr) - block_size + 1, size=n_blocks)
            sampled = np.concatenate([w_arr[s:s + block_size] for s in starts])
            boot_means.append(np.mean(sampled))
        weekly_ci = [float(np.percentile(boot_means, 2.5) * 10000.0),
                     float(np.percentile(boot_means, 97.5) * 10000.0)]  # in bps

    return {
        "strategy": res["strategy"],
        "fee_per_side": res["fee_per_side"],
        "start": res["start"],
        "end": res["end"],
        "cagr_pct": round(cagr, 2) if not math.isnan(cagr) else None,
        "sharpe_rf4": round(sharpe_rf4, 3) if not math.isnan(sharpe_rf4) else None,
        "sharpe_zero": round(sharpe_zero, 3) if not math.isnan(sharpe_zero) else None,
        "sortino": round(sortino, 3) if not math.isnan(sortino) else None,
        "max_dd_pct": round(max_dd, 2),
        "worst_month_pct": round(worst_month, 2),
        "win_rate_pct": round(win_rate, 1),
        "turnover_annual": round(turnover, 2),
        "fees_usd": round(res["fees_usd"], 2),
        "fee_drag_pct": round(fee_drag_pct, 2),
        "funding_usd": round(funding_usd, 2),
        "funding_drag_pct": round(funding_drag_pct, 2),
        "basis_pnl_usd": round(res.get("basis_pnl_usd", 0.0), 2),
        "correlation_btc": round(corr_btc, 3),
        "exposure_pct": exposure_pct,
        "dsr": round(dsr, 3) if not math.isnan(dsr) else None,
        "near_constant_returns": near_constant,
        "max_basis_stress_usd": round(res.get("max_basis_stress_usd", 0.0), 2),
        "weekly_ci_bps": weekly_ci,
        "per_coin_pnl": {s: round(p, 2) for s, p in res["per_coin_pnl"].items()},
        "initial_equity": initial,
        "final_equity": round(final, 2),
    }


def compute_btc_baseline(btc_perp_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    """Computes BTC Buy-and-Hold baseline metrics over [start, end]."""
    sub = btc_perp_df.loc[start:end]
    c0 = float(sub["close"].iloc[0])
    eq = INITIAL_EQUITY * sub["close"] / c0
    eql = np.minimum(INITIAL_EQUITY * sub["low"] / c0, eq)

    years = max(0.01, (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400.0))
    cagr = ((eq.iloc[-1] / INITIAL_EQUITY) ** (1.0 / years) - 1.0) * 100.0

    d_eq = eq.resample("1D").last().dropna()
    d_rets = (d_eq / d_eq.shift(1) - 1.0).dropna()
    rf_daily = RISK_FREE_ANNUAL / 365.0
    excess_d_rets = d_rets - rf_daily

    sharpe_rf4 = float(np.mean(excess_d_rets) / np.std(d_rets) * math.sqrt(365.0)) if len(d_rets) > 1 and np.std(d_rets) > 0 else 0.0
    sharpe_zero = float(np.mean(d_rets) / np.std(d_rets) * math.sqrt(365.0)) if len(d_rets) > 1 and np.std(d_rets) > 0 else 0.0

    downside = np.minimum(excess_d_rets, 0.0)
    downside_std = math.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 0.0
    sortino = float(np.mean(excess_d_rets) / downside_std * math.sqrt(365.0)) if downside_std > 0 else 0.0

    peak = eql.cummax()
    max_dd = float(((peak - eql) / peak).max() * 100.0)

    m_eq = eq.resample("ME").last().dropna()
    m_rets = (m_eq / m_eq.shift(1) - 1.0).dropna()
    worst_month = float(m_rets.min() * 100.0) if len(m_rets) > 0 else 0.0

    return {
        "strategy": "BTC_BUY_AND_HOLD",
        "start": str(sub.index[0]),
        "end": str(sub.index[-1]),
        "cagr_pct": round(cagr, 2),
        "sharpe_rf4": round(sharpe_rf4, 3),
        "sharpe_zero": round(sharpe_zero, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd, 2),
        "worst_month_pct": round(worst_month, 2),
        "equity": eq,
    }
