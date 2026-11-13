"""Dollar Scalper Research Engine.

Pre-registered backtest engine for the "Book $1 net profit on every trade, 4 slots a day" research task.
Implements:
  - Break-even mathematics (Stage 0.2)
  - 4 fixed daily slots (00:00, 06:00, 12:00, 18:00 UTC) executed at the next 5m open
  - 3 entry variants: E1 (random 50/50), E2 (long-only), E3 (trend-aligned 200d SMA)
  - 3 exit variants: X1 (no stop), X2 (disaster stop -$10 net), X3 (12-hour time stop)
  - 2 cost models: M1 (realistic 5/2/2 bps + funding), M2 (double 10/4/4 bps + 2x funding)
  - Accurate double-entry accounting, tick/step size rounding, and vectorized trade execution
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exchange Specifications
# ---------------------------------------------------------------------------
SPECS = {
    "BTCUSDT": {
        "tick_size": 0.10,
        "step_size": 0.001,
        "min_notional": 5.0,
    },
    "ETHUSDT": {
        "tick_size": 0.01,
        "step_size": 0.001,
        "min_notional": 5.0,
    },
}

COST_MODELS = {
    "M1": {
        "taker_entry_bps": 5.0,
        "slippage_entry_bps": 2.0,
        "maker_exit_bps": 2.0,
        "funding_multiplier": 1.0,
        "description": "Realistic: 5 bps taker entry + 2 bps slippage, 2 bps maker exit, 1x funding",
    },
    "M2": {
        "taker_entry_bps": 10.0,
        "slippage_entry_bps": 4.0,
        "maker_exit_bps": 4.0,
        "funding_multiplier": 2.0,
        "description": "Double costs: 10 bps taker entry + 4 bps slippage, 4 bps maker exit, 2x funding",
    },
}


def round_to_step(val: float, step: float) -> float:
    return float(np.round(np.round(val / step) * step, 8))


def round_to_tick(val: float, tick: float, mode: str = "nearest") -> float:
    if mode == "up":
        return float(np.ceil(np.round(val / tick, 8)) * tick)
    elif mode == "down":
        return float(np.floor(np.round(val / tick, 8)) * tick)
    return float(np.round(np.round(val / tick) * tick, 8))


# ---------------------------------------------------------------------------
# Stage 0.2: Break-Even Mathematics
# ---------------------------------------------------------------------------
def compute_breakeven_math(
    df_5m: pd.DataFrame,
    notional: float = 1000.0,
    cost_model: str = "M1",
    target_net_profit: float = 1.0,
    disaster_stop_net: float = 10.0,
) -> Dict[str, Any]:
    """Compute pre-flight break-even math for $1,000 notional."""
    cm = COST_MODELS[cost_model]
    taker_fee_bps = cm["taker_entry_bps"]
    slippage_bps = cm["slippage_entry_bps"]
    maker_fee_bps = cm["maker_exit_bps"]

    # Round trip transaction cost in bps
    rt_cost_bps = taker_fee_bps + slippage_bps + maker_fee_bps
    rt_cost_dollars = notional * (rt_cost_bps / 10000.0)

    # Gross price move needed for +$1.00 net
    gross_needed_dollars = target_net_profit + rt_cost_dollars
    gross_move_bps = (gross_needed_dollars / notional) * 10000.0

    # Stop condition: if required move is smaller than round-trip cost, STOP
    stop_condition_triggered = gross_move_bps < rt_cost_bps

    # Required win rate: required = loss / (win + loss)
    req_win_rate_x2 = disaster_stop_net / (target_net_profit + disaster_stop_net)

    # Share of 5m bars whose high-low range exceeds that move
    bar_ranges_bps = ((df_5m["high"] - df_5m["low"]) / df_5m["open"]) * 10000.0
    share_bars_exceeding = float((bar_ranges_bps >= gross_move_bps).mean())

    return {
        "cost_model": cost_model,
        "notional": notional,
        "round_trip_cost_bps": rt_cost_bps,
        "round_trip_cost_dollars": rt_cost_dollars,
        "gross_needed_dollars": gross_needed_dollars,
        "gross_move_bps": gross_move_bps,
        "stop_condition_triggered": stop_condition_triggered,
        "req_win_rate_x2": req_win_rate_x2,
        "share_bars_exceeding_move": share_bars_exceeding,
    }


# ---------------------------------------------------------------------------
# Position and Trade Records
# ---------------------------------------------------------------------------
@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    direction: int  # +1 Long, -1 Short
    entry_slot: str
    entry_time: pd.Timestamp
    entry_price: float
    entry_fill_price: float
    quantity: float
    tp_price: float
    stop_price: Optional[float]
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str  # "TP", "STOP", "TIME_STOP", "END_OF_SPLIT"
    holding_bars: int
    gross_pnl: float
    entry_fee: float
    entry_slippage_cost: float
    exit_fee: float
    funding_cost: float
    total_costs: float
    net_pnl: float
    max_adverse_excursion: float  # max unrealized dollar drawdown during trade
    max_favorable_excursion: float  # max unrealized dollar profit during trade
    is_closed: bool


# ---------------------------------------------------------------------------
# Core Simulator
# ---------------------------------------------------------------------------
class DollarScalperBacktest:
    def __init__(
        self,
        df_5m: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
        symbol: str = "BTCUSDT",
        initial_equity: float = 10000.0,
        notional: float = 1000.0,
        cost_model: str = "M1",
        entry_variant: str = "E2",  # "E1", "E2", "E3"
        exit_variant: str = "X2",   # "X1", "X2", "X3"
        random_seed: int = 42,
    ):
        self.df = df_5m
        if "open_time" in self.df.columns and not isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.set_index("open_time")
        if not self.df.index.is_monotonic_increasing:
            self.df = self.df.sort_index()

        self.symbol = symbol
        self.spec = SPECS.get(symbol, {"tick_size": 0.01, "step_size": 0.001, "min_notional": 5.0})
        self.initial_equity = initial_equity
        self.notional = notional
        self.cost_model_name = cost_model
        self.cm = COST_MODELS[cost_model]
        self.entry_variant = entry_variant
        self.exit_variant = exit_variant
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # Cache numpy arrays for maximum performance
        self.times = self.df.index
        self.opens = self.df["open"].values
        self.highs = self.df["high"].values
        self.lows = self.df["low"].values
        self.closes = self.df["close"].values
        self.n_bars = len(self.df)
        self.entry_bar_indices = np.where((self.times.minute == 5) & np.isin(self.times.hour, [0, 6, 12, 18]))[0]

        # Pre-align funding array onto 5m bars
        self.funding_array = np.zeros(self.n_bars, dtype=np.float64)
        if funding_df is not None and not funding_df.empty:
            f_df = funding_df
            if "timestamp" in f_df.columns and not isinstance(f_df.index, pd.DatetimeIndex):
                f_df = f_df.set_index("timestamp")
            elif "calc_time" in f_df.columns and not isinstance(f_df.index, pd.DatetimeIndex):
                f_df = f_df.copy()
                f_df["timestamp"] = pd.to_datetime(f_df["calc_time"], unit="ms", utc=True)
                f_df = f_df.set_index("timestamp")
            rate_col = "funding_rate" if "funding_rate" in f_df.columns else "last_funding_rate"
            f_series = f_df[rate_col].reindex(self.times).fillna(0.0)
            self.funding_array = f_series.values * self.cm["funding_multiplier"]

        # Precompute 200-day SMA for E3
        self.sma_200 = None
        if self.entry_variant == "E3":
            self._compute_daily_sma200()

    def _compute_daily_sma200(self):
        """Compute rolling 200-day SMA of completed daily closes."""
        daily_close = self.df["close"].resample("1D").last().dropna()
        sma = daily_close.rolling(200).mean().shift(1)  # strictly completed prior days
        self.sma_200 = sma.reindex(self.df.index, method="ffill").values

    def solve_target_price(
        self,
        entry_fill_price: float,
        quantity: float,
        direction: int,
        entry_fee: float,
        target_net: float = 1.0,
    ) -> float:
        """Solve target limit price so Net PnL is exactly target_net (+ costs)."""
        tick = self.spec["tick_size"]
        maker_rate = self.cm["maker_exit_bps"] / 10000.0

        if direction == 1:
            raw_tp = (target_net + entry_fee + quantity * entry_fill_price) / (quantity * (1.0 - maker_rate))
            tp_rounded = round_to_tick(raw_tp, tick, mode="up")
        else:
            raw_tp = (quantity * entry_fill_price - entry_fee - target_net) / (quantity * (1.0 + maker_rate))
            tp_rounded = round_to_tick(raw_tp, tick, mode="down")
        return tp_rounded

    def solve_stop_price(
        self,
        entry_fill_price: float,
        quantity: float,
        direction: int,
        entry_fee: float,
        disaster_net_loss: float = 10.0,
    ) -> float:
        """Solve disaster stop price for -$10 net loss."""
        tick = self.spec["tick_size"]
        slippage_rate = self.cm["slippage_entry_bps"] / 10000.0
        taker_fee_rate = self.cm["taker_entry_bps"] / 10000.0

        if direction == 1:
            denom = quantity * (1.0 - slippage_rate) * (1.0 - taker_fee_rate)
            raw_stop = (quantity * entry_fill_price + entry_fee - disaster_net_loss) / denom
            stop_rounded = round_to_tick(raw_stop, tick, mode="down")
        else:
            denom = quantity * (1.0 + slippage_rate) * (1.0 + taker_fee_rate)
            raw_stop = (quantity * entry_fill_price - entry_fee + disaster_net_loss) / denom
            stop_rounded = round_to_tick(raw_stop, tick, mode="up")
        return stop_rounded

    def run(self, track_equity_curve: bool = True) -> Dict[str, Any]:
        """Fast vectorized jump simulation across all slots."""
        cash = self.initial_equity
        trades: List[TradeRecord] = []

        tick = self.spec["tick_size"]
        step = self.spec["step_size"]
        taker_bps = self.cm["taker_entry_bps"]
        slippage_bps = self.cm["slippage_entry_bps"]
        maker_bps = self.cm["maker_exit_bps"]

        n_bars = self.n_bars
        times = self.times
        opens = self.opens
        highs = self.highs
        lows = self.lows
        closes = self.closes
        funding_arr = self.funding_array
        entry_bar_indices = self.entry_bar_indices

        total_slots = len(entry_bar_indices)
        skipped_slots = 0
        trade_counter = 0

        # Mark-to-market tracking arrays if requested
        if track_equity_curve:
            equity_curve = np.full(n_bars, cash, dtype=np.float64)
            adverse_equity_curve = np.full(n_bars, cash, dtype=np.float64)

        slot_idx = 0
        n_entries = len(entry_bar_indices)
        unresolved_pos_record = None

        while slot_idx < n_entries:
            e_bar = entry_bar_indices[slot_idx]
            t_entry = times[e_bar]
            o_entry = opens[e_bar]
            slot_str = times[e_bar - 1].strftime("%Y-%m-%d %H:%M") if e_bar > 0 else t_entry.strftime("%Y-%m-%d %H:%M")

            # Entry direction
            if self.entry_variant == "E1":
                direction = 1 if self.rng.rand() >= 0.5 else -1
            elif self.entry_variant == "E2":
                direction = 1
            elif self.entry_variant == "E3":
                if self.sma_200 is not None and not np.isnan(self.sma_200[e_bar]):
                    direction = 1 if closes[e_bar - 1] >= self.sma_200[e_bar] else -1
                else:
                    direction = 1
            else:
                direction = 1

            # Execution with slippage
            if direction == 1:
                fill_entry = o_entry * (1.0 + slippage_bps / 10000.0)
            else:
                fill_entry = o_entry * (1.0 - slippage_bps / 10000.0)

            qty = round_to_step(self.notional / o_entry, step)
            entry_fee = (qty * fill_entry) * (taker_bps / 10000.0)
            entry_slip_cost = qty * abs(fill_entry - o_entry)

            tp_price = self.solve_target_price(fill_entry, qty, direction, entry_fee, target_net=1.0)
            stop_price = self.solve_stop_price(fill_entry, qty, direction, entry_fee, disaster_net_loss=10.0) if self.exit_variant == "X2" else None

            # Lookahead search in bars strictly after entry bar (e_bar + 1 to end)
            search_start = e_bar + 1
            if search_start >= n_bars:
                break

            # Fast chunk of 1000 bars
            chunk_size = 1000
            search_end = min(search_start + chunk_size, n_bars)
            h_sub = highs[search_start:search_end]
            l_sub = lows[search_start:search_end]

            if direction == 1:
                tp_hits = np.where(h_sub >= tp_price + tick)[0]
            else:
                tp_hits = np.where(l_sub <= tp_price - tick)[0]
            first_tp = tp_hits[0] if len(tp_hits) > 0 else 999999999

            first_stop = 999999999
            if self.exit_variant == "X2" and stop_price is not None:
                if direction == 1:
                    stop_hits = np.where(l_sub <= stop_price)[0]
                else:
                    stop_hits = np.where(h_sub >= stop_price)[0]
                first_stop = stop_hits[0] if len(stop_hits) > 0 else 999999999

            first_time = 999999999
            if self.exit_variant == "X3":
                if search_end - search_start >= 144:
                    first_time = 143

            exit_offset = min(first_tp, first_stop, first_time)

            # Fallback to full search only if not exited in first 1000 bars
            if exit_offset >= len(h_sub) and search_end < n_bars:
                h_sub_full = highs[search_start:]
                l_sub_full = lows[search_start:]
                if direction == 1:
                    tp_hits_f = np.where(h_sub_full >= tp_price + tick)[0]
                else:
                    tp_hits_f = np.where(l_sub_full <= tp_price - tick)[0]
                first_tp = tp_hits_f[0] if len(tp_hits_f) > 0 else 999999999

                first_stop = 999999999
                if self.exit_variant == "X2" and stop_price is not None:
                    if direction == 1:
                        stop_hits_f = np.where(l_sub_full <= stop_price)[0]
                    else:
                        stop_hits_f = np.where(h_sub_full >= stop_price)[0]
                    first_stop = stop_hits_f[0] if len(stop_hits_f) > 0 else 999999999

                first_time = 999999999
                if self.exit_variant == "X3":
                    if len(h_sub_full) >= 144:
                        first_time = 143

                exit_offset = min(first_tp, first_stop, first_time)
                max_len = len(h_sub_full)
            else:
                max_len = len(h_sub)

            if exit_offset < max_len:
                # Position exited during backtest
                exit_bar = search_start + exit_offset
                t_exit = times[exit_bar]
                o_exit = opens[exit_bar]
                c_exit = closes[exit_bar]

                # Check priority: Stop wins over Target if same bar
                if first_stop == first_tp and first_stop == exit_offset:
                    reason = "STOP"
                elif first_stop == exit_offset:
                    reason = "STOP"
                elif first_tp == exit_offset:
                    reason = "TP"
                else:
                    reason = "TIME_STOP"

                # Exit pricing & fees
                if reason == "TP":
                    exit_price = tp_price
                    exit_fee = (qty * exit_price) * (maker_bps / 10000.0)
                elif reason == "STOP":
                    exit_price = stop_price * (1.0 - slippage_bps / 10000.0) if direction == 1 else stop_price * (1.0 + slippage_bps / 10000.0)
                    # Gap check
                    if (direction == 1 and o_exit < stop_price) or (direction == -1 and o_exit > stop_price):
                        exit_price = o_exit * (1.0 - slippage_bps / 10000.0) if direction == 1 else o_exit * (1.0 + slippage_bps / 10000.0)
                    exit_fee = (qty * exit_price) * (taker_bps / 10000.0)
                else:  # TIME_STOP
                    exit_price = c_exit * (1.0 - slippage_bps / 10000.0) if direction == 1 else c_exit * (1.0 + slippage_bps / 10000.0)
                    exit_fee = (qty * exit_price) * (taker_bps / 10000.0)

                # Funding cost during hold
                f_rates = funding_arr[e_bar : exit_bar + 1]
                f_cost = direction * np.sum(f_rates) * (qty * exit_price)

                gross_pnl = qty * (exit_price - fill_entry) if direction == 1 else qty * (fill_entry - exit_price)
                total_costs = entry_fee + exit_fee + f_cost
                net_pnl = gross_pnl - total_costs
                is_closed = True

                # MAE & MFE
                sub_l = lows[e_bar : exit_bar + 1]
                sub_h = highs[e_bar : exit_bar + 1]
                if direction == 1:
                    mae = max(0.0, float(np.max((fill_entry - sub_l) * qty)))
                    mfe = max(0.0, float(np.max((sub_h - fill_entry) * qty)))
                else:
                    mae = max(0.0, float(np.max((sub_h - fill_entry) * qty)))
                    mfe = max(0.0, float(np.max((fill_entry - sub_l) * qty)))

                holding_bars = exit_bar - e_bar

                # Mark to market equity fill between e_bar and exit_bar
                if track_equity_curve:
                    # Update equity curve slice
                    for b_i in range(e_bar, exit_bar + 1):
                        cur_c = closes[b_i]
                        cur_l = lows[b_i]
                        cur_h = highs[b_i]
                        adv_p = cur_l if direction == 1 else cur_h
                        u_pnl = qty * (cur_c - fill_entry) if direction == 1 else qty * (fill_entry - cur_c)
                        adv_pnl = qty * (adv_p - fill_entry) if direction == 1 else qty * (fill_entry - adv_p)
                        u_fee = (qty * cur_c) * (maker_bps / 10000.0)
                        equity_curve[b_i] = cash + u_pnl - entry_fee - u_fee
                        adverse_equity_curve[b_i] = cash + adv_pnl - entry_fee - u_fee
                    # Update remaining flat bars until next trade
                    cash += net_pnl
                    equity_curve[exit_bar + 1 :] = cash
                    adverse_equity_curve[exit_bar + 1 :] = cash
                else:
                    cash += net_pnl

            else:
                # Open until the very end of split
                exit_bar = n_bars - 1
                t_exit = times[exit_bar]
                exit_price = closes[exit_bar]
                reason = "END_OF_SPLIT"
                exit_fee = (qty * exit_price) * (maker_bps / 10000.0)
                f_rates = funding_arr[e_bar : exit_bar + 1]
                f_cost = direction * np.sum(f_rates) * (qty * exit_price)

                gross_pnl = qty * (exit_price - fill_entry) if direction == 1 else qty * (fill_entry - exit_price)
                total_costs = entry_fee + exit_fee + f_cost
                net_pnl = gross_pnl - total_costs
                is_closed = False

                sub_l = lows[e_bar : exit_bar + 1]
                sub_h = highs[e_bar : exit_bar + 1]
                if direction == 1:
                    mae = max(0.0, float(np.max((fill_entry - sub_l) * qty)))
                    mfe = max(0.0, float(np.max((sub_h - fill_entry) * qty)))
                else:
                    mae = max(0.0, float(np.max((sub_h - fill_entry) * qty)))
                    mfe = max(0.0, float(np.max((fill_entry - sub_l) * qty)))

                holding_bars = exit_bar - e_bar

                if track_equity_curve:
                    for b_i in range(e_bar, exit_bar + 1):
                        cur_c = closes[b_i]
                        cur_l = lows[b_i]
                        cur_h = highs[b_i]
                        adv_p = cur_l if direction == 1 else cur_h
                        u_pnl = qty * (cur_c - fill_entry) if direction == 1 else qty * (fill_entry - cur_c)
                        adv_pnl = qty * (adv_p - fill_entry) if direction == 1 else qty * (fill_entry - adv_p)
                        u_fee = (qty * cur_c) * (maker_bps / 10000.0)
                        equity_curve[b_i] = cash + u_pnl - entry_fee - u_fee
                        adverse_equity_curve[b_i] = cash + adv_pnl - entry_fee - u_fee
                    cash += net_pnl
                else:
                    cash += net_pnl

            trade_counter += 1
            t_rec = TradeRecord(
                trade_id=trade_counter,
                symbol=self.symbol,
                direction=direction,
                entry_slot=slot_str,
                entry_time=t_entry,
                entry_price=o_entry,
                entry_fill_price=fill_entry,
                quantity=qty,
                tp_price=tp_price,
                stop_price=stop_price,
                exit_time=t_exit,
                exit_price=exit_price,
                exit_reason=reason,
                holding_bars=holding_bars,
                gross_pnl=gross_pnl,
                entry_fee=entry_fee,
                entry_slippage_cost=entry_slip_cost,
                exit_fee=exit_fee,
                funding_cost=f_cost,
                total_costs=total_costs,
                net_pnl=net_pnl,
                max_adverse_excursion=mae,
                max_favorable_excursion=mfe,
                is_closed=is_closed,
            )
            trades.append(t_rec)

            if not is_closed:
                unresolved_pos_record = t_rec
                break

            # Advance slot_idx to the next slot strictly AFTER exit_bar
            next_slots = np.where(entry_bar_indices > exit_bar)[0]
            if len(next_slots) > 0:
                next_slot_idx = next_slots[0]
                skipped_in_this_trade = next_slot_idx - (slot_idx + 1)
                skipped_slots += max(0, skipped_in_this_trade)
                slot_idx = next_slot_idx
            else:
                skipped_slots += (n_entries - (slot_idx + 1))
                break

        if track_equity_curve:
            equity_series = pd.Series(equity_curve, index=times)
            adverse_series = pd.Series(adverse_equity_curve, index=times)
            final_eq = float(equity_series.iloc[-1])
        else:
            equity_series = None
            adverse_series = None
            final_eq = cash

        return {
            "symbol": self.symbol,
            "cost_model": self.cost_model_name,
            "entry_variant": self.entry_variant,
            "exit_variant": self.exit_variant,
            "initial_equity": self.initial_equity,
            "final_equity": final_eq,
            "trades": trades,
            "equity_curve": equity_series,
            "adverse_equity_curve": adverse_series,
            "total_slots": total_slots,
            "skipped_slots": skipped_slots,
            "unresolved_position": unresolved_pos_record,
        }


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------
def compute_variant_metrics(backtest_res: Dict[str, Any], split_name: str = "Dev") -> Dict[str, Any]:
    """Calculate all required closed-trade, equity, and unresolved metrics."""
    trades: List[TradeRecord] = backtest_res["trades"]
    eq = backtest_res["equity_curve"]
    adv_eq = backtest_res["adverse_equity_curve"]
    init_eq = backtest_res["initial_equity"]
    final_eq = backtest_res["final_equity"]

    closed_trades = [t for t in trades if t.is_closed]
    unresolved_trade = backtest_res["unresolved_position"]

    n_closed = len(closed_trades)

    if eq is not None:
        n_days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400.0
    else:
        n_days = (trades[-1].exit_time - trades[0].entry_time).total_seconds() / 86400.0 if trades else 1.0

    trades_per_day = n_closed / n_days if n_days > 0 else 0.0

    if n_closed > 0:
        net_pnls = np.array([t.net_pnl for t in closed_trades])
        wins = net_pnls[net_pnls > 0]
        losses = net_pnls[net_pnls < 0]

        win_rate = float(len(wins) / n_closed)
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        expectancy_dollars = float(net_pnls.mean())
        expectancy_bps = (expectancy_dollars / 1000.0) * 10000.0
        gross_wins = sum(wins) if len(wins) > 0 else 0.0
        gross_losses = abs(sum(losses)) if len(losses) > 0 else 0.0
        profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

        maes = [t.max_adverse_excursion for t in closed_trades]
        mae_median = float(np.median(maes))
        mae_p95 = float(np.percentile(maes, 95))
        mae_p99 = float(np.percentile(maes, 99))
        mae_max = float(np.max(maes))

        holding_hours = [t.holding_bars * (5.0 / 60.0) for t in closed_trades]
        holding_median = float(np.median(holding_hours))
        holding_max = float(np.max(holding_hours))
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        expectancy_dollars = 0.0
        expectancy_bps = 0.0
        profit_factor = 0.0
        mae_median, mae_p95, mae_p99, mae_max = 0.0, 0.0, 0.0, 0.0
        holding_median, holding_max = 0.0, 0.0

    n_years = n_days / 365.25
    cagr = float((final_eq / init_eq) ** (1.0 / n_years) - 1.0) if n_years > 0 and final_eq > 0 else -1.0

    if eq is not None:
        bar_returns = eq.pct_change().dropna()
        bars_per_year = 288 * 365.25
        sharpe = float((bar_returns.mean() / bar_returns.std()) * np.sqrt(bars_per_year)) if len(bar_returns) > 1 and bar_returns.std() > 0 else 0.0

        peak = adv_eq.cummax()
        dd_series = (adv_eq - peak) / peak
        max_drawdown = float(abs(dd_series.min()))
        min_equity_ratio = float(adv_eq.min() / init_eq)

        daily_returns = eq.resample("1D").last().pct_change().dropna()
        worst_day = float(daily_returns.min()) if len(daily_returns) > 0 else 0.0

        monthly_returns = eq.resample("1ME").last().pct_change().dropna()
        worst_month = float(monthly_returns.min()) if len(monthly_returns) > 0 else 0.0

        underwater_amount = peak - eq
        time_loss_20 = float((underwater_amount >= 20.0).mean())
        time_loss_50 = float((underwater_amount >= 50.0).mean())
        time_loss_100 = float((underwater_amount >= 100.0).mean())
    else:
        # Approximate Sharpe from closed trades if equity curve not tracked
        sharpe = float(expectancy_dollars / np.std(net_pnls) * np.sqrt(trades_per_day * 365.25)) if n_closed > 1 and np.std(net_pnls) > 0 else 0.0
        max_drawdown = float(mae_max / init_eq)
        min_equity_ratio = 1.0 - max_drawdown
        worst_day = 0.0
        worst_month = 0.0
        time_loss_20 = 0.0
        time_loss_50 = 0.0
        time_loss_100 = 0.0

    total_fees = sum(t.entry_fee + t.exit_fee for t in trades)
    total_funding = sum(t.funding_cost for t in trades)
    total_slippage = sum(t.entry_slippage_cost for t in trades)

    has_unresolved = unresolved_trade is not None
    unresolved_loss = float(unresolved_trade.net_pnl) if unresolved_trade else 0.0
    largest_open_loss_pct = (mae_max / init_eq) * 100.0

    return {
        "split": split_name,
        "symbol": backtest_res["symbol"],
        "cost_model": backtest_res["cost_model"],
        "entry_variant": backtest_res["entry_variant"],
        "exit_variant": backtest_res["exit_variant"],
        "variant": f"{backtest_res['entry_variant']}_{backtest_res['exit_variant']}",
        "closed_trades": n_closed,
        "trades_per_day": trades_per_day,
        "total_slots": backtest_res["total_slots"],
        "skipped_slots": backtest_res["skipped_slots"],
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy_dollars": expectancy_dollars,
        "expectancy_bps": expectancy_bps,
        "profit_factor": profit_factor,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "min_equity_ratio": min_equity_ratio,
        "worst_day": worst_day,
        "worst_month": worst_month,
        "mae_median": mae_median,
        "mae_p95": mae_p95,
        "mae_p99": mae_p99,
        "mae_max": mae_max,
        "holding_hours_median": holding_median,
        "holding_hours_max": holding_max,
        "total_fees": total_fees,
        "total_funding": total_funding,
        "total_slippage": total_slippage,
        "has_unresolved": has_unresolved,
        "unresolved_loss": unresolved_loss,
        "largest_open_loss_pct": largest_open_loss_pct,
        "time_loss_20": time_loss_20,
        "time_loss_50": time_loss_50,
        "time_loss_100": time_loss_100,
    }
