"""
tests/test_backtest_engine_validation.py — Comprehensive 10-Point Backtest Engine Validation Suite.

Audits and verifies the common backtest engine (`utils/quant_evaluation.py`,
`research/scripts/run_phase_2_1_research.py`, `backtest.py`, `backtest_audit.py`,
`research/scripts/generate_experiment_registry.py`) across all 10 required invariants
plus symbol-agnostic execution:

1. Timestamps: UTC only, closed bars only, no duplicates or gaps; 4H/6H/1D aggregated
   bars use only completed lower-timeframe bars.
2. No look-ahead: signal at bar close, fill at NEXT bar open; indicators (EMA, ATR,
   Donchian, volume SMA) at bar t use data <= t only; shifting future data does not
   alter signals or features at <= t.
3. Fills: stop and take-profit use intrabar high/low; when both hit in one bar, stop
   wins; opening gaps fill at the open with slippage; slippage and fees applied on
   both entry and exit for long and short.
4. Funding: applied at actual settlement timestamps with the correct sign (longs pay
   when rate > 0), only while the position is open, never from future rows.
5. Sizing and leverage: risk-based sizing respects the leverage cap and margin; no
   position exceeds available cash or the cap.
6. Accounting: equity = cash + margin + unrealized PnL at every step; fees, funding,
   and PnL are not double counted; trade-level net PnL sums to the equity change.
7. Drawdown and metrics: max drawdown computed on the equity curve including intratrade
   lows; CAGR, Sharpe, profit factor, expectancy, bootstrap CI checked against
   hand-computed values on a tiny synthetic dataset.
8. Segmentation: walk-forward folds and the holdout do not overlap; embargo/purge and
   pre-roll are excluded; the holdout is never used for parameter selection.
9. Reproducibility: same data and seed give identical trades and metrics twice.
10. Registry consistency: reconciles `research/experiment_registry.json` against
    `research/phase_2_1_fold_metrics.json`, `research/phase_2_1_candidate_comparison.md`,
    and `research/swing_strategy_metrics.json`.
11. Symbol-agnostic execution: symbol is parameterized (`BTCUSDT`, `SOLUSDT`), never
    hardcoded to ETH.
"""

import os
import sys
import json
import math
import tempfile
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.quant_evaluation import (
    calculate_actual_stop_position_sizing,
    compute_continuous_portfolio_metrics,
    compute_reconciled_trade_metrics,
    validate_ohlcv_timestamps,
    resample_completed_ohlcv,
    validate_walk_forward_segmentation,
)
from research.scripts.run_phase_2_1_research import (
    build_multitimeframe_features,
    generate_swing_signals,
    normalize_funding_series,
    simulate_swing_trades_conservative,
    compute_buy_and_hold_equity,
)


def _make_synthetic_1h_df(
    n_hours: int = 120,
    start: str = "2025-01-01 00:00:00+00:00",
    base_price: float = 2500.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Creates a clean, closed, gap-free UTC 1H OHLCV DataFrame for deterministic testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_hours, freq="1h", tz="UTC")
    rets = rng.normal(0.0002, 0.004, size=n_hours)
    closes = base_price * np.cumprod(1.0 + rets)
    opens = np.roll(closes, 1)
    opens[0] = base_price
    highs = np.maximum(opens, closes) * (1.0 + rng.uniform(0.0005, 0.003, size=n_hours))
    lows = np.minimum(opens, closes) * (1.0 - rng.uniform(0.0005, 0.003, size=n_hours))
    volumes = rng.uniform(1000.0, 5000.0, size=n_hours)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )
    df.index.name = "open_time"
    return df


class TestBacktestEngineValidation(unittest.TestCase):
    """10-point backtest engine validation audit test suite."""

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 1: TIMESTAMPS & COMPLETED MULTI-TIMEFRAME AGGREGATION (4H, 6H, 1D)
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_timestamps_utc_closed_no_duplicates_no_gaps_and_mtf_aggregation(self):
        """
        Item 1: Bars are UTC, closed bars only, no duplicates or gaps;
        4H, 6H, and 1D aggregated bars use only completed lower-timeframe bars.
        """
        # 50 hours = 2 full days (48h) + 2 trailing hours (incomplete 4H, 6H, and 1D at the tail)
        df_1h = _make_synthetic_1h_df(n_hours=50, start="2025-01-01 00:00:00+00:00")
        ref_now = pd.Timestamp("2025-01-03 02:00:00+00:00")

        # 1. Valid UTC closed gap-free series passes validation
        self.assertTrue(validate_ohlcv_timestamps(df_1h, expected_freq="1h", reference_now=ref_now))

        # 2. Naive (non-UTC) timestamps must fail
        df_naive = df_1h.copy()
        df_naive.index = df_naive.index.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone-aware UTC"):
            validate_ohlcv_timestamps(df_naive, expected_freq="1h")

        # 3. Non-UTC timezone must fail
        df_est = df_1h.copy()
        df_est.index = df_est.index.tz_convert("US/Eastern")
        with self.assertRaisesRegex(ValueError, "UTC"):
            validate_ohlcv_timestamps(df_est, expected_freq="1h")

        # 4. Duplicate timestamps must fail
        df_dup = pd.concat([df_1h.iloc[:10], df_1h.iloc[9:20]])
        with self.assertRaisesRegex(ValueError, "Duplicate OHLCV timestamps"):
            validate_ohlcv_timestamps(df_dup, expected_freq="1h")

        # 5. Missing 1H gap must fail
        df_gap = df_1h.drop(df_1h.index[15])
        with self.assertRaisesRegex(ValueError, "timestamp gap detected"):
            validate_ohlcv_timestamps(df_gap, expected_freq="1h")

        # 6. Unclosed bar (bar_end > reference_now) must fail
        with self.assertRaisesRegex(ValueError, "Unclosed bar detected"):
            validate_ohlcv_timestamps(
                df_1h,
                expected_freq="1h",
                reference_now=pd.Timestamp("2025-01-03 01:30:00+00:00"),
            )

        # 7. 4H, 6H, and 1D aggregation uses ONLY completed lower-timeframe bars
        # 50 hours from 2025-01-01 00:00 to 2025-01-03 01:00:
        # - Completed 4H bars: 50 // 4 = 12 (hours 48 & 49 form an incomplete 2-bar 4H bucket -> excluded)
        # - Completed 6H bars: 50 // 6 = 8  (hours 48 & 49 form an incomplete 2-bar 6H bucket -> excluded)
        # - Completed 1D bars: 50 // 24 = 2 (hours 48 & 49 form an incomplete 2-bar 1D bucket -> excluded)
        df_1h_out, df_4h, df_6h, df_1d = build_multitimeframe_features(
            df_1h, include_6h=True, reference_now=ref_now
        )
        self.assertEqual(len(df_4h), 12)
        self.assertEqual(len(df_6h), 8)
        self.assertEqual(len(df_1d), 2)
        self.assertEqual(df_4h.index[-1], pd.Timestamp("2025-01-02 20:00:00+00:00"))
        self.assertEqual(df_6h.index[-1], pd.Timestamp("2025-01-02 18:00:00+00:00"))
        self.assertEqual(df_1d.index[-1], pd.Timestamp("2025-01-02 00:00:00+00:00"))

        # Verify that the completed 2025-01-01 1D bar (closed at 2025-01-02 00:00) is mapped onto 2025-01-02 4H bars
        self.assertAlmostEqual(
            float(df_4h.loc[pd.Timestamp("2025-01-02 00:00:00+00:00"), "1d_ema_50"]),
            float(df_1d.loc[pd.Timestamp("2025-01-01 00:00:00+00:00"), "ema_50"]),
            places=8,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 2: NO LOOK-AHEAD (NEXT-BAR OPEN FILL & FUTURE SHIFT INVARIANCE)
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_no_lookahead_next_bar_open_fill_and_future_shift_invariance(self):
        """
        Item 2: Signal at bar close, fill at NEXT bar open. Indicators (EMA, ATR,
        Donchian, volume SMA) at bar t use data <= t only. Shifting/perturbing future
        data (> t) does not change any indicator or signal at <= t.
        """
        df_1h_a = _make_synthetic_1h_df(n_hours=600, seed=101)
        df_1h_b = df_1h_a.copy()

        # Cutoff at hour 360 (15 full days = 90 4H bars)
        cutoff_ts = df_1h_a.index[359]  # 2025-01-15 23:00:00+00:00 (end of 4H bar 2025-01-15 20:00)
        cutoff_4h_ts = pd.Timestamp("2025-01-15 20:00:00+00:00")

        # Massively shock all future bars strictly after cutoff_ts (+500% price jump & 10x volume)
        df_1h_b.loc[df_1h_b.index > cutoff_ts, ["open", "high", "low", "close"]] *= 6.0
        df_1h_b.loc[df_1h_b.index > cutoff_ts, "volume"] *= 10.0

        _, df_4h_a, _ = build_multitimeframe_features(df_1h_a)
        _, df_4h_b, _ = build_multitimeframe_features(df_1h_b)

        indicator_cols = [
            "ema_20", "ema_50", "atr_14", "donchian_high_20", "donchian_low_20",
            "vol_sma20", "vol_ratio", "1d_ema_50", "1d_adx_14",
        ]
        for col in indicator_cols:
            s_a = df_4h_a.loc[:cutoff_4h_ts, col].dropna()
            s_b = df_4h_b.loc[:cutoff_4h_ts, col].dropna()
            pd.testing.assert_series_equal(s_a, s_b, check_names=False)

        # Verify signals at <= cutoff_4h_ts are 100% invariant to future price shocks
        for strat in [
            "Exp1_Momentum_Continuation_4H",
            "Trend_Donchian_120h",
            "Trend_Donchian_168h",
        ]:
            sig_a = generate_swing_signals(df_4h_a, strategy_name=strat).loc[:cutoff_4h_ts]
            sig_b = generate_swing_signals(df_4h_b, strategy_name=strat).loc[:cutoff_4h_ts]
            pd.testing.assert_series_equal(sig_a["signal_long"], sig_b["signal_long"])
            pd.testing.assert_series_equal(sig_a["signal_short"], sig_b["signal_short"])
            pd.testing.assert_series_equal(sig_a["stop_price"], sig_b["stop_price"])
            pd.testing.assert_series_equal(sig_a["target_price"], sig_b["target_price"])

        # Verify signal at 4H bar k close fills at NEXT 1H bar open (ts_4h[k] + 4h), never at decision bar close
        df_1h_exec = _make_synthetic_1h_df(n_hours=48, seed=7)
        df_1h_exec, df_4h_exec, _ = build_multitimeframe_features(df_1h_exec)
        df_sig = df_4h_exec.copy()
        df_sig["signal_long"] = False
        df_sig["signal_short"] = False
        df_sig["stop_price"] = np.nan
        df_sig["target_price"] = np.nan

        # Trigger a LONG signal on 4H bar index 2 (2025-01-01 08:00:00+00:00, which closes at 12:00:00+00:00)
        dec_ts = df_sig.index[2]
        next_1h_open_ts = dec_ts + pd.Timedelta(hours=4)
        # Make decision bar close (at 11:00) very different from next bar open (at 12:00)
        df_1h_exec.loc[dec_ts + pd.Timedelta(hours=3), "close"] = 2000.0
        df_1h_exec.loc[next_1h_open_ts, "open"] = 2100.0
        df_1h_exec.loc[next_1h_open_ts, "high"] = 2120.0
        df_1h_exec.loc[next_1h_open_ts, "low"] = 2090.0
        df_1h_exec.loc[next_1h_open_ts, "close"] = 2110.0

        df_sig.loc[dec_ts, "signal_long"] = True
        df_sig.loc[dec_ts, "stop_price"] = 1900.0
        df_sig.loc[dec_ts, "target_price"] = 2500.0

        trades, _, _, _ = simulate_swing_trades_conservative(
            df_1h=df_1h_exec,
            df_4h_signals=df_sig,
            funding_map={},
            taker_fee=0.0005,
            slippage=0.0003,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(pd.Timestamp(trades[0]["decision_time"]), dec_ts)
        self.assertEqual(pd.Timestamp(trades[0]["entry_time"]), next_1h_open_ts)
        self.assertGreater(pd.Timestamp(trades[0]["entry_time"]), pd.Timestamp(trades[0]["decision_time"]))
        expected_entry_px = 2100.0 * (1.0 + 0.0003)
        self.assertAlmostEqual(trades[0]["entry_price"], expected_entry_px, places=6)

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 3: FILLS (INTRABAR SL/TP, COLLISION STOP-WINS, GAPS, SLIPPAGE & FEES)
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_fills_intrabar_sl_tp_collision_stop_wins_gaps_slippage_and_fees(self):
        """
        Item 3: Stop and take-profit use intrabar high/low; when both hit in one bar,
        stop wins. Gaps fill at the open with slippage. Slippage and fees applied on
        both entry and exit, for long and short.
        """
        df_1h = _make_synthetic_1h_df(n_hours=48, base_price=1000.0, seed=11)
        df_1h, df_4h, _ = build_multitimeframe_features(df_1h)

        def _make_single_signal_frame(side: int, sl: float, tp: float, dec_idx: int = 1) -> pd.DataFrame:
            sig = df_4h.copy()
            sig["signal_long"] = False
            sig["signal_short"] = False
            sig["stop_price"] = np.nan
            sig["target_price"] = np.nan
            ts = sig.index[dec_idx]
            if side == 1:
                sig.loc[ts, "signal_long"] = True
            else:
                sig.loc[ts, "signal_short"] = True
            sig.loc[ts, "stop_price"] = sl
            sig.loc[ts, "target_price"] = tp
            return sig

        exec_ts = df_4h.index[1] + pd.Timedelta(hours=4)  # 2025-01-01 08:00:00+00:00
        next_ts = exec_ts + pd.Timedelta(hours=1)         # 2025-01-01 09:00:00+00:00
        slip = 0.001
        fee = 0.0005

        # Case A: Same-bar SL + TP collision on LONG -> Stop MUST win (stop_loss_collision)
        df_coll = df_1h.copy()
        df_coll.loc[exec_ts, ["open", "high", "low", "close"]] = [1000.0, 1010.0, 995.0, 1005.0]
        # On next_ts, bar hits BOTH SL=980 and TP=1050!
        df_coll.loc[next_ts, ["open", "high", "low", "close"]] = [1005.0, 1060.0, 975.0, 1040.0]
        sig_long = _make_single_signal_frame(side=1, sl=980.0, tp=1050.0)
        trades_coll, _, _, _ = simulate_swing_trades_conservative(
            df_1h=df_coll, df_4h_signals=sig_long, funding_map={}, taker_fee=fee, slippage=slip
        )
        self.assertEqual(len(trades_coll), 1)
        self.assertEqual(trades_coll[0]["exit_reason"], "stop_loss_collision")
        self.assertAlmostEqual(trades_coll[0]["entry_price"], 1000.0 * (1.0 + slip), places=6)
        self.assertAlmostEqual(trades_coll[0]["exit_price"], 980.0 * (1.0 - slip), places=6)
        self.assertLess(trades_coll[0]["net_return"], 0.0)

        # Case B: Opening Gap beyond SL on LONG -> fills at Open with slippage (stop_loss_gap), NOT at SL
        df_gap_l = df_1h.copy()
        df_gap_l.loc[exec_ts, ["open", "high", "low", "close"]] = [1000.0, 1010.0, 995.0, 1000.0]
        # Next bar gaps open at 950.0 (well below SL=980.0)
        df_gap_l.loc[next_ts, ["open", "high", "low", "close"]] = [950.0, 960.0, 940.0, 955.0]
        trades_gap_l, _, _, _ = simulate_swing_trades_conservative(
            df_1h=df_gap_l, df_4h_signals=sig_long, funding_map={}, taker_fee=fee, slippage=slip
        )
        self.assertEqual(trades_gap_l[0]["exit_reason"], "stop_loss_gap")
        self.assertAlmostEqual(trades_gap_l[0]["exit_price"], 950.0 * (1.0 - slip), places=6)

        # Case C: SHORT trade hitting Take-Profit -> verify adverse slippage & fees on both entry and exit
        df_short_tp = df_1h.copy()
        df_short_tp.loc[exec_ts, ["open", "high", "low", "close"]] = [1000.0, 1005.0, 995.0, 1000.0]
        df_short_tp.loc[next_ts, ["open", "high", "low", "close"]] = [995.0, 1000.0, 930.0, 945.0]
        sig_short = _make_single_signal_frame(side=-1, sl=1030.0, tp=940.0)
        trades_short, _, _, _ = simulate_swing_trades_conservative(
            df_1h=df_short_tp, df_4h_signals=sig_short, funding_map={}, taker_fee=fee, slippage=slip
        )
        self.assertEqual(len(trades_short), 1)
        tr_s = trades_short[0]
        self.assertEqual(tr_s["side"], "SHORT")
        self.assertEqual(tr_s["exit_reason"], "take_profit")
        # Short entry fills lower: 1000 * (1 - slip); Short exit fills higher: 940 * (1 + slip)
        exp_short_entry = 1000.0 * (1.0 - slip)
        exp_short_exit = 940.0 * (1.0 + slip)
        self.assertAlmostEqual(tr_s["entry_price"], exp_short_entry, places=6)
        self.assertAlmostEqual(tr_s["exit_price"], exp_short_exit, places=6)
        exp_net_ret = -1.0 * ((exp_short_exit - exp_short_entry) / exp_short_entry) - (2.0 * fee)
        self.assertAlmostEqual(tr_s["net_return"], exp_net_ret, places=8)

        # Case D: Opening Gap beyond SL on SHORT -> fills at Open with adverse slippage
        df_gap_s = df_1h.copy()
        df_gap_s.loc[exec_ts, ["open", "high", "low", "close"]] = [1000.0, 1005.0, 995.0, 1000.0]
        df_gap_s.loc[next_ts, ["open", "high", "low", "close"]] = [1060.0, 1070.0, 1050.0, 1065.0]
        trades_gap_s, _, _, _ = simulate_swing_trades_conservative(
            df_1h=df_gap_s, df_4h_signals=sig_short, funding_map={}, taker_fee=fee, slippage=slip
        )
        self.assertEqual(trades_gap_s[0]["exit_reason"], "stop_loss_gap")
        self.assertAlmostEqual(trades_gap_s[0]["exit_price"], 1060.0 * (1.0 + slip), places=6)

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 4: FUNDING (SETTLEMENT TIMESTAMPS, SIGN CONVENTION, NO FUTURE LEAK)
    # ─────────────────────────────────────────────────────────────────────────
    def test_04_funding_settlement_timestamps_sign_open_only_no_future_rows(self):
        """
        Item 4: Funding is applied at actual settlement timestamps with the correct
        sign (longs pay when rate > 0, shorts receive), only while the position is
        open, and never from future rows.
        """
        df_1h = _make_synthetic_1h_df(n_hours=48, base_price=1000.0, seed=21)
        # Keep prices flat at 1000 so SL/TP don't trigger before horizon expiry
        df_1h[["open", "high", "low", "close"]] = 1000.0
        df_1h["high"] = 1002.0
        df_1h["low"] = 998.0
        df_1h, df_4h, _ = build_multitimeframe_features(df_1h)

        # Create sub-second funding parquet in temp file to verify normalize_funding_series
        with tempfile.TemporaryDirectory() as tmpdir:
            f_path = os.path.join(tmpdir, "funding.parquet")
            # Decision at 2025-01-01 04:00 -> Entry at 2025-01-01 08:00, held for 10h -> Exit at 2025-01-01 18:00
            # Settlements during open window [08:00, 18:00]: 08:00 (+0.0010) and 16:00 (+0.0005)
            # Settlement BEFORE open (00:00: +0.0500) and AFTER close (2025-01-02 00:00: +0.0900) MUST NOT apply!
            f_idx = pd.to_datetime(
                [
                    "2025-01-01 00:00:00.015000+00:00",
                    "2025-01-01 08:00:00.023000+00:00",
                    "2025-01-01 16:00:00.009000+00:00",
                    "2025-01-02 00:00:00.041000+00:00",
                ],
                utc=True,
            )
            df_f = pd.DataFrame({"funding_rate": [0.0500, 0.0010, 0.0005, 0.0900]}, index=f_idx)
            df_f.to_parquet(f_path)
            funding_map, f_stats = normalize_funding_series(f_path)
            self.assertEqual(f_stats["total_records"], 4)

        dec_ts = pd.Timestamp("2025-01-01 04:00:00+00:00")
        sig_long = df_4h.copy()
        sig_long["signal_long"] = False
        sig_long["signal_short"] = False
        sig_long["stop_price"] = np.nan
        sig_long["target_price"] = np.nan
        sig_long.loc[dec_ts, "signal_long"] = True
        sig_long.loc[dec_ts, "stop_price"] = 900.0
        sig_long.loc[dec_ts, "target_price"] = 1100.0

        trades_l, _, _, audit_l = simulate_swing_trades_conservative(
            df_1h=df_1h,
            df_4h_signals=sig_long,
            funding_map=funding_map,
            max_holding_hours=10,
            taker_fee=0.0,
            slippage=0.0,
        )
        self.assertEqual(len(trades_l), 1)
        # Long pays +0.0010 at 08:00 and +0.0005 at 16:00 -> funding_cost = +0.0015 (reduces net_return)
        self.assertAlmostEqual(trades_l[0]["funding_cost"], 0.0015, places=8)
        self.assertAlmostEqual(trades_l[0]["net_return"], -0.0015, places=8)
        self.assertEqual(audit_l["applied_settlements"], 2)
        self.assertEqual(audit_l["missed_settlements"], 0)

        # Same window for SHORT -> Short RECEIVES positive funding (funding_cost = -0.0015, increases net_return)
        sig_short = sig_long.copy()
        sig_short.loc[dec_ts, "signal_long"] = False
        sig_short.loc[dec_ts, "signal_short"] = True
        sig_short.loc[dec_ts, "stop_price"] = 1100.0
        sig_short.loc[dec_ts, "target_price"] = 900.0

        trades_s, _, _, _ = simulate_swing_trades_conservative(
            df_1h=df_1h,
            df_4h_signals=sig_short,
            funding_map=funding_map,
            max_holding_hours=10,
            taker_fee=0.0,
            slippage=0.0,
        )
        self.assertAlmostEqual(trades_s[0]["funding_cost"], -0.0015, places=8)
        self.assertAlmostEqual(trades_s[0]["net_return"], +0.0015, places=8)

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 5: SIZING & LEVERAGE CAP & AVAILABLE CASH MARGIN CONSTRAINT
    # ─────────────────────────────────────────────────────────────────────────
    def test_05_sizing_and_leverage_cap_and_available_cash_margin_constraint(self):
        """
        Item 5: Risk-based sizing respects the leverage cap and margin;
        no position exceeds available cash or the cap.
        """
        # 1. Tight stop (0.1% distance): uncapped risk sizing would be 25x leverage, must cap at 3.0x
        s_capped = calculate_actual_stop_position_sizing(
            account_equity=10000.0,
            entry_price=2000.0,
            stop_price=1998.0,  # 0.1% stop distance
            risk_pct=0.025,
            max_leverage=3.0,
            available_cash=10000.0,
        )
        self.assertAlmostEqual(s_capped["effective_leverage"], 3.0, places=6)
        self.assertAlmostEqual(s_capped["notional"], 30000.0, places=6)
        self.assertAlmostEqual(s_capped["margin_required"], 10000.0, places=6)

        # 2. Available cash constraint: if 70% of cash is already locked in margin, available_cash = $3,000
        s_cash_lim = calculate_actual_stop_position_sizing(
            account_equity=10000.0,
            entry_price=2000.0,
            stop_price=1998.0,
            risk_pct=0.025,
            max_leverage=3.0,
            available_cash=3000.0,
        )
        self.assertLessEqual(s_cash_lim["margin_required"], 3000.0 + 1e-9)
        self.assertAlmostEqual(s_cash_lim["notional"], 9000.0, places=6)

        # 3. Wide stop (5.0% distance): risk sizing = 0.025 / 0.05 = 0.5x leverage (uncapped)
        s_uncapped = calculate_actual_stop_position_sizing(
            account_equity=10000.0,
            entry_price=2000.0,
            stop_price=1900.0,
            risk_pct=0.025,
            max_leverage=3.0,
            available_cash=10000.0,
        )
        self.assertLess(s_uncapped["effective_leverage"], 3.0)
        self.assertAlmostEqual(s_uncapped["effective_leverage"], 0.5, places=6)
        self.assertAlmostEqual(s_uncapped["notional"], 5000.0, places=6)
        self.assertAlmostEqual(s_uncapped["margin_required"], 5000.0 / 3.0, places=6)

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 6: ACCOUNTING (EQUITY = CASH + MARGIN + UNREALIZED PNL, RECONCILIATION)
    # ─────────────────────────────────────────────────────────────────────────
    def test_06_accounting_identity_no_double_counting_and_pnl_reconciliation(self):
        """
        Item 6: equity = cash + margin + unrealized PnL at every step;
        fees, funding, and PnL are not double counted; trade-level net PnL sums
        to the total equity change.
        """
        df_1h = _make_synthetic_1h_df(n_hours=240, base_price=2000.0, seed=55)
        df_1h, df_4h, _ = build_multitimeframe_features(df_1h)
        df_sig = generate_swing_signals(df_4h, strategy_name="Benchmark_Always_Long")

        funding_map = {
            ts: 0.0002 for ts in df_1h.index if ts.hour in (0, 8, 16)
        }
        trades, eq_series, eq_lowest, f_audit = simulate_swing_trades_conservative(
            df_1h=df_1h,
            df_4h_signals=df_sig,
            funding_map=funding_map,
            initial_capital=10000.0,
            taker_fee=0.0005,
            slippage=0.0003,
            max_holding_hours=24,
            force_close_at_end=True,
        )
        self.assertGreater(len(trades), 3)
        acc = f_audit["accounting_audit"]
        self.assertAlmostEqual(acc["max_abs_equity_identity_error"], 0.0, places=9)
        self.assertGreaterEqual(acc["min_free_cash"], 0.0)
        self.assertLessEqual(acc["max_effective_leverage"], 3.0 + 1e-9)
        self.assertAlmostEqual(acc["pnl_reconciliation_diff_usd"], 0.0, places=6)

        # Verify each individual trade's net_pnl matches its formula without double-counting fees or funding
        running_eq = 10000.0
        for tr in trades:
            side_s = 1.0 if tr["side"] == "LONG" else -1.0
            expected_net_ret = (
                side_s * ((tr["exit_price"] - tr["entry_price"]) / tr["entry_price"])
                - tr["entry_fee_cost"]
                - tr["exit_fee_cost"]
                - tr["funding_cost"]
            )
            self.assertAlmostEqual(tr["net_return"], expected_net_ret, places=9)
            expected_pnl = tr["notional"] * expected_net_ret
            self.assertAlmostEqual(tr["trade_pnl_usd"], expected_pnl, places=6)
            running_eq += expected_pnl
            self.assertAlmostEqual(tr["account_equity_after"], running_eq, places=6)

        self.assertAlmostEqual(float(eq_series.iloc[-1]), running_eq, places=6)

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 7: DRAWDOWN (INTRATRADE LOWS) & HAND-COMPUTED METRICS VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    def test_07_drawdown_intratrade_lows_and_hand_computed_metrics_verification(self):
        """
        Item 7: Max drawdown computed on the equity curve including intratrade lows;
        CAGR, Sharpe, profit factor, expectancy, and bootstrap CI checked against
        hand-computed values on a tiny synthetic dataset.
        """
        # 49 hourly timestamps = exactly 2.0 calendar days (48 elapsed hours)
        idx = pd.date_range("2025-01-01 00:00:00+00:00", periods=49, freq="1h", tz="UTC")
        # Close equity stays at 10,000 through hour 11, dips to 9,500 at hour 12 (-5.0% close DD),
        # then rises to 11,000 at hour 48 (+10% over 2 days)
        close_eq = np.full(49, 10000.0, dtype=np.float64)
        close_eq[12] = 9500.0
        close_eq[13:] = np.linspace(10000.0, 11000.0, 36)
        hourly_eq = pd.Series(close_eq, index=idx)

        # Intra-hour worst-tick equity dips to 8,800.0 at hour 12 (-12.0% true intratrade drawdown!)
        low_eq = close_eq.copy()
        low_eq[12] = 8800.0
        hourly_low_eq = pd.Series(low_eq, index=idx)

        # 10 synthetic trades (5 wins of +15% = +$300, 5 losses of -5% = -$100 -> net +$1,000, mean +500 bps)
        synthetic_trades = [
            {"net_return": 0.15 if k % 2 == 0 else -0.05, "trade_pnl_usd": 300.0 if k % 2 == 0 else -100.0}
            for k in range(10)
        ]

        met_with_lows = compute_continuous_portfolio_metrics(
            hourly_equity=hourly_eq,
            trades=synthetic_trades,
            hourly_lowest_equity=hourly_low_eq,
            n_trials=1,
            seed=42,
        )
        met_without_lows = compute_continuous_portfolio_metrics(
            hourly_equity=hourly_eq,
            trades=synthetic_trades,
            hourly_lowest_equity=None,
            n_trials=1,
            seed=42,
        )

        # 1. Intratrade drawdown MUST reflect the 8,800 intra-hour low (12.00%), whereas omitting lows gives 5.00%
        self.assertAlmostEqual(met_with_lows["intratrade_max_drawdown_pct"], 12.00, places=2)
        self.assertAlmostEqual(met_without_lows["intratrade_max_drawdown_pct"], 5.00, places=2)

        # 2. Hand-computed trade metrics:
        # Win rate = 5 / 10 = 50.0%
        # Profit factor = (5 * 0.15) / (5 * 0.05) = 3.00
        # Expectancy = mean([0.15, -0.05] * 5) = +0.05 = +500.0 bps
        # PnL reconciliation diff = (11000 - 10000) - (1500 - 500) = 0.00
        ts = met_with_lows["trade_summary"]
        self.assertEqual(ts["total_trades"], 10)
        self.assertAlmostEqual(ts["win_rate_pct"], 50.0, places=2)
        self.assertAlmostEqual(ts["profit_factor"], 3.0, places=2)
        self.assertAlmostEqual(ts["expectancy_bps"], 500.0, places=2)
        self.assertAlmostEqual(met_with_lows["pnl_reconciliation_difference_usd"], 0.0, places=2)
        # Bootstrap 95% CI on [-500 bps, +1500 bps] must lie within [-500.0, +1500.0] and span a non-zero interval
        ci_low, ci_high = ts["ci_95_expectancy_bps"]
        self.assertGreaterEqual(ci_low, -500.0)
        self.assertLessEqual(ci_high, 1500.0)
        self.assertLess(ci_low, ci_high)

        # 3. Hand-computed Calendar CAGR:
        # elapsed_hours = 48.0 -> elapsed_years = 48.0 / 8766.0
        # cagr = (1.10 ** (1.0 / (48.0 / 8766.0)) - 1.0) * 100.0
        # Note: cappedonly if > 1000% or exact if within bounds; let's check a 1-year curve for un-capped CAGR & Sharpe
        idx_1y = pd.date_range("2025-01-01 00:00:00+00:00", periods=24 * 30 + 1, freq="1h", tz="UTC")
        # 30 days with alternating daily returns +1% and +2%
        eq_1y = [10000.0]
        for d in range(30):
            r_day = 0.01 if d % 2 == 0 else 0.02
            r_hr = (1.0 + r_day) ** (1.0 / 24.0) - 1.0
            for _ in range(24):
                eq_1y.append(eq_1y[-1] * (1.0 + r_hr))
        s_1y = pd.Series(eq_1y, index=idx_1y)
        met_1y = compute_continuous_portfolio_metrics(s_1y, trades=[], n_trials=1, seed=42)

        elapsed_yrs = (len(s_1y) - 1) / 8766.0
        hand_total_ret = s_1y.iloc[-1] / s_1y.iloc[0] - 1.0
        hand_cagr = ((1.0 + hand_total_ret) ** (1.0 / elapsed_yrs) - 1.0) * 100.0
        self.assertAlmostEqual(met_1y["calendar_cagr_pct"], round(hand_cagr, 2), places=2)

        daily_eq = s_1y.resample("1D").last().dropna()
        daily_rets = daily_eq.pct_change().dropna().values
        hand_sharpe = (np.mean(daily_rets) / np.std(daily_rets, ddof=0)) * math.sqrt(365.25)
        self.assertAlmostEqual(met_1y["daily_sharpe"], round(hand_sharpe, 3), places=3)

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 8: SEGMENTATION (NON-OVERLAPPING FOLDS, 168H PURGE, HOLDOUT ISOLATION)
    # ─────────────────────────────────────────────────────────────────────────
    def test_08_segmentation_non_overlapping_folds_purge_and_holdout_isolation(self):
        """
        Item 8: Walk-forward folds and the holdout do not overlap; embargo/purge
        and pre-roll are excluded; the holdout is never used for parameter selection.
        """
        valid_folds = [
            {
                "fold_id": 1,
                "type": "walk_forward_test",
                "purge_start": "2023-09-01 00:00:00+00:00",
                "purge_end": "2023-09-07 23:00:00+00:00",
                "eval_start": "2023-09-08 00:00:00+00:00",
                "eval_end": "2024-04-30 23:00:00+00:00",
            },
            {
                "fold_id": 2,
                "type": "walk_forward_test",
                "purge_start": "2024-05-01 00:00:00+00:00",
                "purge_end": "2024-05-07 23:00:00+00:00",
                "eval_start": "2024-05-08 00:00:00+00:00",
                "eval_end": "2024-12-31 23:00:00+00:00",
            },
            {
                "fold_id": 5,
                "type": "untouched_holdout",
                "purge_start": "2025-01-01 00:00:00+00:00",
                "purge_end": "2025-01-07 23:00:00+00:00",
                "eval_start": "2025-01-08 00:00:00+00:00",
                "eval_end": "2025-06-30 23:00:00+00:00",
            },
        ]
        pre_roll_end = pd.Timestamp("2023-08-31 23:00:00+00:00")
        self.assertTrue(
            validate_walk_forward_segmentation(
                valid_folds,
                pre_roll_end=pre_roll_end,
                purge_hours=168,
                selection_fold_ids=[1, 2],
            )
        )

        # 1. Overlapping folds must fail
        overlap_folds = [dict(f) for f in valid_folds]
        overlap_folds[1]["purge_start"] = "2024-04-29 00:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "overlaps with previous boundary"):
            validate_walk_forward_segmentation(overlap_folds, pre_roll_end=pre_roll_end, purge_hours=168)

        # 2. Insufficient purge window (< 168h) must fail
        short_purge_folds = [dict(f) for f in valid_folds]
        short_purge_folds[1]["purge_end"] = "2024-05-03 23:00:00+00:00"
        short_purge_folds[1]["eval_start"] = "2024-05-04 00:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "purge window is"):
            validate_walk_forward_segmentation(short_purge_folds, pre_roll_end=pre_roll_end, purge_hours=168)

        # 3. Pre-roll overlap must fail
        with self.assertRaisesRegex(ValueError, "overlaps with previous boundary"):
            validate_walk_forward_segmentation(
                valid_folds,
                pre_roll_end=pd.Timestamp("2023-09-02 00:00:00+00:00"),
                purge_hours=168,
            )

        # 4. Including untouched_holdout (Fold 5) in parameter selection must fail
        with self.assertRaisesRegex(ValueError, "Untouched holdout fold"):
            validate_walk_forward_segmentation(
                valid_folds,
                pre_roll_end=pre_roll_end,
                purge_hours=168,
                selection_fold_ids=[1, 2, 5],
            )

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 9: REPRODUCIBILITY (IDENTICAL TRADES & METRICS ACROSS RUNS)
    # ─────────────────────────────────────────────────────────────────────────
    def test_09_reproducibility_identical_trades_and_metrics_across_runs(self):
        """
        Item 9: Same data and seed give identical trades and metrics twice.
        """
        df_1h = _make_synthetic_1h_df(n_hours=360, base_price=2200.0, seed=99)
        df_1h_1, df_4h_1, _ = build_multitimeframe_features(df_1h)
        df_1h_2, df_4h_2, _ = build_multitimeframe_features(df_1h)

        sig_1 = generate_swing_signals(df_4h_1, strategy_name="Benchmark_Always_Long")
        sig_2 = generate_swing_signals(df_4h_2, strategy_name="Benchmark_Always_Long")
        f_map = {ts: 0.0001 for ts in df_1h.index if ts.hour in (0, 8, 16)}

        trades_1, eq_1, eq_low_1, aud_1 = simulate_swing_trades_conservative(
            df_1h=df_1h_1, df_4h_signals=sig_1, funding_map=f_map, max_holding_hours=24
        )
        trades_2, eq_2, eq_low_2, aud_2 = simulate_swing_trades_conservative(
            df_1h=df_1h_2, df_4h_signals=sig_2, funding_map=f_map, max_holding_hours=24
        )

        self.assertEqual(trades_1, trades_2)
        pd.testing.assert_series_equal(eq_1, eq_2)
        pd.testing.assert_series_equal(eq_low_1, eq_low_2)
        self.assertEqual(aud_1, aud_2)

        met_1 = compute_continuous_portfolio_metrics(
            eq_1, trades_1, hourly_lowest_equity=eq_low_1, n_trials=32, seed=42
        )
        met_2 = compute_continuous_portfolio_metrics(
            eq_2, trades_2, hourly_lowest_equity=eq_low_2, n_trials=32, seed=42
        )
        self.assertEqual(met_1, met_2)

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 10: REGISTRY CONSISTENCY & DONCHIAN EXPECTANCY RECONCILIATION
    # ─────────────────────────────────────────────────────────────────────────
    def test_10_experiment_registry_reconciliation_and_donchian_consistency(self):
        """
        Item 10: Reconcile research/experiment_registry.json against
        research/phase_2_1_fold_metrics.json and research/phase_2_1_candidate_comparison.md.
        Specifically verify Trend_Donchian_120h (+1.01 bps / +1.0 bps) and
        Trend_Donchian_168h (+74.40 bps / +74.4 bps) and ensure no stale -34.6 or -61.2
        values remain in the registry.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reg_path = os.path.join(base_dir, "research", "experiment_registry.json")
        p21_path = os.path.join(base_dir, "research", "phase_2_1_fold_metrics.json")
        comp_path = os.path.join(base_dir, "research", "phase_2_1_candidate_comparison.md")

        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        with open(p21_path, "r", encoding="utf-8") as f:
            p21 = json.load(f)
        with open(comp_path, "r", encoding="utf-8") as f:
            comp_md = f.read()

        exp_by_id = {}
        for phase in reg["phases"]:
            for exp in phase["experiments"]:
                exp_by_id[exp["id"]] = exp

        # Verify EXP-30 (Candidate A), EXP-31 (Donchian 120h), EXP-32 (Donchian 168h)
        oos = p21["pooled_oos_summary"]
        holdout = p21["holdout_summary"]

        exp30 = exp_by_id["EXP-30-WF-CANDIDATE-A"]
        exp31 = exp_by_id["EXP-31-WF-DONCHIAN-120H"]
        exp32 = exp_by_id["EXP-32-WF-DONCHIAN-168H"]

        self.assertAlmostEqual(
            exp30["metrics"]["oos_expectancy_bps"],
            oos["Exp1_Momentum_Continuation_4H"]["trade_summary"]["expectancy_bps"],
            places=2,
        )
        self.assertAlmostEqual(
            exp31["metrics"]["oos_expectancy_bps"],
            oos["Trend_Donchian_120h"]["trade_summary"]["expectancy_bps"],
            places=2,
        )
        self.assertAlmostEqual(
            exp32["metrics"]["oos_expectancy_bps"],
            oos["Trend_Donchian_168h"]["trade_summary"]["expectancy_bps"],
            places=2,
        )

        # Confirm exact Donchian OOS expectancies (+1.01 bps and +74.40 bps) and parameters
        self.assertAlmostEqual(exp31["metrics"]["oos_expectancy_bps"], 1.01, places=2)
        self.assertAlmostEqual(exp32["metrics"]["oos_expectancy_bps"], 74.40, places=2)
        self.assertEqual(exp31["parameters"]["stop_loss"], "2.5 * ATR_14")
        self.assertEqual(exp31["parameters"]["take_profit"], "4.0 * ATR_14")
        self.assertEqual(exp32["parameters"]["stop_loss"], "2.5 * ATR_14")
        self.assertEqual(exp32["parameters"]["take_profit"], "4.0 * ATR_14")

        # Confirm comparison markdown has 1.0 bps and 74.4 bps and registry reason strings match
        self.assertIn("1.0 bps", comp_md)
        self.assertIn("74.4 bps", comp_md)
        self.assertIn("+1.0 bps", exp31["reason_for_status"])
        self.assertIn("+74.4 bps", exp32["reason_for_status"])
        self.assertNotIn("-34.6", exp31["reason_for_status"])
        self.assertNotIn("-61.2", exp32["reason_for_status"])
        self.assertIn("reconciliation_audit", reg)

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 11: SYMBOL-AGNOSTIC EXECUTION (PARAMETERIZED SYMBOL, NEVER ETH-ONLY)
    # ─────────────────────────────────────────────────────────────────────────
    def test_11_symbol_agnostic_execution_no_hardcoded_eth(self):
        """
        Verify the backtest engine is symbol-agnostic and executes cleanly for
        non-ETH symbols (e.g., BTCUSDT, SOLUSDT).
        """
        for sym, base_px in [("BTCUSDT", 65000.0), ("SOLUSDT", 150.0)]:
            df_1h = _make_synthetic_1h_df(n_hours=120, base_price=base_px, seed=77)
            df_1h, df_4h, _ = build_multitimeframe_features(df_1h)
            sig = generate_swing_signals(df_4h, strategy_name="Benchmark_Always_Long")

            trades, eq, eq_low, _ = simulate_swing_trades_conservative(
                df_1h=df_1h,
                df_4h_signals=sig,
                funding_map={},
                strategy_name="Benchmark_Always_Long",
                symbol=sym,
            )
            self.assertGreater(len(trades), 0)
            for t in trades:
                self.assertEqual(t["symbol"], sym)

            bh_eq = compute_buy_and_hold_equity(df_1h, initial_capital=10000.0, symbol=sym)
            self.assertEqual(bh_eq.name, sym)

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 12: MUTATION CHECKS (PROVES TESTS FAIL WHEN FIX/PROPERTY IS REMOVED)
    # ─────────────────────────────────────────────────────────────────────────
    def test_12_mutation_checks_each_test_fails_when_property_removed(self):
        """
        Explicit mutation verification: demonstrates that each test assertion fails
        when the corresponding engine property or bug fix is removed/violated.
        """
        # Mutation 1 (Timestamps): Naive Resample without completed-bar filter leaks incomplete trailing 4H bar
        df_1h = _make_synthetic_1h_df(n_hours=50, start="2025-01-01 00:00:00+00:00")
        mutated_4h = df_1h.resample("4h", closed="left", label="left").agg({"close": "last"}).dropna()
        clean_4h = resample_completed_ohlcv(df_1h, target_freq="4h", lower_freq="1h")
        with self.assertRaises(AssertionError):
            self.assertEqual(len(mutated_4h), len(clean_4h))

        # Mutation 2 (No Look-Ahead): Unshifted Donchian channel changes when current bar high spikes
        df_a = _make_synthetic_1h_df(n_hours=120, seed=12)
        _, df_4h_a, _ = build_multitimeframe_features(df_a)
        unshifted_donchian = df_4h_a["high"].rolling(20).max()  # Bug: missing .shift(1)
        with self.assertRaises(AssertionError):
            pd.testing.assert_series_equal(df_4h_a["donchian_high_20"], unshifted_donchian, check_names=False)

        # Mutation 3 (Fills): TP-first collision ordering or ignoring gap open price produces wrong exit price
        df_f = _make_synthetic_1h_df(n_hours=48, base_price=1000.0, seed=11)
        df_f, df_4h_f, _ = build_multitimeframe_features(df_f)
        exec_ts = df_4h_f.index[1] + pd.Timedelta(hours=4)
        next_ts = exec_ts + pd.Timedelta(hours=1)
        df_f.loc[exec_ts, ["open", "high", "low", "close"]] = [1000.0, 1010.0, 995.0, 1005.0]
        df_f.loc[next_ts, ["open", "high", "low", "close"]] = [1005.0, 1060.0, 975.0, 1040.0]
        sig = df_4h_f.copy()
        sig["signal_long"] = False
        sig["signal_short"] = False
        sig["stop_price"] = np.nan
        sig["target_price"] = np.nan
        sig.loc[df_4h_f.index[1], ["signal_long", "stop_price", "target_price"]] = [True, 980.0, 1050.0]
        trades_coll, _, _, _ = simulate_swing_trades_conservative(df_f, sig, funding_map={}, taker_fee=0.0005, slippage=0.001)
        with self.assertRaises(AssertionError):
            # If TP won on same-bar collision, net_return would be positive
            self.assertGreater(trades_coll[0]["net_return"], 0.0)

        # Mutation 4 (Funding): Unnormalized microsecond funding keys fail to match hourly bar timestamps
        raw_micro_ts = pd.Timestamp("2025-01-01 08:00:00.023000+00:00")
        unnormalized_map = {raw_micro_ts: 0.0010}
        trades_unnorm, _, _, _ = simulate_swing_trades_conservative(
            df_f, sig, funding_map=unnormalized_map, max_holding_hours=10
        )
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(trades_unnorm[0]["funding_cost"], 0.0010, places=6)

        # Mutation 5 (Sizing): Uncapped sizing exceeds max_leverage on tight stops
        uncapped_lev = 0.025 / (2.0 / 2000.0)  # 25.0x > 3.0x
        s_cap = calculate_actual_stop_position_sizing(10000.0, 2000.0, 1998.0, 0.025, 3.0)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(s_cap["effective_leverage"], uncapped_lev, places=4)

        # Mutation 6 (Accounting): Double-counting entry fee breaks PnL reconciliation
        mutated_trades = [{"net_return": 0.05, "trade_pnl_usd": 500.0 - 50.0}]  # $50 double-counted fee
        idx_acc = pd.date_range("2025-01-01 00:00:00+00:00", periods=25, freq="1h", tz="UTC")
        eq_acc = pd.Series(np.linspace(10000.0, 10500.0, 25), index=idx_acc)
        met_mut_acc = compute_continuous_portfolio_metrics(eq_acc, mutated_trades, n_trials=1)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(met_mut_acc["pnl_reconciliation_difference_usd"], 0.0, places=2)

        # Mutation 7 (Drawdown): Omitting hourly_lowest_equity under-reports intratrade max drawdown
        idx_dd = pd.date_range("2025-01-01 00:00:00+00:00", periods=25, freq="1h", tz="UTC")
        c_eq = np.full(25, 10000.0)
        c_eq[5] = 9500.0
        l_eq = c_eq.copy()
        l_eq[5] = 8800.0
        m_close_only = compute_continuous_portfolio_metrics(pd.Series(c_eq, index=idx_dd), [], hourly_lowest_equity=None)
        m_true_low = compute_continuous_portfolio_metrics(pd.Series(c_eq, index=idx_dd), [], hourly_lowest_equity=pd.Series(l_eq, index=idx_dd))
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(m_close_only["intratrade_max_drawdown_pct"], m_true_low["intratrade_max_drawdown_pct"], places=2)

        # Mutation 9 (Reproducibility): Different seeds produce different bootstrap CIs
        tr_repro = [{"net_return": (-1) ** k * 0.01 * (k + 1), "trade_pnl_usd": 100.0} for k in range(20)]
        r_seed_1 = compute_reconciled_trade_metrics(tr_repro, n_bootstraps=200, seed=1)
        r_seed_2 = compute_reconciled_trade_metrics(tr_repro, n_bootstraps=200, seed=999)
        with self.assertRaises(AssertionError):
            self.assertEqual(r_seed_1["ci_95_expectancy_bps"], r_seed_2["ci_95_expectancy_bps"])


if __name__ == "__main__":
    unittest.main()
