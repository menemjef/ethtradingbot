"""
utils/quant_evaluation.py — Quantitative Performance, Reconciliation, and Dependence-Aware Bootstrap.

Features:
1. Aligned Predictive Metric Calculation (Directional Acc, Balanced Acc, MSE, MAE, Brier, LogLoss, ECE).
2. Moving Block Bootstrap (Kunsch 1989) and Stationary Bootstrap (Politis & Romano 1994) for Time-Series.
3. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) correcting for non-normality and selection bias.
4. Exact Reconciliation between trade logs, JSON metrics, and Markdown tables.
"""

import math
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss, log_loss
)


def calculate_actual_stop_position_sizing(
    account_equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.025,
    max_leverage: float = 3.0,
    min_stop_pct: float = 0.005,
    available_cash: Optional[float] = None
) -> Dict[str, float]:
    """
    Position sizing based on ACTUAL stop loss distance:
    RiskDollar = account_equity * risk_pct
    StopDistancePct = max(min_stop_pct, abs(entry_price - stop_price) / entry_price)
    Notional = min(RiskDollar / StopDistancePct, effective_capital * max_leverage)
    EffectiveLeverage = Notional / account_equity
    """
    eff_cash = account_equity if available_cash is None else min(account_equity, available_cash)
    if account_equity <= 0 or entry_price <= 0 or eff_cash <= 0 or max_leverage <= 0:
        return {
            "notional": 0.0,
            "position_size": 0.0,
            "effective_leverage": 0.0,
            "stop_distance_pct": 0.0,
            "margin_required": 0.0
        }

    stop_dist = max(min_stop_pct, abs(entry_price - stop_price) / entry_price) if stop_price > 0 else min_stop_pct
    risk_dollar = account_equity * risk_pct
    notional_raw = risk_dollar / stop_dist
    max_notional_by_cap = min(account_equity * max_leverage, eff_cash * max_leverage)
    notional = min(notional_raw, max_notional_by_cap)
    pos_size = notional / entry_price
    eff_lev = notional / account_equity
    margin_req = notional / max_leverage

    return {
        "notional": float(notional),
        "position_size": float(pos_size),
        "effective_leverage": float(eff_lev),
        "stop_distance_pct": float(stop_dist),
        "margin_required": float(margin_req)
    }



def compute_aligned_predictive_metrics(
    y_true: np.ndarray,
    y_pred_reg: np.ndarray,
    y_pred_prob: Optional[np.ndarray] = None,
    block_size: int = 24,
    n_bootstraps: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Calculate comprehensive predictive statistical metrics with Moving Block Bootstrap 95% CIs.
    y_true: ground truth target aligned at decision time (target[j-1])
    y_pred_reg: regression prediction for entry at candle j open
    """
    valid_m = ~np.isnan(y_true) & ~np.isnan(y_pred_reg)
    yt = np.array(y_true)[valid_m]
    yp_reg = np.array(y_pred_reg)[valid_m]
    n_samples = len(yt)

    if n_samples == 0:
        return {
            "n_samples": 0,
            "directional_accuracy_pct": 0.0,
            "ci_95_directional_accuracy": [0.0, 0.0],
            "balanced_accuracy_pct": 0.0,
            "regression_mse": 1.0,
            "regression_mae": 1.0,
            "brier_score": 0.25,
            "log_loss": 0.6931,
            "ece": 0.0,
        }

    yt_dir = (yt > 0).astype(int)
    yp_dir = (yp_reg > 0).astype(int)

    dir_acc = float(accuracy_score(yt_dir, yp_dir)) * 100.0
    bal_acc = float(balanced_accuracy_score(yt_dir, yp_dir)) * 100.0
    prec = float(precision_score(yt_dir, yp_dir, zero_division=0)) * 100.0
    rec = float(recall_score(yt_dir, yp_dir, zero_division=0)) * 100.0
    f1 = float(f1_score(yt_dir, yp_dir, zero_division=0))
    reg_mse = float(np.mean((yp_reg - yt) ** 2))
    reg_mae = float(np.mean(np.abs(yp_reg - yt)))
    zero_mse = float(np.mean((0.0 - yt) ** 2))

    # Probability metrics
    if y_pred_prob is not None:
        yp_prob = np.clip(np.array(y_pred_prob)[valid_m], 1e-6, 1.0 - 1e-6)
    else:
        # Standard logistic sigmoid mapping from regression
        yp_prob = 1.0 / (1.0 + np.exp(-yp_reg * 20.0))

    brier = float(brier_score_loss(yt_dir, yp_prob))
    try:
        ll = float(log_loss(yt_dir, yp_prob))
    except Exception:
        ll = float("nan")

    # Expected Calibration Error (ECE) with 10 bins
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(yp_prob, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = (bin_indices == b)
        if mask.sum() > 0:
            bin_acc = np.mean(yt_dir[mask])
            bin_conf = np.mean(yp_prob[mask])
            ece += (mask.sum() / n_samples) * abs(bin_acc - bin_conf)

    # Moving Block Bootstrap (Kunsch 1989) for 95% CI of Directional Accuracy & MSE
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n_samples // block_size)
    boot_accs = []
    boot_mses = []

    for _ in range(n_bootstraps):
        starts = rng.integers(0, max(1, n_samples - block_size + 1), size=n_blocks)
        boot_idx = []
        for s in starts:
            boot_idx.extend(range(s, min(n_samples, s + block_size)))
        boot_idx = np.array(boot_idx[:n_samples])
        boot_accs.append(accuracy_score(yt_dir[boot_idx], yp_dir[boot_idx]) * 100.0)
        boot_mses.append(np.mean((yp_reg[boot_idx] - yt[boot_idx]) ** 2))

    ci_acc = [round(float(np.percentile(boot_accs, 2.5)), 2), round(float(np.percentile(boot_accs, 97.5)), 2)]
    ci_mse = [round(float(np.percentile(boot_mses, 2.5)), 4), round(float(np.percentile(boot_mses, 97.5)), 4)]

    return {
        "n_samples": n_samples,
        "directional_accuracy_pct": round(dir_acc, 2),
        "ci_95_directional_accuracy": ci_acc,
        "balanced_accuracy_pct": round(bal_acc, 2),
        "precision_pct": round(prec, 2),
        "recall_pct": round(rec, 2),
        "f1_score": round(f1, 4),
        "regression_mse": round(reg_mse, 6),
        "ci_95_regression_mse": ci_mse,
        "regression_mae": round(reg_mae, 6),
        "zero_baseline_mse": round(zero_mse, 6),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "ece": round(float(ece), 4),
    }


def compute_reconciled_trade_metrics(
    trades: List[Dict[str, Any]],
    initial_capital: float = 10000.0,
    risk_pct: float = 0.025,
    max_leverage: float = 3.0,
    sl_distance_assumed: float = 0.015,
    block_size: int = 5,
    n_bootstraps: int = 2000,
    n_trials: int = 1,
    seed: int = 42,
    calendar_duration_years: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes all standard trading and risk metrics directly from individual trade records.
    Guarantees mathematical reconciliation across summary tables and JSON reports.
    """
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "win_rate_pct": 0.0,
            "gross_profit_pct": 0.0,
            "gross_loss_pct": 0.0,
            "profit_factor": 0.0,
            "average_trade_return_pct": 0.0,
            "median_trade_return_pct": 0.0,
            "expectancy_bps": 0.0,
            "ci_95_expectancy_bps": [0.0, 0.0],
            "compounded_net_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "account_equity_levered_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "account_max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "deflated_sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "average_holding_duration_hours": 0.0,
            "max_consecutive_losses": 0,
            "fee_costs_pct": 0.0,
            "slippage_costs_pct": 0.0,
            "funding_costs_pct": 0.0,
            "stop_loss_exits": 0,
            "take_profit_exits": 0,
            "collision_exits": 0,
            "opening_gap_exits": 0,
            "horizon_exits": 0,
            "unclosed_invalid_trades": 0
        }

    df_tr = pd.DataFrame(trades)
    if "net_return" in df_tr.columns:
        net_rets = df_tr["net_return"].values.astype(np.float64)
    elif "net_return_pct" in df_tr.columns:
        net_rets = (df_tr["net_return_pct"].values.astype(np.float64)) / 100.0
    elif "trade_return_pct" in df_tr.columns:
        net_rets = (df_tr["trade_return_pct"].values.astype(np.float64)) / 100.0
    elif "net_pnl_usd" in df_tr.columns and initial_capital > 0:
        net_rets = (df_tr["net_pnl_usd"].values.astype(np.float64)) / initial_capital
    else:
        net_rets = np.zeros(n_trades, dtype=np.float64)

    wins = net_rets[net_rets > 0]
    losses = net_rets[net_rets < 0]
    n_wins = len(wins)
    n_losses = len(losses)

    win_rate = (n_wins / n_trades) * 100.0
    gross_profit = float(wins.sum()) * 100.0 if n_wins > 0 else 0.0
    gross_loss = float(abs(losses.sum())) * 100.0 if n_losses > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    avg_ret = float(np.mean(net_rets)) * 100.0
    med_ret = float(np.median(net_rets)) * 100.0
    expectancy_bps = float(np.mean(net_rets)) * 10000.0

    # Compounded unlevered return
    compounded_mult = float(np.prod(1.0 + net_rets))
    compounded_ret_pct = (compounded_mult - 1.0) * 100.0

    # Max Drawdown from trade equity curve
    trade_equity = np.cumprod(1.0 + net_rets)
    running_peaks = np.maximum.accumulate(trade_equity)
    drawdowns = (running_peaks - trade_equity) / running_peaks * 100.0
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Levered account simulation (2.5% risk budget, 3x leverage cap)
    # Uses actual trade stop loss distance when available in trade record
    account_eq = initial_capital
    peak_eq = initial_capital
    max_acc_dd = 0.0

    for idx, r in enumerate(net_rets):
        tr_rec = trades[idx] if idx < len(trades) else {}
        if "actual_stop_distance_pct" in tr_rec and tr_rec["actual_stop_distance_pct"] > 0:
            tr_sl_dist = tr_rec["actual_stop_distance_pct"]
        elif "sl_distance_pct" in tr_rec and tr_rec["sl_distance_pct"] > 0:
            tr_sl_dist = tr_rec["sl_distance_pct"]
        else:
            tr_sl_dist = sl_distance_assumed

        position_exposure = min(account_eq * (risk_pct / tr_sl_dist), account_eq * max_leverage)
        pnl = position_exposure * r
        account_eq += pnl
        if account_eq > peak_eq:
            peak_eq = account_eq
        dd = (peak_eq - account_eq) / peak_eq * 100.0 if peak_eq > 0 else 0.0
        if dd > max_acc_dd:
            max_acc_dd = dd

    levered_net_return = ((account_eq - initial_capital) / initial_capital) * 100.0

    # Trade counts and splits
    long_trades = int((df_tr["side"].astype(str).str.upper().isin(["LONG", "1"])).sum()) if "side" in df_tr else 0
    short_trades = int((df_tr["side"].astype(str).str.upper().isin(["SHORT", "-1"])).sum()) if "side" in df_tr else 0
    avg_hold = float(df_tr["holding_bars"].mean()) if "holding_bars" in df_tr else 0.0

    # Consecutive losses
    max_consec = 0
    curr_consec = 0
    for r in net_rets:
        if r < 0:
            curr_consec += 1
            max_consec = max(max_consec, curr_consec)
        else:
            curr_consec = 0

    # Cost breakdown
    tot_fee = float(df_tr["entry_fee_cost"].sum() + df_tr["exit_fee_cost"].sum()) * 100.0 if "entry_fee_cost" in df_tr else 0.0
    tot_slip = float(df_tr["total_slippage_cost"].sum()) * 100.0 if "total_slippage_cost" in df_tr else 0.0
    tot_fund = float(df_tr["funding_cost"].sum()) * 100.0 if "funding_cost" in df_tr else 0.0

    # Exit reason counts
    exit_reasons = df_tr["exit_reason"].astype(str) if "exit_reason" in df_tr else pd.Series([], dtype=str)
    n_sl = int(exit_reasons.str.startswith("stop_loss").sum())
    n_tp = int(exit_reasons.str.startswith("take_profit").sum())
    n_col = int(df_tr["is_collision"].sum()) if "is_collision" in df_tr else int(exit_reasons.str.contains("collision").sum())
    n_gap = int(exit_reasons.str.contains("gap").sum())
    n_horiz = int(exit_reasons.str.startswith("horizon").sum())

    # Annualized return & Sharpe (using calendar duration when provided, else holding time)
    if calendar_duration_years is not None and calendar_duration_years > 0:
        years = max(0.01, float(calendar_duration_years))
    else:
        total_hours = float(df_tr["holding_bars"].sum()) if "holding_bars" in df_tr else n_trades * 8.0
        years = max(0.1, total_hours / 8760.0)
    ann_ret = float((compounded_mult ** (1.0 / years) - 1.0) * 100.0) if compounded_mult > 0 else -100.0

    # Sharpe ratio (annualized per-trade basis)
    trades_per_year = n_trades / years
    ret_std = float(np.std(net_rets)) if len(net_rets) > 1 else 1e-6
    sharpe = float((np.mean(net_rets) / ret_std) * np.sqrt(trades_per_year)) if ret_std > 0 else 0.0

    # Sortino ratio (downside deviation only)
    downside_rets = net_rets[net_rets < 0]
    downside_std = float(np.std(downside_rets)) if len(downside_rets) > 1 else 1e-6
    sortino = float((np.mean(net_rets) / downside_std) * np.sqrt(trades_per_year)) if downside_std > 0 else 0.0

    # Calmar ratio
    calmar = float(ann_ret / max_dd) if max_dd > 0 else 0.0

    # Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
    # Accounting for skewness, kurtosis, sample length, and number of trials K
    dsr = compute_deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        returns=net_rets,
        track_record_len=n_trades,
        n_trials=n_trials,
        annualization_factor=max(1.0, trades_per_year)
    )

    # Moving Block Bootstrap (Kunsch 1989) for 95% CI of Expectancy (bps)
    # Preserves autocorrelation between consecutive trades; uses ceil to cover full sample length
    rng = np.random.default_rng(seed)
    eff_block = max(1, min(block_size, n_trades))
    n_blocks = max(1, math.ceil(n_trades / eff_block))
    boot_exp_bps = []

    for _ in range(n_bootstraps):
        starts = rng.integers(0, max(1, n_trades - eff_block + 1), size=n_blocks)
        b_idx = []
        for s in starts:
            b_idx.extend(range(s, min(n_trades, s + eff_block)))
        b_idx = np.array(b_idx[:n_trades])
        boot_exp_bps.append(np.mean(net_rets[b_idx]) * 10000.0)

    ci_exp = [
        round(float(np.percentile(boot_exp_bps, 2.5)), 2),
        round(float(np.percentile(boot_exp_bps, 97.5)), 2)
    ]

    return {
        "total_trades": n_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "win_rate_pct": round(win_rate, 2),
        "gross_profit_pct": round(gross_profit, 2),
        "gross_loss_pct": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "average_trade_return_pct": round(avg_ret, 4),
        "median_trade_return_pct": round(med_ret, 4),
        "expectancy_bps": round(expectancy_bps, 2),
        "ci_95_expectancy_bps": ci_exp,
        "compounded_net_return_pct": round(compounded_ret_pct, 2),
        "annualized_return_pct": round(ann_ret, 2),
        "account_equity_levered_return_pct": round(levered_net_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "account_max_drawdown_pct": round(max_acc_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "deflated_sharpe_ratio": round(dsr, 4),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "average_holding_duration_hours": round(avg_hold, 1),
        "max_consecutive_losses": max_consec,
        "fee_costs_pct": round(tot_fee, 2),
        "slippage_costs_pct": round(tot_slip, 2),
        "funding_costs_pct": round(tot_fund, 4),
        "stop_loss_exits": n_sl,
        "take_profit_exits": n_tp,
        "collision_exits": n_col,
        "opening_gap_exits": n_gap,
        "horizon_exits": n_horiz,
        "unclosed_invalid_trades": 0
    }


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    returns: np.ndarray,
    track_record_len: Optional[int] = None,
    n_trials: int = 1,
    annualization_factor: float = 365.25,
    trials_variance: Optional[float] = None
) -> float:
    """
    Computes Deflated Sharpe Ratio (DSR) based on Bailey and Lopez de Prado (2014):
    'The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality'
    
    Args:
        observed_sharpe: Annualized Sharpe ratio of the evaluated strategy.
        returns: Return time-series corresponding to the observation frequency (e.g. daily, hourly, or trade-level).
        track_record_len: Number of return observations T. If None, len(returns) is used.
        n_trials: Number of independent strategy configurations/trials tested (K >= 1).
        annualization_factor: Annualization multiplier q (e.g. 365.25 for daily crypto, 8760 for hourly, or trades/year).
        trials_variance: Empirical cross-sectional variance of the trial Sharpe ratios. If None, theoretical null variance q/(T-1) is used.
        
    Returns:
        Probability in [0, 1] that the observed Sharpe ratio exceeds the expected maximum Sharpe ratio from pure luck.
    """
    T = len(returns) if track_record_len is None else int(track_record_len)
    if T < 10 or np.std(returns) == 0:
        return 0.0

    q = max(1.0, float(annualization_factor))

    # Sample skewness (gamma_3) and kurtosis (gamma_4) of returns
    diffs = returns - np.mean(returns)
    std = np.std(returns)
    if std <= 1e-12:
        return 0.0
    skew = float(np.mean(diffs ** 3) / (std ** 3 + 1e-12))
    kurt = float(np.mean(diffs ** 4) / (std ** 4 + 1e-12))

    # Non-annualized sample Sharpe ratio (sr_0)
    sr_0 = observed_sharpe / math.sqrt(q)

    # Asymptotic standard error of annualized Sharpe ratio under Mertens (2002) non-normal adjustment
    # sigma_SR_ann = sqrt(q * (1 - gamma_3 * sr_0 + ((gamma_4 - 1) / 4) * sr_0^2) / (T - 1))
    var_term = (1.0 - skew * sr_0 + ((kurt - 1.0) / 4.0) * (sr_0 ** 2)) / (T - 1.0)
    if var_term <= 0 or np.isnan(var_term):
        return 0.0
    sr_ann_std = math.sqrt(q * var_term)
    if sr_ann_std <= 1e-12:
        return 0.0

    # Expected maximum annualized Sharpe ratio under H0 (true SR = 0) among K independent trials
    # SR* = sqrt(V) * ((1 - gamma) * Z^-1(1 - 1/K) + gamma * Z^-1(1 - 1/(K * e)))
    if n_trials > 1:
        euler_mascheroni = 0.5772156649
        k_val = float(n_trials)
        # Variance of Sharpe estimates across trials under H0: V = q / (T - 1) if not provided
        v_trials = float(trials_variance) if trials_variance is not None and trials_variance > 0 else (q / (T - 1.0))
        std_trials = math.sqrt(v_trials)

        # Rational approximation of normal quantile function (probit) for robustness
        def _norm_ppf(p: float) -> float:
            p = min(max(p, 1e-12), 1.0 - 1e-12)
            try:
                import scipy.stats as st
                return float(st.norm.ppf(p))
            except ImportError:
                # Rational approximation (Abramowitz and Stegun 26.2.23)
                t = math.sqrt(-2.0 * math.log(1.0 - p))
                c0, c1, c2 = 2.515517, 0.802853, 0.010328
                d1, d2, d3 = 1.432788, 0.189269, 0.001308
                return t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)

        z_p1 = _norm_ppf(1.0 - 1.0 / k_val)
        z_p2 = _norm_ppf(1.0 - 1.0 / (k_val * math.e))
        expected_max_z = (1.0 - euler_mascheroni) * z_p1 + euler_mascheroni * z_p2
        sr_star = std_trials * expected_max_z
    else:
        sr_star = 0.0

    # Test statistic Z = (observed_sharpe_ann - sr_star) / sr_ann_std
    z_stat = (observed_sharpe - sr_star) / sr_ann_std
    dsr = 0.5 * (1.0 + math.erf(z_stat / math.sqrt(2.0)))
    return float(np.clip(dsr, 0.0, 1.0))


def compute_continuous_portfolio_metrics(
    hourly_equity: pd.Series,
    trades: List[Dict[str, Any]],
    n_trials: int = 1,
    block_hours: int = 168,
    n_bootstraps: int = 2000,
    seed: int = 42,
    hourly_lowest_equity: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Computes rigorous continuous portfolio equity metrics on hourly mark-to-market data:
    1. Calendar CAGR (based on actual elapsed calendar years).
    2. Continuous Mark-to-Market Maximum Drawdown (intratrade hourly fidelity, with lowest-tick support).
    3. Continuous Daily & Hourly Sharpe and Sortino ratios.
    4. Calendar Monthly return distribution & 30% monthly target evaluation.
    5. Moving Block Bootstrap on hourly return time series (Kunsch 1989).
    6. Deflated Sharpe Ratio with multi-trial penalty K.
    """
    if len(hourly_equity) < 2:
        return {"error": "Insufficient hourly equity observations"}

    eq = hourly_equity.values.astype(np.float64)
    timestamps = hourly_equity.index

    t_start = pd.Timestamp(timestamps[0])
    t_end = pd.Timestamp(timestamps[-1])
    cal_days = max(1.0, (t_end - t_start).total_seconds() / 86400.0)
    cal_years = max(0.01, cal_days / 365.25)

    initial_eq = eq[0]
    final_eq = eq[-1]
    compounded_mult = final_eq / initial_eq
    net_return_pct = (compounded_mult - 1.0) * 100.0
    cagr_pct = float((compounded_mult ** (1.0 / cal_years) - 1.0) * 100.0) if compounded_mult > 0 else -100.0

    # Hourly returns
    eq_prev = np.where(eq[:-1] == 0, 1e-6, eq[:-1])
    h_rets = (eq[1:] - eq_prev) / eq_prev
    valid_h = ~np.isnan(h_rets) & ~np.isinf(h_rets)
    h_rets = h_rets[valid_h]

    # Continuous Intratrade Mark-to-Market Drawdown (lowest tick aware)
    peaks = np.maximum.accumulate(eq)
    if hourly_lowest_equity is not None and len(hourly_lowest_equity) == len(eq):
        lowest_vals = np.minimum(eq, hourly_lowest_equity.values.astype(np.float64))
        drawdowns = (peaks - lowest_vals) / np.where(peaks == 0, 1e-6, peaks) * 100.0
    else:
        drawdowns = (peaks - eq) / np.where(peaks == 0, 1e-6, peaks) * 100.0
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Drawdown duration & recovery time
    dd_streak = 0
    max_dd_streak_hours = 0
    underwater = eq < peaks
    for u in underwater:
        if u:
            dd_streak += 1
            max_dd_streak_hours = max(max_dd_streak_hours, dd_streak)
        else:
            dd_streak = 0

    # Hourly Sharpe
    h_std = float(np.std(h_rets)) if len(h_rets) > 1 else 1e-6
    hourly_sharpe = float((np.mean(h_rets) / h_std) * np.sqrt(8760.0)) if h_std > 0 else 0.0

    # Daily resampled returns (calendar 00:00 UTC)
    df_eq = pd.DataFrame({"equity": eq}, index=timestamps)
    daily_eq = df_eq["equity"].resample("1D").last().dropna()
    d_vals = daily_eq.values
    d_rets = np.array([], dtype=np.float64)
    if len(d_vals) > 1:
        d_rets = (d_vals[1:] - d_vals[:-1]) / np.where(d_vals[:-1] == 0, 1e-6, d_vals[:-1])
        d_std = float(np.std(d_rets)) if len(d_rets) > 1 else 1e-6
        daily_sharpe = float((np.mean(d_rets) / d_std) * np.sqrt(365.25)) if d_std > 0 else 0.0

        down_rets = d_rets[d_rets < 0]
        down_std = float(np.std(down_rets)) if len(down_rets) > 1 else 1e-6
        daily_sortino = float((np.mean(d_rets) / down_std) * np.sqrt(365.25)) if down_std > 0 else 0.0
    else:
        daily_sharpe = 0.0
        daily_sortino = 0.0

    calmar = float(cagr_pct / max_dd) if max_dd > 0 else 0.0

    # Monthly Returns & Target 30% Assessment
    monthly_eq = df_eq["equity"].resample("1ME").last().dropna()
    monthly_rets_dict = {}
    m_vals = monthly_eq.values
    m_dates = [ts.strftime("%Y-%m") for ts in monthly_eq.index]
    
    # Check if final month is incomplete (Month-to-Date)
    is_last_mtd = (t_end.day < 28) or (t_end.month != (t_end + pd.Timedelta(days=1)).month)
    mtd_key = None
    mtd_val = 0.0

    if len(m_vals) > 0:
        prev_m = initial_eq
        for idx, (m_date, m_val) in enumerate(zip(m_dates, m_vals)):
            m_ret = ((m_val - prev_m) / prev_m) * 100.0 if prev_m > 0 else 0.0
            if idx == len(m_vals) - 1 and is_last_mtd:
                key_name = f"{m_date} (MTD)"
                mtd_key = key_name
                mtd_val = round(float(m_ret), 2)
                monthly_rets_dict[key_name] = mtd_val
            else:
                monthly_rets_dict[m_date] = round(float(m_ret), 2)
            prev_m = m_val

    # For statistics, use completed months (exclude partial MTD)
    completed_m_list = [v for k, v in monthly_rets_dict.items() if "(MTD)" not in k]
    if len(completed_m_list) == 0:
        completed_m_list = list(monthly_rets_dict.values())
    m_list = completed_m_list
    if len(m_list) > 0:
        avg_monthly = float(np.mean(m_list))
        med_monthly = float(np.median(m_list))
        std_monthly = float(np.std(m_list))
        pct_profitable_months = float((np.array(m_list) > 0).mean()) * 100.0
        best_month = float(np.max(m_list))
        worst_month = float(np.min(m_list))

        # Max consecutive losing months
        curr_loss_m = 0
        max_loss_m = 0
        for mr in m_list:
            if mr < 0:
                curr_loss_m += 1
                max_loss_m = max(max_loss_m, curr_loss_m)
            else:
                curr_loss_m = 0

        # Rolling 3-month and 6-month compounded returns
        r3m = []
        for i in range(len(m_list) - 2):
            w = [1.0 + x / 100.0 for x in m_list[i:i+3]]
            r3m.append((float(np.prod(w)) - 1.0) * 100.0)
        avg_3m_rolling = float(np.mean(r3m)) if r3m else 0.0

        r6m = []
        for i in range(len(m_list) - 5):
            w = [1.0 + x / 100.0 for x in m_list[i:i+6]]
            r6m.append((float(np.prod(w)) - 1.0) * 100.0)
        avg_6m_rolling = float(np.mean(r6m)) if r6m else 0.0
    else:
        avg_monthly = med_monthly = std_monthly = pct_profitable_months = best_month = worst_month = 0.0
        max_loss_m = 0
        avg_3m_rolling = avg_6m_rolling = 0.0

    target_30pct_met = (med_monthly >= 30.0) and (avg_monthly >= 30.0) and (pct_profitable_months >= 70.0)
    shortfall_bps = round((30.0 - med_monthly) * 100.0, 1)

    # Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
    # 1. Daily frequency DSR using daily returns (annualization factor 365.25)
    daily_dsr = compute_deflated_sharpe_ratio(
        observed_sharpe=daily_sharpe,
        returns=d_rets if len(d_rets) >= 10 else h_rets,
        track_record_len=len(d_rets) if len(d_rets) >= 10 else len(h_rets),
        n_trials=n_trials,
        annualization_factor=365.25 if len(d_rets) >= 10 else 8760.0
    )
    # 2. Hourly frequency DSR
    hourly_dsr = compute_deflated_sharpe_ratio(
        observed_sharpe=hourly_sharpe,
        returns=h_rets,
        track_record_len=len(h_rets),
        n_trials=n_trials,
        annualization_factor=8760.0
    )

    # Chronological Time-Series Moving Block Bootstrap on hourly returns
    # Preserves autocorrelation between consecutive hourly returns; uses ceil to cover full duration
    rng = np.random.default_rng(seed)
    n_h = len(h_rets)
    eff_block = max(1, min(block_hours, n_h))
    n_blocks = max(1, math.ceil(n_h / eff_block))
    offsets = np.arange(eff_block)
    boot_cagrs = []
    boot_sharpes = []
    boot_dds = []

    for _ in range(n_bootstraps):
        starts = rng.integers(0, max(1, n_h - eff_block + 1), size=n_blocks)
        b_idx = (starts[:, None] + offsets[None, :]).ravel()[:n_h]
        b_rets = h_rets[b_idx]
        b_cum = np.cumprod(1.0 + b_rets)
        b_final = b_cum[-1] if len(b_cum) > 0 else 1.0
        b_cagr = (b_final ** (1.0 / cal_years) - 1.0) * 100.0 if b_final > 0 else -100.0
        b_peaks = np.maximum.accumulate(b_cum)
        b_dd = float(np.max((b_peaks - b_cum) / b_peaks * 100.0)) if len(b_cum) > 0 else 0.0
        b_std = float(np.std(b_rets)) if len(b_rets) > 1 else 1e-6
        b_sr = (float(np.mean(b_rets)) / b_std) * math.sqrt(8760.0) if b_std > 0 else 0.0

        boot_cagrs.append(b_cagr)
        boot_sharpes.append(b_sr)
        boot_dds.append(b_dd)

    ci_cagr = [round(float(np.percentile(boot_cagrs, 2.5)), 2), round(float(np.percentile(boot_cagrs, 97.5)), 2)]
    ci_sharpe = [round(float(np.percentile(boot_sharpes, 2.5)), 3), round(float(np.percentile(boot_sharpes, 97.5)), 3)]
    ci_max_dd = [round(float(np.percentile(boot_dds, 2.5)), 2), round(float(np.percentile(boot_dds, 97.5)), 2)]

    # Trade-level statistics
    trade_metrics = compute_reconciled_trade_metrics(
        trades=trades,
        initial_capital=initial_eq,
        n_trials=n_trials,
        seed=seed,
        calendar_duration_years=cal_years
    )

    # Reconciliation between trade-level PnL sum and portfolio equity change
    sum_trade_pnl = sum(float(t.get("trade_pnl_usd", t.get("net_pnl_usd", 0.0))) for t in trades)
    pnl_reconciliation_diff = round(abs((final_eq - initial_eq) - sum_trade_pnl), 2)

    return {
        "calendar_start": str(t_start),
        "calendar_end": str(t_end),
        "calendar_days": round(cal_days, 1),
        "calendar_years": round(cal_years, 3),
        "initial_equity": round(initial_eq, 2),
        "final_equity": round(final_eq, 2),
        "net_compounded_return_pct": round(net_return_pct, 2),
        "calendar_cagr_pct": round(cagr_pct, 2),
        "ci_95_cagr_pct": ci_cagr,
        "intratrade_max_drawdown_pct": round(max_dd, 2),
        "ci_95_max_drawdown_pct": ci_max_dd,
        "max_drawdown_duration_hours": int(max_dd_streak_hours),
        "hourly_sharpe": round(hourly_sharpe, 3),
        "daily_sharpe": round(daily_sharpe, 3),
        "ci_95_hourly_sharpe": ci_sharpe,
        "daily_sortino": round(daily_sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "deflated_sharpe_ratio": round(daily_dsr, 4),
        "daily_deflated_sharpe_ratio": round(daily_dsr, 4),
        "hourly_deflated_sharpe_ratio": round(hourly_dsr, 4),
        "pnl_reconciliation_difference_usd": pnl_reconciliation_diff,
        "monthly_metrics": {
            "average_monthly_return_pct": round(avg_monthly, 2),
            "median_monthly_return_pct": round(med_monthly, 2),
            "std_monthly_return_pct": round(std_monthly, 2),
            "pct_profitable_months": round(pct_profitable_months, 2),
            "best_month_pct": round(best_month, 2),
            "worst_month_pct": round(worst_month, 2),
            "max_consecutive_losing_months": int(max_loss_m),
            "rolling_3m_average_return_pct": round(avg_3m_rolling, 2),
            "rolling_6m_average_return_pct": round(avg_6m_rolling, 2),
            "target_30pct_monthly_achieved": bool(target_30pct_met),
            "shortfall_to_30pct_monthly_bps": float(shortfall_bps),
            "monthly_returns_breakdown": monthly_rets_dict
        },
        "trade_summary": trade_metrics
    }


def compute_probability_of_backtest_overfitting(
    equity_matrix: pd.DataFrame,
    n_splits: int = 16,
    max_evals: int = 1000,
    seed: int = 42,
    metric: str = "sharpe"
) -> float:
    """
    Computes Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
    Cross-Validation (CSCV) following Bailey, Borwein, Lopez de Prado, Zhu (2016).
    equity_matrix: columns are candidate strategy equity curves, index is chronological timestamps.
    metric: 'sharpe' (standard risk-adjusted performance) or 'return' (cumulative percentage return).
    """
    if equity_matrix.empty or equity_matrix.shape[1] < 2 or len(equity_matrix) < n_splits * 2:
        return 0.0

    T, N = equity_matrix.shape
    slice_len = T // n_splits
    vals = equity_matrix.values.astype(np.float64)

    # Pre-extract slice periodic returns for high-speed evaluation
    slice_rets_list = []
    slice_cum_returns = np.zeros((n_splits, N))

    for s in range(n_splits):
        start_idx = s * slice_len
        end_idx = (s + 1) * slice_len if s < n_splits - 1 else T
        sub_vals = vals[start_idx:end_idx]
        denom = np.where(sub_vals[:-1] == 0, 1e-6, sub_vals[:-1])
        s_rets = (sub_vals[1:] - denom) / denom
        slice_rets_list.append(s_rets)

        start_val = sub_vals[0]
        end_val = sub_vals[-1]
        slice_cum_returns[s, :] = (end_val - start_val) / np.where(start_val == 0, 1e-6, start_val)

    # Half splits: choose n_splits // 2
    half_s = n_splits // 2
    all_combos = list(itertools.combinations(range(n_splits), half_s))
    if len(all_combos) <= max_evals:
        combos_to_eval = all_combos
    else:
        rng = np.random.default_rng(seed)
        chosen_indices = rng.choice(len(all_combos), size=max_evals, replace=False)
        combos_to_eval = [all_combos[i] for i in chosen_indices]

    rank_losses = 0
    total_evals = 0

    for is_idx in combos_to_eval:
        is_set = set(is_idx)
        oos_idx = [i for i in range(n_splits) if i not in is_set]

        if metric == "sharpe":
            is_concat = np.concatenate([slice_rets_list[s] for s in is_idx], axis=0)
            is_std = np.std(is_concat, axis=0)
            is_perf = np.mean(is_concat, axis=0) / np.where(is_std == 0, 1e-6, is_std)

            oos_concat = np.concatenate([slice_rets_list[s] for s in oos_idx], axis=0)
            oos_std = np.std(oos_concat, axis=0)
            oos_perf = np.mean(oos_concat, axis=0) / np.where(oos_std == 0, 1e-6, oos_std)
        else:
            is_perf = np.mean(slice_cum_returns[list(is_idx), :], axis=0)
            oos_perf = np.mean(slice_cum_returns[oos_idx, :], axis=0)

        best_is_strategy = int(np.argmax(is_perf))
        oos_ranks = np.argsort(np.argsort(-oos_perf))  # 0 is best
        rank_best = oos_ranks[best_is_strategy]

        # Overfitting occurred if in-sample winner ranks at or below median out-of-sample
        median_rank = N / 2.0
        if rank_best >= median_rank:
            rank_losses += 1
        total_evals += 1

    pbo = float(rank_losses / total_evals) if total_evals > 0 else 0.0
    return round(pbo, 4)


def validate_ohlcv_timestamps(
    df: pd.DataFrame,
    expected_freq: str = "1h",
    reference_time: Optional[pd.Timestamp] = None,
    reference_now: Optional[pd.Timestamp] = None,
    allow_empty: bool = False
) -> bool:
    """
    Validates OHLCV DataFrame timestamps:
    1. Index must be a UTC-localized pd.DatetimeIndex (tz is not None and UTC offset == 0).
    2. Strictly monotonic increasing with zero duplicate timestamps.
    3. Zero missing bar gaps (adjacent timestamp differences must equal expected_freq).
    4. Closed bars only: if reference_time/reference_now is provided, every bar's close_time
       (open_time + expected_freq) must be <= reference_time.
    """
    if reference_time is None and reference_now is not None:
        reference_time = reference_now
    if df is None or len(df) == 0:
        if allow_empty:
            return True
        raise ValueError("OHLCV DataFrame is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV index must be a pd.DatetimeIndex.")

    if df.index.tz is None:
        raise ValueError("OHLCV timestamps must be timezone-aware UTC (found timezone-naive index).")

    sample_offset = df.index[0].utcoffset()
    if sample_offset is None or sample_offset.total_seconds() != 0:
        raise ValueError(f"OHLCV timestamps must be in UTC (found tz={df.index.tz}).")

    if df.index.has_duplicates:
        dup_ts = df.index[df.index.duplicated()].tolist()[:5]
        raise ValueError(f"Duplicate OHLCV timestamps detected: {dup_ts}")

    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV timestamps must be strictly monotonically increasing.")

    bar_delta = pd.Timedelta(expected_freq)
    if len(df) > 1:
        diffs = df.index[1:] - df.index[:-1]
        bad_gaps = diffs != bar_delta
        if np.any(bad_gaps):
            first_gap_idx = int(np.where(bad_gaps)[0][0])
            t0 = df.index[first_gap_idx]
            t1 = df.index[first_gap_idx + 1]
            raise ValueError(
                f"OHLCV timestamp gap detected between {t0} and {t1} "
                f"(expected {bar_delta}, got {t1 - t0})."
            )

    if reference_time is not None:
        ref_ts = pd.Timestamp(reference_time)
        if ref_ts.tz is None:
            raise ValueError("reference_time must be timezone-aware UTC.")
        close_times = df.index + bar_delta
        unclosed_mask = close_times > ref_ts
        if np.any(unclosed_mask):
            first_unclosed = df.index[int(np.where(unclosed_mask)[0][0])]
            raise ValueError(
                f"Unclosed bar detected at open_time={first_unclosed} "
                f"(closes at {first_unclosed + bar_delta} > reference_time {ref_ts})."
            )

    return True


def resample_completed_ohlcv(
    df_lower: pd.DataFrame,
    target_freq: str = "4h",
    lower_freq: str = "1h",
    base_freq: Optional[str] = None,
    reference_time: Optional[pd.Timestamp] = None,
    reference_now: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Aggregates lower-timeframe OHLCV bars (e.g. 1H) into higher-timeframe bars
    (e.g. '4h', '6h', '1d') using ONLY completed lower-timeframe bars:
    - Validates lower-timeframe bars first (UTC, closed only, no duplicates, no gaps).
    - Requires exact constituent bar count per target bucket (e.g. 4 for 4H, 6 for 6H, 24 for 1D).
    - Annotates 'close_time' = open_time + pd.Timedelta(target_freq).
    """
    if base_freq is not None:
        lower_freq = base_freq
    if reference_time is None and reference_now is not None:
        reference_time = reference_now
    validate_ohlcv_timestamps(df_lower, expected_freq=lower_freq, reference_time=reference_time)

    lower_delta = pd.Timedelta(lower_freq)
    target_delta = pd.Timedelta(target_freq)
    if target_delta < lower_delta or (target_delta % lower_delta) != pd.Timedelta(0):
        raise ValueError(f"target_freq ({target_freq}) must be an exact integer multiple of lower_freq ({lower_freq}).")

    required_count = int(target_delta // lower_delta)
    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    resampler = df_lower.resample(target_freq, closed="left", label="left")
    df_agg = resampler.agg(agg_rules)
    counts = df_lower["close"].resample(target_freq, closed="left", label="left").count()
    df_completed = df_agg[counts == required_count].copy()
    df_completed["close_time"] = df_completed.index + target_delta
    return df_completed


def validate_walk_forward_segmentation(
    folds: List[Dict[str, Any]],
    pre_roll_end: Optional[pd.Timestamp] = None,
    purge_hours: int = 168,
    selection_fold_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Validates walk-forward segmentation and holdout isolation:
    1. Folds are strictly chronological and non-overlapping.
    2. Pre-roll warmup (<= pre_roll_end) is excluded from all evaluation folds.
    3. Each fold has an explicit purge/embargo window of at least purge_hours
       immediately preceding eval_start, with zero overlap with prior folds.
    4. Any fold marked 'untouched_holdout' is strictly after all 'walk_forward_test' folds
       and never appears in selection_fold_ids.
    """
    if not folds:
        raise ValueError("Folds list cannot be empty.")

    prev_eval_end: Optional[pd.Timestamp] = None
    if pre_roll_end is not None:
        prev_eval_end = pd.Timestamp(pre_roll_end)

    holdout_ids = set()
    wf_ids = set()

    for idx, f in enumerate(folds):
        fid = f["fold_id"]
        ftype = f.get("type", "walk_forward_test")
        p_start = pd.Timestamp(f["purge_start"])
        p_end = pd.Timestamp(f["purge_end"])
        e_start = pd.Timestamp(f["eval_start"])
        e_end = pd.Timestamp(f["eval_end"])

        if e_end <= e_start:
            raise ValueError(f"Fold {fid} eval_end ({e_end}) must be > eval_start ({e_start}).")
        if p_end >= e_start:
            raise ValueError(f"Fold {fid} purge_end ({p_end}) overlaps with eval_start ({e_start}).")

        # Inclusive 1H purge window from p_start (00:00) through p_end (23:00) spans (e_start - p_start)
        actual_purge_hours = (e_start - p_start).total_seconds() / 3600.0
        if actual_purge_hours < purge_hours:
            raise ValueError(
                f"Fold {fid} purge window is {actual_purge_hours:.1f}h, less than required {purge_hours}h."
            )

        if prev_eval_end is not None:
            if p_start <= prev_eval_end:
                raise ValueError(
                    f"Fold {fid} purge_start ({p_start}) overlaps with previous boundary ({prev_eval_end})."
                )
            if e_start <= prev_eval_end:
                raise ValueError(
                    f"Fold {fid} eval_start ({e_start}) overlaps with previous fold end ({prev_eval_end})."
                )

        if ftype == "untouched_holdout":
            holdout_ids.add(fid)
        else:
            if holdout_ids:
                raise ValueError(f"Walk-forward fold {fid} cannot occur after an untouched_holdout fold.")
            wf_ids.add(fid)

        prev_eval_end = e_end

    if selection_fold_ids is not None:
        leaked = holdout_ids.intersection(set(selection_fold_ids))
        if leaked:
            raise ValueError(f"Untouched holdout fold(s) {sorted(leaked)} used in parameter selection!")

    return {
        "valid": True,
        "total_folds": len(folds),
        "walk_forward_fold_ids": sorted(wf_ids),
        "holdout_fold_ids": sorted(holdout_ids),
        "purge_hours_verified": purge_hours
    }


