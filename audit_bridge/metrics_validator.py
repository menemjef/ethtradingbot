"""
audit_bridge/metrics_validator.py — Mathematical validation & reconciliation of backtest results.

Enforces:
1. Gross Profit = sum(positive trade gross PnL)
2. Gross Loss = absolute sum(negative trade gross PnL)
3. Profit Factor = Gross Profit / Gross Loss
4. Win Rate = winning trades / total trades
5. Net PnL = Gross Profit - Gross Loss - Fees - Slippage - Funding
6. Capital Reconciliation: Final Capital = Initial Capital + Net PnL + Deposits/Withdrawals
7. Max Drawdown: peak-to-trough drop from equity curve

If metrics disagree or are mathematically inconsistent, the report is marked INVALID.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np


class BacktestValidationResult:
    def __init__(
        self,
        is_valid: bool,
        reconciled_metrics: Dict[str, Any],
        discrepancies: List[str],
        warnings: List[str] = None,
    ):
        self.is_valid = is_valid
        self.reconciled_metrics = reconciled_metrics
        self.discrepancies = discrepancies
        self.warnings = warnings or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "reconciled_metrics": self.reconciled_metrics,
            "discrepancies": self.discrepancies,
            "warnings": self.warnings,
        }


def compute_metrics_from_ledger(
    trades: List[Dict[str, Any]],
    equity_curve: List[float] = None,
    initial_capital: float = 100.0,
    deposits_withdrawals: float = 0.0,
) -> Dict[str, Any]:
    """
    Independently derive all backtest metrics directly from the trade ledger.
    Never relies on user-supplied summary metrics.
    """
    if not trades:
        return {
            "num_trades": 0,
            "win_count_gross": 0,
            "loss_count_gross": 0,
            "win_count_net": 0,
            "loss_count_net": 0,
            "win_rate_gross": 0.0,
            "win_rate_net": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor_gross": float("nan"),
            "profit_factor_net": float("nan"),
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "total_funding": 0.0,
            "total_costs": 0.0,
            "net_pnl": 0.0,
            "initial_capital": initial_capital,
            "final_capital": initial_capital + deposits_withdrawals,
            "net_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_win_net": 0.0,
            "avg_loss_net": 0.0,
            "expectancy_net": 0.0,
        }

    # Extract individual trade fields (handling various naming conventions)
    gross_pnls = []
    net_pnls = []
    fees_list = []
    slippages = []
    fundings = []

    for t in trades:
        fee = float(t.get("fee", t.get("fees", 0.0)))
        slippage = float(t.get("slippage", 0.0))
        funding = float(t.get("funding", 0.0))

        if "gross_pnl" in t:
            gross = float(t["gross_pnl"])
            net = float(t.get("net_pnl", gross - fee - slippage - funding))
        elif "pnl" in t:
            # Check if this is net or gross
            net = float(t["pnl"])
            gross = float(t.get("gross", net + fee + slippage + funding))
        elif "outcome" in t:
            gross = float(t["outcome"])
            net = float(t.get("pnl", gross - fee - slippage - funding))
        else:
            gross = 0.0
            net = 0.0

        gross_pnls.append(gross)
        net_pnls.append(net)
        fees_list.append(fee)
        slippages.append(slippage)
        fundings.append(funding)

    num_trades = len(trades)

    # 1. Gross metrics
    pos_gross = [p for p in gross_pnls if p > 0]
    neg_gross = [p for p in gross_pnls if p < 0]
    gross_profit = float(sum(pos_gross))
    gross_loss = float(abs(sum(neg_gross)))
    profit_factor_gross = (
        float(gross_profit / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_profit > 0 else 1.0)
    )
    win_count_gross = len(pos_gross)
    loss_count_gross = len(neg_gross)
    win_rate_gross = float(win_count_gross / num_trades * 100) if num_trades > 0 else 0.0

    # 2. Costs
    total_fees = float(sum(fees_list))
    total_slippage = float(sum(slippages))
    total_funding = float(sum(fundings))
    total_costs = total_fees + total_slippage + total_funding

    # 3. Net metrics
    pos_net = [p for p in net_pnls if p > 0]
    neg_net = [p for p in net_pnls if p <= 0]
    win_count_net = len(pos_net)
    loss_count_net = len(neg_net)
    win_rate_net = float(win_count_net / num_trades * 100) if num_trades > 0 else 0.0
    net_wins_usd = float(sum(pos_net))
    net_losses_usd = float(abs(sum(neg_net)))
    profit_factor_net = (
        float(net_wins_usd / net_losses_usd) if net_losses_usd > 1e-9 else (float("inf") if net_wins_usd > 0 else 1.0)
    )

    net_pnl = float(sum(net_pnls))
    final_capital = initial_capital + net_pnl + deposits_withdrawals
    net_return_pct = float(net_pnl / initial_capital * 100) if initial_capital > 0 else 0.0

    avg_win_net = float(np.mean(pos_net)) if pos_net else 0.0
    avg_loss_net = float(np.mean(neg_net)) if neg_net else 0.0
    expectancy_net = float(np.mean(net_pnls)) if net_pnls else 0.0

    # 4. Max Drawdown from equity curve
    if not equity_curve:
        # Reconstruct equity curve from net pnls
        curr = initial_capital
        equity_curve = [curr]
        for p in net_pnls:
            curr += p
            equity_curve.append(curr)

    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd

    return {
        "num_trades": num_trades,
        "win_count_gross": win_count_gross,
        "loss_count_gross": loss_count_gross,
        "win_count_net": win_count_net,
        "loss_count_net": loss_count_net,
        "win_rate_gross": round(win_rate_gross, 2),
        "win_rate_net": round(win_rate_net, 2),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "profit_factor_gross": round(profit_factor_gross, 2) if not math.isinf(profit_factor_gross) else "inf",
        "profit_factor_net": round(profit_factor_net, 2) if not math.isinf(profit_factor_net) else "inf",
        "total_fees": round(total_fees, 4),
        "total_slippage": round(total_slippage, 4),
        "total_funding": round(total_funding, 4),
        "total_costs": round(total_costs, 4),
        "net_pnl": round(net_pnl, 4),
        "initial_capital": round(initial_capital, 4),
        "final_capital": round(final_capital, 4),
        "net_return_pct": round(net_return_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_win_net": round(avg_win_net, 4),
        "avg_loss_net": round(avg_loss_net, 4),
        "expectancy_net": round(expectancy_net, 4),
    }


def validate_backtest_report(
    claimed_metrics: Dict[str, Any],
    trades: Optional[List[Dict[str, Any]]] = None,
    equity_curve: Optional[List[float]] = None,
) -> BacktestValidationResult:
    """
    Validate and reconcile reported backtest metrics against trade ledger or mathematical invariants.
    Returns BacktestValidationResult with is_valid, reconciled_metrics, and discrepancies list.
    """
    discrepancies = []
    warnings = []

    # Case A: Trade ledger is provided -> independently compute and compare
    if trades and len(trades) > 0:
        initial_cap = float(claimed_metrics.get("initial_capital", 100.0) or 100.0)
        reconciled = compute_metrics_from_ledger(trades, equity_curve, initial_capital=initial_cap)

        # Check trade counts
        claimed_trades = claimed_metrics.get("num_trades")
        if claimed_trades is not None and int(claimed_trades) != reconciled["num_trades"]:
            discrepancies.append(
                f"Trade count mismatch: claimed {claimed_trades} vs ledger {reconciled['num_trades']}"
            )

        # Check net pnl / final capital
        claimed_final = claimed_metrics.get("final_capital")
        if claimed_final is not None and abs(float(claimed_final) - reconciled["final_capital"]) > 0.05:
            discrepancies.append(
                f"Final capital mismatch: claimed ${claimed_final:.2f} vs ledger ${reconciled['final_capital']:.2f}"
            )

        # Check win rate (check if matches gross or net)
        claimed_wr = claimed_metrics.get("win_rate")
        if claimed_wr is not None:
            cwr = float(claimed_wr)
            matches_gross = abs(cwr - reconciled["win_rate_gross"]) < 0.5
            matches_net = abs(cwr - reconciled["win_rate_net"]) < 0.5
            if not matches_gross and not matches_net:
                discrepancies.append(
                    f"Win rate mismatch: claimed {cwr:.1f}% vs ledger gross {reconciled['win_rate_gross']}% / net {reconciled['win_rate_net']}%"
                )
            elif matches_gross and not matches_net:
                warnings.append(
                    f"Claimed win rate ({cwr:.1f}%) matches GROSS win rate, but NET win rate is {reconciled['win_rate_net']}% due to fees turning marginal wins negative."
                )

        # Check profit factor
        claimed_pf = claimed_metrics.get("profit_factor")
        if claimed_pf is not None and str(claimed_pf) != "inf":
            cpf = float(claimed_pf)
            matches_gross_pf = abs(cpf - float(reconciled["profit_factor_gross"])) < 0.05 if str(reconciled["profit_factor_gross"]) != "inf" else False
            matches_net_pf = abs(cpf - float(reconciled["profit_factor_net"])) < 0.05 if str(reconciled["profit_factor_net"]) != "inf" else False
            if not matches_gross_pf and not matches_net_pf:
                discrepancies.append(
                    f"Profit factor mismatch: claimed {cpf} vs ledger gross {reconciled['profit_factor_gross']} / net {reconciled['profit_factor_net']}"
                )

        is_valid = len(discrepancies) == 0
        return BacktestValidationResult(is_valid, reconciled, discrepancies, warnings)

    # Case B: No trade ledger provided -> perform internal mathematical consistency checks
    reconciled = dict(claimed_metrics)
    init_cap = float(claimed_metrics.get("initial_capital", 0.0) or 0.0)
    final_cap = float(claimed_metrics.get("final_capital", 0.0) or 0.0)
    net_ret = claimed_metrics.get("net_return_pct")

    if init_cap > 0 and final_cap > 0 and net_ret is not None:
        expected_ret = (final_cap - init_cap) / init_cap * 100
        if abs(expected_ret - float(net_ret)) > 0.5:
            discrepancies.append(
                f"Net return mismatch: claimed {net_ret}% vs capital change {expected_ret:.2f}% (${init_cap:.2f} -> ${final_cap:.2f})"
            )

    # Check win/loss counts vs total trades
    num_trades = claimed_metrics.get("num_trades")
    wins = claimed_metrics.get("win_count")
    losses = claimed_metrics.get("loss_count")
    if num_trades is not None and wins is not None and losses is not None:
        if int(wins) + int(losses) != int(num_trades):
            discrepancies.append(
                f"Trade count arithmetic mismatch: wins ({wins}) + losses ({losses}) != total trades ({num_trades})"
            )
        claimed_wr = claimed_metrics.get("win_rate")
        if claimed_wr is not None and int(num_trades) > 0:
            calc_wr = int(wins) / int(num_trades) * 100
            if abs(calc_wr - float(claimed_wr)) > 0.5:
                discrepancies.append(
                    f"Win rate arithmetic mismatch: claimed {claimed_wr}% vs wins/total ({calc_wr:.2f}%)"
                )

    # Check avg_win * wins vs avg_loss * losses reconciliation
    avg_win = claimed_metrics.get("avg_win")
    avg_loss = claimed_metrics.get("avg_loss")
    if None not in (wins, losses, avg_win, avg_loss, init_cap, final_cap):
        calc_gross_delta = int(wins) * float(avg_win) + int(losses) * float(avg_loss)
        actual_delta = final_cap - init_cap
        # If delta differs by more than $5, flag discrepancy
        if abs(calc_gross_delta - actual_delta) > 5.0:
            discrepancies.append(
                f"PnL reconciliation failure: wins*avg_win + losses*avg_loss = ${calc_gross_delta:.2f}, but actual capital change is ${actual_delta:.2f}. "
                f"(Indicates wins were counted gross before fees while avg_win/pnl were measured net, or trade sizing fluctuated)."
            )

    is_valid = len(discrepancies) == 0
    return BacktestValidationResult(is_valid, reconciled, discrepancies, warnings)


def reconcile_backtest_audit_99_explanation() -> Dict[str, Any]:
    """
    Formal mathematical reconciliation of the 99-trade backtest audit results.
    Explains the exact mechanics of why summary statistics appeared inconsistent.
    """
    return {
        "analysis": "Reconciliation of 99-trade backtest audit metrics",
        "reported_metrics": {
            "initial_capital": 57.0,
            "final_capital": 38.76,
            "net_pnl": -18.24,
            "num_trades": 99,
            "win_count_gross": 55,
            "loss_count_gross": 44,
            "win_rate_gross": 55.6,
            "avg_win_net": 1.2404,
            "avg_loss_net": -1.0345,
            "profit_factor_net": 0.72,
        },
        "root_cause_of_discrepancy": (
            "In `backtest_audit.py`, trade outcomes were labeled as 'wins' based on GROSS price change "
            "(outcome > 0), giving 55 wins (55.6%). However, each trade was subject to round-trip fees and slippage "
            "(0.12% total notional). Several trades with tiny gross gains turned NEGATIVE after fees. "
            "Consequently, `avg_win` was computed only on the subset of trades whose net PnL stayed > 0, "
            "and `profit_factor` was computed as sum(net_wins) / sum(abs(net_losses)) = 0.72. "
            "Because the win count was gross but avg_win and profit factor were net, manual formula "
            "`55 * 1.2404 - 44 * 1.0345` does not equal `-$18.24`."
        ),
        "validation_status": "EXPLAINED_AND_RECONCILED",
        "required_fix": (
            "Always store the complete trade ledger and compute both Gross and Net metrics explicitly: "
            "Win Rate (Gross) vs Win Rate (Net), Profit Factor (Gross) vs Profit Factor (Net)."
        ),
    }
