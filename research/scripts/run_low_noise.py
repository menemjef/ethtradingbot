"""
research/scripts/run_low_noise.py

Stages:
  dev      development symbols only (data truncated at 2024-12-31); sanity checks + sensitivity grid
  dryrun   exercises the holdout code path on DEVELOPMENT data only (nothing written)
  holdout  the SINGLE final holdout run (refuses to run twice; logs a timestamp)
  report   renders research/low_noise_report.md from the two result files

No tuning: every rule/parameter lives in low_noise_strategies.py and was fixed before results.
"""
import os
import sys
import json
import time
import hashlib
import subprocess
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (BASE_DIR, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import low_noise_strategies as L
from utils.quant_evaluation import (
    compute_continuous_portfolio_metrics,
    compute_reconciled_trade_metrics,
)

RESEARCH_DIR = os.path.join(BASE_DIR, "research")
DEV_JSON = os.path.join(RESEARCH_DIR, "low_noise_dev_results.json")
HOLDOUT_JSON = os.path.join(RESEARCH_DIR, "low_noise_holdout_results.json")
HOLDOUT_LOG = os.path.join(RESEARCH_DIR, "low_noise_holdout_run_log.json")
REPORT_MD = os.path.join(RESEARCH_DIR, "low_noise_report.md")

LABELS = {"F0": "F0 (daily trend score, long/short)", "F0_LO": "F0-LO (daily, long-only)",
          "G0": "G0 (6H Donchian+EMA100)", "G1": "G1 (G0 + ER gate)", "G2": "G2 (G0 on EMA5-smoothed close)"}


# ───────────────────────────── metrics helpers ─────────────────────────────
def summarize(res: Dict[str, Any], n_boot: int = 2000) -> Dict[str, Any]:
    m = compute_continuous_portfolio_metrics(res["equity"], res["trades"],
                                             hourly_lowest_equity=res["equity_low"],
                                             n_trials=L.N_TRIALS, n_bootstraps=n_boot, seed=42)
    ts = m["trade_summary"]
    trades = res["trades"]
    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]
    hold = float(np.mean([t["holding_bars"] for t in trades])) if trades else 0.0
    return {
        "symbol": res["symbol"], "strategy": res["strategy"], "fee_per_side": res["fee_rate_per_side"],
        "start": str(res["equity"].index[0]), "end": str(res["equity"].index[-1]),
        "trades": ts["total_trades"], "cagr_pct": m["calendar_cagr_pct"],
        "sharpe": m["daily_sharpe"], "sortino": m["daily_sortino"],
        "max_dd_pct": m["intratrade_max_drawdown_pct"], "profit_factor": ts["profit_factor"],
        "win_rate_pct": ts["win_rate_pct"], "expectancy_bps": ts["expectancy_bps"],
        "expectancy_ci95_bps": ts["ci_95_expectancy_bps"], "dsr": m["deflated_sharpe_ratio"],
        "exposure_pct": round(res["exposure_frac"] * 100.0, 1),
        "avg_gross_leverage_in_market": round(res["avg_gross_leverage_in_market"], 3),
        "avg_holding_hours": round(hold, 1),
        "fee_drag_pct_of_initial": round(res["fees_usd"] / L.INITIAL_EQUITY * 100.0, 2),
        "funding_drag_pct_of_initial": round(res["funding_usd"] / L.INITIAL_EQUITY * 100.0, 2),
        "long_trades": len(longs), "short_trades": len(shorts),
        "long_pnl_usd": round(sum(t["trade_pnl_usd"] for t in longs), 2),
        "short_pnl_usd": round(sum(t["trade_pnl_usd"] for t in shorts), 2),
        "final_equity": round(float(res["equity"].iloc[-1]), 2),
        "pnl_reconciliation_diff_usd": m.get("pnl_reconciliation_difference_usd"),
    }


def pool_equity(results: List[Dict[str, Any]]):
    """Equal-weight portfolio, rebalanced hourly across symbols that have data at that hour.
    Worst-tick series assumes every symbol hits its hourly worst tick simultaneously (conservative)."""
    rets, lows = {}, {}
    for r in results:
        e = r["equity"]
        prev = e.shift(1)
        rets[r["symbol"]] = (e / prev - 1.0).iloc[1:]
        lows[r["symbol"]] = (r["equity_low"] / prev - 1.0).iloc[1:]
    rdf = pd.DataFrame(rets).sort_index()
    ldf = pd.DataFrame(lows).reindex(rdf.index)
    pr = rdf.mean(axis=1, skipna=True)
    pl = ldf.mean(axis=1, skipna=True)
    first = min(r["equity"].index[0] for r in results)
    index = pd.DatetimeIndex([first]).append(rdf.index)
    eq = pd.Series(L.INITIAL_EQUITY * np.concatenate([[1.0], np.cumprod(1.0 + pr.values)]), index=index)
    prev_eq = eq.shift(1).iloc[1:]
    eql = pd.concat([pd.Series([L.INITIAL_EQUITY], index=[first]),
                     prev_eq * (1.0 + pl)])
    eql = np.minimum(eql, eq)
    return eq, eql


def pooled_summary(results: List[Dict[str, Any]], n_boot: int = 2000) -> Dict[str, Any]:
    eq, eql = pool_equity(results)
    m = compute_continuous_portfolio_metrics(eq, [], hourly_lowest_equity=eql, n_trials=L.N_TRIALS,
                                             n_bootstraps=n_boot, seed=42)
    all_trades = [t for r in results for t in r["trades"]]
    years = max(0.01, (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400.0))
    tm = compute_reconciled_trade_metrics(all_trades, n_trials=L.N_TRIALS, calendar_duration_years=years,
                                          n_bootstraps=n_boot, seed=42)
    longs = [t for t in all_trades if t["side"] == "LONG"]
    shorts = [t for t in all_trades if t["side"] == "SHORT"]
    return {
        "strategy": results[0]["strategy"], "fee_per_side": results[0]["fee_rate_per_side"],
        "symbols": [r["symbol"] for r in results],
        "start": str(eq.index[0]), "end": str(eq.index[-1]),
        "trades": tm["total_trades"], "cagr_pct": m["calendar_cagr_pct"], "sharpe": m["daily_sharpe"],
        "sortino": m["daily_sortino"], "max_dd_pct": m["intratrade_max_drawdown_pct"],
        "profit_factor": tm["profit_factor"], "win_rate_pct": tm["win_rate_pct"],
        "expectancy_bps": tm["expectancy_bps"], "expectancy_ci95_bps": tm["ci_95_expectancy_bps"],
        "dsr": m["deflated_sharpe_ratio"],
        "exposure_pct": round(float(np.mean([r["exposure_frac"] for r in results])) * 100.0, 1),
        "avg_holding_hours": round(float(np.mean([t["holding_bars"] for t in all_trades])), 1) if all_trades else 0.0,
        "fee_drag_pct_of_initial": round(float(np.mean([r["fees_usd"] for r in results])) / L.INITIAL_EQUITY * 100.0, 2),
        "funding_drag_pct_of_initial": round(float(np.mean([r["funding_usd"] for r in results])) / L.INITIAL_EQUITY * 100.0, 2),
        "long_trades": len(longs), "short_trades": len(shorts),
        "long_pnl_usd": round(sum(t["trade_pnl_usd"] for t in longs), 2),
        "short_pnl_usd": round(sum(t["trade_pnl_usd"] for t in shorts), 2),
        "n_symbols_positive_cagr": None,
    }


def bh_result(df_1h: pd.DataFrame, symbol: str, trade_start: Optional[pd.Timestamp]) -> Dict[str, Any]:
    i0 = 0 if trade_start is None else int(df_1h.index.searchsorted(trade_start))
    d = df_1h.iloc[i0:]
    c0 = float(d["close"].iloc[0])
    eq = L.INITIAL_EQUITY * d["close"] / c0
    eql = np.minimum(L.INITIAL_EQUITY * d["low"] / c0, eq)
    return {"symbol": symbol, "strategy": "BUY_AND_HOLD", "equity": eq, "equity_low": eql, "trades": [],
            "fees_usd": 0.0, "funding_usd": 0.0, "fee_rate_per_side": 0.0, "exposure_frac": 1.0,
            "avg_gross_leverage": 1.0, "avg_gross_leverage_in_market": 1.0}


def bh_summary(res, n_boot=2000):
    s = summarize(res, n_boot)
    return s


def basket_summary(bh_results: List[Dict[str, Any]], n_boot=2000):
    eq, eql = pool_equity(bh_results)
    m = compute_continuous_portfolio_metrics(eq, [], hourly_lowest_equity=eql, n_trials=L.N_TRIALS,
                                             n_bootstraps=n_boot, seed=42)
    return {"symbols": [r["symbol"] for r in bh_results], "start": str(eq.index[0]), "end": str(eq.index[-1]),
            "cagr_pct": m["calendar_cagr_pct"], "sharpe": m["daily_sharpe"], "sortino": m["daily_sortino"],
            "max_dd_pct": m["intratrade_max_drawdown_pct"]}


# ───────────────────────────── a full evaluation block ─────────────────────────────
def evaluate_block(load_fn: Callable, symbols: List[str], trade_start: Optional[pd.Timestamp],
                   strategies: List[str], log=print, keep_eth_slice: bool = False) -> Dict[str, Any]:
    data = {s: load_fn(s) for s in symbols}
    out: Dict[str, Any] = {"symbols": symbols, "trade_start": str(trade_start) if trade_start is not None else None,
                           "per_symbol": {}, "pooled": {}, "buy_and_hold": {}, "eth_already_seen": {}}
    bh = {s: bh_result(data[s][0], s, trade_start) for s in symbols}
    out["buy_and_hold"] = {s: bh_summary(bh[s]) for s in symbols}
    out["basket_buy_and_hold"] = basket_summary(list(bh.values()))
    for strat in strategies:
        out["per_symbol"][strat] = {}
        out["pooled"][strat] = {}
        for tag, fee, nb in (("base", L.FEE_BASE, 2000), ("double", L.FEE_DOUBLE, 400)):
            results = []
            for s in symbols:
                df, fd = data[s]
                results.append(L.run_strategy(strat, df, fd, symbol=s, fee=fee, trade_start=trade_start))
            out["per_symbol"][strat][tag] = {r["symbol"]: summarize(r, nb) for r in results}
            ps = pooled_summary(results, nb)
            ps["n_symbols_positive_cagr"] = int(sum(1 for r in out["per_symbol"][strat][tag].values()
                                                  if r["cagr_pct"] > 0))
            ps["n_symbols"] = len(symbols)
            out["pooled"][strat][tag] = ps
            if keep_eth_slice and "ETHUSDT" in symbols:
                eth = [r for r in results if r["symbol"] == "ETHUSDT"][0]
                e = eth["equity"].loc[L.ETH_SEEN_START:]
                if len(e) > 100:
                    scale = L.INITIAL_EQUITY / float(e.iloc[0])
                    el = eth["equity_low"].loc[L.ETH_SEEN_START:]
                    tr = []
                    for t in eth["trades"]:
                        if pd.Timestamp(t["entry_time"]) >= L.ETH_SEEN_START:
                            t2 = dict(t)
                            t2["trade_pnl_usd"] = t["trade_pnl_usd"] * scale
                            t2["fee_usd"] = t.get("fee_usd", 0.0) * scale
                            t2["funding_usd"] = t.get("funding_usd", 0.0) * scale
                            tr.append(t2)
                    fees_usd = sum(t["fee_usd"] for t in tr)
                    fund_usd = sum(t["funding_usd"] for t in tr)
                    sl = {"symbol": "ETHUSDT", "strategy": strat, "equity": e * scale, "equity_low": el * scale,
                          "trades": tr, "fees_usd": fees_usd, "funding_usd": fund_usd,
                          "fee_rate_per_side": fee, "exposure_frac": eth["exposure_frac"],
                          "avg_gross_leverage": eth["avg_gross_leverage"],
                          "avg_gross_leverage_in_market": eth["avg_gross_leverage_in_market"]}
                    if strat not in out["eth_already_seen"]:
                        out["eth_already_seen"][strat] = {}
                    out["eth_already_seen"][strat][tag] = summarize(sl, 1000 if tag == "base" else 400)
                    if "BUY_AND_HOLD" not in out["eth_already_seen"]:
                        ebh = bh_result(data["ETHUSDT"][0], "ETHUSDT", L.ETH_SEEN_START)
                        out["eth_already_seen"]["BUY_AND_HOLD"] = summarize(ebh, 1000)
            log(f"  {strat:6s} {tag:6s} pooled: trades={ps['trades']} cagr={ps['cagr_pct']} "
                f"sharpe={ps['sharpe']} dd={ps['max_dd_pct']} +sym={ps['n_symbols_positive_cagr']}/{len(symbols)}")
    return out


def pass_rules(block: Dict[str, Any], strat: str) -> Dict[str, Any]:
    base = block["pooled"][strat]["base"]
    dbl = block["pooled"][strat]["double"]
    bh_dd = block["basket_buy_and_hold"]["max_dd_pct"]
    ci_lo = base["expectancy_ci95_bps"][0]
    c = {
        "sharpe_gt_0.7": bool(base["sharpe"] > 0.7),
        "cagr_positive_at_double_costs": bool(dbl["cagr_pct"] > 0),
        "maxdd_lt_50pct_of_buy_and_hold": bool(base["max_dd_pct"] < 0.5 * bh_dd),
        "expectancy_ci_lower_gt_0_or_(trades>=150_and_expectancy>0)": bool(
            ci_lo > 0 or (base["trades"] >= 150 and base["expectancy_bps"] > 0)),
        "positive_on_>=60pct_of_symbols": bool(base["n_symbols_positive_cagr"] / base["n_symbols"] >= 0.6),
    }
    return {"criteria": c, "passed": all(c.values()),
            "values": {"sharpe": base["sharpe"], "cagr_double_cost_pct": dbl["cagr_pct"],
                       "max_dd_pct": base["max_dd_pct"], "buy_and_hold_basket_max_dd_pct": bh_dd,
                       "50pct_of_bh_dd": round(0.5 * bh_dd, 2), "expectancy_bps": base["expectancy_bps"],
                       "expectancy_ci95_bps": base["expectancy_ci95_bps"], "pooled_trades": base["trades"],
                       "symbols_positive": f"{base['n_symbols_positive_cagr']}/{base['n_symbols']}"}}


# ───────────────────────────── stages ─────────────────────────────
def stage_dev(log=print) -> Dict[str, Any]:
    assert not L.holdout_unlocked()
    t0 = time.time()
    out = {"generated_utc": str(datetime.now(timezone.utc)), "dev_end": str(L.DEV_END),
           "symbols": L.DEV_SYMBOLS}
    log("DEV main runs")
    out["main"] = evaluate_block(L.load_symbol, L.DEV_SYMBOLS, None, L.STRATEGIES, log)
    # Sensitivity grid (development only; whole grid reported)
    log("DEV sensitivity grid")
    data = {s: L.load_symbol(s) for s in L.DEV_SYMBOLS}
    grid = []
    for strat in ("G0", "G1", "G2"):
        for dn in (15, 20, 30):
            for am in (2.5, 3.0, 4.0):
                res = [L.run_strategy(strat, data[s][0], data[s][1], symbol=s, donchian_n=dn, atr_mult=am)
                       for s in L.DEV_SYMBOLS]
                ps = pooled_summary(res, 500)
                grid.append({"strategy": strat, "donchian_n": dn, "atr_mult": am, **{k: ps[k] for k in
                             ("trades", "cagr_pct", "sharpe", "max_dd_pct", "expectancy_bps", "profit_factor")}})
                log(f"  grid {strat} N={dn} ATRx={am}: sharpe={ps['sharpe']} cagr={ps['cagr_pct']}")
    for strat in ("F0", "F0_LO"):
        for vt in (0.20, 0.25, 0.30):
            res = [L.run_strategy(strat, data[s][0], data[s][1], symbol=s, vol_target=vt) for s in L.DEV_SYMBOLS]
            ps = pooled_summary(res, 500)
            grid.append({"strategy": strat, "vol_target": vt, **{k: ps[k] for k in
                         ("trades", "cagr_pct", "sharpe", "max_dd_pct", "expectancy_bps", "profit_factor")}})
            log(f"  grid {strat} vt={vt}: sharpe={ps['sharpe']} cagr={ps['cagr_pct']}")
    out["sensitivity_grid"] = grid
    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open(DEV_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Wrote {DEV_JSON}")
    return out


def holdout_pipeline(load_fn, set_a: List[str], set_b: List[str], start_b: pd.Timestamp, log=print,
                     keep_eth_slice=True) -> Dict[str, Any]:
    res = {"A": evaluate_block(load_fn, set_a, None, L.STRATEGIES, log),
           "B": evaluate_block(load_fn, set_b, start_b, L.STRATEGIES, log, keep_eth_slice=keep_eth_slice)}
    res["verdicts"] = {}
    for strat in L.STRATEGIES:
        a = pass_rules(res["A"], strat)
        b = pass_rules(res["B"], strat)
        res["verdicts"][strat] = {"A": a, "B": b, "verdict": "PASSED" if (a["passed"] and b["passed"]) else "FAILED"}
    return res


def stage_dryrun(log=print):
    assert not L.holdout_unlocked()
    r = holdout_pipeline(L.load_symbol, L.DEV_SYMBOLS[:2], L.DEV_SYMBOLS, pd.Timestamp("2023-01-01", tz="UTC"), log)
    for s, v in r["verdicts"].items():
        log(s, v["verdict"], {k: v["A"]["criteria"][k] for k in v["A"]["criteria"]})
    log("dry-run complete (nothing written)")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def stage_holdout(log=print):
    if os.path.exists(HOLDOUT_LOG) or os.path.exists(HOLDOUT_JSON):
        raise SystemExit("Holdout has already been run (log exists). Refusing to run a second time.")
    if not os.path.exists(DEV_JSON):
        raise SystemExit("Run the dev stage first.")
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    except Exception:
        head = "unknown"
    started = str(datetime.now(timezone.utc))
    files = {f"{s.lower()}_{k}_full.parquet": None for s in L.ALL_SYMBOLS for k in ("1h", "funding")}
    for fn in files:
        files[fn] = _sha(os.path.join(L.CACHE_DIR, fn))
    with open(HOLDOUT_LOG, "w", encoding="utf-8") as f:
        json.dump({"status": "started", "started_utc": started, "git_head_at_start": head,
                   "dev_results_sha256": _sha(DEV_JSON), "data_sha256": files,
                   "note": "SINGLE holdout run. Presence of this file blocks any re-run."}, f, indent=2)
    L.unlock_holdout()
    res = holdout_pipeline(L.load_symbol, L.HOLDOUT_ONLY_SYMBOLS, L.ALL_SYMBOLS, L.HOLDOUT_B_START, log)
    res["run_started_utc"] = started
    res["run_finished_utc"] = str(datetime.now(timezone.utc))
    with open(HOLDOUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    with open(HOLDOUT_LOG, "r", encoding="utf-8") as f:
        lg = json.load(f)
    lg["status"] = "finished"
    lg["finished_utc"] = res["run_finished_utc"]
    lg["holdout_results_sha256"] = _sha(HOLDOUT_JSON)
    with open(HOLDOUT_LOG, "w", encoding="utf-8") as f:
        json.dump(lg, f, indent=2)
    log(f"Wrote {HOLDOUT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "dryrun", "holdout", "report"], required=True)
    a = ap.parse_args()
    if a.stage == "dev":
        stage_dev()
    elif a.stage == "dryrun":
        stage_dryrun()
    elif a.stage == "holdout":
        stage_holdout()
    else:
        from low_noise_report import write_report
        write_report()
