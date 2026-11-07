"""
research/scripts/run_market_neutral.py

Execution pipeline for Market-Neutral strategies N1 and N2:
  stage dev_val: Runs Development (2019-09 to 2022-12) and Validation (2023-01 to 2024-12).
  stage dryrun:  Tests final-test pipeline on validation data without unlocking final test.
  stage final:   SINGLE final test run (2025-01 to end); cryptographically pinned and locked.
  stage report:  Renders research/market_neutral_report.md.
"""
import os
import sys
import json
import time
import hashlib
import subprocess
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import research.scripts.market_neutral_strategies as M

RESEARCH_DIR = os.path.join(BASE_DIR, "research")
DEV_VAL_JSON = os.path.join(RESEARCH_DIR, "market_neutral_dev_val_results.json")
FINAL_JSON = os.path.join(RESEARCH_DIR, "market_neutral_final_results.json")
FINAL_LOG = os.path.join(RESEARCH_DIR, "market_neutral_final_run_log.json")
REPORT_MD = os.path.join(RESEARCH_DIR, "market_neutral_report.md")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slice_simulation_result(res: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    """Extracts a sub-period from a simulation result and rescales initial equity."""
    eq_sub = res["equity"].loc[start:end]
    eq_low_sub = res["equity_low"].loc[start:end]
    if len(eq_sub) == 0:
        raise ValueError(f"No equity data between {start} and {end}")

    scale = M.INITIAL_EQUITY / float(eq_sub.iloc[0])
    scaled_eq = eq_sub * scale
    scaled_eq_low = eq_low_sub * scale

    # Weekly / 8h returns in this period
    sub_res = dict(res)
    sub_res["start"] = str(scaled_eq.index[0])
    sub_res["end"] = str(scaled_eq.index[-1])
    sub_res["equity"] = scaled_eq
    sub_res["equity_low"] = scaled_eq_low
    sub_res["initial_equity"] = M.INITIAL_EQUITY
    sub_res["final_equity"] = float(scaled_eq.iloc[-1])

    # Slice weekly returns
    if "weekly_returns" in res:
        # Approximate weekly returns within this window
        w_eq = scaled_eq.resample("W-SUN").last().dropna()
        sub_res["weekly_returns"] = list((w_eq / w_eq.shift(1) - 1.0).dropna().values)

    if "eight_hour_returns" in res:
        h_eq = scaled_eq.resample("8h").last().dropna()
        sub_res["eight_hour_returns"] = list((h_eq / h_eq.shift(1) - 1.0).dropna().values)

    return sub_res


def evaluate_split(market_data: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp,
                   split_name: str, log=print) -> Dict[str, Any]:
    """Evaluates N1 (14d, 28d, 56d at base fee, 28d at double fee) and N2 (base and double fee)
    plus BTC baseline over [start, end]."""
    log(f"--- Evaluating {split_name} ({start.strftime('%Y-%m')} -> {end.strftime('%Y-%m')}) ---")
    btc_perp = market_data["perp"]["BTCUSDT"]
    btc_base = M.compute_btc_baseline(btc_perp, start, end)

    out = {
        "split": split_name,
        "start": str(start),
        "end": str(end),
        "btc_buy_and_hold": btc_base,
        "N1": {},
        "N2": {},
    }

    # N1 runs: 28d (primary, base & double), 14d & 56d (sensitivity, base)
    for lb in (28, 14, 56):
        fees = (M.FEE_BASE, M.FEE_DOUBLE) if lb == 28 else (M.FEE_BASE,)
        for fee in fees:
            tag = "base" if fee == M.FEE_BASE else "double"
            key = f"{lb}d_{tag}"
            sim = M.simulate_n1(market_data, start_time=start, end_time=end, lookback_days=lb, fee=fee)
            metrics = M.compute_strategy_metrics(sim, btc_base["equity"])
            out["N1"][key] = metrics
            log(f"  N1 ({lb}d, {tag:6s}): CAGR={metrics['cagr_pct']:+6.2f}% | Sharpe={metrics['sharpe_rf4']:+5.2f} "
                f"| MaxDD={metrics['max_dd_pct']:5.2f}% | BTC Corr={metrics['correlation_btc']:+5.2f}")

    # N2 runs: base & double
    for fee in (M.FEE_BASE, M.FEE_DOUBLE):
        tag = "base" if fee == M.FEE_BASE else "double"
        sim = M.simulate_n2(market_data, start_time=start, end_time=end, fee=fee)
        metrics = M.compute_strategy_metrics(sim, btc_base["equity"])
        out["N2"][tag] = metrics
        log(f"  N2 ({tag:6s}):      CAGR={metrics['cagr_pct']:+6.2f}% | Sharpe={metrics['sharpe_rf4']:+5.2f} "
            f"| MaxDD={metrics['max_dd_pct']:5.2f}% | Funding=${metrics['funding_usd']:+7.1f} | Basis=${metrics['basis_pnl_usd']:+7.1f}")

    return out


def evaluate_n1_pass_rules(val_metrics_base: dict, val_metrics_dbl: dict,
                           final_metrics_base: Optional[dict] = None,
                           final_metrics_dbl: Optional[dict] = None,
                           pooled_weekly_ci: Optional[list] = None) -> Dict[str, Any]:
    """Pass rules for N1:
    In BOTH validation and final test:
      - Sharpe > 0.7 (rf 4%)
      - positive CAGR at double costs
      - max drawdown < 30%
      - |correlation to BTC| < 0.3
    And pooled validation+final block-bootstrap 95% CI of mean weekly return has lower bound > 0.
    """
    v_c = {
        "val_sharpe_gt_0.7": bool(val_metrics_base["sharpe_rf4"] > 0.7),
        "val_cagr_positive_double_cost": bool(val_metrics_dbl["cagr_pct"] > 0),
        "val_maxdd_lt_30pct": bool(val_metrics_base["max_dd_pct"] < 30.0),
        "val_abs_btc_corr_lt_0.3": bool(abs(val_metrics_base["correlation_btc"]) < 0.3),
    }

    f_c = {}
    if final_metrics_base and final_metrics_dbl:
        f_c = {
            "final_sharpe_gt_0.7": bool(final_metrics_base["sharpe_rf4"] > 0.7),
            "final_cagr_positive_double_cost": bool(final_metrics_dbl["cagr_pct"] > 0),
            "final_maxdd_lt_30pct": bool(final_metrics_base["max_dd_pct"] < 30.0),
            "final_abs_btc_corr_lt_0.3": bool(abs(final_metrics_base["correlation_btc"]) < 0.3),
        }

    ci_pass = False
    if pooled_weekly_ci and len(pooled_weekly_ci) == 2:
        ci_pass = bool(pooled_weekly_ci[0] > 0.0)

    all_criteria = {**v_c, **f_c}
    if pooled_weekly_ci:
        all_criteria["pooled_weekly_ci_lower_gt_0"] = ci_pass

    passed = all(all_criteria.values())
    return {"criteria": all_criteria, "passed": passed}


def evaluate_n2_pass_rules(val_metrics_base: dict, val_metrics_dbl: dict,
                           final_metrics_base: Optional[dict] = None,
                           final_metrics_dbl: Optional[dict] = None) -> Dict[str, Any]:
    """Pass rules for N2:
    In BOTH validation and final test:
      - Sharpe > 0.7 (rf 4%)
      - positive CAGR at double costs
      - max drawdown < 10%
      - share of PnL from funding exceeds share from basis PnL (funding_usd > basis_pnl_usd).
    """
    v_c = {
        "val_sharpe_gt_0.7": bool(val_metrics_base["sharpe_rf4"] > 0.7),
        "val_cagr_positive_double_cost": bool(val_metrics_dbl["cagr_pct"] > 0),
        "val_maxdd_lt_10pct": bool(val_metrics_base["max_dd_pct"] < 10.0),
        "val_funding_pnl_gt_basis_pnl": bool(val_metrics_base["funding_usd"] > val_metrics_base["basis_pnl_usd"]),
    }

    f_c = {}
    if final_metrics_base and final_metrics_dbl:
        f_c = {
            "final_sharpe_gt_0.7": bool(final_metrics_base["sharpe_rf4"] > 0.7),
            "final_cagr_positive_double_cost": bool(final_metrics_dbl["cagr_pct"] > 0),
            "final_maxdd_lt_10pct": bool(final_metrics_base["max_dd_pct"] < 10.0),
            "final_funding_pnl_gt_basis_pnl": bool(final_metrics_base["funding_usd"] > final_metrics_base["basis_pnl_usd"]),
        }

    all_criteria = {**v_c, **f_c}
    passed = all(all_criteria.values())
    return {"criteria": all_criteria, "passed": passed}


def stage_dev_val(log=print):
    """Executes Development and Validation splits (data <= 2024-12-31)."""
    assert not M.final_unlocked()
    t0 = time.time()
    log("Loading market data for Dev & Val splits (<= 2024-12-31) ...")
    market_data = M.load_all_market_data()

    dev_res = evaluate_split(market_data, M.DEV_START, M.DEV_END, "Development", log)
    val_res = evaluate_split(market_data, M.VAL_START, M.VAL_END, "Validation", log)

    # Validation pass rule check
    n1_val_pass = evaluate_n1_pass_rules(val_res["N1"]["28d_base"], val_res["N1"]["28d_double"])
    n2_val_pass = evaluate_n2_pass_rules(val_res["N2"]["base"], val_res["N2"]["double"])

    out = {
        "generated_utc": str(datetime.now(timezone.utc)),
        "splits": {
            "Development": dev_res,
            "Validation": val_res,
        },
        "validation_verdicts": {
            "N1": n1_val_pass,
            "N2": n2_val_pass,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    with open(DEV_VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Wrote {DEV_VAL_JSON} ({out['elapsed_sec']}s)")


def stage_dryrun(log=print):
    """Exercises the final pipeline on validation data only without unlocking or writing files."""
    assert not M.final_unlocked()
    log("Running dry-run of final pipeline on validation data ...")
    market_data = M.load_all_market_data()
    # Mock final split on 2024
    mock_start = pd.Timestamp("2024-01-01 00:00:00+00:00")
    mock_end = M.VAL_END
    mock_res = evaluate_split(market_data, mock_start, mock_end, "DryRun_MockFinal", log)
    log("Dry-run complete (nothing written to disk).")


def stage_final(log=print):
    """The SINGLE final test run across 2025-01-01 to end of data.
    Pins all data files, hashes dev_val results, records git head, logs timestamp."""
    if os.path.exists(FINAL_LOG) or os.path.exists(FINAL_JSON):
        raise SystemExit("Final test has already been run (log exists). Refusing to run a second time.")
    if not os.path.exists(DEV_VAL_JSON):
        raise SystemExit("Run the dev_val stage first.")

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    except Exception:
        head = "unknown"

    started = str(datetime.now(timezone.utc))
    files = {}
    for s in M.SYMBOLS:
        for k in ("1h", "spot_1h", "funding"):
            fn = f"{s.lower()}_{k}_full.parquet"
            fp = os.path.join(M.CACHE_DIR, fn)
            files[fn] = sha256_file(fp)

    with open(FINAL_LOG, "w", encoding="utf-8") as f:
        json.dump({
            "status": "started",
            "started_utc": started,
            "git_head_at_start": head,
            "dev_val_results_sha256": sha256_file(DEV_VAL_JSON),
            "data_sha256": files,
            "note": "SINGLE final test run. Presence of this file blocks any re-run."
        }, f, indent=2)

    M.unlock_final()
    t0 = time.time()
    log("Loading full market data for Final Test split (2025-01 to end) ...")
    market_data = M.load_all_market_data(truncate_at=None)

    # Determine end of data
    end_time = min(df.index[-1] for df in market_data["perp"].values())
    final_res = evaluate_split(market_data, M.FINAL_START, end_time, "Final_Test", log)

    # Also evaluate full pooled backtest across all splits (DEV_START -> end_time)
    pooled_res = evaluate_split(market_data, M.DEV_START, end_time, "Full_Pooled", log)

    # Compute pooled validation+final block-bootstrap 95% CI of mean weekly return for N1
    val_final_start = M.VAL_START
    n1_val_final_sim = M.simulate_n1(market_data, start_time=val_final_start, end_time=end_time,
                                     lookback_days=28, fee=M.FEE_BASE)
    w_arr = np.array(n1_val_final_sim["weekly_returns"])
    rng = np.random.default_rng(42)
    boot_means = []
    block_size = 4
    n_blocks = max(1, len(w_arr) // block_size)
    for _ in range(2000):
        starts = rng.integers(0, len(w_arr) - block_size + 1, size=n_blocks)
        sampled = np.concatenate([w_arr[s:s + block_size] for s in starts])
        boot_means.append(np.mean(sampled))
    pooled_weekly_ci = [float(np.percentile(boot_means, 2.5) * 10000.0),
                        float(np.percentile(boot_means, 97.5) * 10000.0)]

    # Load validation results to evaluate pass rules
    with open(DEV_VAL_JSON, "r", encoding="utf-8") as f:
        dev_val_data = json.load(f)
    val_res = dev_val_data["splits"]["Validation"]

    n1_verdict = evaluate_n1_pass_rules(val_res["N1"]["28d_base"], val_res["N1"]["28d_double"],
                                        final_res["N1"]["28d_base"], final_res["N1"]["28d_double"],
                                        pooled_weekly_ci)
    n2_verdict = evaluate_n2_pass_rules(val_res["N2"]["base"], val_res["N2"]["double"],
                                        final_res["N2"]["base"], final_res["N2"]["double"])

    finished = str(datetime.now(timezone.utc))
    out = {
        "run_started_utc": started,
        "run_finished_utc": finished,
        "final_start": str(M.FINAL_START),
        "final_end": str(end_time),
        "splits": {
            "Final_Test": final_res,
            "Full_Pooled": pooled_res,
        },
        "pooled_val_final_n1_weekly_ci_bps": pooled_weekly_ci,
        "verdicts": {
            "N1": n1_verdict,
            "N2": n2_verdict,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    with open(FINAL_LOG, "r", encoding="utf-8") as f:
        lg = json.load(f)
    lg["status"] = "finished"
    lg["finished_utc"] = finished
    lg["final_results_sha256"] = sha256_file(FINAL_JSON)
    with open(FINAL_LOG, "w", encoding="utf-8") as f:
        json.dump(lg, f, indent=2)

    log(f"Wrote {FINAL_JSON} and finalized {FINAL_LOG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev_val", "dryrun", "final", "report"], required=True)
    a = ap.parse_args()

    if a.stage == "dev_val":
        stage_dev_val()
    elif a.stage == "dryrun":
        stage_dryrun()
    elif a.stage == "final":
        stage_final()
    else:
        try:
            from research.scripts.market_neutral_report import write_report
        except ImportError:
            from market_neutral_report import write_report
        write_report()
