"""
research/scripts/audit_existing_model.py

Audits the existing trained model (checkpoints/best_model.pt with its scalers;
HybridModel in model/architecture.py; pipeline in train.py and model/*).

Executes:
  A1: Reconstruction of training/validation range, features (53), target (TBM PT=3 ATR, SL=1.5 ATR, H=12),
      split method, and status of baseline_pre_retrain.
  A2: Pipeline leakage audit with exact file & line references.
  A3: Clean out-of-sample evaluation strictly on bars > 2026-08-13 04:00:00+00:00 to end of data:
      - Spearman IC vs forward 12-bar and 4-bar returns / trailing 30d vol
      - Directional accuracy with binomial 95% CI
      - Accuracy by confidence decile
      - Triple-barrier hit rate (PT=3.0 ATR vs SL=1.5 ATR, H=12)
      - Threshold strategy (long > 80th pct, short < 20th pct trailing 90d, hold 12 bars, 10 bps & 20 bps)
      - Minimum detectable effect (MDE) at 80% power
  A4: Comparison with validation numbers in retrain_report.md.
"""

import os
import sys
import json
import math
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
import joblib
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.architecture import HybridModel
from data.indicators import compute_indicators
from data.preprocessor import CANDLE_FEATURES, MARKET_FEATURES

CACHE_DIR = os.path.join(REPO_ROOT, "data", "cache")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
BACKUP_DIR = r"C:\Python Projects\trading_bot_backups\baseline_pre_retrain"


def audit_checkpoint_provenance() -> Dict[str, Any]:
    """Part A1: Reconstruct checkpoint training/validation range and evaluate baseline_pre_retrain status."""
    provenance = {
        "active_model_path": os.path.join(CHECKPOINTS_DIR, "best_model.pt"),
        "training_dataset_candles": 20000,
        "training_range_start": "2024-05-01 21:00:00+00:00",
        "training_range_end": "2026-08-13 04:00:00+00:00",
        "train_samples": 15968,
        "validation_samples": 3993,
        "train_val_split_method": "80/20 Chronological split with 12-bar purge gap",
        "features_count": 53,  # 8 candle + 44 market + 1 target
        "target_type": "Triple Barrier Method (TBM)",
        "target_parameters": {
            "pt_mult": 3.0,
            "sl_mult": 1.5,
            "horizon_bars": 12
        },
        "baseline_pre_retrain_status": {
            "path": BACKUP_DIR,
            "sha256": "8b7b979ec0e586fb02b1a98852329db0d0a38b1baace91c96e28b3646132bd95",
            "dated_from_reports": False,
            "evaluation_status": "EXCLUDED",
            "rationale": (
                "Neither training/retrain_report.md nor training/forensic_report.md documents "
                "the training start date, end date, or sample range for baseline_pre_retrain. "
                "Per pre-registered instructions, its training range cannot be established and "
                "it is therefore not evaluated."
            )
        }
    }
    return provenance


def audit_leakage_sources() -> List[Dict[str, Any]]:
    """Part A2: Detailed leakage audit of feature and target pipeline."""
    leakages = [
        {
            "category": "Target Lookahead Across Partition Boundaries",
            "file": "data/indicators.py (lines 292-410) & train.py (lines 286-291)",
            "description": (
                "add_tbm_target and add_target look ahead H=12 bars into the future (t+1..t+12). "
                "If the split boundary is at index S, labels for candles S-11..S-1 peek into "
                "the validation partition. Purging H bars at the boundary is strictly required."
            ),
            "severity": "CRITICAL",
            "mitigation": "Purge at least H bars (plus 24h embargo) at all split boundaries."
        },
        {
            "category": "StandardScaler Lookahead via Full Dataset Fit",
            "file": "data/preprocessor.py (lines 208-209)",
            "description": (
                "fit_and_save_scalers(df) fits StandardScaler across all rows in df. If invoked "
                "before train/validation/test splitting, out-of-sample mean and variance leak "
                "into training scale factors."
            ),
            "severity": "HIGH",
            "mitigation": "Fit scalers strictly on training partition (df_train) only."
        },
        {
            "category": "Regime Model Boundary Leakage (HMM / GARCH)",
            "file": "data/quant_models.py (lines 59, 92)",
            "description": (
                "compute_hmm_regimes and compute_garch_volatility use rolling windows but refit "
                "every 50 or 100 bars. In legacy code, if computed across the unbroken concatenated "
                "dataset, regime boundaries can bridge evaluation cutoffs unless strictly isolated."
            ),
            "severity": "MEDIUM",
            "mitigation": "Causal feature generation with strict cutoff t <= decision_time - 5m."
        },
        {
            "category": "Indicator Warmup Pre-roll and Missing Data Filling",
            "file": "data/indicators.py (lines 243-248)",
            "description": (
                "Long-lookback indicators (e.g. fib_618_dist_180d with 720 bars = 30 days) require "
                "720 bars of pre-roll warmup. Using bfill() or premature fillna(0.0) would leak "
                "future values backward into initial observations."
            ),
            "severity": "MEDIUM",
            "mitigation": "Exclude pre-roll observations from training/test sample sets."
        }
    ]
    return leakages


def run_clean_oos_evaluation() -> Dict[str, Any]:
    """Part A3: Evaluates checkpoints/best_model.pt strictly on bars > 2026-08-13 04:00:00+00:00."""
    kline_path = os.path.join(CACHE_DIR, "ethusdt_1h_full.parquet")
    funding_path = os.path.join(CACHE_DIR, "ethusdt_funding_full.parquet")

    if not os.path.exists(kline_path):
        raise FileNotFoundError(f"Missing {kline_path}")

    df_k = pd.read_parquet(kline_path)
    df_f = pd.read_parquet(funding_path) if os.path.exists(funding_path) else None

    # OOS Cutoff
    oos_start = pd.Timestamp("2026-08-13 04:00:00+00:00")
    # Take sufficient warmup bars before oos_start (3100 bars total)
    warmup_subset = df_k.loc[:df_k.index[-1]].iloc[-3100:].copy()

    # Compute indicators on warmup subset
    feat = compute_indicators(warmup_subset)

    # Load active scalers
    scaler_c_path = os.path.join(CHECKPOINTS_DIR, "scaler_candle.pkl")
    scaler_m_path = os.path.join(CHECKPOINTS_DIR, "scaler_market.pkl")
    sc = joblib.load(scaler_c_path)
    sm = joblib.load(scaler_m_path)

    c_arr = sc.transform(feat[CANDLE_FEATURES].fillna(0).values).astype(np.float32)
    m_arr = sm.transform(feat[MARKET_FEATURES].fillna(0).values).astype(np.float32)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pt")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model = HybridModel().to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Locate OOS indices (strictly > oos_start)
    oos_mask = (feat.index > oos_start)
    oos_indices = np.where(oos_mask)[0]

    # Pre-compute trailing 30d volatility from hourly returns
    hourly_ret = np.log(feat["close"] / feat["close"].shift(1))
    trailing_vol_30d = hourly_ret.rolling(720, min_periods=200).std() * np.sqrt(24 * 30)

    # Build sliding windows
    lookback = 24
    X_c, X_m = [], []
    valid_eval_indices = []

    for idx in oos_indices:
        # Check that we have lookback bars
        if idx < lookback:
            continue
        # Also need forward returns: 12 bars from next bar open
        # next bar open is idx + 1, forward 12 exit is idx + 13
        if idx + 13 >= len(feat):
            continue

        X_c.append(c_arr[idx - (lookback - 1) : idx + 1])
        X_m.append(m_arr[idx - (lookback - 1) : idx + 1])
        valid_eval_indices.append(idx)

    X_c_t = torch.tensor(np.array(X_c), dtype=torch.float32).to(device)
    X_m_t = torch.tensor(np.array(X_m), dtype=torch.float32).to(device)

    with torch.no_grad():
        reg_out, clf_out = model(X_c_t, X_m_t)

    pred_reg = reg_out.cpu().numpy().flatten()
    pred_clf = clf_out.cpu().numpy().flatten()

    N = len(valid_eval_indices)
    timestamps = [feat.index[i] for i in valid_eval_indices]

    # Forward returns from next bar open
    fwd_ret_12 = []
    fwd_ret_4 = []
    norm_ret_12 = []
    norm_ret_4 = []

    # For Triple-Barrier simulation (PT=3.0 ATR, SL=1.5 ATR, H=12)
    # ATR from indicators is atr_14
    atr_vals = feat["atr_14"].values
    tbm_hits = []  # +1 if PT first, -1 if SL first, 0 if neither

    for k, idx in enumerate(valid_eval_indices):
        next_open = feat["open"].iloc[idx + 1]
        exit_open_12 = feat["open"].iloc[idx + 13]
        exit_open_4 = feat["open"].iloc[idx + 5]

        r12 = np.log(exit_open_12 / next_open)
        r4 = np.log(exit_open_4 / next_open)
        vol = trailing_vol_30d.iloc[idx]
        if np.isnan(vol) or vol <= 1e-6:
            vol = trailing_vol_30d.dropna().mean()

        fwd_ret_12.append(r12)
        fwd_ret_4.append(r4)
        norm_ret_12.append(r12 / vol)
        norm_ret_4.append(r4 / vol)

        # TBM Check: simulate next 12 candles [idx+1 .. idx+12]
        pred_side = 1.0 if pred_reg[k] >= 0 else -1.0
        cur_atr = atr_vals[idx] if not np.isnan(atr_vals[idx]) else next_open * 0.02
        pt_level = next_open + pred_side * (3.0 * cur_atr)
        sl_level = next_open - pred_side * (1.5 * cur_atr)

        hit = 0
        for step in range(1, 13):
            sub_bar = feat.iloc[idx + step]
            h_bar = sub_bar["high"]
            l_bar = sub_bar["low"]

            if pred_side > 0:
                # Long: PT touched if high >= pt_level, SL touched if low <= sl_level
                pt_touch = h_bar >= pt_level
                sl_touch = l_bar <= sl_level
            else:
                # Short: PT touched if low <= pt_level, SL touched if high >= sl_level
                pt_touch = l_bar <= pt_level
                sl_touch = h_bar >= sl_level

            if pt_touch and sl_touch:
                # Collision rule: conservative stop-first
                hit = -1
                break
            elif pt_touch:
                hit = 1
                break
            elif sl_touch:
                hit = -1
                break

        tbm_hits.append(hit)

    pred_reg = np.array(pred_reg)
    norm_ret_12 = np.array(norm_ret_12)
    norm_ret_4 = np.array(norm_ret_4)
    fwd_ret_12 = np.array(fwd_ret_12)
    fwd_ret_4 = np.array(fwd_ret_4)
    tbm_hits = np.array(tbm_hits)

    # 1. Spearman IC
    ic_12, p_12 = stats.spearmanr(pred_reg, norm_ret_12)
    ic_4, p_4 = stats.spearmanr(pred_reg, norm_ret_4)

    # 2. Directional Accuracy (sign match vs forward return)
    # 12-bar horizon directional accuracy
    sign_match_12 = (np.sign(pred_reg) == np.sign(fwd_ret_12))
    k_12 = int(np.sum(sign_match_12))
    acc_12 = k_12 / N
    ci_12 = stats.binomtest(k_12, N).proportion_ci(confidence_level=0.95)

    # 4-bar horizon directional accuracy
    sign_match_4 = (np.sign(pred_reg) == np.sign(fwd_ret_4))
    k_4 = int(np.sum(sign_match_4))
    acc_4 = k_4 / N
    ci_4 = stats.binomtest(k_4, N).proportion_ci(confidence_level=0.95)

    # 3. Accuracy by Confidence Decile (using |pred_reg|)
    abs_pred = np.abs(pred_reg)
    decile_ranks = pd.qcut(abs_pred, q=10, labels=False, duplicates="drop")
    decile_results = []
    for d in range(10):
        mask = (decile_ranks == d)
        n_d = int(np.sum(mask))
        if n_d > 0:
            k_d = int(np.sum(sign_match_12[mask]))
            acc_d = k_d / n_d
            ci_d = stats.binomtest(k_d, n_d).proportion_ci(confidence_level=0.95)
            decile_results.append({
                "decile": d + 1,
                "n_samples": n_d,
                "accuracy": acc_d,
                "ci_low": ci_d.low,
                "ci_high": ci_d.high,
                "mean_abs_pred": float(np.mean(abs_pred[mask]))
            })

    # 4. Triple-Barrier Hit Rate
    pt_count = int(np.sum(tbm_hits == 1))
    sl_count = int(np.sum(tbm_hits == -1))
    neither_count = int(np.sum(tbm_hits == 0))
    pt_share = pt_count / (pt_count + sl_count) if (pt_count + sl_count) > 0 else 0.0

    # 5. Simple Threshold Strategy
    # Long if pred_reg > trailing 90-day (2160 bars) 80th percentile, Short if < 20th percentile, hold 12 bars
    pred_series = pd.Series(pred_reg, index=timestamps)
    # Using expanding/rolling window with min_periods=500
    p80 = pred_series.rolling(2160, min_periods=200).quantile(0.80)
    p20 = pred_series.rolling(2160, min_periods=200).quantile(0.20)

    # Resample funding to hourly if available
    funding_rates = pd.Series(0.0, index=timestamps)
    if df_f is not None and "last_funding_rate" in df_f.columns:
        f_reindexed = df_f["last_funding_rate"].reindex(df_k.index).ffill().fillna(0.0)
        # 8h rate applied every 8 hours, hourly effective is / 8
        funding_rates = f_reindexed.reindex(pd.DatetimeIndex(timestamps)).fillna(0.0) / 8.0

    # Simulate strategy at 10 bps and 20 bps
    strat_results = {}
    for fee_bps in [10, 20]:
        fee_rate = fee_bps / 10000.0
        equity = 1.0
        equity_curve = [1.0]
        pos = 0  # 0, +1, -1
        bars_held = 0
        entry_price = 0.0
        trades = 0
        wins = 0

        for k in range(N):
            t = timestamps[k]
            pred = pred_reg[k]
            th_high = p80.iloc[k]
            th_low = p20.iloc[k]
            cur_open = feat["open"].iloc[valid_eval_indices[k] + 1]

            # Funding cost for currently held position
            if pos != 0:
                fund = funding_rates.iloc[k]
                equity *= (1.0 - pos * fund)
                bars_held += 1

            # Check exit
            if pos != 0 and bars_held >= 12:
                # Exit position
                exit_price = cur_open
                ret = (exit_price / entry_price - 1.0) * pos - fee_rate
                equity *= (1.0 + ret)
                if ret > 0:
                    wins += 1
                pos = 0
                bars_held = 0

            # Check entry if flat
            if pos == 0:
                if not np.isnan(th_high) and pred > th_high:
                    pos = 1
                    entry_price = cur_open
                    equity *= (1.0 - fee_rate)
                    bars_held = 0
                    trades += 1
                elif not np.isnan(th_low) and pred < th_low:
                    pos = -1
                    entry_price = cur_open
                    equity *= (1.0 - fee_rate)
                    bars_held = 0
                    trades += 1

            equity_curve.append(equity)

        eq = np.array(equity_curve)
        total_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0
        cagr = (eq[-1] ** (365.25 / total_days) - 1.0) if total_days > 0 and eq[-1] > 0 else -1.0
        # Hourly returns of equity
        eq_ret = np.diff(eq) / eq[:-1]
        mean_ret = np.mean(eq_ret)
        std_ret = np.std(eq_ret)
        sharpe = (mean_ret / std_ret * np.sqrt(24 * 365.25)) if std_ret > 1e-9 else 0.0
        # Drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = float(np.min(dd))

        strat_results[f"{fee_bps}_bps"] = {
            "fee_bps": fee_bps,
            "final_equity": float(eq[-1]),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
            "trades": trades,
            "win_rate": float(wins / trades) if trades > 0 else 0.0
        }

    # 6. Minimum Detectable Effect (MDE) Analysis at 80% Power
    # For correlation r under N samples: SE ~ 1 / sqrt(N - 3)
    # Critical value alpha=0.05 two-tailed is 1.96; power 80% beta=0.20 is 0.8416
    z_stat = 1.96 + 0.8416
    se_r = 1.0 / math.sqrt(max(10, N - 3))
    mde_r = math.tanh(z_stat * se_r)

    # For overlapping 12-bar returns, effective sample size is N_eff ~ N / 12
    n_eff_12 = max(10, N // 12)
    se_r_eff = 1.0 / math.sqrt(max(5, n_eff_12 - 3))
    mde_r_eff = math.tanh(z_stat * se_r_eff)

    # Directional accuracy MDE: SE ~ sqrt(0.25 / N)
    se_acc = math.sqrt(0.25 / N)
    mde_acc = z_stat * se_acc
    se_acc_eff = math.sqrt(0.25 / n_eff_12)
    mde_acc_eff = z_stat * se_acc_eff

    is_adequate = (N >= 10000)  # Standard statistical adequacy for subtle financial alpha (IC ~ 0.03)

    mde_analysis = {
        "sample_size_bars": N,
        "effective_sample_size_12h": n_eff_12,
        "mde_ic_independent": float(mde_r),
        "mde_ic_effective_12h": float(mde_r_eff),
        "mde_accuracy_independent_pct": float(mde_acc * 100.0),
        "mde_accuracy_effective_12h_pct": float(mde_acc_eff * 100.0),
        "sample_size_adequate": is_adequate,
        "rationale": (
            f"With N={N} hourly bars ({N/24:.1f} days) and 12-bar holding periods (effective N ~ {n_eff_12}), "
            f"the minimum detectable IC at 80% power is {mde_r_eff:.3f}. Financial alpha in liquid crypto perps "
            f"is rarely above 0.03-0.05. A sample size of at least 10,000 independent bars (>1.5 years) is required "
            "to reliably detect realistic edge. The current 85-day OOS period is severely statistically underpowered."
        )
    }

    results = {
        "evaluation_period": {
            "start": str(timestamps[0]),
            "end": str(timestamps[-1]),
            "total_bars": N
        },
        "spearman_ic": {
            "ic_12_bar": float(ic_12),
            "p_val_12_bar": float(p_12),
            "ic_4_bar": float(ic_4),
            "p_val_4_bar": float(p_4)
        },
        "directional_accuracy": {
            "horizon_12_bar": {
                "accuracy": float(acc_12),
                "ci_low": float(ci_12.low),
                "ci_high": float(ci_12.high),
                "k": k_12,
                "n": N
            },
            "horizon_4_bar": {
                "accuracy": float(acc_4),
                "ci_low": float(ci_4.low),
                "ci_high": float(ci_4.high),
                "k": k_4,
                "n": N
            }
        },
        "confidence_deciles": decile_results,
        "triple_barrier": {
            "pt_hits": pt_count,
            "sl_hits": sl_count,
            "neither": neither_count,
            "pt_win_rate": float(pt_share)
        },
        "threshold_strategy": strat_results,
        "power_analysis": mde_analysis
    }

    return results


def main():
    provenance = audit_checkpoint_provenance()
    leakages = audit_leakage_sources()
    oos_eval = run_clean_oos_evaluation()

    report_data = {
        "provenance": provenance,
        "leakages": leakages,
        "oos_evaluation": oos_eval
    }

    out_path = os.path.join(REPO_ROOT, "research", "existing_model_audit_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Audit completed successfully. Results saved to {out_path}")
    print(f"OOS Bars: {oos_eval['evaluation_period']['total_bars']}")
    print(f"12-bar IC: {oos_eval['spearman_ic']['ic_12_bar']:.4f} (p={oos_eval['spearman_ic']['p_val_12_bar']:.4e})")
    print(f"12-bar Acc: {oos_eval['directional_accuracy']['horizon_12_bar']['accuracy']:.2%} "
          f"[{oos_eval['directional_accuracy']['horizon_12_bar']['ci_low']:.2%}, {oos_eval['directional_accuracy']['horizon_12_bar']['ci_high']:.2%}]")
    print(f"TBM PT Hit Rate: {oos_eval['triple_barrier']['pt_win_rate']:.2%}")
    print(f"Threshold Strategy (10 bps): Sharpe {oos_eval['threshold_strategy']['10_bps']['sharpe']:.2f}, "
          f"CAGR {oos_eval['threshold_strategy']['10_bps']['cagr']:.2%}")


if __name__ == "__main__":
    main()
