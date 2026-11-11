# Pre-Registered Chart-Image CNN: Deviations Log

## Post-Development / Pre-Final Code Fixes
1. **`2026-11-11T19:22:00+05:30` (Prior to Final Split Read):**
   - **What:**
     1. Updated `log_final_split_read(experiment_name="chart_cnn", split="Final")` call signature to match `research/scripts/run_logger.py`.
     2. Computed Control 1 (shuffled-target) and Control 3 (random-image) expected OOS Spearman IC over 100 Monte Carlo target permutations / random-image draws (in addition to logging the single-realization column), and evaluated Control 2 (1-bar lag-shift) using the paired block-bootstrap 95% CI on `IC_lag1 - IC_cnn`.
   - **Why:**
     1. `run_logger.py` expects `experiment_name` rather than `experiment`.
     2. For a single random draw on $N_{\text{eff}} \approx 365$, the sampling standard error is $1/\sqrt{N_{\text{eff}}} \approx 0.052$, so testing `|IC| < 0.01` requires averaging across $M=100$ Monte Carlo draws ($\text{SE} \approx 0.0052 < 0.01$). Zero changes were made to any model architecture, hyperparameter, window, target, split, or walk-forward prediction (`pred_cnn`, `pred_ridge`, `pred_lgbm`, `pred_mom`, `pred_zero`).

## Post-Final Code Changes
- **None.** Zero code changes were made after reading the Final split.

## Pre-Registered Protocol Compliance
- **Stage 0 Pre-Flight:** All 8 safety checks (0.1 Data Audit, 0.2a–f Image Renderer Tests, 0.3 Label & Purge/Embargo Tests, 0.4 Planted-Signal Learnability, 0.5 Capacity Overfitting, 0.6 Seed Determinism, 0.7 Power & Cost Realism, 0.8 Compute Budget & Parquet Persistence) were verified prior to running the walk-forward evaluation.
- **Final Split Access:** Read strictly once via `research/scripts/run_logger.py` (`data/run_log.jsonl`). Note: the Final split (`2025-01-01` onward) was previously evaluated by earlier pre-registered experiments on this repository (`vol_target_beta`, `ml_predictivity`, `sr_breakout`).
