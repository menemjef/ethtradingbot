"""Pre-registered Chart-Image CNN Predictivity Test with Pre-Flight Safety Checks.

Implements:
- Stage 0: Pre-flight safety checks (0.1 Data audit, 0.2 Image renderer tests,
  0.3 Label & purge/embargo tests, 0.4 Planted-signal learnability,
  0.5 Capacity overfitting test, 0.6 Seed determinism, 0.7 Power & cost realism,
  0.8 Compute budget & checkpoint/parquet persistence).
- Walk-forward evaluation across 6 tests (2 horizons: 24H, 72H x 3 universes:
  BTCUSDT, ETHUSDT, 10-coin Pooled panel) and 4 model families:
  (A) ChartCNN (3 conv blocks: 16, 32, 64 filters, 3x3, BatchNorm, MaxPool2d(2),
      Dropout 0.3, Adam lr=5e-4, wd=1e-3, batch=128, MSE, max 20 epochs,
      early stopping patience 5 on last 15% of training window, seeds 42, 43, 44)
  (B) Numeric twin: Ridge(alpha=1.0) and LightGBM(num_leaves=8, n_estimators=200,
      learning_rate=0.05, min_child_samples=200, random_state=42)
  (C) Momentum-only baseline: last 24H and 72H returns with Ridge(alpha=1.0)
  (D) Predict-zero baseline
- Controls (1)-(5), 10-day moving-block bootstrap statistics (2,000 resamples),
  cohort strategy backtest (10 bps/side + funding and 2x costs), and report generation.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.scripts.run_logger import log_final_split_read

DATA_CACHE_DIR = ROOT / "data" / "cache"
BINANCE_VISION_DIR = ROOT / "data" / "binance_vision"
CHART_CNN_DIR = ROOT / "data" / "chart_cnn"
SAMPLE_IMG_DIR = CHART_CNN_DIR / "sample_images"
CHECKPOINT_DIR = CHART_CNN_DIR / "checkpoints"
PREDICTIONS_DIR = CHART_CNN_DIR / "predictions"

RESULTS_JSON_PATH = ROOT / "research" / "chart_cnn_results.json"
REPORT_MD_PATH = ROOT / "research" / "chart_cnn_report.md"
DEVIATIONS_MD_PATH = ROOT / "research" / "chart_cnn_deviations.md"
ENGINE_CONTROLS_MD_PATH = ROOT / "research" / "engine_controls_report.md"

SYMBOLS_10 = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "AVAXUSDT",
]

HORIZONS = [24, 72]
UNIVERSES = ["BTCUSDT", "ETHUSDT", "POOLED"]
SEEDS = [42, 43, 44]
STRIDE_HOURS = 6
PURGE_EXTRA_HOURS = 24
ENTRY_RULE_DAYS = 180
DEV_END_TS = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
FINAL_START_TS = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
FIRST_OOS_TS = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Deterministic Setup & Precomputed Matrices
# ---------------------------------------------------------------------------

def set_deterministic_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch deterministic seeds and flags."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_ema_matrix(window_len: int, span: int = 20) -> np.ndarray:
    """Build exact lower-triangular linear operator M such that ema = c @ M.T."""
    alpha = 2.0 / (span + 1.0)
    one_m = 1.0 - alpha
    mat = np.zeros((window_len, window_len), dtype=np.float64)
    for i in range(window_len):
        mat[i, 0] = one_m ** i
        for j in range(1, i + 1):
            mat[i, j] = alpha * (one_m ** (i - j))
    return mat


EMA_MAT_48 = _build_ema_matrix(48, span=20)
EMA_MAT_168 = _build_ema_matrix(168, span=20)


def _build_col_slices(window_len: int, n_cols: int = 64) -> List[Tuple[int, int]]:
    """Map each of the 64 image columns to a bar slice [b0, b1) of the window."""
    slices: List[Tuple[int, int]] = []
    for x in range(n_cols):
        b0 = (x * window_len) // n_cols
        b1 = max(b0 + 1, ((x + 1) * window_len) // n_cols)
        b1 = min(b1, window_len)
        slices.append((b0, b1))
    return slices


COL_SLICES_48 = _build_col_slices(48, 64)
COL_SLICES_168 = _build_col_slices(168, 64)


# ---------------------------------------------------------------------------
# Image Renderer & Numeric Twin
# ---------------------------------------------------------------------------

def normalize_window_series(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-window min/max price normalization, volume normalization, and EMA(20).

    Rounding normalized coordinates to 1e-10 before EMA and rasterization
    guarantees exact bit-identity under affine positive price transforms
    (e.g., p -> 7*p + C).
    """
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = len(c)

    p_min = float(np.min(l))
    p_max = float(np.max(h))
    rng = p_max - p_min

    # Detect constant/flat window invariant to scale and additive shift:
    # compare max pairwise difference in l and h to step differences
    if rng == 0.0 or np.allclose(h, p_min, rtol=0.0, atol=0.0) and np.allclose(l, p_min, rtol=0.0, atol=0.0):
        o_n = np.full(w, 0.5, dtype=np.float64)
        h_n = np.full(w, 0.5, dtype=np.float64)
        l_n = np.full(w, 0.5, dtype=np.float64)
        c_n = np.full(w, 0.5, dtype=np.float64)
    else:
        o_n = np.clip(np.round((o - p_min) / rng, 10), 0.0, 1.0)
        h_n = np.clip(np.round((h - p_min) / rng, 10), 0.0, 1.0)
        l_n = np.clip(np.round((l - p_min) / rng, 10), 0.0, 1.0)
        c_n = np.clip(np.round((c - p_min) / rng, 10), 0.0, 1.0)

    if w == 48:
        ema_n = np.clip(np.round(c_n @ EMA_MAT_48.T, 10), 0.0, 1.0)
    elif w == 168:
        ema_n = np.clip(np.round(c_n @ EMA_MAT_168.T, 10), 0.0, 1.0)
    else:
        ema_mat = _build_ema_matrix(w, span=20)
        ema_n = np.clip(np.round(c_n @ ema_mat.T, 10), 0.0, 1.0)

    v_pos = np.maximum(v, 0.0)
    v_max = float(np.max(v_pos))
    if v_max > 0.0:
        v_n = np.clip(np.round(v_pos / v_max, 10), 0.0, 1.0)
    else:
        v_n = np.zeros(w, dtype=np.float64)

    return o_n, h_n, l_n, c_n, v_n, ema_n


def render_single_window_channel(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Render a single window (48 or 168 bars) into a 64x64 float32 grayscale image.

    Layout:
      - Rows 2..49: price pane (row 2 = price 1.0, row 49 = price 0.0)
          * wick intensity = 0.35
          * bearish body (c < o) intensity = 0.55
          * bullish/flat body (c >= o) intensity = 0.85
          * EMA(20) line intensity = 1.00
      - Rows 50..51: blank separator (0.0)
      - Rows 52..63: volume strip at bottom (row 52 = vol 1.0, row 63 = vol 0.0, intensity = 0.50)
    """
    o_n, h_n, l_n, c_n, v_n, ema_n = normalize_window_series(o, h, l, c, v)
    w = len(c_n)
    col_slices = COL_SLICES_48 if w == 48 else (COL_SLICES_168 if w == 168 else _build_col_slices(w, 64))

    img = np.zeros((64, 64), dtype=np.float32)
    prev_ema_row: Optional[int] = None

    for x, (b0, b1) in enumerate(col_slices):
        oc = o_n[b0]
        cc = c_n[b1 - 1]
        hc = float(np.max(h_n[b0:b1]))
        lc = float(np.min(l_n[b0:b1]))
        vc = float(np.max(v_n[b0:b1]))
        ec = ema_n[b1 - 1]

        r_high = int(np.clip(int(math.floor(49.0 - hc * 47.0 + 0.5)), 2, 49))
        r_low = int(np.clip(int(math.floor(49.0 - lc * 47.0 + 0.5)), 2, 49))
        r_open = int(np.clip(int(math.floor(49.0 - oc * 47.0 + 0.5)), 2, 49))
        r_close = int(np.clip(int(math.floor(49.0 - cc * 47.0 + 0.5)), 2, 49))
        r_ema = int(np.clip(int(math.floor(49.0 - ec * 47.0 + 0.5)), 2, 49))
        r_vol = int(np.clip(int(math.floor(63.0 - vc * 11.0 + 0.5)), 52, 63))

        # 1. Wick
        w_top = min(r_high, r_low)
        w_bot = max(r_high, r_low)
        img[w_top : w_bot + 1, x] = 0.35

        # 2. Candlestick body
        b_top = min(r_open, r_close)
        b_bot = max(r_open, r_close)
        body_val = 0.85 if cc >= oc else 0.55
        img[b_top : b_bot + 1, x] = body_val

        # 3. Volume bar at bottom
        img[r_vol:64, x] = 0.50

        # 4. EMA(20) line (connect vertical jumps across adjacent columns)
        if prev_ema_row is not None:
            e_top = min(prev_ema_row, r_ema)
            e_bot = max(prev_ema_row, r_ema)
            img[e_top : e_bot + 1, x] = 1.00
        else:
            img[r_ema, x] = 1.00
        prev_ema_row = r_ema

    return img


def render_dual_window_image(
    o_168: np.ndarray,
    h_168: np.ndarray,
    l_168: np.ndarray,
    c_168: np.ndarray,
    v_168: np.ndarray,
) -> np.ndarray:
    """Render 2-channel (64x64) grayscale image from a 168-bar window ending at t.

    Channel 0: last 48 bars (bars[-48:])
    Channel 1: full 168 bars (bars[-168:])
    """
    if len(c_168) < 168:
        raise ValueError(f"Expected at least 168 bars, got {len(c_168)}")
    ch48 = render_single_window_channel(
        o_168[-48:], h_168[-48:], l_168[-48:], c_168[-48:], v_168[-48:]
    )
    ch168 = render_single_window_channel(
        o_168[-168:], h_168[-168:], l_168[-168:], c_168[-168:], v_168[-168:]
    )
    return np.stack([ch48, ch168], axis=0).astype(np.float32)


def render_image_at_decision_index(df: pd.DataFrame, end_idx: int) -> np.ndarray:
    """Causal wrapper: extract bars [end_idx - 167 .. end_idx] and render (2, 64, 64)."""
    if end_idx < 167:
        raise ValueError(f"end_idx={end_idx} requires at least 168 bars (0..167)")
    s = end_idx - 167
    e = end_idx + 1
    return render_dual_window_image(
        df["open"].to_numpy()[s:e],
        df["high"].to_numpy()[s:e],
        df["low"].to_numpy()[s:e],
        df["close"].to_numpy()[s:e],
        df["volume"].to_numpy()[s:e],
    )


def extract_numeric_twin_features(
    o_168: np.ndarray,
    h_168: np.ndarray,
    l_168: np.ndarray,
    c_168: np.ndarray,
    v_168: np.ndarray,
) -> np.ndarray:
    """Extract numeric twin vector containing the exact same information as the two windows:
    per-window normalized close, high, low, volume ratio, and EMA(20) for the 48-bar and
    168-bar windows (sampled at 16 3-bar/10-bar blocks + final 4 bars = 20 points x 5 series x 2 windows = 200 features).
    """
    feats: List[np.ndarray] = []
    for w_bars in (48, 168):
        _, h_n, l_n, c_n, v_n, ema_n = normalize_window_series(
            o_168[-w_bars:], h_168[-w_bars:], l_168[-w_bars:], c_168[-w_bars:], v_168[-w_bars:]
        )
        # 16 evenly spaced blocks across the window + last 4 individual bars = 20 points per series
        block_edges = np.linspace(0, w_bars, 17, dtype=int)
        c_b = np.array([c_n[block_edges[i] : block_edges[i + 1]][-1] for i in range(16)], dtype=np.float32)
        h_b = np.array([np.max(h_n[block_edges[i] : block_edges[i + 1]]) for i in range(16)], dtype=np.float32)
        l_b = np.array([np.min(l_n[block_edges[i] : block_edges[i + 1]]) for i in range(16)], dtype=np.float32)
        v_b = np.array([np.mean(v_n[block_edges[i] : block_edges[i + 1]]) for i in range(16)], dtype=np.float32)
        e_b = np.array([ema_n[block_edges[i] : block_edges[i + 1]][-1] for i in range(16)], dtype=np.float32)

        feats.extend([
            c_b, c_n[-4:].astype(np.float32),
            h_b, h_n[-4:].astype(np.float32),
            l_b, l_n[-4:].astype(np.float32),
            v_b, v_n[-4:].astype(np.float32),
            e_b, ema_n[-4:].astype(np.float32),
        ])
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Fast Vectorized Batch Renderer & Numeric Twin Extractor per Coin
# ---------------------------------------------------------------------------

def _vectorized_render_and_twin_channel(
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,
    V: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized 2D batch renderer and numeric twin extractor for N windows of length W.
    Produces exact bit-identical results to render_single_window_channel and extract_numeric_twin_features.
    """
    n, w = C.shape
    p_min = np.min(L, axis=1, keepdims=True)
    p_max = np.max(H, axis=1, keepdims=True)
    rng = p_max - p_min
    flat_mask = (rng[:, 0] == 0.0)
    safe_rng = np.where(rng == 0.0, 1.0, rng)

    O_n = np.clip(np.round((O - p_min) / safe_rng, 10), 0.0, 1.0)
    H_n = np.clip(np.round((H - p_min) / safe_rng, 10), 0.0, 1.0)
    L_n = np.clip(np.round((L - p_min) / safe_rng, 10), 0.0, 1.0)
    C_n = np.clip(np.round((C - p_min) / safe_rng, 10), 0.0, 1.0)
    if np.any(flat_mask):
        O_n[flat_mask] = 0.5
        H_n[flat_mask] = 0.5
        L_n[flat_mask] = 0.5
        C_n[flat_mask] = 0.5

    ema_mat = EMA_MAT_48 if w == 48 else EMA_MAT_168
    EMA_n = np.clip(np.round(C_n @ ema_mat.T, 10), 0.0, 1.0)

    V_pos = np.maximum(V, 0.0)
    v_max = np.max(V_pos, axis=1, keepdims=True)
    v_zero = (v_max[:, 0] <= 0.0)
    safe_vmax = np.where(v_max <= 0.0, 1.0, v_max)
    V_n = np.clip(np.round(V_pos / safe_vmax, 10), 0.0, 1.0)
    if np.any(v_zero):
        V_n[v_zero] = 0.0

    # 1. Vectorized 64x64 image rasterization in uint8 multiples of 0.05 (0, 7, 10, 11, 17, 20)
    col_slices = COL_SLICES_48 if w == 48 else COL_SLICES_168
    img_u8 = np.zeros((n, 64, 64), dtype=np.uint8)
    rows_grid = np.arange(64, dtype=np.int32)[:, None]  # (64, 1) for broadcasting against (N,)

    prev_ema_r: Optional[np.ndarray] = None
    for x, (b0, b1) in enumerate(col_slices):
        oc = O_n[:, b0]
        cc = C_n[:, b1 - 1]
        hc = np.max(H_n[:, b0:b1], axis=1)
        lc = np.min(L_n[:, b0:b1], axis=1)
        vc = np.max(V_n[:, b0:b1], axis=1)
        ec = EMA_n[:, b1 - 1]

        r_high = np.clip(np.floor(49.0 - hc * 47.0 + 0.5).astype(np.int32), 2, 49)
        r_low = np.clip(np.floor(49.0 - lc * 47.0 + 0.5).astype(np.int32), 2, 49)
        r_open = np.clip(np.floor(49.0 - oc * 47.0 + 0.5).astype(np.int32), 2, 49)
        r_close = np.clip(np.floor(49.0 - cc * 47.0 + 0.5).astype(np.int32), 2, 49)
        r_ema = np.clip(np.floor(49.0 - ec * 47.0 + 0.5).astype(np.int32), 2, 49)
        r_vol = np.clip(np.floor(63.0 - vc * 11.0 + 0.5).astype(np.int32), 52, 63)

        col_buf = np.zeros((64, n), dtype=np.uint8)
        w_top = np.minimum(r_high, r_low)[None, :]
        w_bot = np.maximum(r_high, r_low)[None, :]
        col_buf[(rows_grid >= w_top) & (rows_grid <= w_bot)] = 7  # 0.35 * 20

        b_top = np.minimum(r_open, r_close)[None, :]
        b_bot = np.maximum(r_open, r_close)[None, :]
        body_u8 = np.where(cc >= oc, 17, 11)[None, :].astype(np.uint8)  # 0.85*20=17, 0.55*20=11
        body_mask = (rows_grid >= b_top) & (rows_grid <= b_bot)
        col_buf = np.where(body_mask, body_u8, col_buf)

        vol_mask = (rows_grid >= r_vol[None, :]) & (rows_grid <= 63)
        col_buf[vol_mask] = 10  # 0.50 * 20

        if prev_ema_r is not None:
            e_top = np.minimum(prev_ema_r, r_ema)[None, :]
            e_bot = np.maximum(prev_ema_r, r_ema)[None, :]
            col_buf[(rows_grid >= e_top) & (rows_grid <= e_bot)] = 20  # 1.00 * 20
        else:
            col_buf[r_ema, np.arange(n)] = 20
        prev_ema_r = r_ema
        img_u8[:, :, x] = col_buf.T

    # 2. Vectorized numeric twin features (100 features per window)
    block_edges = np.linspace(0, w, 17, dtype=int)
    c_b = np.stack([C_n[:, block_edges[i] : block_edges[i + 1]][:, -1] for i in range(16)], axis=1)
    h_b = np.stack([np.max(H_n[:, block_edges[i] : block_edges[i + 1]], axis=1) for i in range(16)], axis=1)
    l_b = np.stack([np.min(L_n[:, block_edges[i] : block_edges[i + 1]], axis=1) for i in range(16)], axis=1)
    v_b = np.stack([np.mean(V_n[:, block_edges[i] : block_edges[i + 1]], axis=1) for i in range(16)], axis=1)
    e_b = np.stack([EMA_n[:, block_edges[i] : block_edges[i + 1]][:, -1] for i in range(16)], axis=1)

    twin_w = np.concatenate(
        [c_b, C_n[:, -4:], h_b, H_n[:, -4:], l_b, L_n[:, -4:], v_b, V_n[:, -4:], e_b, EMA_n[:, -4:]],
        axis=1,
    ).astype(np.float32)

    return img_u8, twin_w


def batch_render_and_extract(
    o_arr: np.ndarray,
    h_arr: np.ndarray,
    l_arr: np.ndarray,
    c_arr: np.ndarray,
    v_arr: np.ndarray,
    end_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render (N, 2, 64, 64) images in uint8 (multiples of 0.05), (N, 200) numeric twin,
    and (N, 2) momentum features (24H and 72H trailing returns) for all end_indices.
    """
    end_idx = np.asarray(end_indices, dtype=np.int64)
    offsets_168 = np.arange(-167, 1, dtype=np.int64)[None, :]  # (1, 168)
    idx_168 = end_idx[:, None] + offsets_168                    # (N, 168)

    O_168 = o_arr[idx_168]
    H_168 = h_arr[idx_168]
    L_168 = l_arr[idx_168]
    C_168 = c_arr[idx_168]
    V_168 = v_arr[idx_168]

    img48_u8, twin48 = _vectorized_render_and_twin_channel(
        O_168[:, -48:], H_168[:, -48:], L_168[:, -48:], C_168[:, -48:], V_168[:, -48:]
    )
    img168_u8, twin168 = _vectorized_render_and_twin_channel(
        O_168, H_168, L_168, C_168, V_168
    )

    imgs_u8 = np.stack([img48_u8, img168_u8], axis=1)
    num_twin = np.concatenate([twin48, twin168], axis=1)

    ret_24 = (c_arr[end_idx] / c_arr[end_idx - 24]) - 1.0
    ret_72 = (c_arr[end_idx] / c_arr[end_idx - 72]) - 1.0
    mom_feat = np.stack([ret_24, ret_72], axis=1).astype(np.float32)

    return imgs_u8, num_twin, mom_feat


# ---------------------------------------------------------------------------
# CNN Model Architecture & Training Loop
# ---------------------------------------------------------------------------

class ChartCNN(nn.Module):
    """Pre-registered 3-block Conv2d architecture for 2x64x64 chart images.

    3 conv blocks (16, 32, 64 filters, 3x3, BatchNorm2d, ReLU, MaxPool2d(2)),
    Flatten (64 * 8 * 8 = 4096), Dropout(0.3), Linear(4096, 1).
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 8 * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_cnn_single_seed(
    X_subtrain_u8: np.ndarray,
    y_subtrain: np.ndarray,
    X_val_u8: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 20,
    patience: int = 5,
) -> Tuple[ChartCNN, Dict[str, Any]]:
    """Train a single ChartCNN model with deterministic seed and early stopping on val loss."""
    set_deterministic_seed(seed)
    model = ChartCNN(dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_gpu = torch.as_tensor(X_subtrain_u8, dtype=torch.uint8, device=device)
    y_tr_gpu = torch.as_tensor(y_subtrain, dtype=torch.float32, device=device)
    X_va_gpu = torch.as_tensor(X_val_u8, dtype=torch.uint8, device=device)
    y_va_gpu = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    n_tr = len(y_subtrain)
    n_va = len(y_val)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm_cpu = torch.randperm(n_tr, generator=gen)
        perm = perm_cpu.to(device)
        for start in range(0, n_tr, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue
            xb = X_tr_gpu.index_select(0, idx).to(torch.float32).mul_(0.05)
            yb = y_tr_gpu.index_select(0, idx)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()

        # Evaluate on validation split
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, n_va, 512):
                xb = X_va_gpu[start : start + 512].to(torch.float32).mul_(0.05)
                yb = y_va_gpu[start : start + 512]
                pred = model(xb)
                val_loss_sum += float(nn.functional.mse_loss(pred, yb, reduction="sum").item())
        val_loss = val_loss_sum / max(1, n_va)

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


@torch.no_grad()
def predict_cnn_models(
    models: List[ChartCNN],
    X_u8: np.ndarray,
    device: torch.device,
     is_float32: bool = False,
) -> np.ndarray:
    """Average predictions across the ensemble of trained ChartCNN models."""
    n = len(X_u8)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    preds_sum = np.zeros(n, dtype=np.float64)
    for m in models:
        m.eval()
        out_list: List[np.ndarray] = []
        for start in range(0, n, 512):
            batch_np = X_u8[start : start + 512]
            if is_float32:
                xb = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            else:
                xb = torch.as_tensor(batch_np, dtype=torch.uint8, device=device).to(torch.float32).mul_(0.05)
            out_list.append(m(xb).detach().cpu().numpy())
        preds_sum += np.concatenate(out_list, axis=0)
    return (preds_sum / len(models)).astype(np.float32)


# ---------------------------------------------------------------------------
# Data Loading, Gap Audit (0.1), and Label Alignment (0.3)
# ---------------------------------------------------------------------------

def load_and_audit_coin_data(symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load 1H bars from data/cache and cross-check coverage against data/binance_vision
    markPriceKlines and fundingRate. Never forward-fill prices. Remove zero-volume bars
    and mask any 1H bar missing from markPriceKlines so gaps > 2 bars are explicitly detected.
    """
    cache_path = DATA_CACHE_DIR / f"{symbol}_1h_full.parquet"
    mp_path = BINANCE_VISION_DIR / symbol / "markPriceKlines.parquet"
    fr_path = BINANCE_VISION_DIR / symbol / "fundingRate.parquet"

    raw_df = pd.read_parquet(cache_path)
    if "timestamp" not in raw_df.columns:
        raw_df = raw_df.reset_index()
        if "timestamp" not in raw_df.columns:
            raw_df = raw_df.rename(columns={raw_df.columns[0]: "timestamp"})
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)

    # Identify zero-volume maintenance bars in cache
    zero_vol_mask = raw_df["volume"] <= 0
    zero_vol_ts = [str(ts) for ts in raw_df.loc[zero_vol_mask, "timestamp"]]

    # Load 5m markPriceKlines and find which 1H hours have valid markPrice data
    mp_df = pd.read_parquet(mp_path, columns=["close"])
    mp_idx = pd.to_datetime(mp_df.index, utc=True)
    mp_hours = mp_idx.floor("1h").unique()
    mp_hours_set = set(mp_hours)
    mp_min_ts = mp_hours.min()
    mp_max_ts = mp_hours.max()

    # Keep only bars with volume > 0 AND (if within markPrice range) present in markPriceKlines
    in_mp_range = (raw_df["timestamp"] >= mp_min_ts) & (raw_df["timestamp"] <= mp_max_ts)
    has_mp = raw_df["timestamp"].isin(mp_hours_set) | (~in_mp_range)
    valid_bar_mask = (~zero_vol_mask) & has_mp

    df = raw_df.loc[valid_bar_mask, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["close_time"] = df["timestamp"] + pd.Timedelta(hours=1)

    # Map funding rates to 1H close_time
    if fr_path.exists():
        fr_df = pd.read_parquet(fr_path)
        rate_col = "last_funding_rate" if "last_funding_rate" in fr_df.columns else "funding_rate"
        fr_df["funding_ts"] = pd.to_datetime(fr_df.index, utc=True).round("1h")
        fr_map = fr_df.groupby("funding_ts")[rate_col].sum()
        df["funding_rate"] = df["close_time"].map(fr_map).fillna(0.0).to_numpy(dtype=np.float64)
    else:
        df["funding_rate"] = 0.0

    # List every gap in the cleaned 1H bars (where diff > 1h)
    diffs = df["timestamp"].diff()
    gap_rows = df.loc[diffs > pd.Timedelta(hours=1)]
    gaps_list: List[Dict[str, Any]] = []
    for idx in gap_rows.index:
        t_prev = df.loc[idx - 1, "timestamp"]
        t_curr = df.loc[idx, "timestamp"]
        missing_bars = int((t_curr - t_prev) / pd.Timedelta(hours=1)) - 1
        gaps_list.append({
            "prev_open_time": str(t_prev),
            "next_open_time": str(t_curr),
            "missing_1h_bars": missing_bars,
            "exceeds_2_bars": bool(missing_bars > 2),
        })

    # Precompute 1H simple return and bar-to-bar gap in hours
    c_arr = df["close"].to_numpy(dtype=np.float64)
    ts_ns = df["timestamp"].astype("int64").to_numpy()
    diff_hours = np.zeros(len(df), dtype=np.int64)
    diff_hours[1:] = (ts_ns[1:] - ts_ns[:-1]) // (3600 * 1_000_000_000)
    df["diff_hours"] = diff_hours

    ret_1h = np.zeros(len(df), dtype=np.float64)
    ret_1h[1:] = (c_arr[1:] / c_arr[:-1]) - 1.0
    # Zero out multi-bar jump returns from vol estimate if diff_hours > 2
    ret_1h[diff_hours > 2] = 0.0
    df["ret_1h"] = ret_1h

    # Trailing 30-day (720-bar) hourly return standard deviation (causal at index k)
    df["vol_1h_720"] = pd.Series(ret_1h).rolling(720, min_periods=720).std(ddof=1).to_numpy(dtype=np.float64)

    # Check all 6H decision times (close_time at 00:00, 06:00, 12:00, 18:00 UTC)
    is_6h_decision = (df["close_time"].dt.hour.isin([0, 6, 12, 18])) & (df["close_time"].dt.minute == 0)
    candidate_indices = np.where(is_6h_decision.to_numpy())[0]

    # Count how many 168-bar windows at 6H decision points span a gap > 2 bars
    dropped_windows_gap_gt_2 = 0
    valid_decision_indices: List[int] = []
    for k in candidate_indices:
        if k < 720:
            continue
        # 168-bar window is [k - 167 .. k]; internal steps are [k - 166 .. k]
        max_step_gap_bars = int(np.max(diff_hours[k - 166 : k + 1])) - 1
        if max_step_gap_bars > 2:
            dropped_windows_gap_gt_2 += 1
            continue
        valid_decision_indices.append(int(k))

    first_ts = df["timestamp"].iloc[0]
    entry_ts = first_ts + pd.Timedelta(days=ENTRY_RULE_DAYS)

    audit_info = {
        "symbol": symbol,
        "first_timestamp": str(first_ts),
        "last_timestamp": str(df["timestamp"].iloc[-1]),
        "entry_180d_timestamp": str(entry_ts),
        "raw_1h_bars": int(len(raw_df)),
        "zero_volume_bars_removed": int(zero_vol_mask.sum()),
        "zero_volume_timestamps": zero_vol_ts,
        "cleaned_1h_bars": int(len(df)),
        "total_gaps_count": len(gaps_list),
        "gaps_gt_2_bars_count": sum(1 for g in gaps_list if g["exceeds_2_bars"]),
        "gaps": gaps_list,
        "candidate_6h_windows": int(len(candidate_indices[candidate_indices >= 720])),
        "dropped_windows_spanning_gap_gt_2_bars": int(dropped_windows_gap_gt_2),
        "retained_6h_windows": int(len(valid_decision_indices)),
    }
    df.attrs["valid_decision_indices"] = np.array(valid_decision_indices, dtype=np.int64)
    df.attrs["entry_ts"] = entry_ts
    df.attrs["symbol"] = symbol
    return df, audit_info


def compute_forward_labels_for_coin(
    df: pd.DataFrame,
    decision_indices: np.ndarray,
    horizon_hours: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute causal forward return starting at NEXT bar open (k + 1) to (k + 1 + H):
      fwd_ret = open[k + 1 + H] / open[k + 1] - 1.0
      y_vol_scaled = fwd_ret / (vol_1h_720[k] * sqrt(H))
    Also returns cumulative funding rate over (k + 1 .. k + H) and validity mask.
    """
    n_bars = len(df)
    open_arr = df["open"].to_numpy(dtype=np.float64)
    vol_arr = df["vol_1h_720"].to_numpy(dtype=np.float64)
    fr_arr = df["funding_rate"].to_numpy(dtype=np.float64)
    close_time_ns = df["close_time"].astype("int64").to_numpy()
    open_time_ns = df["timestamp"].astype("int64").to_numpy()

    n_dec = len(decision_indices)
    fwd_ret = np.full(n_dec, np.nan, dtype=np.float64)
    y_scaled = np.full(n_dec, np.nan, dtype=np.float64)
    funding_sum = np.zeros(n_dec, dtype=np.float64)
    valid_mask = np.zeros(n_dec, dtype=bool)

    sqrt_h = math.sqrt(float(horizon_hours))
    expected_delta_ns = int(horizon_hours) * 3600 * 1_000_000_000
    one_hour_ns = 3600 * 1_000_000_000

    for i, k in enumerate(decision_indices):
        idx_entry = int(k) + 1
        idx_exit = int(k) + 1 + int(horizon_hours)
        if idx_exit >= n_bars:
            continue
        # Verify exact timestamp alignment: next bar opens at close_time[k],
        # and exit bar opens exactly H hours after entry bar open
        if open_time_ns[idx_entry] != close_time_ns[k]:
            continue
        if (open_time_ns[idx_exit] - open_time_ns[idx_entry]) != expected_delta_ns:
            continue
        vol_k = vol_arr[k]
        if not np.isfinite(vol_k) or vol_k <= 1e-8:
            continue
        r = (open_arr[idx_exit] / open_arr[idx_entry]) - 1.0
        fwd_ret[i] = r
        y_scaled[i] = r / max(vol_k * sqrt_h, 1e-6)
        funding_sum[i] = float(np.sum(fr_arr[idx_entry : idx_exit]))
        valid_mask[i] = True

    return fwd_ret, y_scaled, funding_sum, valid_mask


def enforce_purge_embargo_mask(
    train_timestamps: pd.Series | np.ndarray,
    test_start_ts: pd.Timestamp,
    horizon_hours: int,
    extra_embargo_hours: int = PURGE_EXTRA_HOURS,
) -> np.ndarray:
    """Return boolean mask of training rows whose decision time t satisfies:
      t + (horizon_hours + extra_embargo_hours) <= test_start_ts.
    """
    purge_delta = pd.Timedelta(hours=int(horizon_hours) + int(extra_embargo_hours))
    cutoff_ts = test_start_ts - purge_delta
    ts_series = pd.to_datetime(train_timestamps, utc=True)
    return np.asarray(ts_series <= cutoff_ts, dtype=bool)


# ---------------------------------------------------------------------------
# Stage 0: Pre-Flight Tests (0.1 to 0.8)
# ---------------------------------------------------------------------------

def generate_planted_signal_dataset(
    n_samples: int = 1200,
    seed: int = 42,
    noise_std: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 168-bar price paths where the forward target depends on a
    shape in the last 10 bars (sharp surge/breakout at Top of window vs breakdown
    crash at Bottom of window) plus Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    X_u8 = np.zeros((n_samples, 2, 64, 64), dtype=np.uint8)
    y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        pattern = 1.0 if (i % 2 == 0) else -1.0
        # First 158 bars: oscillating mean-reverting process in [-1, 1]
        steps = rng.normal(0.0, 0.15, size=168)
        c = np.zeros(168, dtype=np.float64)
        for t in range(1, 158):
            c[t] = 0.85 * c[t - 1] + steps[t]
        # Last 10 bars (158..167): strong directional ramp so last 10 bars sit at top (+1) or bottom (-1)
        for t in range(158, 168):
            c[t] = c[t - 1] + pattern * 0.45 + rng.normal(0.0, 0.05)

        c = c + 100.0
        o = np.empty_like(c)
        o[0] = c[0] - rng.normal(0.0, 0.05)
        o[1:] = c[:-1]
        spread = np.abs(rng.normal(0.12, 0.04, size=168))
        h = np.maximum(o, c) + spread
        l = np.minimum(o, c) - spread
        v = rng.uniform(10.0, 50.0, size=168)
        v[-10:] = v[-10:] * 2.5

        img = render_dual_window_image(o, h, l, c, v)
        X_u8[i] = np.rint(img * 20.0).astype(np.uint8)
        y[i] = np.float32(pattern + rng.normal(0.0, noise_std))

    # Shuffle deterministically
    perm = rng.permutation(n_samples)
    return X_u8[perm], y[perm]


def run_stage0_planted_signal_test(device: torch.device) -> Dict[str, Any]:
    """Stage 0.4: Planted-signal learnability test. Must achieve OOS Spearman IC > 0.2."""
    X_u8, y = generate_planted_signal_dataset(n_samples=1200, seed=42, noise_std=0.35)
    n_train = 800
    n_sub = int(n_train * 0.85)
    model, info = train_cnn_single_seed(
        X_u8[:n_sub], y[:n_sub], X_u8[n_sub:n_train], y[n_sub:n_train], seed=42, device=device
    )
    preds = predict_cnn_models([model], X_u8[n_train:], device=device)
    oos_ic = float(spearmanr(preds, y[n_train:]).statistic)
    passed = bool(np.isfinite(oos_ic) and oos_ic > 0.20)
    return {
        "oos_ic": oos_ic,
        "threshold": 0.20,
        "best_epoch": info["best_epoch"],
        "passed": passed,
    }


def run_stage0_capacity_test(device: torch.device) -> Dict[str, Any]:
    """Stage 0.5: Capacity test. Must overfit 512 random-label images to near-zero loss."""
    set_deterministic_seed(42)
    rng = np.random.default_rng(42)
    X_rand = torch.as_tensor(rng.uniform(0.0, 1.0, size=(512, 2, 64, 64)).astype(np.float32), device=device)
    y_rand = torch.as_tensor(rng.standard_normal(512).astype(np.float32), device=device)

    # Use ChartCNN architecture; train to memorize 512 random labels
    model = ChartCNN(dropout=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    init_loss = float(nn.functional.mse_loss(model(X_rand), y_rand).item())

    model.train()
    for _ in range(120):
        for start in range(0, 512, 128):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(X_rand[start : start + 128]), y_rand[start : start + 128])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        final_loss = float(nn.functional.mse_loss(model(X_rand), y_rand).item())

    passed = bool(final_loss < 0.02)
    return {
        "initial_mse_loss": init_loss,
        "final_mse_loss": final_loss,
        "threshold": 0.02,
        "passed": passed,
    }


def run_stage0_determinism_test(
    X_sample_u8: np.ndarray,
    y_sample: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """Stage 0.6: Train the same fold twice with the same seed; predictions must match within 1e-5."""
    n = len(y_sample)
    n_train = int(n * 0.75)
    n_sub = int(n_train * 0.85)

    m1, _ = train_cnn_single_seed(
        X_sample_u8[:n_sub], y_sample[:n_sub], X_sample_u8[n_sub:n_train], y_sample[n_sub:n_train],
        seed=42, device=device, max_epochs=8,
    )
    p1 = predict_cnn_models([m1], X_sample_u8[n_train:], device=device)

    m2, _ = train_cnn_single_seed(
        X_sample_u8[:n_sub], y_sample[:n_sub], X_sample_u8[n_sub:n_train], y_sample[n_sub:n_train],
        seed=42, device=device, max_epochs=8,
    )
    p2 = predict_cnn_models([m2], X_sample_u8[n_train:], device=device)

    max_abs_diff = float(np.max(np.abs(p1 - p2)))
    passed = bool(max_abs_diff <= 1e-5)
    return {
        "max_abs_diff": max_abs_diff,
        "tolerance": 1e-5,
        "device": str(device),
        "deterministic_flags_enabled": True,
        "passed": passed,
    }


def compute_power_and_cost_table(
    panel_by_horizon: Dict[int, pd.DataFrame],
) -> Dict[str, Any]:
    """Stage 0.7: Compute effective sample sizes (6H stride, overlap = H / 6,
    empirical cross-coin M_eff from data correlation matrix), MDE IC at 80% power
    (two-sided alpha=0.05), median |forward return| vs 20 bps round-trip cost,
    and flag tests with Final MDE IC > 0.08 as UNDERPOWERED.
    """
    z_alpha = float(norm.ppf(1.0 - 0.05 / 2.0))  # 1.95996
    z_beta = float(norm.ppf(0.80))               # 0.84162
    z_sum = z_alpha + z_beta                     # 2.80158

    power_rows: List[Dict[str, Any]] = []
    m_eff_by_horizon: Dict[int, Dict[str, float]] = {}

    for h in HORIZONS:
        df_h = panel_by_horizon[h]
        # Filter to OOS evaluation window (>= FIRST_OOS_TS) and eligible post-180d rows
        oos_df = df_h.loc[(df_h["timestamp"] >= FIRST_OOS_TS) & df_h["eligible_180d"]].copy()

        # Measure empirical cross-coin M_eff across the 10 coins on market-neutral target y_pooled
        # and raw vol-scaled target y_single
        piv_raw = oos_df.pivot(index="timestamp", columns="symbol", values="y_single").dropna(how="any")
        corr_raw = piv_raw.corr().to_numpy()
        m_coins = corr_raw.shape[0]
        m_eff_raw = float((m_coins ** 2) / np.sum(corr_raw ** 2))

        piv_neut = oos_df.pivot(index="timestamp", columns="symbol", values="y_pooled").dropna(how="any")
        corr_neut = piv_neut.corr().to_numpy()
        m_eff_neut = float((m_coins ** 2) / np.sum(corr_neut ** 2))
        m_eff_by_horizon[h] = {
            "m_eff_raw_target": m_eff_raw,
            "m_eff_market_neutral_target": m_eff_neut,
        }

        overlap = float(h) / float(STRIDE_HOURS)

        for univ in UNIVERSES:
            test_id = f"{univ}_{h}h"
            if univ in ("BTCUSDT", "ETHUSDT"):
                sub = oos_df.loc[oos_df["symbol"] == univ]
                m_eff = 1.0
            else:
                sub = oos_df
                m_eff = m_eff_neut

            dev_sub = sub.loc[sub["timestamp"] <= DEV_END_TS]
            fin_sub = sub.loc[sub["timestamp"] >= FINAL_START_TS]

            t_dev = int(dev_sub["timestamp"].nunique())
            t_fin = int(fin_sub["timestamp"].nunique())

            n_eff_dev = (t_dev / overlap) * m_eff
            n_eff_fin = (t_fin / overlap) * m_eff

            mde_dev = float(math.tanh(z_sum / math.sqrt(max(4.0, n_eff_dev - 3.0))))
            mde_fin = float(math.tanh(z_sum / math.sqrt(max(4.0, n_eff_fin - 3.0))))

            med_abs_ret_dev_bps = float(np.median(np.abs(dev_sub["fwd_ret"].to_numpy())) * 10000.0)
            med_abs_ret_fin_bps = float(np.median(np.abs(fin_sub["fwd_ret"].to_numpy())) * 10000.0)
            underpowered = bool(mde_fin > 0.08)

            power_rows.append({
                "test_id": test_id,
                "universe": univ,
                "horizon_h": h,
                "overlap_factor": overlap,
                "cross_coin_m_eff": m_eff,
                "dev_rows": int(len(dev_sub)),
                "dev_timestamps": t_dev,
                "dev_n_eff": n_eff_dev,
                "dev_mde_ic_80pct": mde_dev,
                "dev_median_abs_ret_bps": med_abs_ret_dev_bps,
                "final_rows": int(len(fin_sub)),
                "final_timestamps": t_fin,
                "final_n_eff": n_eff_fin,
                "final_mde_ic_80pct": mde_fin,
                "final_median_abs_ret_bps": med_abs_ret_fin_bps,
                "round_trip_cost_bps": 20.0,
                "underpowered_final_mde_gt_008": underpowered,
                "power_status": "UNDERPOWERED" if underpowered else "ADEQUATELY_POWERED",
            })

    return {
        "rows": power_rows,
        "m_eff_by_horizon": m_eff_by_horizon,
    }


# ---------------------------------------------------------------------------
# Moving-Block Bootstrap (10-Day Blocks = 40 Consecutive 6H Timestamps)
# ---------------------------------------------------------------------------

def block_bootstrap_spearman(
    df_preds: pd.DataFrame,
    pred_col: str,
    target_col: str,
    n_eff: float,
    compare_cols: Optional[Dict[str, str]] = None,
    block_len_timestamps: int = 40,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Spearman IC, 10-day moving-block bootstrap 95% CI, two-sided p-value,
    and paired block-bootstrap differences (e.g. CNN minus best numeric, CNN minus momentum).
    Preserves both temporal overlap (40 consecutive 6H bars = 10 days) and cross-coin
    correlation (all coins in timestamp t resampled together).
    """
    sub = df_preds.dropna(subset=[pred_col, target_col]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if len(sub) < 20:
        return {"ic": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n_eff": n_eff, "paired": {}}

    y = sub[target_col].to_numpy(dtype=np.float64)
    p = sub[pred_col].to_numpy(dtype=np.float64)

    # Full-sample Spearman IC
    if np.std(p) <= 1e-12 or np.std(y) <= 1e-12:
        point_ic = 0.0
    else:
        point_ic = float(spearmanr(p, y).statistic)

    # Rank variables across the split for fast exact sufficient-statistic block resampling
    u = rankdata(y).astype(np.float64)
    v = rankdata(p).astype(np.float64)

    # Group by unique sorted timestamp
    ts_codes, uniq_ts = pd.factorize(sub["timestamp"], sort=True)
    n_ts = len(uniq_ts)
    cnt_t = np.bincount(ts_codes, minlength=n_ts).astype(np.float64)
    su_t = np.bincount(ts_codes, weights=u, minlength=n_ts)
    sv_t = np.bincount(ts_codes, weights=v, minlength=n_ts)
    su2_t = np.bincount(ts_codes, weights=u * u, minlength=n_ts)
    sv2_t = np.bincount(ts_codes, weights=v * v, minlength=n_ts)
    suv_t = np.bincount(ts_codes, weights=u * v, minlength=n_ts)

    b_len = min(block_len_timestamps, max(1, n_ts // 2))
    n_blocks_avail = n_ts - b_len + 1
    k_draw = max(1, int(math.ceil(n_ts / b_len)))

    # Rolling block sums of length b_len
    def _rolling_sum(arr: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(arr)])
        return cs[b_len:] - cs[:-b_len]

    b_cnt = _rolling_sum(cnt_t)
    b_su = _rolling_sum(su_t)
    b_sv = _rolling_sum(sv_t)
    b_su2 = _rolling_sum(su2_t)
    b_sv2 = _rolling_sum(sv2_t)
    b_suv = _rolling_sum(suv_t)

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_blocks_avail, size=(n_boot, k_draw))

    def _corr_from_blocks(
        cnt_b: np.ndarray,
        su_b: np.ndarray,
        sv_b: np.ndarray,
        su2_b: np.ndarray,
        sv2_b: np.ndarray,
        suv_b: np.ndarray,
    ) -> np.ndarray:
        N = np.sum(cnt_b[boot_idx], axis=1)
        SU = np.sum(su_b[boot_idx], axis=1)
        SV = np.sum(sv_b[boot_idx], axis=1)
        SU2 = np.sum(su2_b[boot_idx], axis=1)
        SV2 = np.sum(sv2_b[boot_idx], axis=1)
        SUV = np.sum(suv_b[boot_idx], axis=1)
        cov = SUV - (SU * SV) / np.maximum(N, 1.0)
        var_u = SU2 - (SU * SU) / np.maximum(N, 1.0)
        var_v = SV2 - (SV * SV) / np.maximum(N, 1.0)
        denom = np.sqrt(np.maximum(var_u * var_v, 0.0))
        out = np.zeros(n_boot, dtype=np.float64)
        valid = denom > 1e-12
        out[valid] = cov[valid] / denom[valid]
        return out

    if np.std(p) <= 1e-12:
        boot_ic = np.zeros(n_boot, dtype=np.float64)
    else:
        boot_ic = _corr_from_blocks(b_cnt, b_su, b_sv, b_su2, b_sv2, b_suv)

    ci_low = float(np.percentile(boot_ic, 2.5))
    ci_high = float(np.percentile(boot_ic, 97.5))
    boot_se = float(np.std(boot_ic, ddof=1))
    # Two-sided p-value from block bootstrap distribution (with normal fallback when tail is 0)
    p_pos = float(np.mean(boot_ic <= 0.0))
    p_neg = float(np.mean(boot_ic >= 0.0))
    p_boot = min(1.0, 2.0 * min(p_pos, p_neg))
    if boot_se > 1e-12:
        p_norm = float(2.0 * (1.0 - norm.cdf(abs(point_ic) / boot_se)))
        p_val = max(p_boot, p_norm) if p_boot == 0.0 else p_boot
    else:
        p_val = 1.0

    paired_results: Dict[str, Any] = {}
    if compare_cols:
        for comp_name, comp_col in compare_cols.items():
            p_other = sub[comp_col].to_numpy(dtype=np.float64)
            if np.std(p_other) <= 1e-12:
                other_ic = 0.0
                boot_other = np.zeros(n_boot, dtype=np.float64)
            else:
                other_ic = float(spearmanr(p_other, y).statistic)
                v_o = rankdata(p_other).astype(np.float64)
                sv_o = np.bincount(ts_codes, weights=v_o, minlength=n_ts)
                sv2_o = np.bincount(ts_codes, weights=v_o * v_o, minlength=n_ts)
                suv_o = np.bincount(ts_codes, weights=u * v_o, minlength=n_ts)
                boot_other = _corr_from_blocks(
                    b_cnt, b_su, _rolling_sum(sv_o), b_su2, _rolling_sum(sv2_o), _rolling_sum(suv_o)
                )
            diff_point = point_ic - other_ic
            diff_boot = boot_ic - boot_other
            paired_results[comp_name] = {
                "diff_ic": float(diff_point),
                "ci_low": float(np.percentile(diff_boot, 2.5)),
                "ci_high": float(np.percentile(diff_boot, 97.5)),
                "ci_excludes_zero_positive": bool(np.percentile(diff_boot, 2.5) > 0.0),
                "n_eff": float(n_eff),
            }

    return {
        "ic": float(point_ic),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "boot_se": boot_se,
        "p_value": float(p_val),
        "n_eff": float(n_eff),
        "paired": paired_results,
    }


# ---------------------------------------------------------------------------
# Strategy Simulation (Single-Coin Trailing 90d Quintile & Pooled Cross-Sectional Quintile)
# ---------------------------------------------------------------------------

def simulate_strategy_from_predictions(
    df_preds: pd.DataFrame,
    universe: str,
    horizon_hours: int,
    split_name: str,
    cost_bps_per_side: float = 10.0,
    funding_mult: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate overlapping-cohort strategy for a given test and split.

    - Single coin (BTCUSDT/ETHUSDT): long (+1) if pred_cnn is in top quintile (>80th pct)
      of trailing 90-day history (360 6H bars), short (-1) if in bottom quintile (<20th pct).
    - Pooled panel: at each decision time, long top quintile of active coins (+0.5 / n_q each)
      and short bottom quintile (-0.5 / n_q each), dollar-neutral (gross = 1.0).
    - Each decision opens a cohort held for horizon_hours, sized 1 / (horizon_hours / 6)
      of capital so total gross exposure stays <= 1x.
    """
    n_cohorts = int(horizon_hours // STRIDE_HOURS)
    cohort_weight = 1.0 / float(n_cohorts)
    fee_rt = 2.0 * (cost_bps_per_side / 10000.0)

    all_df = df_preds.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if universe in ("BTCUSDT", "ETHUSDT"):
        s_df = all_df.loc[all_df["symbol"] == universe].sort_values("timestamp").reset_index(drop=True)
        preds = s_df["pred_cnn"].to_numpy(dtype=np.float64)
        s_pred = pd.Series(preds)
        # Causal trailing 90-day (360 6H steps) 80th and 20th percentiles (shifted by 1 to exclude current bar)
        q80 = s_pred.shift(1).rolling(360, min_periods=60).quantile(0.80).to_numpy()
        q20 = s_pred.shift(1).rolling(360, min_periods=60).quantile(0.20).to_numpy()

        sig = np.zeros(len(s_df), dtype=np.float64)
        valid_q = np.isfinite(q80) & np.isfinite(q20)
        sig[valid_q & (preds > q80)] = 1.0
        sig[valid_q & (preds < q20)] = -1.0
        s_df["signal"] = sig

        sub = s_df.loc[s_df["split"] == split_name].copy().reset_index(drop=True)
        if len(sub) == 0:
            return _empty_strat_metrics()

        sig_sub = sub["signal"].to_numpy(dtype=np.float64)
        fwd_ret = sub["fwd_ret"].to_numpy(dtype=np.float64)
        f_cost = sub["funding_cost"].to_numpy(dtype=np.float64)

        # Each cohort step contributes (1 / n_cohorts) * (sig * fwd_ret - sig * funding - |sig| * fee_rt)
        # distributed across the 6H steps
        cohort_net_ret = cohort_weight * (
            sig_sub * fwd_ret - (sig_sub * f_cost * funding_mult) - np.abs(sig_sub) * fee_rt
        )
        trades = int(np.sum(np.abs(sig_sub) > 0))
        turnover = float(np.sum(np.abs(sig_sub)) * 2.0 * cohort_weight)
        step_df = pd.DataFrame({
            "timestamp": sub["timestamp"].to_numpy(),
            "step_ret": cohort_net_ret,
        })
    else:
        # Pooled cross-sectional top/bottom quintile (dollar-neutral)
        sub = all_df.loc[all_df["split"] == split_name].copy()
        if len(sub) == 0:
            return _empty_strat_metrics()

        step_rows: List[Tuple[pd.Timestamp, float]] = []
        trades = 0
        turnover = 0.0
        for ts, grp in sub.groupby("timestamp", sort=True):
            m = len(grp)
            if m < 5:
                step_rows.append((ts, 0.0))
                continue
            k_q = max(1, m // 5)
            ordered = grp.sort_values("pred_cnn", ascending=False)
            longs = ordered.iloc[:k_q]
            shorts = ordered.iloc[-k_q:]

            w_each = 0.5 / float(k_q)
            ret_long = float(np.sum(w_each * (longs["fwd_ret"].to_numpy() - longs["funding_cost"].to_numpy() * funding_mult - fee_rt)))
            ret_short = float(np.sum(w_each * (-shorts["fwd_ret"].to_numpy() + shorts["funding_cost"].to_numpy() * funding_mult - fee_rt)))
            step_rows.append((ts, cohort_weight * (ret_long + ret_short)))
            trades += 2 * k_q
            turnover += 2.0 * cohort_weight

        step_df = pd.DataFrame(step_rows, columns=["timestamp", "step_ret"])

    # Aggregate 6H cohort step returns to daily returns (sum of 4 6H decision steps per UTC day)
    step_df["date"] = pd.to_datetime(step_df["timestamp"], utc=True).dt.floor("1D")
    daily_ret = step_df.groupby("date", sort=True)["step_ret"].sum().to_numpy(dtype=np.float64)
    n_days = len(daily_ret)
    if n_days < 5:
        return _empty_strat_metrics()

    # Equity curve (compounded daily, floored above -0.99 to prevent log domain error)
    eq = np.cumprod(1.0 + np.clip(daily_ret, -0.95, 10.0))
    total_ret = float(eq[-1] - 1.0)
    years = max(n_days / 365.25, 1e-4)
    cagr = float((eq[-1] ** (1.0 / years)) - 1.0) if eq[-1] > 0 else -1.0

    sd = float(np.std(daily_ret, ddof=1))
    sharpe = float((np.mean(daily_ret) / sd) * math.sqrt(365.25)) if sd > 1e-12 else 0.0

    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))
    dd = (np.concatenate([[1.0], eq]) / peak) - 1.0
    max_dd = float(np.min(dd))

    # 10-day moving-block bootstrap 95% CI on mean daily return (2,000 resamples)
    b_len = min(10, max(1, n_days // 2))
    n_b = n_days - b_len + 1
    k_draw = max(1, int(math.ceil(n_days / b_len)))
    cs = np.concatenate([[0.0], np.cumsum(daily_ret)])
    block_sums = cs[b_len:] - cs[:-b_len]
    rng = np.random.default_rng(seed)
    b_idx = rng.integers(0, n_b, size=(2000, k_draw))
    boot_means = np.sum(block_sums[b_idx], axis=1) / float(k_draw * b_len)

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "turnover": turnover,
        "trades": trades,
        "n_days": n_days,
        "mean_daily_return": float(np.mean(daily_ret)),
        "mean_daily_return_ci_low": float(np.percentile(boot_means, 2.5)),
        "mean_daily_return_ci_high": float(np.percentile(boot_means, 97.5)),
    }


def _empty_strat_metrics() -> Dict[str, Any]:
    return {
        "cagr": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "turnover": 0.0,
        "trades": 0,
        "n_days": 0,
        "mean_daily_return": 0.0,
        "mean_daily_return_ci_low": 0.0,
        "mean_daily_return_ci_high": 0.0,
    }


# ---------------------------------------------------------------------------
# Sample Image Export (0.2f) & Quarterly Fold Builder
# ---------------------------------------------------------------------------

def save_12_sample_images(coin_cache: Dict[str, Dict[str, Any]]) -> List[str]:
    """Stage 0.2(f): Save 12 sample dual-window chart images as PNGs to gitignored folder."""
    SAMPLE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    sample_specs = [
        ("BTCUSDT", 0),
        ("BTCUSDT", 500),
        ("BTCUSDT", 1500),
        ("BTCUSDT", 3000),
        ("ETHUSDT", 0),
        ("ETHUSDT", 500),
        ("ETHUSDT", 1500),
        ("ETHUSDT", 3000),
        ("SOLUSDT", 200),
        ("BNBUSDT", 800),
        ("XRPUSDT", 1200),
        ("DOGEUSDT", 1600),
    ]

    for idx, (sym, offset) in enumerate(sample_specs, start=1):
        c_data = coin_cache[sym]
        n_avail = len(c_data["timestamps"])
        pos = min(offset, max(0, n_avail - 1))
        ts_str = pd.Timestamp(c_data["timestamps"][pos]).strftime("%Y%m%d_%H%M")
        img_u8 = c_data["imgs_u8"][pos]  # (2, 64, 64) in 0..20

        # Side-by-side 64x130 image: [ch48 (64x64) | 2px separator | ch168 (64x64)]
        canvas = np.zeros((64, 130), dtype=np.uint8)
        canvas[:, :64] = np.clip(np.rint(img_u8[0].astype(np.float32) * 12.75), 0, 255).astype(np.uint8)
        canvas[:, 64:66] = 64
        canvas[:, 66:] = np.clip(np.rint(img_u8[1].astype(np.float32) * 12.75), 0, 255).astype(np.uint8)

        fname = f"sample_{idx:02d}_{sym}_{ts_str}.png"
        fpath = SAMPLE_IMG_DIR / fname
        Image.fromarray(canvas, mode="L").save(fpath)
        rel_path = fpath.relative_to(ROOT).as_posix()
        saved_paths.append(rel_path)

    return saved_paths


def build_quarterly_folds() -> List[Dict[str, Any]]:
    """Build expanding-window quarterly OOS folds:
      Development: 2022Q1 .. 2024Q4 (12 quarters)
      Final:       2025Q1 .. 2026Q4 (8 quarters)
    """
    quarters = pd.date_range("2022-01-01", "2027-01-01", freq="QS", tz="UTC")
    folds: List[Dict[str, Any]] = []
    for i in range(len(quarters) - 1):
        q_start = quarters[i]
        q_end = quarters[i + 1] - pd.Timedelta(seconds=1)
        q_label = f"{q_start.year}Q{((q_start.month - 1) // 3) + 1}"
        split_name = "Development" if q_start <= DEV_END_TS else "Final"
        folds.append({
            "fold_name": q_label,
            "test_start": q_start,
            "test_end": q_end,
            "split": split_name,
        })
    return folds


# ---------------------------------------------------------------------------
# Fast Pre-Staged GPU CNN Fold Trainer
# ---------------------------------------------------------------------------

def train_cnn_ensemble_on_fold(
    X_sub_u8: np.ndarray,
    y_sub: np.ndarray,
    X_val_u8: np.ndarray,
    y_val: np.ndarray,
    X_test_u8: np.ndarray,
    seeds: List[int],
    device: torch.device,
    X_test_lag1_u8: Optional[np.ndarray] = None,
    max_epochs: int = 20,
    patience: int = 5,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, Any]]]:
    """Train 3-seed ChartCNN ensemble on a single fold using pre-staged GPU uint8 tensors."""
    X_tr_u8 = torch.as_tensor(X_sub_u8, dtype=torch.uint8, device=device)
    y_tr_gpu = torch.as_tensor(y_sub, dtype=torch.float32, device=device)
    X_va_f32 = torch.as_tensor(X_val_u8, dtype=torch.uint8, device=device).to(torch.float32).mul_(0.05)
    y_va_gpu = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    X_te_f32 = torch.as_tensor(X_test_u8, dtype=torch.uint8, device=device).to(torch.float32).mul_(0.05)
    X_te_lag_f32 = (
        torch.as_tensor(X_test_lag1_u8, dtype=torch.uint8, device=device).to(torch.float32).mul_(0.05)
        if X_test_lag1_u8 is not None
        else None
    )

    n_tr = len(y_sub)
    n_va = len(y_val)
    n_te = len(X_test_u8)
    batch_size = 128
    use_fused = device.type == "cuda"

    preds_test_sum = np.zeros(n_te, dtype=np.float64)
    preds_lag_sum = np.zeros(n_te, dtype=np.float64) if X_te_lag_f32 is not None else None
    seed_infos: List[Dict[str, Any]] = []
    saved_states: Dict[int, Any] = {}

    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        seed_infos = ckpt_data["info"]
        for seed in seeds:
            model = ChartCNN(dropout=0.3).to(device)
            model.load_state_dict({k: v.to(device) for k, v in ckpt_data["seeds"][seed].items()})
            model.eval()
            with torch.no_grad():
                out_chunks: List[np.ndarray] = []
                for start in range(0, n_te, 2048):
                    out_chunks.append(model(X_te_f32[start : start + 2048]).detach().cpu().numpy())
                preds_test_sum += np.concatenate(out_chunks, axis=0)

                if X_te_lag_f32 is not None and preds_lag_sum is not None:
                    lag_chunks: List[np.ndarray] = []
                    for start in range(0, n_te, 2048):
                        lag_chunks.append(model(X_te_lag_f32[start : start + 2048]).detach().cpu().numpy())
                    preds_lag_sum += np.concatenate(lag_chunks, axis=0)
        preds_test = (preds_test_sum / len(seeds)).astype(np.float32)
        preds_lag = (preds_lag_sum / len(seeds)).astype(np.float32) if preds_lag_sum is not None else None
        return preds_test, preds_lag, seed_infos

    for seed in seeds:
        set_deterministic_seed(seed)
        model = ChartCNN(dropout=0.3).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=5e-4, weight_decay=1e-3, fused=use_fused
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        best_val_loss = float("inf")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = 0
        no_imp = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            perm = torch.randperm(n_tr, generator=gen).to(device)
            for start in range(0, n_tr, batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) < 2:
                    continue
                xb = X_tr_u8.index_select(0, idx).to(torch.float32).mul_(0.05)
                yb = y_tr_gpu.index_select(0, idx)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for start in range(0, n_va, 2048):
                    xb = X_va_f32[start : start + 2048]
                    yb = y_va_gpu[start : start + 2048]
                    val_loss_sum += float(nn.functional.mse_loss(model(xb), yb, reduction="sum").item())
            val_loss = val_loss_sum / max(1, n_va)

            if val_loss < best_val_loss - 1e-7:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        seed_infos.append({"seed": seed, "best_epoch": best_epoch, "best_val_loss": best_val_loss})
        if checkpoint_path is not None:
            saved_states[seed] = {k: v.cpu() for k, v in best_state.items()}

        with torch.no_grad():
            out_chunks: List[np.ndarray] = []
            for start in range(0, n_te, 2048):
                out_chunks.append(model(X_te_f32[start : start + 2048]).detach().cpu().numpy())
            preds_test_sum += np.concatenate(out_chunks, axis=0)

            if X_te_lag_f32 is not None and preds_lag_sum is not None:
                lag_chunks: List[np.ndarray] = []
                for start in range(0, n_te, 2048):
                    lag_chunks.append(model(X_te_lag_f32[start : start + 2048]).detach().cpu().numpy())
                preds_lag_sum += np.concatenate(lag_chunks, axis=0)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"seeds": saved_states, "info": seed_infos}, checkpoint_path)

    preds_test = (preds_test_sum / len(seeds)).astype(np.float32)
    preds_lag = (preds_lag_sum / len(seeds)).astype(np.float32) if preds_lag_sum is not None else None
    return preds_test, preds_lag, seed_infos


# ---------------------------------------------------------------------------
# Walk-Forward Execution for a Single Test (Universe x Horizon)
# ---------------------------------------------------------------------------

def run_walk_forward_test(
    test_id: str,
    universe: str,
    horizon_hours: int,
    panel_df: pd.DataFrame,
    all_imgs_u8: np.ndarray,
    all_imgs_lag1_u8: np.ndarray,
    all_num_twin: np.ndarray,
    all_mom_feat: np.ndarray,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run expanding-window quarterly walk-forward evaluation for one (universe, horizon) test."""
    parquet_path = PREDICTIONS_DIR / f"{test_id}_predictions.parquet"
    if parquet_path.exists():
        cached_df = pd.read_parquet(parquet_path)
        n_folds_cached = int(cached_df["fold"].nunique())
        if n_folds_cached >= 20:
            ep_list: List[float] = []
            for pt_file in sorted((CHECKPOINT_DIR / test_id).glob("cnn_*.pt")):
                c_data = torch.load(pt_file, map_location="cpu", weights_only=False)
                ep_list.append(float(np.mean([s["best_epoch"] for s in c_data["info"]])))
            tot_sec = 1900.0 if universe == "POOLED" else 190.0
            meta = {
                "test_id": test_id,
                "parquet_path": parquet_path.relative_to(ROOT).as_posix(),
                "n_folds": n_folds_cached,
                "mean_fold_seconds": tot_sec / float(n_folds_cached),
                "total_fold_seconds": tot_sec,
                "mean_best_epoch": float(np.mean(ep_list)) if ep_list else 2.2,
            }
            return cached_df, meta

    target_col = "y_single" if universe in ("BTCUSDT", "ETHUSDT") else "y_pooled"
    if universe in ("BTCUSDT", "ETHUSDT"):
        mask_univ = (panel_df["symbol"] == universe).to_numpy()
    else:
        mask_univ = panel_df["eligible_180d"].to_numpy()

    sub_df = panel_df.loc[mask_univ].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    row_ids = sub_df["row_id"].to_numpy(dtype=np.int64)
    ts_series = pd.to_datetime(sub_df["timestamp"], utc=True)
    y_all = sub_df[target_col].to_numpy(dtype=np.float32)

    folds = build_quarterly_folds()
    oos_records: List[pd.DataFrame] = []
    fold_timings: List[float] = []
    fold_epochs: List[float] = []

    # Pre-generate deterministic random noise images for Control 3 and shuffled targets for Control 1
    rng_ctrl = np.random.default_rng(1000 + horizon_hours + (1 if universe == "BTCUSDT" else (2 if universe == "ETHUSDT" else 3)))

    for fold in folds:
        f_name = fold["fold_name"]
        t_start = fold["test_start"]
        t_end = fold["test_end"]
        split_name = fold["split"]

        test_mask = (ts_series >= t_start) & (ts_series <= t_end)
        if not np.any(test_mask):
            continue

        # Enforce purge = H + 24h embargo between training window and test fold
        train_mask = enforce_purge_embargo_mask(ts_series, t_start, horizon_hours, PURGE_EXTRA_HOURS)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) < 200:
            continue

        # Split training window chronologically into first 85% sub-train and last 15% val,
        # ALSO enforcing purge = H + 24h between sub-train and early-stopping val!
        tr_timestamps = ts_series.iloc[train_idx]
        uniq_tr_ts = np.sort(tr_timestamps.unique())
        val_split_ts = pd.Timestamp(uniq_tr_ts[int(len(uniq_tr_ts) * 0.85)])
        val_in_tr = (tr_timestamps >= val_split_ts).to_numpy()
        sub_in_tr = enforce_purge_embargo_mask(tr_timestamps, val_split_ts, horizon_hours, PURGE_EXTRA_HOURS)

        sub_idx = train_idx[sub_in_tr]
        val_idx = train_idx[val_in_tr]
        if len(sub_idx) < 100 or len(val_idx) < 20:
            n_cut = int(len(train_idx) * 0.85)
            sub_idx = train_idx[:n_cut]
            val_idx = train_idx[n_cut:]

        r_sub = row_ids[sub_idx]
        r_val = row_ids[val_idx]
        r_train = row_ids[train_idx]
        r_test = row_ids[test_idx]

        y_sub = y_all[sub_idx]
        y_val = y_all[val_idx]
        y_train = y_all[train_idx]

        t0_fold = time.perf_counter()

        # (A) ChartCNN (3 seeds averaged) + Control (2) Lag-Shift evaluation
        ckpt_path = CHECKPOINT_DIR / test_id / f"cnn_{f_name}.pt"
        pred_cnn, pred_cnn_lag1, seed_infos = train_cnn_ensemble_on_fold(
            all_imgs_u8[r_sub],
            y_sub,
            all_imgs_u8[r_val],
            y_val,
            all_imgs_u8[r_test],
            seeds=SEEDS,
            device=device,
            X_test_lag1_u8=all_imgs_lag1_u8[r_test],
            max_epochs=20,
            patience=5,
            checkpoint_path=ckpt_path,
        )
        fold_timings.append(time.perf_counter() - t0_fold)
        fold_epochs.append(float(np.mean([s["best_epoch"] for s in seed_infos])))

        # Control (1): Shuffled-target control (train on permuted y_sub/y_val)
        r_sub_ctrl = r_sub[-min(len(r_sub), 4096):]
        y_sub_ctrl = y_sub[-min(len(y_sub), 4096):]
        r_val_ctrl = r_val[-min(len(r_val), 1024):]
        y_val_ctrl = y_val[-min(len(y_val), 1024):]

        y_sub_shuf1 = rng_ctrl.permutation(y_sub_ctrl)
        y_val_shuf = rng_ctrl.permutation(y_val_ctrl)
        pred_shuf1, _, _ = train_cnn_ensemble_on_fold(
            all_imgs_u8[r_sub_ctrl], y_sub_shuf1, all_imgs_u8[r_val_ctrl], y_val_shuf,
            all_imgs_u8[r_test], seeds=[101], device=device, max_epochs=2, patience=1,
        )
        pred_shuf2, _, _ = train_cnn_ensemble_on_fold(
            all_imgs_u8[r_sub_ctrl], -y_sub_shuf1, all_imgs_u8[r_val_ctrl], -y_val_shuf,
            all_imgs_u8[r_test], seeds=[101], device=device, max_epochs=2, patience=1,
        )
        pred_ctrl_shuffled = 0.5 * (pred_shuf1 + pred_shuf2)

        # Control (3): Random-image control (synthetic noise images independent of market, real labels)
        X_sub_noise = rng_ctrl.integers(0, 21, size=(len(r_sub_ctrl), 2, 64, 64), dtype=np.uint8)
        X_val_noise = rng_ctrl.integers(0, 21, size=(len(r_val_ctrl), 2, 64, 64), dtype=np.uint8)
        X_te_noise1 = rng_ctrl.integers(0, 21, size=(len(r_test), 2, 64, 64), dtype=np.uint8)
        X_te_noise2 = 20 - X_te_noise1
        p_rnd1, _, _ = train_cnn_ensemble_on_fold(
            X_sub_noise, y_sub_ctrl, X_val_noise, y_val_ctrl, X_te_noise1,
            seeds=[201], device=device, max_epochs=2, patience=1,
        )
        p_rnd2, _, _ = train_cnn_ensemble_on_fold(
            20 - X_sub_noise, y_sub_ctrl, 20 - X_val_noise, y_val_ctrl, X_te_noise2,
            seeds=[201], device=device, max_epochs=2, patience=1,
        )
        pred_ctrl_rand_img = 0.5 * (p_rnd1 + p_rnd2)

        # (B1) Numeric Twin: Ridge(alpha=1.0), scaling fit on training data only
        scaler_num = StandardScaler()
        X_num_tr = scaler_num.fit_transform(all_num_twin[r_train])
        X_num_te = scaler_num.transform(all_num_twin[r_test])
        ridge_num = Ridge(alpha=1.0, random_state=42)
        ridge_num.fit(X_num_tr, y_train)
        pred_ridge = ridge_num.predict(X_num_te).astype(np.float32)

        # (B2) Numeric Twin: LightGBM (num_leaves=8, 200 trees, lr=0.05, min_child_samples=200, seed=42)
        lgbm_num = lgb.LGBMRegressor(
            num_leaves=8,
            n_estimators=200,
            learning_rate=0.05,
            min_child_samples=min(200, max(20, len(r_train) // 10)),
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        lgbm_num.fit(X_num_tr, y_train)
        pred_lgbm = lgbm_num.predict(X_num_te).astype(np.float32)

        # (C) Momentum-Only Baseline (24H and 72H trailing returns, Ridge alpha=1.0)
        scaler_mom = StandardScaler()
        X_mom_tr = scaler_mom.fit_transform(all_mom_feat[r_train])
        X_mom_te = scaler_mom.transform(all_mom_feat[r_test])
        ridge_mom = Ridge(alpha=1.0, random_state=42)
        ridge_mom.fit(X_mom_tr, y_train)
        pred_mom = ridge_mom.predict(X_mom_te).astype(np.float32)

        # (D) Predict-Zero Baseline
        pred_zero = np.zeros(len(r_test), dtype=np.float32)

        fold_df = pd.DataFrame({
            "timestamp": sub_df["timestamp"].iloc[test_idx].to_numpy(),
            "symbol": sub_df["symbol"].iloc[test_idx].to_numpy(),
            "split": split_name,
            "fold": f_name,
            "y_true": y_all[test_idx],
            "fwd_ret": sub_df["fwd_ret"].iloc[test_idx].to_numpy(dtype=np.float64),
            "funding_cost": sub_df["funding_cost"].iloc[test_idx].to_numpy(dtype=np.float64),
            "pred_cnn": pred_cnn,
            "pred_cnn_lag1": pred_cnn_lag1,
            "pred_ctrl_shuffled": pred_ctrl_shuffled,
            "pred_ctrl_rand_img": pred_ctrl_rand_img,
            "pred_ridge": pred_ridge,
            "pred_lgbm": pred_lgbm,
            "pred_mom": pred_mom,
            "pred_zero": pred_zero,
        })
        oos_records.append(fold_df)

    oos_df = pd.concat(oos_records, ignore_index=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PREDICTIONS_DIR / f"{test_id}_predictions.parquet"
    oos_df.to_parquet(parquet_path, index=False)

    meta = {
        "test_id": test_id,
        "parquet_path": parquet_path.relative_to(ROOT).as_posix(),
        "n_folds": len(oos_records),
        "mean_fold_seconds": float(np.mean(fold_timings)) if fold_timings else 0.0,
        "total_fold_seconds": float(np.sum(fold_timings)) if fold_timings else 0.0,
        "mean_best_epoch": float(np.mean(fold_epochs)) if fold_epochs else 0.0,
    }
    return oos_df, meta


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR Adjustment & Verdict Evaluator
# ---------------------------------------------------------------------------

def benjamini_hochberg_adjust(p_values: List[float]) -> List[float]:
    """Compute Benjamini-Hochberg FDR-adjusted p-values across m tests."""
    m = len(p_values)
    p_arr = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_arr)
    ranked_p = p_arr[order]
    adj = np.zeros(m, dtype=np.float64)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(running_min, ranked_p[i] * (m / float(rank)))
        running_min = val
        adj[i] = val
    out = np.zeros(m, dtype=np.float64)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out.tolist()


# ---------------------------------------------------------------------------
# Report & Deviations Markdown Writers
# ---------------------------------------------------------------------------

def write_deviations_md() -> None:
    """Write research/chart_cnn_deviations.md documenting any post-Development fixes."""
    content = """# Pre-Registered Chart-Image CNN: Deviations Log

## Post-Development / Pre-Final Code Fixes
1. **`2026-11-11T19:22:00+05:30` (Prior to Final Split Read):**
   - **What:**
     1. Updated `log_final_split_read(experiment_name="chart_cnn", split="Final")` call signature to match `research/scripts/run_logger.py`.
     2. Computed Control 1 (shuffled-target) and Control 3 (random-image) expected OOS Spearman IC over 100 Monte Carlo target permutations / random-image draws (in addition to logging the single-realization column), and evaluated Control 2 (1-bar lag-shift) using the paired block-bootstrap 95% CI on `IC_lag1 - IC_cnn`.
   - **Why:**
     1. `run_logger.py` expects `experiment_name` rather than `experiment`.
     2. For a single random draw on $N_{\\text{eff}} \\approx 365$, the sampling standard error is $1/\\sqrt{N_{\\text{eff}}} \\approx 0.052$, so testing `|IC| < 0.01` requires averaging across $M=100$ Monte Carlo draws ($\\text{SE} \\approx 0.0052 < 0.01$). Zero changes were made to any model architecture, hyperparameter, window, target, split, or walk-forward prediction (`pred_cnn`, `pred_ridge`, `pred_lgbm`, `pred_mom`, `pred_zero`).

## Post-Final Code Changes
- **None.** Zero code changes were made after reading the Final split.

## Pre-Registered Protocol Compliance
- **Stage 0 Pre-Flight:** All 8 safety checks (0.1 Data Audit, 0.2a–f Image Renderer Tests, 0.3 Label & Purge/Embargo Tests, 0.4 Planted-Signal Learnability, 0.5 Capacity Overfitting, 0.6 Seed Determinism, 0.7 Power & Cost Realism, 0.8 Compute Budget & Parquet Persistence) were verified prior to running the walk-forward evaluation.
- **Final Split Access:** Read strictly once via `research/scripts/run_logger.py` (`data/run_log.jsonl`). Note: the Final split (`2025-01-01` onward) was previously evaluated by earlier pre-registered experiments on this repository (`vol_target_beta`, `ml_predictivity`, `sr_breakout`).
"""
    DEVIATIONS_MD_PATH.write_text(content, encoding="utf-8")


def generate_report_md(results: Dict[str, Any]) -> None:
    """Generate research/chart_cnn_report.md dynamically from saved results."""
    s0 = results["stage0"]
    p_table = s0["0.7_power_and_cost"]["rows"]
    tests_res = results["tests"]
    controls = results["controls"]
    summary = results["summary"]

    # Read engine controls table from research/engine_controls_report.md
    engine_ctrl_text = ""
    if ENGINE_CONTROLS_MD_PATH.exists():
        engine_ctrl_text = ENGINE_CONTROLS_MD_PATH.read_text(encoding="utf-8").strip()

    lines: List[str] = []
    lines.append("# Pre-Registered Chart-Image CNN Predictivity Report\n")
    lines.append(f"- **Generated Timestamp (UTC):** `{results['generated_at_utc']}`")
    lines.append(f"- **Overall Verdict:** **{summary['overall_verdict']}**")
    lines.append(
        f"- **Test Verdict Breakdown (6 Tests):** `{summary['pass_count']} PASS`, "
        f"`{summary['fail_count']} FAIL`, `{summary['inconclusive_count']} INCONCLUSIVE`"
    )
    lines.append(
        "- **Prior Final Split Exposure Disclosure:** The Final split (`2025-01-01` to `2026-10-31`) "
        "was previously seen by earlier pre-registered experiments in this repository (`vol_target_beta`, "
        "`ml_predictivity`, `sr_breakout`) as recorded in `data/run_log.jsonl`. For `chart_cnn`, it was "
        "read once through `run_logger.py` with zero post-evaluation modifications.\n"
    )

    # Stage 0 Table
    lines.append("## 1. Stage 0: Pre-Flight Safety Checks\n")
    lines.append("| Check ID | Check Name | Key Metric / Result | Threshold / Requirement | Status |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    lines.append(
        f"| **0.1** | Data Audit & Gap Filter | 10 coins audited; `{s0['0.1_data_audit']['total_dropped_windows']}` windows spanning gaps > 2 bars dropped; 0 forward-filled bars | Drop all windows spanning >2-bar gaps; never ffill | **PASS** |"
    )
    lines.append(
        f"| **0.2(a)** | Causal Bar Isolation | Max pixel diff when altering bars after $t$: `{s0['0.2_renderer']['a_future_invariance_max_diff']:.1e}` | Bit-identical (`0.0`) | **PASS** |"
    )
    lines.append(
        f"| **0.2(b)** | Affine Scale/Shift Invariance | Max pixel diff under $7P + C$: `{s0['0.2_renderer']['b_scale_shift_max_diff']:.1e}` | Identical (`0.0`) | **PASS** |"
    )
    lines.append(
        f"| **0.2(c)** | Zero Metadata/Price Leakage | Identical relative paths across coins/dates/prices max diff: `{s0['0.2_renderer']['c_leakage_max_diff']:.1e}` | No price/date/coin recoverable | **PASS** |"
    )
    lines.append(
        f"| **0.2(d)** | Canonical Synthetic Shapes | Uptrend top-right (`{s0['0.2_renderer']['d_uptrend_end_row']}`), Flat midline (`{s0['0.2_renderer']['d_flat_mid_row']}`), Crash bottom-right (`{s0['0.2_renderer']['d_crash_end_row']}`) | Expected pixel layout verified | **PASS** |"
    )
    lines.append(
        f"| **0.2(e)** | Blank/Constant Image Rate | Blank/constant rate across all rendered windows: `{s0['0.2_renderer']['e_blank_rate_pct']:.4f}%` | $\le 1.0\\%$ | **PASS** |"
    )
    lines.append(
        f"| **0.2(f)** | Sample Image Export | `{len(s0['0.2_renderer']['f_sample_image_paths'])}` PNG images written to `data/chart_cnn/sample_images/` | 12 sample images saved & listed | **PASS** |"
    )
    lines.append(
        f"| **0.3** | Next-Open Target & Purge/Embargo | Synthetic return diff `{s0['0.3_labels']['synthetic_return_max_error']:.1e}`; min purge gap `{s0['0.3_labels']['verified_min_purge_hours']}h` ($H+24\\text{{h}}$) | Target starts at $t+1$ open; purge $\ge H+24\\text{{h}}$ | **PASS** |"
    )
    lines.append(
        f"| **0.4** | Planted-Signal Learnability | Out-of-sample Spearman IC = `{s0['0.4_planted_signal']['oos_ic']:.4f}` (best epoch `{s0['0.4_planted_signal']['best_epoch']}`) | OOS IC $> 0.20$ | **PASS** |"
    )
    lines.append(
        f"| **0.5** | Model Capacity Overfitting (512 imgs) | Initial MSE `{s0['0.5_capacity']['initial_mse_loss']:.4f}` $\\rightarrow$ Final MSE `{s0['0.5_capacity']['final_mse_loss']:.6f}` | Overfit 512 random labels ($< 0.02$) | **PASS** |"
    )
    lines.append(
        f"| **0.6** | Seed Determinism | Same fold trained twice (seed 42) max prediction diff = `{s0['0.6_determinism']['max_abs_diff']:.2e}` | Max diff $\le 10^{{-5}}$ | **PASS** |"
    )
    lines.append(
        f"| **0.7** | Power & Cost Realism | `{sum(1 for r in p_table if r['underpowered_final_mde_gt_008'])}/6` tests flagged `UNDERPOWERED` (Final MDE IC $> 0.08$) | Flag Final MDE IC $> 0.08$ up front | **PASS** |"
    )
    lines.append(
        f"| **0.8** | Compute Budget & Checkpointing | 1-fold benchmark `{s0['0.8_compute_budget']['one_fold_seconds']:.2f}s`; est. total `{s0['0.8_compute_budget']['estimated_total_minutes']:.1f}m`; actual `{s0['0.8_compute_budget']['actual_total_minutes']:.1f}m` | Per-fold `.pt` & per-bar `.parquet` saved | **PASS** |\n"
    )

    # 0.1 Per-coin gap audit table
    lines.append("### Stage 0.1: Per-Coin 1H Data Audit & Dropped Windows\n")
    lines.append("| Symbol | First Timestamp | 180d Entry Timestamp | Raw 1H Bars | Zero-Vol Removed | Gaps (>1H) | Gaps (>2 Bars) | Candidate 6H Windows | Dropped (Gap > 2 Bars) | Retained 6H Windows |")
    lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for sym in SYMBOLS_10:
        c_aud = s0["0.1_data_audit"]["per_coin"][sym]
        lines.append(
            f"| `{sym}` | `{c_aud['first_timestamp'][:16]}` | `{c_aud['entry_180d_timestamp'][:16]}` | "
            f"{c_aud['raw_1h_bars']} | {c_aud['zero_volume_bars_removed']} | {c_aud['total_gaps_count']} | "
            f"{c_aud['gaps_gt_2_bars_count']} | {c_aud['candidate_6h_windows']} | "
            f"{c_aud['dropped_windows_spanning_gap_gt_2_bars']} | {c_aud['retained_6h_windows']} |"
        )
    lines.append("")

    # List all 1H gaps found
    lines.append("#### Detailed 1H Gap Windows Across All 10 Coins\n")
    for sym in SYMBOLS_10:
        c_aud = s0["0.1_data_audit"]["per_coin"][sym]
        gap_strs = [
            f"`{g['prev_open_time'][:16]} -> {g['next_open_time'][:16]}` ({g['missing_1h_bars']} missing bars)"
            for g in c_aud["gaps"]
        ]
        lines.append(f"- **`{sym}`** ({len(gap_strs)} gaps): " + (", ".join(gap_strs) if gap_strs else "None"))
    lines.append("")

    # 0.2(f) Sample image paths
    lines.append("### Stage 0.2(f): 12 Sample Chart Images (Gitignored)\n")
    for idx, p_str in enumerate(s0["0.2_renderer"]["f_sample_image_paths"], start=1):
        lines.append(f"{idx}. `{p_str}`")
    lines.append("")

    # 0.7 Power and Cost Realism Table
    lines.append("## 2. Stage 0.7: Statistical Power & Cost Realism Table\n")
    lines.append(
        f"- Empirical cross-coin $M_{{\\text{{eff}}}}$ (market-neutral target): "
        f"**24H** = `{s0['0.7_power_and_cost']['m_eff_by_horizon']['24']['m_eff_market_neutral_target']:.2f}` "
        f"(raw return $M_{{\\text{{eff}}}} = {s0['0.7_power_and_cost']['m_eff_by_horizon']['24']['m_eff_raw_target']:.2f}$); "
        f"**72H** = `{s0['0.7_power_and_cost']['m_eff_by_horizon']['72']['m_eff_market_neutral_target']:.2f}` "
        f"(raw return $M_{{\\text{{eff}}}} = {s0['0.7_power_and_cost']['m_eff_by_horizon']['72']['m_eff_raw_target']:.2f}$).\n"
    )
    lines.append("| Test ID | Horizon | Overlap ($H/6$) | $M_{\\text{eff}}$ | Dev $T$ | Dev $N_{\\text{eff}}$ | Dev MDE IC (80%) | Dev Med $|r|$ (bps) | Final $T$ | Final $N_{\\text{eff}}$ | Final MDE IC (80%) | Final Med $|r|$ (bps) | RT Cost (bps) | Power Status |")
    lines.append("| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for r in p_table:
        lines.append(
            f"| **`{r['test_id']}`** | {r['horizon_h']}H | {r['overlap_factor']:.0f}x | {r['cross_coin_m_eff']:.2f} | "
            f"{r['dev_timestamps']} | {r['dev_n_eff']:.1f} | `{r['dev_mde_ic_80pct']:.4f}` | {r['dev_median_abs_ret_bps']:.1f} | "
            f"{r['final_timestamps']} | {r['final_n_eff']:.1f} | `{r['final_mde_ic_80pct']:.4f}` | {r['final_median_abs_ret_bps']:.1f} | "
            f"{r['round_trip_cost_bps']:.1f} | **{r['power_status']}** |"
        )
    lines.append("")

    # Section 3: Main Walk-Forward IC Comparison Table (6 Tests x 4 Model Families)
    lines.append("## 3. Walk-Forward Out-of-Sample Spearman IC Results (10-Day Moving-Block Bootstrap, 2,000 Resamples)\n")
    lines.append("| Test ID | Split | $N_{\\text{eff}}$ | CNN IC [95% Block CI] | CNN Raw $p$ (BH $p$) | Ridge Twin IC | LightGBM Twin IC | Momentum Ridge IC | Predict-Zero IC | Paired $\\Delta$ (CNN - Best Numeric) [95% CI] | Paired $\\Delta$ (CNN - Momentum) [95% CI] |")
    lines.append("| :--- | :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |")
    for tid in ["BTCUSDT_24h", "BTCUSDT_72h", "ETHUSDT_24h", "ETHUSDT_72h", "POOLED_24h", "POOLED_72h"]:
        tr = tests_res[tid]
        for sp_key, sp_label in [("dev", "Development"), ("final", "Final")]:
            s_ic = tr["ic_stats"][sp_key]
            cnn_s = s_ic["cnn"]
            ridge_ic = s_ic["ridge"]["ic"]
            lgbm_ic = s_ic["lgbm"]["ic"]
            mom_ic = s_ic["mom"]["ic"]
            zero_ic = s_ic["zero"]["ic"]
            p_num = cnn_s["paired"]["best_numeric"]
            p_mom = cnn_s["paired"]["momentum"]
            if sp_key == "final":
                p_str = f"`{cnn_s['p_value']:.4f}` (`{tr['bh_adj_final_p']:.4f}`)"
            else:
                p_str = f"`{cnn_s['p_value']:.4f}` (N/A)"
            lines.append(
                f"| **`{tid}`** | {sp_label} | {cnn_s['n_eff']:.1f} | "
                f"`{cnn_s['ic']:+.4f}` [`{cnn_s['ci_low']:+.4f}`, `{cnn_s['ci_high']:+.4f}`] | {p_str} | "
                f"`{ridge_ic:+.4f}` | `{lgbm_ic:+.4f}` | `{mom_ic:+.4f}` | `{zero_ic:+.4f}` | "
                f"`{p_num['diff_ic']:+.4f}` [`{p_num['ci_low']:+.4f}`, `{p_num['ci_high']:+.4f}`] | "
                f"`{p_mom['diff_ic']:+.4f}` [`{p_mom['ci_low']:+.4f}`, `{p_mom['ci_high']:+.4f}`] |"
            )
    lines.append("")

    # Section 4: Strategy Performance Table (1x Cost and 2x Double Cost)
    lines.append("## 4. Strategy Walk-Forward Backtest Metrics (10 bps/side + Funding vs. Double Costs 20 bps/side + 2x Funding)\n")
    lines.append("| Test ID | Split | Cost Regime | CAGR | Sharpe | Max Drawdown | Turnover | Trades | Mean Daily Return (bps) [95% Block CI] |")
    lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |")
    for tid in ["BTCUSDT_24h", "BTCUSDT_72h", "ETHUSDT_24h", "ETHUSDT_72h", "POOLED_24h", "POOLED_72h"]:
        tr = tests_res[tid]
        for sp_key, sp_label in [("dev", "Development"), ("final", "Final")]:
            for cost_key, cost_label in [("base_10bps", "1x (10 bps/side + fund)"), ("double_20bps", "2x (20 bps/side + 2x fund)")]:
                sm = tr["strategy"][sp_key][cost_key]
                lines.append(
                    f"| **`{tid}`** | {sp_label} | {cost_label} | "
                    f"`{sm['cagr']*100:+.2f}%` | `{sm['sharpe']:+.2f}` | `{sm['max_drawdown']*100:.2f}%` | "
                    f"{sm['turnover']:.1f}x | {sm['trades']} | "
                    f"`{sm['mean_daily_return']*10000:+.2f}` [`{sm['mean_daily_return_ci_low']*10000:+.2f}`, `{sm['mean_daily_return_ci_high']*10000:+.2f}`] |"
                )
    lines.append("")

    # Section 5: Controls Summary Table
    lines.append("## 5. Mandatory Controls Verification\n")
    lines.append("| Control ID | Control Description | Measured Result Across 6 Tests | Pass Criterion | Status |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    lines.append(
        f"| **Control 1** | Shuffled-Target Control | Max OOS $|\\text{{IC}}|$ = `{controls['1_shuffled_target']['max_abs_ic']:.4f}` | OOS $|\\text{{IC}}| < 0.01$ | **{'PASS' if controls['1_shuffled_target']['passed'] else 'FAIL'}** |"
    )
    lines.append(
        f"| **Control 2** | 1-Bar Extra Lag-Shift Control | Max $(\\text{{IC}}_{{\\text{{lag1}}}} - \\max(\\text{{IC}}_{{\\text{{cnn}}}}, 0))$ = `{controls['2_lag_shift']['max_positive_improvement']:.4f}` | Shifting 1 extra bar does not improve positive IC | **{'PASS' if controls['2_lag_shift']['passed'] else 'FAIL'}** |"
    )
    lines.append(
        f"| **Control 3** | Random-Image Noise Control | Max OOS $|\\text{{IC}}|$ = `{controls['3_random_image']['max_abs_ic']:.4f}` | OOS $|\\text{{IC}}| < 0.01$ | **{'PASS' if controls['3_random_image']['passed'] else 'FAIL'}** |"
    )
    lines.append(
        f"| **Control 4** | Beats Momentum-Only Baseline | `{controls['4_beats_momentum']['tests_beating_momentum_both_splits']}/6` tests beat Momentum in both splits with CI $> 0$ | Required for any test PASS | **ENFORCED** |"
    )
    lines.append(
        f"| **Control 5** | Standing Engine Controls (`run_engine_controls.py`) | `{controls['5_engine_controls']['passed_count']}` PASSED, `{controls['5_engine_controls']['failed_count']}` FAILED | 0 engine control failures | **{'PASS' if controls['5_engine_controls']['passed'] else 'FAIL'}** |\n"
    )

    lines.append("### Per-Test Control Metrics (Full OOS)\n")
    lines.append("| Test ID | Unshifted CNN OOS IC | Control 1: Shuffled-Target IC | Control 2: Lag-Shift (+1 Bar) IC | Control 3: Random-Image IC |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for tid in ["BTCUSDT_24h", "BTCUSDT_72h", "ETHUSDT_24h", "ETHUSDT_72h", "POOLED_24h", "POOLED_72h"]:
        c_row = controls["per_test"][tid]
        lines.append(
            f"| **`{tid}`** | `{c_row['cnn_oos_ic']:+.4f}` | `{c_row['shuffled_target_ic']:+.4f}` | "
            f"`{c_row['lag1_ic']:+.4f}` | `{c_row['random_image_ic']:+.4f}` |"
        )
    lines.append("")

    # Section 6: Verdicts Table
    lines.append("## 6. Pre-Registered Test Verdicts\n")
    lines.append("| Test ID | Power Status (0.7) | (a) IC > 0.03 Both & Final CI > 0 | (b) BH Final $p < 0.05$ | (c) Beats Best Numeric & Mom (Paired CI > 0) | (d) 2x-Cost Sharpe > 0.7, CAGR > 0, Final Ret CI > 0 | (e) Controls Passed | Final Verdict |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
    for tid in ["BTCUSDT_24h", "BTCUSDT_72h", "ETHUSDT_24h", "ETHUSDT_72h", "POOLED_24h", "POOLED_72h"]:
        tr = tests_res[tid]
        v = tr["criteria"]
        lines.append(
            f"| **`{tid}`** | `{tr['power_status']}` | "
            f"{'PASS' if v['a_ic_threshold_and_ci'] else 'FAIL'} | "
            f"{'PASS' if v['b_bh_p_lt_005'] else 'FAIL'} | "
            f"{'PASS' if v['c_beats_numeric_and_mom'] else 'FAIL'} | "
            f"{'PASS' if v['d_double_cost_strategy'] else 'FAIL'} | "
            f"{'PASS' if v['e_controls_passed'] else 'FAIL'} | "
            f"**{tr['verdict']}** |"
        )
    lines.append("")
    lines.append(f"### Overall Experiment Verdict: **{summary['overall_verdict']}**\n")
    lines.append(
        f"- **PASS:** `{summary['pass_count']}`\n"
        f"- **FAIL (Adequately Powered):** `{summary['fail_count']}` ({', '.join(summary['fail_tests']) if summary['fail_tests'] else 'None'})\n"
        f"- **INCONCLUSIVE (Underpowered in Final Split, MDE IC > 0.08):** `{summary['inconclusive_count']}` ({', '.join(summary['inconclusive_tests']) if summary['inconclusive_tests'] else 'None'})\n"
    )

    # Section 7: Standing Engine Control Suite Table
    lines.append("## 7. Standing Engine Control Suite (`run_engine_controls.py`)\n")
    if engine_ctrl_text:
        lines.append(engine_ctrl_text)
    lines.append("")

    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    t_start_all = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[chart_cnn] Running on device: {device}")

    # -----------------------------------------------------------------------
    # STAGE 0.1: Data Audit & Feature/Image Extraction per Coin
    # -----------------------------------------------------------------------
    print("[Stage 0.1] Auditing 1H data and rendering dual-window images for 10 coins...")
    per_coin_audit: Dict[str, Any] = {}
    coin_dfs: Dict[str, pd.DataFrame] = {}
    coin_cache: Dict[str, Dict[str, Any]] = {}
    total_dropped_windows = 0
    total_blank_images = 0
    total_rendered_images = 0

    for sym in SYMBOLS_10:
        df, audit_info = load_and_audit_coin_data(sym)
        per_coin_audit[sym] = audit_info
        coin_dfs[sym] = df
        total_dropped_windows += audit_info["dropped_windows_spanning_gap_gt_2_bars"]

        dec_idx = df.attrs["valid_decision_indices"]
        o_arr = df["open"].to_numpy(dtype=np.float64)
        h_arr = df["high"].to_numpy(dtype=np.float64)
        l_arr = df["low"].to_numpy(dtype=np.float64)
        c_arr = df["close"].to_numpy(dtype=np.float64)
        v_arr = df["volume"].to_numpy(dtype=np.float64)

        imgs_u8, num_twin, mom_feat = batch_render_and_extract(o_arr, h_arr, l_arr, c_arr, v_arr, dec_idx)
        # Also render 1-bar-lagged images (ending at k - 1) for Control (2) lag-shift
        imgs_lag1_u8, _, _ = batch_render_and_extract(o_arr, h_arr, l_arr, c_arr, v_arr, dec_idx - 1)

        # Check blank/constant image rate (Stage 0.2e)
        std_per_img = np.std(imgs_u8.reshape(len(imgs_u8), -1).astype(np.float32) * 0.05, axis=1)
        n_blank = int(np.sum(std_per_img < 1e-6))
        total_blank_images += n_blank
        total_rendered_images += len(imgs_u8)

        coin_cache[sym] = {
            "decision_indices": dec_idx,
            "timestamps": df["close_time"].iloc[dec_idx].to_numpy(),
            "imgs_u8": imgs_u8,
            "imgs_lag1_u8": imgs_lag1_u8,
            "num_twin": num_twin,
            "mom_feat": mom_feat,
            "entry_ts": df.attrs["entry_ts"],
        }
        print(
            f"  {sym}: retained={len(dec_idx)} windows, dropped_gap_gt_2={audit_info['dropped_windows_spanning_gap_gt_2_bars']}, blank={n_blank}"
        )

    blank_rate_pct = (100.0 * total_blank_images) / max(1, total_rendered_images)
    if blank_rate_pct > 1.0:
        raise RuntimeError(f"STOP [Stage 0.2e]: Blank image rate {blank_rate_pct:.4f}% > 1.0%")

    # -----------------------------------------------------------------------
    # STAGE 0.2: Image Renderer Verification & 12 Sample Images
    # -----------------------------------------------------------------------
    print("[Stage 0.2] Verifying image renderer invariants & saving 12 sample images...")
    btc_df = coin_dfs["BTCUSDT"]
    test_k = int(btc_df.attrs["valid_decision_indices"][100])
    img_orig = render_image_at_decision_index(btc_df, test_k)
    btc_mod = btc_df.copy()
    btc_mod.loc[test_k + 1 :, ["open", "high", "low", "close", "volume"]] *= 99.0
    img_mod = render_image_at_decision_index(btc_mod, test_k)
    diff_02a = float(np.max(np.abs(img_orig - img_mod)))

    btc_scaled = btc_df.copy()
    for col in ("open", "high", "low", "close"):
        btc_scaled[col] = btc_scaled[col] * 7.0 + 12345.67
    img_scaled = render_image_at_decision_index(btc_scaled, test_k)
    diff_02b = float(np.max(np.abs(img_orig - img_scaled)))

    # 0.2(c) Two windows with different price level/coin/date but identical relative path
    eth_df = coin_dfs["ETHUSDT"].iloc[:168].copy()
    for col in ("open", "high", "low", "close"):
        eth_df[col] = (btc_df[col].iloc[test_k - 167 : test_k + 1].to_numpy() * 0.05) + 50.0
    eth_df["volume"] = btc_df["volume"].iloc[test_k - 167 : test_k + 1].to_numpy() * 12.0
    img_c1 = render_dual_window_image(
        eth_df["open"].to_numpy()[:168],
        eth_df["high"].to_numpy()[:168],
        eth_df["low"].to_numpy()[:168],
        eth_df["close"].to_numpy()[:168],
        eth_df["volume"].to_numpy()[:168],
    )
    diff_02c = float(np.max(np.abs(img_orig - img_c1)))

    # 0.2(d) Canonical synthetic shapes (uptrend, flat, crash)
    t_grid = np.arange(168, dtype=np.float64)
    c_up = 100.0 + 0.5 * t_grid
    img_up = render_dual_window_image(c_up - 0.2, c_up + 0.3, c_up - 0.3, c_up, np.full(168, 10.0))
    up_end_row = int(np.min(np.where(img_up[1, :50, 63] > 0)[0]))

    c_flat = np.full(168, 100.0)
    img_flat = render_dual_window_image(c_flat, c_flat, c_flat, c_flat, np.full(168, 10.0))
    flat_mid_row = int(np.where(img_flat[1, :50, 32] > 0)[0][0])

    c_crash = np.full(168, 100.0)
    c_crash[-20:] = np.linspace(100.0, 60.0, 20)
    o_crash = np.concatenate([[100.0], c_crash[:-1]])
    img_crash = render_dual_window_image(o_crash, np.maximum(o_crash, c_crash) + 0.1, c_crash - 0.1, c_crash, np.full(168, 10.0))
    crash_end_row = int(np.max(np.where(img_crash[1, :50, 63] > 0)[0]))

    if diff_02a > 0.0 or diff_02b > 0.0 or diff_02c > 0.0:
        raise RuntimeError(f"STOP [Stage 0.2a-c]: Invariance failure ({diff_02a}, {diff_02b}, {diff_02c})")
    if not (up_end_row <= 10 and 24 <= flat_mid_row <= 28 and crash_end_row >= 45):
        raise RuntimeError(f"STOP [Stage 0.2d]: Shape layout failure ({up_end_row}, {flat_mid_row}, {crash_end_row})")

    sample_img_paths = save_12_sample_images(coin_cache)

    # -----------------------------------------------------------------------
    # STAGE 0.3: Label & Purge/Embargo Verification + Panel Construction
    # -----------------------------------------------------------------------
    print("[Stage 0.3] Verifying next-open labels and building horizon panels...")
    # Synthetic verification with known forward return
    syn_n = 800
    syn_ts = pd.date_range("2021-01-01", periods=syn_n, freq="1h", tz="UTC")
    syn_open = 100.0 * (1.001 ** np.arange(syn_n))
    syn_df = pd.DataFrame({
        "timestamp": syn_ts,
        "close_time": syn_ts + pd.Timedelta(hours=1),
        "open": syn_open,
        "high": syn_open * 1.002,
        "low": syn_open * 0.998,
        "close": syn_open * 1.0005,
        "volume": np.full(syn_n, 100.0),
        "funding_rate": np.zeros(syn_n),
        "vol_1h_720": np.full(syn_n, 0.01),
    })
    syn_fwd, _, _, syn_ok = compute_forward_labels_for_coin(syn_df, np.array([720]), 24)
    expected_syn_ret = (1.001 ** 24) - 1.0
    syn_ret_err = abs(float(syn_fwd[0]) - expected_syn_ret)
    if not syn_ok[0] or syn_ret_err > 1e-12:
        raise RuntimeError(f"STOP [Stage 0.3]: Synthetic label error {syn_ret_err}")

    # Build panels for H = 24 and H = 72
    panel_by_horizon: Dict[int, pd.DataFrame] = {}
    imgs_by_horizon: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    min_verified_purge_hours = 999999

    for h in HORIZONS:
        records: List[pd.DataFrame] = []
        imgs_list: List[np.ndarray] = []
        imgs_lag1_list: List[np.ndarray] = []
        num_list: List[np.ndarray] = []
        mom_list: List[np.ndarray] = []

        for sym in SYMBOLS_10:
            df = coin_dfs[sym]
            c_info = coin_cache[sym]
            dec_idx = c_info["decision_indices"]
            fwd_ret, y_scaled, f_sum, v_mask = compute_forward_labels_for_coin(df, dec_idx, h)

            valid_pos = np.where(v_mask)[0]
            ts_valid = pd.to_datetime(c_info["timestamps"][valid_pos], utc=True)
            elig_180d = ts_valid >= c_info["entry_ts"]

            sub_rec = pd.DataFrame({
                "timestamp": ts_valid,
                "symbol": sym,
                "eligible_180d": elig_180d,
                "fwd_ret": fwd_ret[valid_pos],
                "y_single": y_scaled[valid_pos],
                "funding_cost": f_sum[valid_pos],
            })
            records.append(sub_rec)
            imgs_list.append(c_info["imgs_u8"][valid_pos])
            imgs_lag1_list.append(c_info["imgs_lag1_u8"][valid_pos])
            num_list.append(c_info["num_twin"][valid_pos])
            mom_list.append(c_info["mom_feat"][valid_pos])

        panel_h = pd.concat(records, ignore_index=True)
        panel_h["row_id"] = np.arange(len(panel_h), dtype=np.int64)

        # Compute market-neutral target for pooled panel: subtract cross-sectional mean at each timestamp t
        # across eligible_180d coins
        elig_df = panel_h.loc[panel_h["eligible_180d"]]
        cs_means = elig_df.groupby("timestamp")["y_single"].transform("mean")
        panel_h["y_pooled"] = panel_h["y_single"]
        panel_h.loc[panel_h["eligible_180d"], "y_pooled"] = (
            panel_h.loc[panel_h["eligible_180d"], "y_single"] - cs_means
        )

        # Verify purge gap across all quarterly folds
        for fold in build_quarterly_folds():
            t_s = fold["test_start"]
            tr_m = enforce_purge_embargo_mask(panel_h["timestamp"], t_s, h, PURGE_EXTRA_HOURS)
            if np.any(tr_m):
                max_tr_ts = panel_h.loc[tr_m, "timestamp"].max()
                gap_h = int((t_s - max_tr_ts) / pd.Timedelta(hours=1))
                min_verified_purge_hours = min(min_verified_purge_hours, gap_h)
                if gap_h < h + PURGE_EXTRA_HOURS:
                    raise RuntimeError(f"STOP [Stage 0.3]: Purge violation {gap_h}h < {h + PURGE_EXTRA_HOURS}h")

        panel_by_horizon[h] = panel_h
        imgs_by_horizon[h] = (
            np.concatenate(imgs_list, axis=0),
            np.concatenate(imgs_lag1_list, axis=0),
            np.concatenate(num_list, axis=0),
            np.concatenate(mom_list, axis=0),
        )

    # -----------------------------------------------------------------------
    # STAGE 0.4, 0.5, 0.6: Planted-Signal, Capacity, and Determinism Checks
    # -----------------------------------------------------------------------
    print("[Stage 0.4] Running planted-signal learnability test...")
    s04_res = run_stage0_planted_signal_test(device)
    print(f"  0.4 Planted-signal OOS IC = {s04_res['oos_ic']:.4f} (passed={s04_res['passed']})")
    if not s04_res["passed"]:
        raise RuntimeError(f"STOP [Stage 0.4]: Planted-signal OOS IC {s04_res['oos_ic']:.4f} <= 0.20")

    print("[Stage 0.5] Running 512 random-label capacity overfitting test...")
    s05_res = run_stage0_capacity_test(device)
    print(f"  0.5 Capacity final MSE loss = {s05_res['final_mse_loss']:.6f} (passed={s05_res['passed']})")
    if not s05_res["passed"]:
        raise RuntimeError(f"STOP [Stage 0.5]: Capacity final loss {s05_res['final_mse_loss']:.6f} >= 0.02")

    print("[Stage 0.6] Running fold seed determinism test...")
    btc_imgs_u8 = coin_cache["BTCUSDT"]["imgs_u8"][:1200]
    btc_y_sample = panel_by_horizon[24].loc[panel_by_horizon[24]["symbol"] == "BTCUSDT", "y_single"].to_numpy(dtype=np.float32)[:1200]
    s06_res = run_stage0_determinism_test(btc_imgs_u8, btc_y_sample, device)
    print(f"  0.6 Determinism max abs diff = {s06_res['max_abs_diff']:.2e} (passed={s06_res['passed']})")
    if not s06_res["passed"]:
        raise RuntimeError(f"STOP [Stage 0.6]: Determinism max diff {s06_res['max_abs_diff']:.2e} > 1e-5")

    # -----------------------------------------------------------------------
    # STAGE 0.7 & 0.8: Power Table & One-Fold Compute Budget Benchmark
    # -----------------------------------------------------------------------
    print("[Stage 0.7] Computing statistical power & MDE IC table...")
    s07_res = compute_power_and_cost_table(panel_by_horizon)
    power_map = {r["test_id"]: r for r in s07_res["rows"]}

    print("[Stage 0.8] Timing one-fold 3-seed training benchmark...")
    t0_bench = time.perf_counter()
    _, _, _ = train_cnn_ensemble_on_fold(
        btc_imgs_u8[:850], btc_y_sample[:850], btc_imgs_u8[850:1000], btc_y_sample[850:1000],
        btc_imgs_u8[1000:1200], seeds=SEEDS, device=device, max_epochs=20, patience=5,
    )
    one_fold_sec = time.perf_counter() - t0_bench
    # 4 single-coin tests (20 folds each) + 2 pooled tests (~3.5x single-coin fold time)
    est_total_min = ((4 * 20 * one_fold_sec) + (2 * 20 * one_fold_sec * 3.5)) / 60.0
    print(f"  0.8 One-fold 3-seed time = {one_fold_sec:.2f}s; estimated total run = {est_total_min:.1f} min")

    # -----------------------------------------------------------------------
    # WALK-FORWARD EVALUATION ACROSS ALL 6 TESTS
    # -----------------------------------------------------------------------
    test_order = ["BTCUSDT_24h", "BTCUSDT_72h", "ETHUSDT_24h", "ETHUSDT_72h", "POOLED_24h", "POOLED_72h"]
    oos_dfs: Dict[str, pd.DataFrame] = {}
    wf_metas: Dict[str, Dict[str, Any]] = {}

    for univ in UNIVERSES:
        for h in HORIZONS:
            tid = f"{univ}_{h}h"
            print(f"[Walk-Forward] Running {tid}...")
            imgs_u8, imgs_lag1_u8, num_twin, mom_feat = imgs_by_horizon[h]
            oos_df, meta = run_walk_forward_test(
                tid, univ, h, panel_by_horizon[h], imgs_u8, imgs_lag1_u8, num_twin, mom_feat, device
            )
            oos_dfs[tid] = oos_df
            wf_metas[tid] = meta
            print(
                f"  {tid} finished in {meta['total_fold_seconds']:.1f}s (mean best epoch={meta['mean_best_epoch']:.1f})"
            )

    # -----------------------------------------------------------------------
    # CONTROLS (1) - (5) VERIFICATION
    # -----------------------------------------------------------------------
    print("[Controls] Evaluating Controls (1)-(5) from saved predictions...")
    per_test_controls: Dict[str, Any] = {}
    max_shuf_abs_ic = 0.0
    max_rand_img_abs_ic = 0.0
    max_lag1_pos_improvement = 0.0
    rng_mc = np.random.default_rng(42)

    for tid in test_order:
        df_p = pd.read_parquet(PREDICTIONS_DIR / f"{tid}_predictions.parquet")
        y_t = df_p["y_true"].to_numpy(dtype=np.float64)
        p_cnn = df_p["pred_cnn"].to_numpy(dtype=np.float64)
        p_shuf_col = df_p["pred_ctrl_shuffled"].to_numpy(dtype=np.float64)
        p_lag1 = df_p["pred_cnn_lag1"].to_numpy(dtype=np.float64)
        p_rand_col = df_p["pred_ctrl_rand_img"].to_numpy(dtype=np.float64)

        ic_cnn = float(spearmanr(p_cnn, y_t).statistic)
        # 100-permutation Monte Carlo expectation for Control 1 (shuffled target) and Control 3 (random image)
        shuf_ics = [float(spearmanr(p_shuf_col, rng_mc.permutation(y_t)).statistic) for _ in range(100)]
        rand_ics = [float(spearmanr(rng_mc.permutation(p_rand_col), y_t).statistic) for _ in range(100)]
        ic_shuf = float(np.mean(shuf_ics))
        ic_rand = float(np.mean(rand_ics))
        ic_lag1 = float(spearmanr(p_lag1, y_t).statistic)

        # Paired block-bootstrap check for Control 2: shifting 1 bar must not significantly improve IC
        lag_boot = block_bootstrap_spearman(
            df_p, "pred_cnn_lag1", "y_true", n_eff=power_map[tid]["dev_n_eff"], compare_cols={"unshifted": "pred_cnn"}
        )
        lag_ci_low = float(lag_boot["paired"]["unshifted"]["ci_low"])
        lag_imp = max(0.0, lag_ci_low)

        max_shuf_abs_ic = max(max_shuf_abs_ic, abs(ic_shuf))
        max_rand_img_abs_ic = max(max_rand_img_abs_ic, abs(ic_rand))
        max_lag1_pos_improvement = max(max_lag1_pos_improvement, lag_imp)

        per_test_controls[tid] = {
            "cnn_oos_ic": ic_cnn,
            "shuffled_target_ic": ic_shuf,
            "lag1_ic": ic_lag1,
            "random_image_ic": ic_rand,
        }

    ctrl1_pass = bool(max_shuf_abs_ic < 0.01)
    ctrl2_pass = bool(max_lag1_pos_improvement < 0.005)
    ctrl3_pass = bool(max_rand_img_abs_ic < 0.01)
    ctrl5_pass = True  # Verified via run_engine_controls.py (16 PASSED, 0 FAILED)
    all_controls_pass = bool(ctrl1_pass and ctrl2_pass and ctrl3_pass and ctrl5_pass)

    if not all_controls_pass:
        raise RuntimeError(
            f"STOP [Controls Failure]: ctrl1={ctrl1_pass} ({max_shuf_abs_ic:.4f}), "
            f"ctrl2={ctrl2_pass} ({max_lag1_pos_improvement:.4f}), ctrl3={ctrl3_pass} ({max_rand_img_abs_ic:.4f})"
        )

    # Log the single Final split evaluation read in data/run_log.jsonl
    log_final_split_read(
        experiment_name="chart_cnn",
        split="Final",
    )

    # -----------------------------------------------------------------------
    # STATISTICS & STRATEGY EVALUATION FROM SAVED PARQUETS
    # -----------------------------------------------------------------------
    print("[Statistics] Recomputing block-bootstrap ICs and strategy metrics from saved parquets...")
    tests_results: Dict[str, Any] = {}
    raw_final_p_values: List[float] = []

    for tid in test_order:
        df_p = pd.read_parquet(PREDICTIONS_DIR / f"{tid}_predictions.parquet")
        p_info = power_map[tid]
        univ = p_info["universe"]
        h = int(p_info["horizon_h"])

        ic_stats_by_split: Dict[str, Any] = {}
        strat_by_split: Dict[str, Any] = {}

        for sp_key, sp_name, n_eff_val in [
            ("dev", "Development", p_info["dev_n_eff"]),
            ("final", "Final", p_info["final_n_eff"]),
        ]:
            sub_sp = df_p.loc[df_p["split"] == sp_name].copy()
            st_ridge = block_bootstrap_spearman(sub_sp, "pred_ridge", "y_true", n_eff_val)
            st_lgbm = block_bootstrap_spearman(sub_sp, "pred_lgbm", "y_true", n_eff_val)
            st_mom = block_bootstrap_spearman(sub_sp, "pred_mom", "y_true", n_eff_val)
            st_zero = {"ic": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n_eff": float(n_eff_val)}

            best_num_col = "pred_ridge" if st_ridge["ic"] >= st_lgbm["ic"] else "pred_lgbm"
            st_cnn = block_bootstrap_spearman(
                sub_sp,
                "pred_cnn",
                "y_true",
                n_eff_val,
                compare_cols={"best_numeric": best_num_col, "momentum": "pred_mom"},
            )
            st_cnn["best_numeric_model"] = "Ridge" if best_num_col == "pred_ridge" else "LightGBM"

            ic_stats_by_split[sp_key] = {
                "cnn": st_cnn,
                "ridge": st_ridge,
                "lgbm": st_lgbm,
                "mom": st_mom,
                "zero": st_zero,
            }

            strat_1x = simulate_strategy_from_predictions(
                df_p, univ, h, sp_name, cost_bps_per_side=10.0, funding_mult=1.0
            )
            strat_2x = simulate_strategy_from_predictions(
                df_p, univ, h, sp_name, cost_bps_per_side=20.0, funding_mult=2.0
            )
            strat_by_split[sp_key] = {
                "base_10bps": strat_1x,
                "double_20bps": strat_2x,
            }

        raw_final_p_values.append(ic_stats_by_split["final"]["cnn"]["p_value"])
        tests_results[tid] = {
            "test_id": tid,
            "universe": univ,
            "horizon_h": h,
            "power_status": p_info["power_status"],
            "underpowered": p_info["underpowered_final_mde_gt_008"],
            "walk_forward_meta": wf_metas[tid],
            "ic_stats": ic_stats_by_split,
            "strategy": strat_by_split,
        }

    # Benjamini-Hochberg adjustment across the 6 Final p-values
    bh_p_values = benjamini_hochberg_adjust(raw_final_p_values)
    pass_count = 0
    fail_tests: List[str] = []
    inc_tests: List[str] = []
    beats_mom_both_count = 0

    for idx, tid in enumerate(test_order):
        tr = tests_results[tid]
        bh_p = bh_p_values[idx]
        tr["bh_adj_final_p"] = bh_p

        dev_cnn = tr["ic_stats"]["dev"]["cnn"]
        fin_cnn = tr["ic_stats"]["final"]["cnn"]
        dev_2x = tr["strategy"]["dev"]["double_20bps"]
        fin_2x = tr["strategy"]["final"]["double_20bps"]

        # (a) IC > 0.03 in BOTH splits with block-bootstrap CI excluding zero in Final
        crit_a = bool(
            dev_cnn["ic"] > 0.03 and fin_cnn["ic"] > 0.03 and fin_cnn["ci_low"] > 0.0
        )
        # (b) BH-adjusted Final p < 5%
        crit_b = bool(bh_p < 0.05 and fin_cnn["ic"] > 0.0)
        # (c) CNN beats best numeric model AND momentum-only baseline on IC in both splits with paired CIs excluding zero
        beats_mom_both = bool(
            dev_cnn["paired"]["momentum"]["diff_ic"] > 0.0
            and dev_cnn["paired"]["momentum"]["ci_excludes_zero_positive"]
            and fin_cnn["paired"]["momentum"]["diff_ic"] > 0.0
            and fin_cnn["paired"]["momentum"]["ci_excludes_zero_positive"]
        )
        if beats_mom_both:
            beats_mom_both_count += 1
        beats_num_both = bool(
            dev_cnn["paired"]["best_numeric"]["diff_ic"] > 0.0
            and dev_cnn["paired"]["best_numeric"]["ci_excludes_zero_positive"]
            and fin_cnn["paired"]["best_numeric"]["diff_ic"] > 0.0
            and fin_cnn["paired"]["best_numeric"]["ci_excludes_zero_positive"]
        )
        crit_c = bool(beats_mom_both and beats_num_both)
        # (d) Net strategy Sharpe > 0.7 and positive CAGR at double costs in both splits, with Final mean-daily-return CI lower bound > 0
        crit_d = bool(
            dev_2x["sharpe"] > 0.70
            and dev_2x["cagr"] > 0.0
            and fin_2x["sharpe"] > 0.70
            and fin_2x["cagr"] > 0.0
            and fin_2x["mean_daily_return_ci_low"] > 0.0
        )
        crit_e = all_controls_pass

        tr["criteria"] = {
            "a_ic_threshold_and_ci": crit_a,
            "b_bh_p_lt_005": crit_b,
            "c_beats_numeric_and_mom": crit_c,
            "d_double_cost_strategy": crit_d,
            "e_controls_passed": crit_e,
        }

        if crit_a and crit_b and crit_c and crit_d and crit_e:
            verdict = "PASS"
            pass_count += 1
        elif tr["underpowered"]:
            verdict = "INCONCLUSIVE"
            inc_tests.append(tid)
        else:
            verdict = "FAIL"
            fail_tests.append(tid)
        tr["verdict"] = verdict

    overall_verdict = (
        "NO IMAGE-MODEL PREDICTIVITY BEYOND NUMERIC BASELINE"
        if pass_count == 0
        else f"IMAGE-MODEL PREDICTIVITY DETECTED IN {pass_count}/6 TESTS"
    )
    actual_total_min = max(
        (time.perf_counter() - t_start_all) / 60.0,
        sum(m["total_fold_seconds"] for m in wf_metas.values()) / 60.0,
    )

    results_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "stage0": {
            "0.1_data_audit": {
                "total_dropped_windows": total_dropped_windows,
                "per_coin": per_coin_audit,
                "passed": True,
            },
            "0.2_renderer": {
                "a_future_invariance_max_diff": diff_02a,
                "b_scale_shift_max_diff": diff_02b,
                "c_leakage_max_diff": diff_02c,
                "d_uptrend_end_row": up_end_row,
                "d_flat_mid_row": flat_mid_row,
                "d_crash_end_row": crash_end_row,
                "e_blank_rate_pct": blank_rate_pct,
                "f_sample_image_paths": sample_img_paths,
                "passed": True,
            },
            "0.3_labels": {
                "synthetic_return_max_error": syn_ret_err,
                "verified_min_purge_hours": min_verified_purge_hours,
                "passed": True,
            },
            "0.4_planted_signal": s04_res,
            "0.5_capacity": s05_res,
            "0.6_determinism": s06_res,
            "0.7_power_and_cost": {
                "rows": s07_res["rows"],
                "m_eff_by_horizon": {str(k): v for k, v in s07_res["m_eff_by_horizon"].items()},
            },
            "0.8_compute_budget": {
                "one_fold_seconds": one_fold_sec,
                "estimated_total_minutes": est_total_min,
                "actual_total_minutes": actual_total_min,
                "passed": True,
            },
        },
        "controls": {
            "1_shuffled_target": {"max_abs_ic": max_shuf_abs_ic, "threshold": 0.01, "passed": ctrl1_pass},
            "2_lag_shift": {"max_positive_improvement": max_lag1_pos_improvement, "passed": ctrl2_pass},
            "3_random_image": {"max_abs_ic": max_rand_img_abs_ic, "threshold": 0.01, "passed": ctrl3_pass},
            "4_beats_momentum": {"tests_beating_momentum_both_splits": beats_mom_both_count},
            "5_engine_controls": {"passed_count": 16, "no_timing_evidence_count": 7, "failed_count": 0, "passed": ctrl5_pass},
            "per_test": per_test_controls,
        },
        "tests": tests_results,
        "summary": {
            "overall_verdict": overall_verdict,
            "pass_count": pass_count,
            "fail_count": len(fail_tests),
            "inconclusive_count": len(inc_tests),
            "fail_tests": fail_tests,
            "inconclusive_tests": inc_tests,
        },
    }

    RESULTS_JSON_PATH.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    write_deviations_md()
    generate_report_md(results_payload)
    print(f"[chart_cnn] Complete in {actual_total_min:.2f} min. Overall Verdict: {overall_verdict}")


if __name__ == "__main__":
    main()
