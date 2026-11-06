"""
research/scripts/download_spot_klines.py

Downloads PUBLIC Binance SPOT 1h klines (no credentials, public endpoint)
from 2019-09-01 (or coin listing date) to now for the ten pre-declared symbols:
    BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT,
    ADAUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, AVAXUSDT
Saves:
    data/cache/<symbol>_spot_1h_full.parquet
Validates each file with validate_ohlcv_timestamps (closed bars only, UTC, deduplicated).
Updates research/data_manifest.json with spot file metadata and gaps.
"""
import os
import sys
import json
import time
import hashlib
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.quant_evaluation import validate_ohlcv_timestamps

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
           "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT"]
SPOT_BASE_URL = "https://api.binance.com"
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
MANIFEST_PATH = os.path.join(BASE_DIR, "research", "data_manifest.json")
HOUR_MS = 3_600_000
START_MS = 1567296000000  # 2019-09-01 00:00:00 UTC


def _get_spot(path: str, params: dict) -> list:
    for attempt in range(6):
        try:
            r = requests.get(SPOT_BASE_URL + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
        except requests.exceptions.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed {path} {params}")


def fetch_spot_klines(symbol: str, now_ms: int) -> pd.DataFrame:
    rows, start = [], START_MS
    while True:
        data = _get_spot("/api/v3/klines", {"symbol": symbol, "interval": "1h",
                                             "startTime": start, "limit": 1000})
        if not data:
            break
        rows.extend(data)
        start = data[-1][0] + HOUR_MS
        if len(data) < 1000 or start > now_ms:
            break
        time.sleep(0.08)
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qv", "n", "tb", "tq", "ig"])
    # Keep only completed closed bars
    df = df[df["open_time"] + HOUR_MS <= now_ms].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]].astype("float64")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def find_gaps(idx: pd.DatetimeIndex) -> list:
    gaps = []
    diffs = idx[1:] - idx[:-1]
    for i in range(len(diffs)):
        if diffs[i] != pd.Timedelta("1h"):
            missing = int(diffs[i] / pd.Timedelta("1h")) - 1
            gaps.append({
                "last_bar_before_gap": str(idx[i]),
                "first_bar_after_gap": str(idx[i + 1]),
                "missing_bars": missing,
                "handling": "NOT filled; bars absent from file. Consumers must treat as a break.",
            })
    return gaps


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    now_ms = int(now.timestamp() * 1000)

    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"symbols": {}}

    manifest["spot_source"] = "Binance Spot public REST (/api/v3/klines); no credentials"
    manifest["spot_generated_utc"] = str(now)
    if "spot_symbols" not in manifest:
        manifest["spot_symbols"] = {}

    total_spot_bytes = 0
    for sym in SYMBOLS:
        s_path = os.path.join(CACHE_DIR, f"{sym.lower()}_spot_1h_full.parquet")
        if os.path.exists(s_path):
            print(f"[{sym}] Spot file already exists at {s_path}, reading ...", flush=True)
            df = pd.read_parquet(s_path)
        else:
            print(f"[{sym}] Downloading Binance Spot 1h klines ...", flush=True)
            df = fetch_spot_klines(sym, now_ms)
            df.to_parquet(s_path)
            print(f"[{sym}] Downloaded {len(df)} spot bars ({df.index[0]} -> {df.index[-1]})", flush=True)

        total_spot_bytes += os.path.getsize(s_path)
        gaps = find_gaps(df.index)

        try:
            validate_ohlcv_timestamps(df, expected_freq="1h", reference_now=now)
            validator = "PASS"
        except ValueError as e:
            validator = f"FAIL (expected when gaps exist): {e}"

        seg_ok = True
        breaks = [0] + [int(i) + 1 for i in
                        (df.index[1:] - df.index[:-1] != pd.Timedelta("1h")).nonzero()[0]] + [len(df)]
        for a, b in zip(breaks[:-1], breaks[1:]):
            try:
                validate_ohlcv_timestamps(df.iloc[a:b], expected_freq="1h", reference_now=now)
            except ValueError:
                seg_ok = False

        manifest["spot_symbols"][sym] = {
            "symbol": sym,
            "market_type": "SPOT",
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "bar_count": int(len(df)),
            "expected_bars_if_contiguous": int((df.index[-1] - df.index[0]) / pd.Timedelta("1h")) + 1,
            "gap_count": len(gaps),
            "gaps": gaps,
            "validate_ohlcv_timestamps_full_series": validator,
            "validate_ohlcv_timestamps_all_contiguous_segments": "PASS" if seg_ok else "FAIL",
            "spot_file": os.path.relpath(s_path, BASE_DIR).replace("\\", "/"),
            "spot_sha256": sha256_file(s_path),
        }
        print(f"[{sym}] Spot validated | bars={len(df)} | gaps={len(gaps)} | segments_valid={seg_ok}", flush=True)

    manifest["total_spot_parquet_bytes"] = total_spot_bytes
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Updated {MANIFEST_PATH} with spot data; total spot size {total_spot_bytes / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
