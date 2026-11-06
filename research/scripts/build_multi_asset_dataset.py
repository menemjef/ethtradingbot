"""
research/scripts/build_multi_asset_dataset.py

Downloads PUBLIC Binance USDT-perpetual 1h klines and funding-rate history (no API keys,
no accounts, no orders) from the earliest available date to now, for a fixed pre-declared
symbol list (NOT selected by performance). Saves:
    data/cache/<symbol>_1h_full.parquet
    data/cache/<symbol>_funding_full.parquet
Never overwrites existing files. Writes research/data_manifest.json.

Gaps are reported, never forward-filled. Only closed bars are kept.
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
BASE_URL = "https://fapi.binance.com"
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
HOUR_MS = 3_600_000
START_MS = 1567296000000  # 2019-09-01 00:00 UTC (before any Binance USDT-M listing)


def _get(path: str, params: dict) -> list:
    for attempt in range(6):
        r = requests.get(BASE_URL + path, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed {path} {params}")


def fetch_klines(symbol: str, now_ms: int) -> pd.DataFrame:
    rows, start = [], START_MS
    while True:
        data = _get("/fapi/v1/klines", {"symbol": symbol, "interval": "1h",
                                        "startTime": start, "limit": 1500})
        if not data:
            break
        rows.extend(data)
        start = data[-1][0] + HOUR_MS
        if len(data) < 1500 or start > now_ms:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qv", "n", "tb", "tq", "ig"])
    # closed bars only
    df = df[df["open_time"] + HOUR_MS <= now_ms].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]].astype("float64")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def fetch_funding(symbol: str, now_ms: int) -> pd.DataFrame:
    rows, start = [], START_MS
    while True:
        data = _get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start, "limit": 1000})
        if not data:
            break
        rows.extend(data)
        start = data[-1]["fundingTime"] + 1
        if start > now_ms:
            break
        time.sleep(0.15)
    df = pd.DataFrame({
        "funding_time_ms": [int(x["fundingTime"]) for x in rows],
        "funding_rate": [float(x["fundingRate"]) for x in rows],
    })
    df.index = pd.to_datetime(df["funding_time_ms"], unit="ms", utc=True)
    df.index.name = "timestamp"
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
                "handling": "NOT filled; bars absent from file. Consumers must treat as a break "
                            "(split into contiguous segments).",
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
    manifest = {"generated_utc": str(now), "source": "Binance USDT-M futures public REST "
                "(/fapi/v1/klines, /fapi/v1/fundingRate); no credentials",
                "symbol_selection": "Fixed pre-declared list; not selected on performance.",
                "symbols": {}}
    total_bytes = 0
    for sym in SYMBOLS:
        k_path = os.path.join(CACHE_DIR, f"{sym.lower()}_1h_full.parquet")
        f_path = os.path.join(CACHE_DIR, f"{sym.lower()}_funding_full.parquet")
        if os.path.exists(k_path) or os.path.exists(f_path):
            raise FileExistsError(f"Refusing to overwrite existing file for {sym}")
        print(f"[{sym}] downloading klines ...", flush=True)
        kl = fetch_klines(sym, now_ms)
        print(f"[{sym}] {len(kl)} bars; downloading funding ...", flush=True)
        fd = fetch_funding(sym, now_ms)
        kl.to_parquet(k_path)
        fd.to_parquet(f_path)
        total_bytes += os.path.getsize(k_path) + os.path.getsize(f_path)

        gaps = find_gaps(kl.index)
        try:
            validate_ohlcv_timestamps(kl, expected_freq="1h", reference_now=now)
            validator = "PASS"
        except ValueError as e:
            validator = f"FAIL (expected when gaps exist): {e}"
        # Validate each contiguous segment independently
        seg_ok = True
        breaks = [0] + [int(i) + 1 for i in
                        (kl.index[1:] - kl.index[:-1] != pd.Timedelta("1h")).nonzero()[0]] + [len(kl)]
        for a, b in zip(breaks[:-1], breaks[1:]):
            try:
                validate_ohlcv_timestamps(kl.iloc[a:b], expected_freq="1h", reference_now=now)
            except ValueError:
                seg_ok = False
        manifest["symbols"][sym] = {
            "symbol": sym,
            "listing_start_first_bar": str(kl.index[0]),
            "start": str(kl.index[0]),
            "end": str(kl.index[-1]),
            "bar_count": int(len(kl)),
            "expected_bars_if_contiguous": int((kl.index[-1] - kl.index[0]) / pd.Timedelta("1h")) + 1,
            "gap_count": len(gaps),
            "gaps": gaps,
            "validate_ohlcv_timestamps_full_series": validator,
            "validate_ohlcv_timestamps_all_contiguous_segments": "PASS" if seg_ok else "FAIL",
            "ohlcv_file": os.path.relpath(k_path, BASE_DIR).replace("\\", "/"),
            "ohlcv_sha256": sha256_file(k_path),
            "funding_file": os.path.relpath(f_path, BASE_DIR).replace("\\", "/"),
            "funding_row_count": int(len(fd)),
            "funding_start": str(fd.index[0]) if len(fd) else None,
            "funding_end": str(fd.index[-1]) if len(fd) else None,
            "funding_sha256": sha256_file(f_path),
        }
        print(f"[{sym}] {kl.index[0]} -> {kl.index[-1]} | gaps={len(gaps)} | funding={len(fd)}", flush=True)
    manifest["total_parquet_bytes"] = total_bytes
    out = os.path.join(BASE_DIR, "research", "data_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out}; total parquet size {total_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
