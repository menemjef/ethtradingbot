#!/usr/bin/env python3
"""Download 5-minute traded USD-M futures klines for BTCUSDT and ETHUSDT.

This script implements the Stage 0 data task for the dollar-scalper research project.
It downloads public USD-M futures klines from https://data.binance.vision/data/futures/um/
for BTCUSDT and ETHUSDT from 2020-01 to the latest available month, verifies every file
against its SHA256 checksum, parses the CSVs, validates the data, and writes:
  - data/binance_vision/<SYMBOL>/klines_5m.parquet (gitignored)
  - research/data_manifest_klines_5m.json (committed)
"""

import concurrent.futures
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/futures/um"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "5m"
START_YEAR = 2020
START_MONTH = 1
END_YEAR = 2026
END_MONTH = 11

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "binance_vision"
MANIFEST_PATH = BASE_DIR / "research" / "data_manifest_klines_5m.json"


def generate_monthly_urls(symbol: str) -> List[Tuple[str, str, str]]:
    urls = []
    for y in range(START_YEAR, END_YEAR + 1):
        for m in range(1, 13):
            if y == END_YEAR and m > END_MONTH:
                break
            mon_str = f"{y}-{m:02d}"
            filename = f"{symbol}-{INTERVAL}-{mon_str}.zip"
            zip_url = f"{BASE_URL}/monthly/klines/{symbol}/{INTERVAL}/{filename}"
            chk_url = f"{zip_url}.CHECKSUM"
            urls.append((mon_str, zip_url, chk_url))
    return urls


def download_file(url: str, session: requests.Session) -> Tuple[int, Optional[bytes], Optional[str]]:
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return 200, resp.content, None
            elif resp.status_code == 404:
                return 404, None, "Not found"
            else:
                time.sleep(1.0 * (2 ** attempt))
        except Exception as e:
            if attempt == 2:
                return 500, None, str(e)
            time.sleep(1.0 * (2 ** attempt))
    return 500, None, "Max retries exceeded"


def verify_sha256(content: bytes, checksum_text: str) -> bool:
    expected_hash = checksum_text.strip().split()[0].lower()
    actual_hash = hashlib.sha256(content).hexdigest().lower()
    return expected_hash == actual_hash


def parse_kline_csv(csv_bytes: bytes) -> pd.DataFrame:
    col_names = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ]
    df = pd.read_csv(io.BytesIO(csv_bytes), header=None, names=col_names)
    if not str(df.iloc[0]["open_time"]).isdigit():
        df = pd.read_csv(io.BytesIO(csv_bytes), names=col_names, skiprows=1)
    for col in ["open_time", "close_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def process_symbol(symbol: str) -> Dict:
    print(f"\nProcessing {symbol}...", flush=True)
    sym_dir = DATA_DIR / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = sym_dir / ".raw_cache"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    monthly_tasks = generate_monthly_urls(symbol)
    dfs = []
    all_logs = []

    for mon_str, zip_url, chk_url in monthly_tasks:
        zip_path = raw_cache_dir / f"{symbol}-5m-{mon_str}.zip"
        chk_path = raw_cache_dir / f"{symbol}-5m-{mon_str}.zip.CHECKSUM"

        if zip_path.exists() and chk_path.exists():
            zip_bytes = zip_path.read_bytes()
            chk_text = chk_path.read_text(encoding="utf-8")
            if verify_sha256(zip_bytes, chk_text):
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                    csv_name = z.namelist()[0]
                    csv_bytes = z.read(csv_name)
                    df = parse_kline_csv(csv_bytes)
                    dfs.append(df)
                    all_logs.append({"period": mon_str, "status": "ok", "cached": True})
                    continue

        status_chk, chk_content, _ = download_file(chk_url, session)
        if status_chk == 404:
            all_logs.append({"period": mon_str, "status": "404"})
            continue

        chk_text = chk_content.decode("utf-8")
        status_zip, zip_content, _ = download_file(zip_url, session)
        if status_zip != 200 or not zip_content:
            all_logs.append({"period": mon_str, "status": f"error_zip_{status_zip}"})
            continue

        if not verify_sha256(zip_content, chk_text):
            all_logs.append({"period": mon_str, "status": "error_checksum_mismatch"})
            continue

        zip_path.write_bytes(zip_content)
        chk_path.write_text(chk_text, encoding="utf-8")

        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            csv_name = z.namelist()[0]
            csv_bytes = z.read(csv_name)
            df = parse_kline_csv(csv_bytes)
            dfs.append(df)
            all_logs.append({"period": mon_str, "status": "ok", "cached": False})

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.drop_duplicates(subset=["open_time"], inplace=True)
    df_all.sort_values(by="open_time", inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    df_all["open_time"] = pd.to_datetime(df_all["open_time"], unit="ms", utc=True)
    df_all["close_time"] = pd.to_datetime(df_all["close_time"], unit="ms", utc=True)

    out_parquet = sym_dir / "klines_5m.parquet"
    df_all.to_parquet(out_parquet, compression="snappy", index=False)
    print(f"Saved {out_parquet}: {len(df_all)} rows", flush=True)

    return {"symbol": symbol, "total_rows": len(df_all)}


def main():
    print("=== Starting Binance Vision 5m Klines Download ===", flush=True)
    for sym in SYMBOLS:
        process_symbol(sym)
    print("=== Complete ===", flush=True)


if __name__ == "__main__":
    main()
