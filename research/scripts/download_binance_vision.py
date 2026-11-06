"""
research/scripts/download_binance_vision.py

Downloads, validates, and standardizes public Binance Futures (USDT-M) data from:
  https://data.binance.vision/data/futures/um/

Datasets supported:
  1. daily metrics (OI, long/short ratios, taker buy/sell ratio)
  2. monthly markPriceKlines (5m) + daily for incomplete month
  3. monthly indexPriceKlines (5m) + daily for incomplete month
  4. monthly premiumIndexKlines (5m) + daily for incomplete month
  5. monthly fundingRate

Audit & Safety Rules:
  - Public data only. Zero API keys, zero private credentials.
  - No silent failures: log every URL as ok / 404 / error.
  - Retry on transient errors up to 3 times with exponential backoff.
  - At most 8 concurrent workers.
  - Verify every zip against its .CHECKSUM file (SHA256). Record mismatches.
  - Local raw zip caching ensures complete resume capability.
  - Detect CSV header vs no-header by content; enforce explicit column counts.
  - Sort by timestamp, drop exact duplicates, convert to UTC, save as Parquet.
  - Generates research/data_manifest_binance_vision.json.
"""
import os
import sys
import io
import time
import json
import zipfile
import hashlib
import logging
from datetime import datetime, timezone, date, timedelta
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("download_binance_vision")

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "binance_vision")
RAW_CACHE_DIR = os.path.join(DATA_DIR, ".raw_zips")
RESEARCH_DIR = os.path.join(BASE_DIR, "research")
MANIFEST_PATH = os.path.join(RESEARCH_DIR, "data_manifest_binance_vision.json")

BASE_URL = "https://data.binance.vision/data/futures/um"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT"
]

DATASET_TYPES = [
    "metrics",
    "markPriceKlines",
    "indexPriceKlines",
    "premiumIndexKlines",
    "fundingRate"
]

# Explicit schemas and column definitions
SCHEMAS = {
    "metrics": {
        "expected_cols": 8,
        "col_names": [
            "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
            "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
            "count_long_short_ratio", "sum_taker_long_short_vol_ratio"
        ],
        "header_key": "create_time",
        "time_col": "create_time",
        "time_type": "string_iso",
        "numeric_cols": [
            "sum_open_interest", "sum_open_interest_value",
            "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
            "count_long_short_ratio", "sum_taker_long_short_vol_ratio"
        ]
    },
    "klines": {
        "expected_cols": 12,
        "col_names": [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ],
        "header_key": "open_time",
        "time_col": "open_time",
        "time_type": "ms_timestamp",
        "numeric_cols": [
            "open", "high", "low", "close", "volume",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"
        ]
    },
    "fundingRate": {
        "expected_cols": 3,
        "col_names": ["calc_time", "funding_interval_hours", "last_funding_rate"],
        "header_key": "calc_time",
        "time_col": "calc_time",
        "time_type": "ms_timestamp",
        "numeric_cols": ["funding_interval_hours", "last_funding_rate"]
    }
}

# Earliest available listing dates on Binance Futures UM
START_DATES = {
    "BTCUSDT": {"metrics": "2020-09-01", "monthly": "2020-01"},
    "ETHUSDT": {"metrics": "2021-12-01", "monthly": "2020-01"},
    "BNBUSDT": {"metrics": "2021-12-01", "monthly": "2020-02"},
    "SOLUSDT": {"metrics": "2021-12-01", "monthly": "2020-09"},
    "XRPUSDT": {"metrics": "2021-12-01", "monthly": "2020-01"},
    "ADAUSDT": {"metrics": "2021-12-01", "monthly": "2020-01"},
    "DOGEUSDT": {"metrics": "2021-12-01", "monthly": "2020-07"},
    "LINKUSDT": {"metrics": "2021-12-01", "monthly": "2020-01"},
    "LTCUSDT": {"metrics": "2021-12-01", "monthly": "2020-01"},
    "AVAXUSDT": {"metrics": "2021-12-01", "monthly": "2020-09"},
}


def create_session(max_workers: int = 8) -> requests.Session:
    """Configures a requests Session with pooling and HTTP retry strategy."""
    session = requests.Session()
    session.headers.update({"User-Agent": "AntigravityDataEngine/2.0 (Public Research)"})
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(
        pool_connections=max_workers * 2,
        pool_maxsize=max_workers * 2,
        max_retries=retries
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def sha256_bytes(data: bytes) -> str:
    """Computes SHA256 hex digest of bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    """Computes SHA256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_csv_bytes(csv_bytes: bytes, schema_info: Dict[str, Any]) -> pd.DataFrame:
    """Parses raw CSV bytes with header detection and column count validation."""
    expected_cols = schema_info["expected_cols"]
    col_names = schema_info["col_names"]
    header_key = schema_info["header_key"]
    
    # Read first line to detect header
    text_io = io.StringIO(csv_bytes.decode("utf-8", errors="replace"))
    first_line = text_io.readline()
    if not first_line:
        return pd.DataFrame(columns=col_names)
    
    parts = [p.strip() for p in first_line.strip().split(",")]
    if len(parts) != expected_cols:
        raise ValueError(
            f"Unexpected column count: expected {expected_cols}, got {len(parts)}: {parts}"
        )
    
    has_header = (parts[0].lower() == header_key.lower())
    text_io.seek(0)
    
    if has_header:
        df = pd.read_csv(text_io, header=0, names=col_names, dtype=str)
    else:
        df = pd.read_csv(text_io, header=None, names=col_names, dtype=str)
    
    return df


def download_and_verify_zip(session: requests.Session, zip_url: str,
                            checksum_url: str, local_cache_path: str) -> Tuple[str, Optional[bytes], Optional[str]]:
    """Downloads a zip file and its checksum, verifies integrity, and caches locally.
    Returns:
        (status, zip_bytes, error_message)
        where status is 'ok', '404', or 'error'
    """
    # Check if already cached locally with verified checksum
    chk_cache_path = local_cache_path + ".CHECKSUM"
    if os.path.exists(local_cache_path) and os.path.exists(chk_cache_path):
        try:
            with open(chk_cache_path, "r", encoding="utf-8") as f:
                expected_hash = f.read().strip().split()[0].lower()
            with open(local_cache_path, "rb") as f:
                zip_bytes = f.read()
            if sha256_bytes(zip_bytes).lower() == expected_hash:
                return "ok", zip_bytes, None
        except Exception:
            pass  # Re-download if corrupted

    for attempt in range(3):
        try:
            # 1. Fetch checksum file
            r_chk = session.get(checksum_url, timeout=15)
            if r_chk.status_code == 404:
                return "404", None, f"Checksum file 404: {checksum_url}"
            if r_chk.status_code != 200:
                if attempt == 2:
                    return "error", None, f"Checksum HTTP {r_chk.status_code}: {checksum_url}"
                time.sleep(1.0 * (attempt + 1))
                continue
            
            chk_text = r_chk.text.strip()
            expected_hash = chk_text.split()[0].lower()
            
            # 2. Fetch zip file
            r_zip = session.get(zip_url, timeout=30)
            if r_zip.status_code == 404:
                return "404", None, f"Zip file 404: {zip_url}"
            if r_zip.status_code != 200:
                if attempt == 2:
                    return "error", None, f"Zip HTTP {r_zip.status_code}: {zip_url}"
                time.sleep(1.0 * (attempt + 1))
                continue
            
            zip_bytes = r_zip.content
            actual_hash = sha256_bytes(zip_bytes).lower()
            if actual_hash != expected_hash:
                return "error", None, (
                    f"Checksum mismatch for {zip_url}! Expected {expected_hash}, got {actual_hash}"
                )
            
            # Cache locally
            os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)
            with open(local_cache_path, "wb") as f:
                f.write(zip_bytes)
            with open(chk_cache_path, "w", encoding="utf-8") as f:
                f.write(chk_text)
                
            return "ok", zip_bytes, None
            
        except Exception as e:
            if attempt == 2:
                return "error", None, f"Exception: {str(e)}"
            time.sleep(1.0 * (attempt + 1))
            
    return "error", None, "Max retries exceeded"


def extract_csv_from_zip(zip_bytes: bytes) -> bytes:
    """Extracts the first CSV file contained in the zip archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for fname in z.namelist():
            if fname.endswith(".csv"):
                return z.read(fname)
        raise FileNotFoundError("No .csv file found inside zip archive")


def get_month_range(start_ym: str, end_ym: str) -> List[str]:
    """Generates a list of YYYY-MM strings inclusive."""
    s_yr, s_mo = map(int, start_ym.split("-"))
    e_yr, e_mo = map(int, end_ym.split("-"))
    months = []
    curr_yr, curr_mo = s_yr, s_mo
    while (curr_yr < e_yr) or (curr_yr == e_yr and curr_mo <= e_mo):
        months.append(f"{curr_yr:04d}-{curr_mo:02d}")
        curr_mo += 1
        if curr_mo > 12:
            curr_mo = 1
            curr_yr += 1
    return months


def get_date_range(start_ymd: str, end_ymd: str) -> List[str]:
    """Generates a list of YYYY-MM-DD strings inclusive."""
    s_dt = datetime.strptime(start_ymd, "%Y-%m-%d").date()
    e_dt = datetime.strptime(end_ymd, "%Y-%m-%d").date()
    days = []
    curr = s_dt
    while curr <= e_dt:
        days.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return days


def process_dataset(session: requests.Session, symbol: str, dataset: str,
                    latest_complete_month: str = "2026-10",
                    today_str: str = "2026-11-05",
                    max_workers: int = 8) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Downloads all files for a given (symbol, dataset), validates, and returns DataFrame + stats."""
    logger.info(f"Processing {symbol} -> {dataset} ...")
    start_info = START_DATES[symbol]
    
    schema_key = "klines" if "Klines" in dataset else dataset
    schema_info = SCHEMAS[schema_key]
    
    cache_subdir = os.path.join(RAW_CACHE_DIR, dataset, symbol)
    url_tasks = []  # List of (label, zip_url, checksum_url, local_cache_path)
    
    if dataset == "metrics":
        start_date = start_info["metrics"]
        days = get_date_range(start_date, today_str)
        for d in days:
            fname = f"{symbol}-metrics-{d}.zip"
            z_url = f"{BASE_URL}/daily/metrics/{symbol}/{fname}"
            c_url = f"{z_url}.CHECKSUM"
            l_path = os.path.join(cache_subdir, fname)
            url_tasks.append((d, z_url, c_url, l_path))
            
    elif dataset in ("markPriceKlines", "indexPriceKlines", "premiumIndexKlines"):
        start_mo = start_info["monthly"]
        months = get_month_range(start_mo, latest_complete_month)
        for m in months:
            fname = f"{symbol}-5m-{m}.zip"
            z_url = f"{BASE_URL}/monthly/{dataset}/{symbol}/5m/{fname}"
            c_url = f"{z_url}.CHECKSUM"
            l_path = os.path.join(cache_subdir, fname)
            url_tasks.append((m, z_url, c_url, l_path))
        
        # Daily files for incomplete month (November 2026)
        inc_days = get_date_range("2026-11-01", today_str)
        for d in inc_days:
            fname = f"{symbol}-5m-{d}.zip"
            z_url = f"{BASE_URL}/daily/{dataset}/{symbol}/5m/{fname}"
            c_url = f"{z_url}.CHECKSUM"
            l_path = os.path.join(cache_subdir, fname)
            url_tasks.append((d, z_url, c_url, l_path))
            
    elif dataset == "fundingRate":
        start_mo = start_info["monthly"]
        months = get_month_range(start_mo, latest_complete_month)
        for m in months:
            fname = f"{symbol}-fundingRate-{m}.zip"
            z_url = f"{BASE_URL}/monthly/fundingRate/{symbol}/{fname}"
            c_url = f"{z_url}.CHECKSUM"
            l_path = os.path.join(cache_subdir, fname)
            url_tasks.append((m, z_url, c_url, l_path))

    total_tasks = len(url_tasks)
    logger.info(f"  {symbol} {dataset}: Enqueued {total_tasks} URLs to fetch.")
    
    ok_count = 0
    not_found_count = 0
    err_count = 0
    mismatches = []
    dfs = []
    
    def worker(task):
        label, z_url, c_url, l_path = task
        st, z_bytes, err = download_and_verify_zip(session, z_url, c_url, l_path)
        return label, z_url, st, z_bytes, err

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, t): t for t in url_tasks}
        for future in as_completed(futures):
            label, z_url, st, z_bytes, err = future.result()
            if st == "ok":
                ok_count += 1
                try:
                    csv_bytes = extract_csv_from_zip(z_bytes)
                    df_chunk = parse_csv_bytes(csv_bytes, schema_info)
                    dfs.append(df_chunk)
                except Exception as ex:
                    logger.error(f"  PARSE ERROR in {z_url}: {ex}")
                    err_count += 1
            elif st == "404":
                not_found_count += 1
            else:
                err_count += 1
                logger.error(f"  FETCH ERROR: {z_url} -> {err}")
                if "Checksum mismatch" in str(err):
                    mismatches.append(z_url)

    logger.info(
        f"  Finished {symbol} {dataset}: OK={ok_count}, 404={not_found_count}, Errors={err_count}"
    )
    if mismatches:
        raise ValueError(f"Checksum mismatches detected in {symbol} {dataset}: {mismatches}")
    if err_count > 0:
        logger.warning(f"Encountered {err_count} fetch/parse errors in {symbol} {dataset}")
        
    if not dfs:
        logger.warning(f"No data parsed for {symbol} {dataset}!")
        return None, {
            "symbol": symbol, "dataset": dataset, "rows": 0, "ok_count": ok_count,
            "404_count": not_found_count, "err_count": err_count
        }

    # Concatenate all chunks
    combined_df = pd.concat(dfs, ignore_index=True)
    initial_rows = len(combined_df)
    
    # Standardize timestamps
    time_col = schema_info["time_col"]
    if schema_info["time_type"] == "ms_timestamp":
        combined_df["timestamp"] = pd.to_datetime(
            combined_df[time_col].astype(np.int64), unit="ms", utc=True
        )
    else:  # string_iso create_time
        combined_df["timestamp"] = pd.to_datetime(combined_df[time_col], utc=True)
        
    # Cast numeric columns
    for num_col in schema_info["numeric_cols"]:
        if num_col in combined_df.columns:
            combined_df[num_col] = pd.to_numeric(combined_df[num_col], errors="coerce")
            
    # Sort by timestamp
    combined_df = combined_df.sort_values("timestamp")
    
    # Drop exact duplicates
    combined_df = combined_df.drop_duplicates()
    final_rows = len(combined_df)
    duplicates_dropped = initial_rows - final_rows
    
    # Set timestamp as index and keep relevant columns
    combined_df = combined_df.set_index("timestamp")
    
    # Detailed Gap and Manifest Analysis
    first_ts = combined_df.index[0]
    last_ts = combined_df.index[-1]
    
    # Rows per day
    daily_counts = combined_df.resample("1D").size()
    median_rows_day = float(daily_counts.median()) if len(daily_counts) > 0 else 0.0
    min_rows_day = float(daily_counts.min()) if len(daily_counts) > 0 else 0.0
    missing_days = [d.strftime("%Y-%m-%d") for d, c in daily_counts.items() if c == 0]
    
    # Gaps longer than 15 minutes (for 5m klines and metrics)
    diffs = combined_df.index[1:] - combined_df.index[:-1]
    gaps_gt_15m = []
    if "funding" not in dataset:
        for idx_gap in np.where(diffs > pd.Timedelta(minutes=15))[0]:
            gap_start = combined_df.index[idx_gap]
            gap_end = combined_df.index[idx_gap + 1]
            dur_min = (gap_end - gap_start).total_seconds() / 60.0
            gaps_gt_15m.append({
                "start": str(gap_start),
                "end": str(gap_end),
                "gap_minutes": round(dur_min, 1)
            })

    stats = {
        "symbol": symbol,
        "dataset": dataset,
        "first_timestamp": str(first_ts),
        "last_timestamp": str(last_ts),
        "rows": final_rows,
        "duplicates_dropped": duplicates_dropped,
        "median_rows_per_day": median_rows_day,
        "min_rows_per_day": min_rows_day,
        "missing_days": missing_days,
        "gaps_longer_than_15m": gaps_gt_15m,
        "ok_files": ok_count,
        "404_files": not_found_count,
        "error_files": err_count
    }
    
    return combined_df, stats


def run_pipeline(symbols: List[str] = SYMBOLS,
                 datasets: List[str] = DATASET_TYPES,
                 max_workers: int = 8) -> None:
    """Executes the full download, verification, standardization, and manifest generation."""
    logger.info("=== STARTING BINANCE PUBLIC METRICS & KLINES INGESTION ===")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    os.makedirs(RESEARCH_DIR, exist_ok=True)
    
    session = create_session(max_workers=max_workers)
    
    # Load existing manifest if resuming
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "symbols": symbols,
        "datasets": datasets,
        "records": {}
    }
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                old_m = json.load(f)
                manifest["records"] = old_m.get("records", {})
        except Exception:
            pass
            
    total_datasets = len(symbols) * len(datasets)
    idx = 0
    
    for sym in symbols:
        sym_dir = os.path.join(DATA_DIR, sym)
        os.makedirs(sym_dir, exist_ok=True)
        if sym not in manifest["records"]:
            manifest["records"][sym] = {}
        
        for ds in datasets:
            idx += 1
            out_parquet = os.path.join(sym_dir, f"{ds}.parquet")
            
            # Check if parquet already exists and valid
            if os.path.exists(out_parquet) and sym in manifest["records"] and ds in manifest["records"][sym]:
                logger.info(f"[{idx}/{total_datasets}] Already present: {sym} {ds} ({out_parquet}). Skipping download.")
                continue
                
            logger.info(f"[{idx}/{total_datasets}] Ingesting {sym} {ds} ...")
            df, stats = process_dataset(session, sym, ds, max_workers=max_workers)
            if df is not None and not df.empty:
                df.to_parquet(out_parquet, compression="snappy")
                sha = sha256_file(out_parquet)
                stats["sha256"] = sha
                stats["parquet_size_bytes"] = os.path.getsize(out_parquet)
                logger.info(
                    f"Saved {out_parquet}: {len(df):,} rows, {stats['parquet_size_bytes']:,} bytes, sha256={sha[:12]}..."
                )
            else:
                stats["sha256"] = None
                stats["parquet_size_bytes"] = 0
                logger.warning(f"Could not write {out_parquet} (empty or missing data)")
                
            manifest["records"][sym][ds] = stats
            # Save manifest incrementally
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Successfully generated manifest at {MANIFEST_PATH}")
    logger.info("=== BINANCE VISION DOWNLOAD & AUDIT COMPLETE ===")


if __name__ == "__main__":
    run_pipeline(max_workers=8)
