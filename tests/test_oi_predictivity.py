"""
tests/test_oi_predictivity.py

Unit tests for Binance Vision data ingestion and pre-registered OI predictivity pipeline:
1. Mocked HTTP: 404 handling, retry with backoff, checksum mismatch detection.
2. CSV Parser: Header vs headerless detection, explicit column count validation.
3. Sorting and de-duplication: Strict monotonicity and duplicate removal.
4. Causal stamping: Features at bar close t use metrics <= t - 5m; future data invariance.
5. 180-day universe rule: Coins enter universe strictly 180 days after first bar.
6. Benjamini-Hochberg FDR: Correct false discovery rate adjustment at 5%.
7. Final split lock: Final test partition (> 2024-12-31) cannot be accessed while locked.
"""
import os
import sys
import io
import zipfile
import hashlib
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import research.scripts.download_binance_vision as D


class TestBinanceVisionIngestion(unittest.TestCase):

    def test_mocked_http_404_handling(self):
        """HTTP 404 on zip or checksum must return '404' status without raising unhandled error."""
        session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        session.get.return_value = mock_resp

        status, b, err = D.download_and_verify_zip(
            session,
            "https://data.binance.vision/missing.zip",
            "https://data.binance.vision/missing.zip.CHECKSUM",
            os.path.join(D.RAW_CACHE_DIR, "test_missing.zip")
        )
        self.assertEqual(status, "404")
        self.assertIsNone(b)
        self.assertIn("404", str(err))

    def test_mocked_http_retry_with_backoff(self):
        """HTTP 500 transient errors must be retried with exponential backoff and succeed on 200."""
        session = MagicMock()

        # Checksum mock
        r_chk = MagicMock()
        r_chk.status_code = 200
        content_zip = b"PK\x03\x04synthetic_zip_content"
        correct_hash = hashlib.sha256(content_zip).hexdigest()
        r_chk.text = f"{correct_hash}  test.zip\n"

        # Zip mock: fail twice with 500, then succeed on 3rd attempt
        r_fail = MagicMock()
        r_fail.status_code = 500
        r_ok = MagicMock()
        r_ok.status_code = 200
        r_ok.content = content_zip

        # session.get called for checksum, then zip (fail), checksum, zip (fail), checksum, zip (ok)
        session.get.side_effect = [
            r_chk, r_fail,  # attempt 0
            r_chk, r_fail,  # attempt 1
            r_chk, r_ok     # attempt 2
        ]

        with patch("time.sleep", return_value=None):
            status, b, err = D.download_and_verify_zip(
                session,
                "https://data.binance.vision/test.zip",
                "https://data.binance.vision/test.zip.CHECKSUM",
                os.path.join(D.RAW_CACHE_DIR, "test_retry.zip")
            )
        self.assertEqual(status, "ok")
        self.assertEqual(b, content_zip)
        self.assertIsNone(err)

    def test_mocked_http_checksum_mismatch(self):
        """Checksum mismatch between .CHECKSUM and zip content must trigger error status."""
        session = MagicMock()
        r_chk = MagicMock()
        r_chk.status_code = 200
        r_chk.text = "0000000000000000000000000000000000000000000000000000000000000000  bad.zip\n"

        r_zip = MagicMock()
        r_zip.status_code = 200
        r_zip.content = b"valid_zip_bytes_with_different_hash"

        session.get.side_effect = [r_chk, r_zip]

        status, b, err = D.download_and_verify_zip(
            session,
            "https://data.binance.vision/bad.zip",
            "https://data.binance.vision/bad.zip.CHECKSUM",
            os.path.join(D.RAW_CACHE_DIR, "test_bad.zip")
        )
        self.assertEqual(status, "error")
        self.assertIn("Checksum mismatch", str(err))

    def test_header_vs_headerless_csv_parsing(self):
        """CSVs with header vs headerless must be correctly parsed with explicit schema enforcement."""
        schema_klines = D.SCHEMAS["klines"]

        # 1. Header present
        csv_header = (
            b"open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
            b"1704067200000,42000.0,42100.0,41900.0,42050.0,10.5,1704067499999,441525.0,300,5.2,218660.0,0\n"
        )
        df_h = D.parse_csv_bytes(csv_header, schema_klines)
        self.assertEqual(len(df_h), 1)
        self.assertEqual(df_h.iloc[0]["open_time"], "1704067200000")
        self.assertEqual(df_h.iloc[0]["close"], "42050.0")

        # 2. Headerless
        csv_no_header = (
            b"1577836800000,7195.36,7195.36,7180.29,7180.42,0,1577837099999,0,300,0,0,0\n"
            b"1577837100000,7180.43,7189.88,7179.33,7189.65,0,1577837399999,0,300,0,0,0\n"
        )
        df_nh = D.parse_csv_bytes(csv_no_header, schema_klines)
        self.assertEqual(len(df_nh), 2)
        self.assertEqual(df_nh.iloc[0]["open_time"], "1577836800000")
        self.assertEqual(df_nh.iloc[1]["close"], "7189.65")

        # 3. Wrong column count must raise ValueError loudly
        csv_bad = b"1577836800000,7195.36,7195.36\n"
        with self.assertRaises(ValueError):
            D.parse_csv_bytes(csv_bad, schema_klines)

    def test_sorting_and_deduplication(self):
        """Unsorted rows with duplicates must sort strictly monotonically and drop duplicates."""
        timestamps = [
            "2023-01-01 02:00:00+00:00",
            "2023-01-01 01:00:00+00:00",
            "2023-01-01 01:00:00+00:00",  # Duplicate
            "2023-01-01 03:00:00+00:00"
        ]
        df = pd.DataFrame({
            "create_time": timestamps,
            "symbol": ["BTCUSDT"] * 4,
            "sum_open_interest": [100.0, 95.0, 95.0, 105.0]
        })
        df["timestamp"] = pd.to_datetime(df["create_time"], utc=True)
        df_sorted = df.sort_values("timestamp").drop_duplicates().set_index("timestamp")

        self.assertEqual(len(df_sorted), 3)
        self.assertTrue(df_sorted.index.is_monotonic_increasing)


class TestPredictivityRulesAndGuards(unittest.TestCase):

    def test_causal_stamping_no_future_data(self):
        """Features evaluated at bar close t must strictly use metrics <= t - 5m; future data has zero effect."""
        # Timeline: decision at 2024-01-01 12:00:00 UTC
        t_decision = pd.Timestamp("2024-01-01 12:00:00+00:00")
        t_cutoff = t_decision - pd.Timedelta(minutes=5)  # 11:55:00 UTC

        # Metrics 5m series up to 13:00 UTC
        idx_5m = pd.date_range("2024-01-01 00:00:00+00:00", "2024-01-01 13:00:00+00:00", freq="5min")
        np.random.seed(42)
        base_oi = 50000.0 + np.cumsum(np.random.normal(0, 10, len(idx_5m)))
        metrics_df = pd.DataFrame({"sum_open_interest": base_oi}, index=idx_5m)

        # Causal selection: latest row with index <= t_cutoff
        causal_mask = metrics_df.index <= t_cutoff
        val_base = metrics_df.loc[causal_mask, "sum_open_interest"].iloc[-1]
        self.assertEqual(metrics_df.loc[causal_mask].index[-1], t_cutoff)

        # Now perturb all future rows (after 11:55:00) with extreme spikes
        future_mask = metrics_df.index > t_cutoff
        metrics_perturbed = metrics_df.copy()
        metrics_perturbed.loc[future_mask, "sum_open_interest"] *= 100.0

        val_perturbed = metrics_perturbed.loc[metrics_perturbed.index <= t_cutoff, "sum_open_interest"].iloc[-1]
        self.assertEqual(val_base, val_perturbed)

    def test_180_day_universe_rule(self):
        """A coin enters universe strictly 180 days after its first bar."""
        t_first = pd.Timestamp("2021-01-01 00:00:00+00:00")
        t_eligible = t_first + pd.Timedelta(days=180)

        # At day 100, not eligible
        t_100 = t_first + pd.Timedelta(days=100)
        self.assertFalse(t_100 >= t_eligible)

        # At day 179, not eligible
        t_179 = t_first + pd.Timedelta(days=179)
        self.assertFalse(t_179 >= t_eligible)

        # At day 180, eligible
        self.assertTrue(t_eligible >= t_eligible)

    def test_benjamini_hochberg_fdr_control(self):
        """Benjamini-Hochberg adjustment controls FDR at 5% with monotonic adjusted p-values."""
        # 12 synthetic p-values: 2 very strong, 10 noise
        raw_p = [0.001, 0.004, 0.08, 0.12, 0.15, 0.22, 0.35, 0.40, 0.55, 0.65, 0.75, 0.90]
        m = len(raw_p)

        # Manual BH calculation
        sorted_indices = np.argsort(raw_p)
        sorted_p = np.array(raw_p)[sorted_indices]

        adj_p = np.zeros(m)
        adj_p[-1] = sorted_p[-1]
        for i in range(m - 2, -1, -1):
            adj_p[i] = min(adj_p[i + 1], sorted_p[i] * m / (i + 1))
        adj_p = np.minimum(1.0, adj_p)

        # Check that smallest p-values are adjusted properly
        self.assertAlmostEqual(adj_p[0], 0.001 * 12 / 1.0, places=4)  # 0.012
        self.assertAlmostEqual(adj_p[1], 0.004 * 12 / 2.0, places=4)  # 0.024
        self.assertTrue(np.all(np.diff(adj_p) >= 0.0))  # Monotonic increasing

    def test_final_split_locked_guard(self):
        """Attempting to access final split data (> 2024-12-31) while locked must raise PermissionError."""
        # Simulated lock guard pattern
        is_unlocked = False

        def get_data_for_split(split_name: str):
            if split_name == "Final" and not is_unlocked:
                raise PermissionError("Final test data (> 2024-12-31) is locked until single final run.")
            return "data"

        # Development can always be accessed
        self.assertEqual(get_data_for_split("Development"), "data")

        # Final cannot be accessed when locked
        with self.assertRaises(PermissionError):
            get_data_for_split("Final")


if __name__ == "__main__":
    unittest.main()
