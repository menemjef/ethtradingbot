"""
tests/test_engine_controls.py

Unit test suite for the Standing Backtest Engine Control Suite.
Verifies all 8 institutional engine controls across BTCUSDT and ETHUSDT 1H data:
  1. Random-signal control
  2. Perfect-foresight control
  3. Shifted-date control (tightened 50 shifts between 30 and 365 days; synthetic skill/noskill)
  4. Buy-and-hold replication
  5. Cost monotonicity
  6. Timing control
  7. Accounting control
  8. Report-vs-code control
"""
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from research.scripts.run_engine_controls import (
    run_control_1_random_signal,
    run_control_2_perfect_foresight,
    run_control_3_shifted_date,
    evaluate_synthetic_timing_signal,
    run_control_4_buy_and_hold_replication,
    run_control_5_cost_monotonicity,
    run_control_6_timing,
    run_control_7_accounting,
    run_control_8_report_vs_code,
)


class TestEngineControls(unittest.TestCase):
    """Standing engine control tests."""

    def test_control_1_random_signal_btc(self):
        """Control 1: Random signal baseline on BTCUSDT (mean <= 0 and close to -costs)."""
        res = run_control_1_random_signal(symbol="BTCUSDT", n_repeats=300, max_bars=8000)
        self.assertEqual(res.status, "PASS", f"Control 1 BTC failed: {res.diagnostics}")

    def test_control_1_random_signal_eth(self):
        """Control 1: Random signal baseline on ETHUSDT (mean <= 0 and close to -costs)."""
        res = run_control_1_random_signal(symbol="ETHUSDT", n_repeats=300, max_bars=8000)
        self.assertEqual(res.status, "PASS", f"Control 1 ETH failed: {res.diagnostics}")

    def test_control_2_perfect_foresight_btc(self):
        """Control 2: Perfect foresight lookahead on BTCUSDT produces massive profit."""
        res = run_control_2_perfect_foresight(symbol="BTCUSDT", max_bars=3000)
        self.assertEqual(res.status, "PASS", f"Control 2 BTC failed: {res.diagnostics}")

    def test_control_2_perfect_foresight_eth(self):
        """Control 2: Perfect foresight lookahead on ETHUSDT produces massive profit."""
        res = run_control_2_perfect_foresight(symbol="ETHUSDT", max_bars=3000)
        self.assertEqual(res.status, "PASS", f"Control 2 ETH failed: {res.diagnostics}")

    def test_control_3_synthetic_with_timing_skill_passes(self):
        """Control 3: Synthetic signal with true timing skill must pass tightened shifted-date test."""
        res = evaluate_synthetic_timing_signal(has_skill=True, n_bars=3000, n_shifts=50, seed=42)
        self.assertEqual(res.status, "PASS", f"Synthetic timing skill should pass: {res.diagnostics}")

    def test_control_3_synthetic_without_timing_skill_flags_no_timing_evidence(self):
        """Control 3: Synthetic signal without timing skill must produce NO TIMING EVIDENCE."""
        res = evaluate_synthetic_timing_signal(has_skill=False, n_bars=3000, n_shifts=50, seed=42)
        self.assertEqual(res.status, "NO TIMING EVIDENCE", f"Synthetic no-skill should flag NO TIMING EVIDENCE: {res.diagnostics}")

    def test_control_3_shifted_date_btc_f0(self):
        """Control 3: Shifted-date control on BTCUSDT F0 satisfies tightened timing criteria."""
        res = run_control_3_shifted_date(symbol="BTCUSDT", strategy="F0", n_shifts=50, seed=42)
        self.assertEqual(res.status, "PASS", f"BTC F0 expected to pass: {res.diagnostics}")

    def test_control_3_shifted_date_btc_g0_finding(self):
        """Control 3: Shifted-date control on BTCUSDT G0 flags NO TIMING EVIDENCE (empirical finding)."""
        res = run_control_3_shifted_date(symbol="BTCUSDT", strategy="G0", n_shifts=50, seed=42)
        self.assertEqual(res.status, "NO TIMING EVIDENCE", f"BTC G0 expected to flag NO TIMING EVIDENCE: {res.diagnostics}")

    def test_control_4_buy_and_hold_replication_btc(self):
        """Control 4: Buy-and-hold matches analytical formula within 0.1% on BTCUSDT."""
        res = run_control_4_buy_and_hold_replication(symbol="BTCUSDT", max_bars=8000)
        self.assertEqual(res.status, "PASS", f"Control 4 BTC failed: {res.diagnostics}")

    def test_control_4_buy_and_hold_replication_eth(self):
        """Control 4: Buy-and-hold matches analytical formula within 0.1% on ETHUSDT."""
        res = run_control_4_buy_and_hold_replication(symbol="ETHUSDT", max_bars=8000)
        self.assertEqual(res.status, "PASS", f"Control 4 ETH failed: {res.diagnostics}")

    def test_control_5_cost_monotonicity_btc(self):
        """Control 5: Performance strictly decreases from 0 -> 10 -> 20 bps on BTCUSDT."""
        res = run_control_5_cost_monotonicity(symbol="BTCUSDT", max_bars=6000)
        self.assertEqual(res.status, "PASS", f"Control 5 BTC failed: {res.diagnostics}")

    def test_control_5_cost_monotonicity_eth(self):
        """Control 5: Performance strictly decreases from 0 -> 10 -> 20 bps on ETHUSDT."""
        res = run_control_5_cost_monotonicity(symbol="ETHUSDT", max_bars=6000)
        self.assertEqual(res.status, "PASS", f"Control 5 ETH failed: {res.diagnostics}")

    def test_control_6_timing_next_open_vs_bar_close_btc(self):
        """Control 6: Fill at next open vs bar close changes result on BTCUSDT."""
        res = run_control_6_timing(symbol="BTCUSDT", max_bars=4000)
        self.assertEqual(res.status, "PASS", f"Control 6 BTC failed: {res.diagnostics}")

    def test_control_6_timing_next_open_vs_bar_close_eth(self):
        """Control 6: Fill at next open vs bar close changes result on ETHUSDT."""
        res = run_control_6_timing(symbol="ETHUSDT", max_bars=4000)
        self.assertEqual(res.status, "PASS", f"Control 6 ETH failed: {res.diagnostics}")

    def test_control_7_accounting_exact_cent_btc(self):
        """Control 7: Sum of trade PnLs equals delta equity to the cent on BTCUSDT."""
        res = run_control_7_accounting(symbol="BTCUSDT", max_bars=8000)
        self.assertEqual(res.status, "PASS", f"Control 7 BTC failed: {res.diagnostics}")

    def test_control_7_accounting_exact_cent_eth(self):
        """Control 7: Sum of trade PnLs equals delta equity to the cent on ETHUSDT."""
        res = run_control_7_accounting(symbol="ETHUSDT", max_bars=8000)
        self.assertEqual(res.status, "PASS", f"Control 7 ETH failed: {res.diagnostics}")

    def test_control_8_report_vs_code_zero_discrepancies(self):
        """Control 8: Recomputed metrics across research reports match saved JSON results."""
        res = run_control_8_report_vs_code()
        self.assertEqual(res.status, "PASS", f"Control 8 failed: {res.diagnostics}")


if __name__ == "__main__":
    unittest.main()
