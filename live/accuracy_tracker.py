"""
live/accuracy_tracker.py — Tracks prediction accuracy across live candle closes.

On each candle close:
  - Compares previous prediction direction with actual price movement
  - Updates win rate and cumulative stats
  - Appends a row to signals.csv
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import SIGNALS_CSV, SIGNAL_THRESHOLD

logger = logging.getLogger(__name__)

CSV_HEADERS = [
    "candle_close_utc",
    "close_price",
    "prediction",
    "direction",
    "threshold_met",
    "actual_log_return",
    "correct",
    "total_signals",
    "wins",
    "win_rate_pct",
]


@dataclass
class AccuracyTracker:
    """
    Stateful tracker for live prediction accuracy.

    Usage:
        tracker = AccuracyTracker()
        # After each candle close:
        tracker.update(close_time, prediction, previous_close_price, current_close_price)
    """

    total: int = 0
    wins: int = 0
    threshold_total: int = 0
    threshold_wins: int = 0

    # State carried across candles
    _prev_prediction: Optional[float] = field(default=None, repr=False)
    _prev_close: Optional[float] = field(default=None, repr=False)
    _prev_close_time: Optional[datetime] = field(default=None, repr=False)

    def __post_init__(self):
        self._ensure_csv()

    def _ensure_csv(self):
        """Create CSV with headers if it doesn't exist."""
        path = Path(SIGNALS_CSV)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()

    def record_signal(
        self,
        close_time: datetime,
        prediction: float,
        current_close: float,
    ):
        """
        Called at each candle close with the new prediction.

        - Evaluates the PREVIOUS prediction against the price move from prev_close → current_close.
        - Records the CURRENT prediction for future evaluation.
        """
        row: dict = {}

        if self._prev_prediction is not None and self._prev_close is not None:
            actual_log_return = _log_return(self._prev_close, current_close)
            pred_dir = _direction(self._prev_prediction)
            actual_dir = _direction(actual_log_return)
            correct = int(pred_dir == actual_dir)
            threshold_met = abs(self._prev_prediction) >= SIGNAL_THRESHOLD

            self.total += 1
            self.wins += correct

            if threshold_met:
                self.threshold_total += 1
                self.threshold_wins += correct

            win_rate = self.wins / self.total * 100 if self.total > 0 else 0.0
            thresh_rate = self.threshold_wins / self.threshold_total * 100 if self.threshold_total > 0 else 0.0

            row = {
                "candle_close_utc": self._prev_close_time.isoformat() if self._prev_close_time else "",
                "close_price": round(self._prev_close, 2) if self._prev_close else 0.0,
                "prediction": round(self._prev_prediction, 6),
                "direction": pred_dir,
                "threshold_met": int(threshold_met),
                "actual_log_return": round(actual_log_return, 6),
                "correct": correct,
                "total_signals": self.total,
                "wins": self.wins,
                "win_rate_pct": round(win_rate, 2),
            }

            self._append_csv(row)
            self._print_stats(row, thresh_rate)

        # Save current state for next candle
        self._prev_prediction = prediction
        self._prev_close = current_close
        self._prev_close_time = close_time

    def _append_csv(self, row: dict):
        with open(SIGNALS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row)

    def _print_stats(self, row: dict, thresh_rate: float):
        logger.info(
            "\n+-- RESULT -----------------------------------------------+\n"
            "  Prev Predicted: %+.6f  (%s)  |  Threshold: %s\n"
            "  Actual move:    %+.6f  (%s)\n"
            "  Correct:        %s\n"
            "  Cumulative win rate:       %.1f%%  (%d/%d signals)\n"
            "  Above-threshold win rate:  %.1f%%  (%d/%d signals)\n"
            "+---------------------------------------------------------+",
            row["prediction"],
            row["direction"],
            "[YES]" if row["threshold_met"] else "[NO]",
            row["actual_log_return"],
            _direction(row["actual_log_return"]),
            "[WIN]" if row["correct"] else "[LOSS]",
            row["win_rate_pct"],
            self.wins,
            self.total,
            thresh_rate,
            self.threshold_wins,
            self.threshold_total,
        )

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0

    @property
    def threshold_win_rate(self) -> float:
        return self.threshold_wins / self.threshold_total if self.threshold_total > 0 else 0.0


def _log_return(prev: float, curr: float) -> float:
    import math
    if prev <= 0 or curr <= 0:
        return 0.0
    return math.log(curr / prev)


def _direction(value: float) -> str:
    if value > 0:
        return "LONG"
    elif value < 0:
        return "SHORT"
    return "FLAT"
