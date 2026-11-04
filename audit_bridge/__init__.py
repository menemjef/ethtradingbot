"""
audit_bridge — Lightweight development audit and observability system.

Provides structured logging, change tracking, and backtest reporting
for external AI auditor inspection. Completely separate from trading logic.

Usage:
    from audit_bridge import log_activity, log_backtest, log_test_result, update_status, snapshot_changes
"""

from audit_bridge.logger import log_activity
from audit_bridge.backtest_reporter import log_backtest
from audit_bridge.test_reporter import log_test_result
from audit_bridge.status import update_status
from audit_bridge.tracker import snapshot_changes

__all__ = [
    "log_activity",
    "log_backtest",
    "log_test_result",
    "update_status",
    "snapshot_changes",
]
