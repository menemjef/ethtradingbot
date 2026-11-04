"""
audit_bridge/test_reporter.py — Test result reporting.

Captures test execution results in structured JSON format.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
LATEST_TESTS = BRIDGE_DIR / "latest_tests.json"


def log_test_result(
    test_name: str,
    *,
    passed: bool,
    total: int = None,
    failures: int = None,
    errors: int = None,
    skipped: int = None,
    duration_seconds: float = None,
    command: str = None,
    output: str = None,
    error_output: str = None,
    test_details: list[dict] = None,
    git_commit: str = None,
) -> dict:
    """
    Save a test execution result.

    Args:
        test_name: Name/path of the test suite.
        passed: Overall pass/fail status.
        total: Total test count.
        failures: Number of failures.
        errors: Number of errors.
        skipped: Number of skipped tests.
        duration_seconds: Execution time.
        command: The command used to run tests.
        output: Stdout (truncated to 3000 chars).
        error_output: Stderr (truncated to 3000 chars).
        test_details: Per-test results [{name, passed, duration, error}].
        git_commit: Git commit at time of test.

    Returns:
        The result dict.
    """
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_name": test_name,
        "overall_result": "PASS" if passed else "FAIL",
        "total": total,
        "passed_count": (total or 0) - (failures or 0) - (errors or 0) - (skipped or 0),
        "failures": failures or 0,
        "errors": errors or 0,
        "skipped": skipped or 0,
        "duration_seconds": duration_seconds,
        "command": command,
        "output": output[:3000] if output else None,
        "error_output": error_output[:3000] if error_output else None,
        "test_details": test_details,
        "git_commit": git_commit,
    }

    # Load existing results and append
    existing = []
    if LATEST_TESTS.exists():
        try:
            existing = json.loads(LATEST_TESTS.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing = [existing]
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(result)

    # Keep last 50 test runs
    if len(existing) > 50:
        existing = existing[-50:]

    LATEST_TESTS.write_text(
        json.dumps(existing, indent=2, default=str),
        encoding="utf-8",
    )

    return result
