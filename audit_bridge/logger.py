"""
audit_bridge/logger.py — Structured activity logging in JSON Lines format.

Every meaningful development operation produces a record appended to agent_activity.jsonl.
Secrets are never logged. Publication safety is verified before writing.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from audit_bridge.sanitizer import sanitize_text, sanitize_data, assert_safe_for_publication

BRIDGE_DIR = Path(__file__).resolve().parent
ACTIVITY_LOG = BRIDGE_DIR / "agent_activity.jsonl"
_COUNTER_FILE = BRIDGE_DIR / ".task_counter"


def _next_task_id() -> str:
    """Generate a monotonically increasing task ID."""
    try:
        count = int(_COUNTER_FILE.read_text().strip()) if _COUNTER_FILE.exists() else 0
    except (ValueError, OSError):
        count = 0
    count += 1
    _COUNTER_FILE.write_text(str(count))
    return f"TASK-{count:04d}"


def log_activity(
    operation: str,
    description: str,
    *,
    task_id: str = None,
    files_affected: list[str] = None,
    command: str = None,
    exit_status: int = None,
    test_result: str = None,
    output: str = None,
    error: str = None,
    git_commit: str = None,
    explanation: str = None,
    metadata: dict = None,
) -> dict:
    """
    Append a structured activity record to agent_activity.jsonl.
    Sanitizes all fields and verifies publication safety before writing.
    """
    if task_id is None:
        task_id = _next_task_id()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "operation": operation,
        "description": sanitize_text(description),
        "files_affected": files_affected or [],
        "command": sanitize_text(command) if command else None,
        "exit_status": exit_status,
        "test_result": test_result,
        "output": sanitize_text(output[:2500]) if output else None,
        "error": sanitize_text(error[:2500]) if error else None,
        "git_commit": git_commit,
        "explanation": sanitize_text(explanation) if explanation else None,
    }

    if metadata:
        record["metadata"] = sanitize_data(metadata)

    # Enforce publication safety gating
    assert_safe_for_publication(record, context_name=f"Activity {task_id}")

    # Atomic append
    ACTIVITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVITY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    return record
