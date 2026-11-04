"""
audit_bridge/inspector.py — Read-only inspection utilities for external auditors.

Provides safe, read-only access to project artifacts without exposing
secrets, trading controls, or filesystem traversal.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BRIDGE_DIR.parent

# Directories an auditor is allowed to read from
_ALLOWED_DIRS = {
    PROJECT_ROOT / "audit_bridge",
    PROJECT_ROOT / "model",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "live",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "utils",
}

# File extensions an auditor is allowed to read
_ALLOWED_EXTENSIONS = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".toml", ".cfg", ".ini"}

# Files explicitly blocked
_BLOCKED_FILES = {".env", "ethbotmk2.pem", ".pem"}
_BLOCKED_PATTERNS = {"api_key", "secret", "password", "credential", "token"}


def _is_safe_path(filepath: str) -> bool:
    """Check if a file path is safe for auditor access."""
    p = Path(filepath).resolve()
    name_lower = p.name.lower()

    # Block known secret files
    if name_lower in _BLOCKED_FILES or p.suffix in _BLOCKED_FILES:
        return False

    # Block files with secret-like names
    for pattern in _BLOCKED_PATTERNS:
        if pattern in name_lower:
            return False

    # Must be within project root
    try:
        p.relative_to(PROJECT_ROOT)
    except ValueError:
        return False

    # Check extension
    if p.suffix.lower() not in _ALLOWED_EXTENSIONS:
        return False

    # Block checkpoints directory (binary files)
    if "checkpoints" in p.parts and p.suffix not in {".json", ".txt", ".md"}:
        return False

    return True


def read_source_file(filepath: str) -> dict:
    """
    Read a project source file safely.

    Returns:
        {"path": str, "content": str, "size": int, "lines": int}
        or {"error": str} if access is denied.
    """
    if not _is_safe_path(filepath):
        return {"error": f"Access denied: {filepath}"}

    p = Path(filepath)
    if not p.exists():
        return {"error": f"File not found: {filepath}"}

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        return {
            "path": str(p),
            "content": content,
            "size": p.stat().st_size,
            "lines": content.count("\n") + 1,
        }
    except OSError as e:
        return {"error": str(e)}


def get_activity_log(last_n: int = 50) -> list[dict]:
    """Read the last N activity log entries."""
    log_path = BRIDGE_DIR / "agent_activity.jsonl"
    if not log_path.exists():
        return []

    entries = []
    try:
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines[-last_n:]:
            if line.strip():
                entries.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass

    return entries


def get_latest_backtest() -> dict:
    """Read the latest backtest report."""
    path = BRIDGE_DIR / "latest_backtest.json"
    if not path.exists():
        return {"error": "No backtest report available"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"error": str(e)}


def get_latest_tests() -> list[dict]:
    """Read the latest test results."""
    path = BRIDGE_DIR / "latest_tests.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, OSError):
        return []


def get_latest_status() -> dict:
    """Read the latest project status."""
    path = BRIDGE_DIR / "latest_status.json"
    if not path.exists():
        return {"error": "No status available"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"error": str(e)}


def get_change_history(last_n: int = 20) -> list[dict]:
    """Read the last N change snapshots."""
    path = BRIDGE_DIR / "change_history.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data[-last_n:] if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def get_audit_index() -> dict:
    """
    Generate a concise index of all available audit artifacts.

    This is the entry point for an external auditor to understand
    what information is available.
    """
    def _file_info(p: Path) -> dict:
        if p.exists():
            stat = p.stat()
            return {
                "path": str(p.relative_to(PROJECT_ROOT)),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "available": True,
            }
        return {"path": str(p.relative_to(PROJECT_ROOT)), "available": False}

    # Count activity log entries
    activity_count = 0
    activity_path = BRIDGE_DIR / "agent_activity.jsonl"
    if activity_path.exists():
        activity_count = sum(1 for line in activity_path.read_text(encoding="utf-8").split("\n") if line.strip())

    # List report files
    reports_dir = BRIDGE_DIR / "reports"
    report_files = []
    if reports_dir.exists():
        report_files = sorted([f.name for f in reports_dir.iterdir() if f.suffix == ".json"])

    # List snapshot files
    snapshots_dir = BRIDGE_DIR / "snapshots"
    snapshot_files = []
    if snapshots_dir.exists():
        snapshot_files = sorted([f.name for f in snapshots_dir.iterdir()])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "artifacts": {
            "activity_log": {
                **_file_info(activity_path),
                "format": "JSON Lines",
                "entry_count": activity_count,
            },
            "latest_status": _file_info(BRIDGE_DIR / "latest_status.json"),
            "latest_backtest": _file_info(BRIDGE_DIR / "latest_backtest.json"),
            "latest_tests": _file_info(BRIDGE_DIR / "latest_tests.json"),
            "change_history": _file_info(BRIDGE_DIR / "change_history.json"),
            "readme": _file_info(BRIDGE_DIR / "README.md"),
        },
        "reports": report_files,
        "snapshots": snapshot_files,
        "source_directories": [
            str(d.relative_to(PROJECT_ROOT))
            for d in sorted(_ALLOWED_DIRS)
            if d.exists()
        ],
    }
