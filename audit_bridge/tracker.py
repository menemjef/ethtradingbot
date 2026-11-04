"""
audit_bridge/tracker.py — Git-based code change tracking.

Records commit history, diffs, and file-level changes.
Never commits secrets, checkpoints, or large binaries.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BRIDGE_DIR.parent
CHANGE_HISTORY = BRIDGE_DIR / "change_history.json"


def _run_git(*args: str, cwd: str = None) -> tuple[int, str]:
    """Run a git command and return (exit_code, stdout)."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=cwd or str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return -1, str(e)


def get_current_commit() -> str:
    """Get the current HEAD commit hash."""
    code, out = _run_git("rev-parse", "--short", "HEAD")
    return out if code == 0 else "unknown"


def get_recent_commits(n: int = 10) -> list[dict]:
    """Get the last N commits as structured records."""
    fmt = "%H|%h|%ai|%an|%s"
    code, out = _run_git("log", f"-{n}", f"--format={fmt}")
    if code != 0 or not out:
        return []

    commits = []
    for line in out.split("\n"):
        parts = line.split("|", 4)
        if len(parts) == 5:
            commits.append({
                "hash": parts[0],
                "short_hash": parts[1],
                "date": parts[2],
                "author": parts[3],
                "message": parts[4],
            })
    return commits


def get_changed_files(from_commit: str = "HEAD~1", to_commit: str = "HEAD") -> dict:
    """Get files changed between two commits."""
    code, out = _run_git("diff", "--name-status", from_commit, to_commit)
    if code != 0:
        return {"added": [], "modified": [], "deleted": []}

    added, modified, deleted = [], [], []
    for line in out.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, filepath = parts
        if status.startswith("A"):
            added.append(filepath)
        elif status.startswith("M"):
            modified.append(filepath)
        elif status.startswith("D"):
            deleted.append(filepath)

    return {"added": added, "modified": modified, "deleted": deleted}


def get_diff_stat(from_commit: str = "HEAD~1", to_commit: str = "HEAD") -> str:
    """Get a compact diff stat summary."""
    code, out = _run_git("diff", "--stat", from_commit, to_commit)
    return out if code == 0 else ""


def get_working_tree_status() -> dict:
    """Get current working tree status (staged, unstaged, untracked)."""
    code, out = _run_git("status", "--porcelain")
    if code != 0:
        return {"staged": [], "unstaged": [], "untracked": []}

    staged, unstaged, untracked = [], [], []
    for line in out.split("\n"):
        if len(line) < 3:
            continue
        idx_status = line[0]
        wt_status = line[1]
        filepath = line[3:]

        if idx_status == "?":
            untracked.append(filepath)
        elif idx_status != " ":
            staged.append(filepath)
        if wt_status != " " and wt_status != "?":
            unstaged.append(filepath)

    return {"staged": staged, "unstaged": unstaged, "untracked": untracked}


def snapshot_changes(
    description: str,
    from_commit: str = None,
    to_commit: str = "HEAD",
) -> dict:
    """
    Record a change snapshot to change_history.json.

    Args:
        description: Human-readable description of the change set.
        from_commit: Starting commit (default: HEAD~1).
        to_commit: Ending commit (default: HEAD).

    Returns:
        The snapshot record.
    """
    if from_commit is None:
        from_commit = "HEAD~1"

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "from_commit": from_commit,
        "to_commit": to_commit,
        "current_commit": get_current_commit(),
        "changed_files": get_changed_files(from_commit, to_commit),
        "diff_stat": get_diff_stat(from_commit, to_commit),
        "working_tree": get_working_tree_status(),
    }

    # Load existing history and append
    history = []
    if CHANGE_HISTORY.exists():
        try:
            history = json.loads(CHANGE_HISTORY.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []

    history.append(snapshot)

    # Keep last 100 snapshots
    if len(history) > 100:
        history = history[-100:]

    CHANGE_HISTORY.write_text(
        json.dumps(history, indent=2, default=str),
        encoding="utf-8",
    )

    return snapshot
