"""
research/scripts/run_logger.py

Append-only run logger for managing and auditing access to the Final test split.
Records every read of the Final split into data/run_log.jsonl with:
  - timestamp (ISO-8601 UTC)
  - experiment_name
  - git_commit
  - split ("Final")

If the Final split has already been logged for a given experiment_name, raises
PermissionError to enforce strict single-read holdout discipline.
"""

import os
import json
import subprocess
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RUN_LOG_PATH = os.path.join(REPO_ROOT, "data", "run_log.jsonl")


def get_current_git_commit(cwd: Optional[str] = None) -> str:
    """Retrieves current HEAD commit SHA via git."""
    target_cwd = cwd or REPO_ROOT
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=target_cwd,
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "UNKNOWN_COMMIT"


def check_final_split_already_logged(
    experiment_name: str,
    split: str = "Final",
    log_path: Optional[str] = None
) -> bool:
    """Returns True if the specified experiment has already logged a read of split."""
    target_log_path = log_path or DEFAULT_RUN_LOG_PATH
    if not os.path.exists(target_log_path):
        return False

    with open(target_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if (
                    entry.get("experiment_name") == experiment_name
                    and entry.get("split", "").lower() == split.lower()
                ):
                    return True
            except json.JSONDecodeError:
                continue
    return False


def log_final_split_read(
    experiment_name: str,
    split: str = "Final",
    log_path: Optional[str] = None,
    git_commit: Optional[str] = None
) -> Dict[str, Any]:
    """Records read of the Final split into the append-only JSONL log.

    If the Final split has already been logged for experiment_name, raises PermissionError.
    Returns the newly appended record dict.
    """
    target_log_path = log_path or DEFAULT_RUN_LOG_PATH

    # Check if already logged
    if check_final_split_already_logged(experiment_name, split=split, log_path=target_log_path):
        raise PermissionError(
            f"Final split has already been logged for experiment '{experiment_name}' in {target_log_path}. "
            "Re-execution or further reads of the Final split are strictly blocked."
        )

    os.makedirs(os.path.dirname(os.path.abspath(target_log_path)), exist_ok=True)

    if git_commit is None:
        git_commit = get_current_git_commit()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "git_commit": git_commit,
        "split": split,
    }

    with open(target_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def get_all_logged_runs(log_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns list of all logged runs from data/run_log.jsonl."""
    target_log_path = log_path or DEFAULT_RUN_LOG_PATH
    if not os.path.exists(target_log_path):
        return []

    runs = []
    with open(target_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs
