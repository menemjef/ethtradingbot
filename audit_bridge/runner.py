"""
audit_bridge/runner.py — Standalone command execution wrapper.

Allows executing shell commands with automatic audit capture:
- Captures command, start time, duration, exit code
- Captures stdout/stderr (sanitizing secrets)
- Captures git status / modified files before and after
- Records to agent_activity.jsonl
- If command is a test, also records to latest_tests.json

Usage:
    python -m audit_bridge.runner pytest tests/
    python -m audit_bridge.runner python backtest_audit.py
"""

import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_bridge.logger import log_activity
from audit_bridge.test_reporter import log_test_result
from audit_bridge.tracker import get_current_commit, get_working_tree_status


def run_command_with_audit(cmd_args: list[str]) -> int:
    if not cmd_args:
        print("Usage: python -m audit_bridge.runner <command> [args...]")
        return 1

    cmd_str = " ".join(cmd_args)
    commit = get_current_commit()
    wt_before = get_working_tree_status()

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd_args,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        exit_code = proc.returncode
        stdout_txt = proc.stdout
        stderr_txt = proc.stderr
    except Exception as e:
        exit_code = -1
        stdout_txt = ""
        stderr_txt = str(e)

    duration = time.time() - t0
    wt_after = get_working_tree_status()

    # Determine modified files
    modified_files = list(
        set(wt_after.get("unstaged", []) + wt_after.get("staged", []))
        - set(wt_before.get("unstaged", []) + wt_before.get("staged", []))
    )

    is_test = any("test" in arg.lower() for arg in cmd_args) or "pytest" in cmd_str

    # Print output to user console
    if stdout_txt:
        sys.stdout.write(stdout_txt)
    if stderr_txt:
        sys.stderr.write(stderr_txt)

    # 1. Log Activity
    log_activity(
        operation="COMMAND_RUN",
        description=f"Wrapped execution: {cmd_str[:120]}",
        files_affected=modified_files,
        command=cmd_str,
        exit_status=exit_code,
        output=stdout_txt[:2000],
        error=stderr_txt[:2000] if exit_code != 0 else None,
        git_commit=commit,
        explanation=f"Command exited with code {exit_code} in {duration:.2f}s",
        metadata={"duration_seconds": round(duration, 3)},
    )

    # 2. If test, also log to test_reporter
    if is_test:
        passed = (exit_code == 0)
        log_test_result(
            test_name=cmd_str,
            passed=passed,
            duration_seconds=round(duration, 3),
            command=cmd_str,
            output=stdout_txt,
            error_output=stderr_txt if not passed else None,
            git_commit=commit,
        )

    return exit_code


if __name__ == "__main__":
    code = run_command_with_audit(sys.argv[1:])
    sys.exit(code)
