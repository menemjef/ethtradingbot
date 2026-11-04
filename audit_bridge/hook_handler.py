"""
audit_bridge/hook_handler.py — Native Antigravity lifecycle hook handler.

Invoked automatically by Antigravity during agent execution loops.
Receives event context on stdin (JSON), records the activity via audit_bridge.logger,
and outputs expected JSON on stdout.

Supported events:
- PostToolUse: captures run_command, write_file, edit_file tool steps, exit codes, and errors
- Stop: captures completion of an agent turn
"""

import sys
import json
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_bridge.logger import log_activity
from audit_bridge.tracker import get_current_commit


def handle_hook():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print("{}")
            return

        payload = json.loads(raw_input)
    except Exception as e:
        # Fallback empty JSON response
        print("{}")
        return

    try:
        commit = get_current_commit()

        # Case 1: PostToolUse event
        if "toolCall" in payload:
            tool_call = payload.get("toolCall", {})
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})
            error = payload.get("error")
            exit_status = 1 if error else 0

            operation = "TOOL_USE"
            description = f"Executed {tool_name}"
            files_affected = []
            command = None

            if tool_name == "run_command":
                operation = "COMMAND_EXEC"
                command = tool_args.get("CommandLine", "")
                summary = tool_args.get("toolSummary", "")
                description = f"Run: {summary or command}"
            elif tool_name in ("write_file", "edit_file"):
                operation = "CODE_CHANGE"
                file_path = tool_args.get("Path", "")
                if file_path:
                    files_affected = [file_path]
                description = f"{tool_name}: {file_path}"
            elif tool_name == "ask_question":
                operation = "USER_INTERACTION"
                description = "User consultation prompt"

            log_activity(
                operation=operation,
                description=description,
                files_affected=files_affected,
                command=command,
                exit_status=exit_status,
                error=str(error) if error else None,
                git_commit=commit,
                metadata={
                    "step_idx": payload.get("stepIdx"),
                    "conversation_id": payload.get("conversationId"),
                    "model_name": payload.get("modelName"),
                },
            )

        # Case 2: Stop event
        elif "terminationReason" in payload:
            reason = payload.get("terminationReason", "COMPLETED")
            log_activity(
                operation="AGENT_LIFECYCLE",
                description=f"Agent turn terminated: {reason}",
                git_commit=commit,
                metadata={"termination_reason": reason},
            )

    except Exception as e:
        # Avoid crashing the agent loop on logging failure
        pass

    # Expected stdout for hook contract
    print("{}")


if __name__ == "__main__":
    handle_hook()
