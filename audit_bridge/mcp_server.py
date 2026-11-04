"""
audit_bridge/mcp_server.py — Read-only Model Context Protocol (MCP) Server.

Provides a standard JSON-RPC 2.0 stdio MCP server for independent AI auditors (Claude Desktop, etc.).
Strictly read-only:
- Exposes only audit queries and safe file inspections
- NO shell execution
- NO write capabilities
- NO trading or exchange operations
- Path-restricted to allowed repository source folders
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_bridge.inspector import (
    get_audit_index,
    get_latest_status,
    get_latest_backtest,
    get_activity_log,
    read_source_file,
    get_change_history,
)

SERVER_INFO = {
    "name": "trading-bot-audit-bridge",
    "version": "2.0.0",
}

TOOLS = [
    {
        "name": "get_audit_index",
        "description": "Get index of all available audit artifacts, sizes, and timestamps.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_project_status",
        "description": "Get latest project status, phase, model checkpoint SHA256, and bot state.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_latest_backtest",
        "description": "Get latest validated backtest report including metrics and reconciliation.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_activity_log",
        "description": "Get recent development activity log entries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "last_n": {"type": "integer", "description": "Number of recent entries", "default": 20}
            },
        },
    },
    {
        "name": "get_change_history",
        "description": "Get git commit change snapshots and working tree status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "last_n": {"type": "integer", "description": "Number of recent commits", "default": 10}
            },
        },
    },
    {
        "name": "read_source_file",
        "description": "Safely read a permitted project source file (secrets and binaries blocked).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Relative path to file in repo"}
            },
            "required": ["filepath"],
        },
    },
]


def handle_request(req: dict) -> dict:
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})

        if name == "get_audit_index":
            content = get_audit_index()
        elif name == "get_project_status":
            content = get_latest_status()
        elif name == "get_latest_backtest":
            content = get_latest_backtest()
        elif name == "get_activity_log":
            content = get_activity_log(args.get("last_n", 20))
        elif name == "get_change_history":
            content = get_change_history(args.get("last_n", 10))
        elif name == "read_source_file":
            content = read_source_file(args.get("filepath", ""))
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method or tool '{name}' not found"},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(content, indent=2, default=str)}]
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unhandled method '{method}'"},
    }


def run_stdio_server():
    """Run JSON-RPC stdio loop."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
