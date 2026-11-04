"""
tests/test_audit_bridge.py — Comprehensive test suite for Audit Bridge V2.

Verifies:
1. Secret redaction across every category (Bearer tokens, JWTs, API keys, Binance credentials,
   Telegram tokens, passwords, private keys, database connection strings, Authorization headers).
2. Embedded secrets in logs, terminal outputs, and JSON payloads.
3. Publication safety gate (blocking artifacts with un-redacted secrets).
4. Mathematical validation & consistency checks for backtest metrics.
5. Independent calculation from trade ledger vs invalid manual metrics.
6. Execution capture via runner and hook handler.
7. Inspector read-only security restrictions.
8. Model Context Protocol (MCP) server JSON-RPC protocol compliance.
"""

import io
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_secret_redaction_all_categories():
    """Verify redaction for each required secret category."""
    from audit_bridge.sanitizer import sanitize_text

    # 1. Bearer tokens (standalone)
    s1 = sanitize_text("Authorization failed with Bearer secret_jwt_token_12345 in header")
    assert "Bearer [REDACTED]" in s1
    assert "secret_jwt_token_12345" not in s1

    # 2. Authorization header
    s2 = sanitize_text("Headers: Authorization: Bearer abcdef1234567890xyz")
    assert "abcdef1234567890xyz" not in s2
    assert "Authorization: Bearer [REDACTED]" in s2 or "[REDACTED]" in s2

    # 3. JWTs
    s3 = sanitize_text("User session: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c")
    assert "[REDACTED_JWT]" in s3
    assert "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c" not in s3

    # 4. API keys (generic and Binance)
    s4 = sanitize_text("Configuring with BINANCE_API_KEY=64charHexAlphanumericSecretKey1234567890abcdef1234567890abcdef and API_SECRET='mySuperSecretApiVal99'")
    assert "64charHexAlphanumericSecretKey1234567890abcdef1234567890abcdef" not in s4
    assert "mySuperSecretApiVal99" not in s4
    assert "[REDACTED]" in s4

    # 5. Telegram bot token
    s5 = sanitize_text("Alert sent via 123456789:ABCdefGHIjklMNOpqrSTUvwxYZ-1234567 to channel")
    assert "[REDACTED_TELEGRAM_TOKEN]" in s5
    assert "ABCdefGHIjklMNOpqrSTUvwxYZ" not in s5

    # 6. Passwords
    s6 = sanitize_text("Database login: PASSWORD=my_super_secret_password_123")
    assert "my_super_secret_password_123" not in s6
    assert "[REDACTED]" in s6

    # 7. Private keys (full multiline PEM blocks)
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Y1+examplePrivateKeystringcontentnotreal1234567890\n"
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890==\n"
        "-----END RSA PRIVATE KEY-----"
    )
    s7 = sanitize_text(f"Key loaded:\n{pem}\nProceeding...")
    assert "[REDACTED_PRIVATE_KEY]" in s7
    assert "examplePrivateKeystring" not in s7

    # 8. Database connection strings with embedded credentials
    s8 = sanitize_text("Connecting to postgresql://postgres:superSecretDbPass123@db.internal:5432/trading_db")
    assert "superSecretDbPass123" not in s8
    assert "postgresql://postgres:[REDACTED]@db.internal:5432/trading_db" in s8

    print("  [PASS] All secret categories cleanly redacted")


def test_embedded_secrets_in_json_and_logs():
    """Verify redaction inside nested dictionaries and formatted logs."""
    from audit_bridge.sanitizer import sanitize_data

    payload = {
        "status": "error",
        "env": {
            "BINANCE_API_KEY": "live_real_key_abc_123_xyz",
            "PORT": 8080,
            "DATABASE_URL": "mysql://app_user:dbSecretPassword456@10.0.0.5:3306/bot",
        },
        "headers": ["Authorization: Bearer secret_jwt_token_value_999"],
        "message": "Connection to wss://fstream.binance.com failed: TELEGRAM_TOKEN=987654321:AAHdefGHIjklMNOpqrSTUvwxYZ-1234567",
    }

    sanitized = sanitize_data(payload)

    # Check key-level redaction
    assert sanitized["env"]["BINANCE_API_KEY"] == "[REDACTED]"
    assert "live_real_key_abc_123_xyz" not in json.dumps(sanitized)
    assert "dbSecretPassword456" not in json.dumps(sanitized)
    assert "secret_jwt_token_value_999" not in json.dumps(sanitized)
    assert "AAHdefGHIjklMNOpqrSTUvwxYZ" not in json.dumps(sanitized)
    assert sanitized["env"]["PORT"] == 8080  # Non-secrets preserved

    print("  [PASS] Nested JSON and logs sanitized properly")


def test_publication_safety_gate():
    """Verify that artifacts containing unredacted secrets are blocked from publication."""
    from audit_bridge.sanitizer import assert_safe_for_publication, PublicationSafetyError

    safe_report = {
        "report_id": "BT-TEST-01",
        "git_commit": "f26020a",
        "results": {"net_pnl": 12.5, "win_rate": 55.0},
    }
    # Safe report passes
    assert assert_safe_for_publication(safe_report) is True

    unsafe_report = {
        "report_id": "BT-LEAK-01",
        "api_key": "BINANCE_API_KEY=leak_secret_value_1234567",
    }
    # Unsafe report must raise PublicationSafetyError
    try:
        assert_safe_for_publication(unsafe_report)
        assert False, "Should have raised PublicationSafetyError"
    except PublicationSafetyError:
        pass

    print("  [PASS] Publication safety gate correctly blocks leaked secrets")


def test_backtest_ledger_metrics_consistency():
    """Verify mathematical metrics computation from trade ledger."""
    from audit_bridge.metrics_validator import compute_metrics_from_ledger, validate_backtest_report

    # Synthetic ledger with 4 trades: 2 wins, 2 losses
    trades = [
        {"gross_pnl": 2.0, "fee": 0.1, "net_pnl": 1.9},   # Win
        {"gross_pnl": 3.0, "fee": 0.2, "net_pnl": 2.8},   # Win
        {"gross_pnl": -1.5, "fee": 0.1, "net_pnl": -1.6}, # Loss
        {"gross_pnl": -1.0, "fee": 0.1, "net_pnl": -1.1}, # Loss
    ]
    initial_cap = 50.0

    metrics = compute_metrics_from_ledger(trades, initial_capital=initial_cap)

    assert metrics["num_trades"] == 4
    assert metrics["gross_profit"] == 5.0  # 2.0 + 3.0
    assert metrics["gross_loss"] == 2.5    # 1.5 + 1.0
    assert metrics["profit_factor_gross"] == 2.0  # 5.0 / 2.5
    assert metrics["win_rate_gross"] == 50.0
    assert metrics["total_fees"] == 0.5    # 0.1 + 0.2 + 0.1 + 0.1
    assert round(metrics["net_pnl"], 2) == 2.0  # (1.9 + 2.8 - 1.6 - 1.1) = 2.0
    assert round(metrics["final_capital"], 2) == 52.0  # 50 + 2.0

    # Validate against valid claimed metrics
    claimed = {
        "initial_capital": 50.0,
        "final_capital": 52.0,
        "num_trades": 4,
        "win_rate": 50.0,
        "profit_factor": 2.0,
    }
    res = validate_backtest_report(claimed, trades=trades)
    assert res.is_valid is True
    assert len(res.discrepancies) == 0

    # Inconsistent claimed metrics must fail validation
    bad_claimed = {
        "initial_capital": 50.0,
        "final_capital": 65.0,  # False capital
        "num_trades": 4,
        "win_rate": 90.0,       # False win rate
        "profit_factor": 10.0,  # False PF
    }
    bad_res = validate_backtest_report(bad_claimed, trades=trades)
    assert bad_res.is_valid is False
    assert len(bad_res.discrepancies) >= 2

    print("  [PASS] Backtest metric validation & consistency checks passed")


def test_99_trade_reconciliation_explanation():
    """Verify the formal reconciliation for the 99-trade audit."""
    from audit_bridge.metrics_validator import reconcile_backtest_audit_99_explanation

    reconciliation = reconcile_backtest_audit_99_explanation()
    assert reconciliation["validation_status"] == "EXPLAINED_AND_RECONCILED"
    assert "gross" in reconciliation["root_cause_of_discrepancy"].lower()
    assert "net" in reconciliation["root_cause_of_discrepancy"].lower()

    print("  [PASS] 99-trade audit reconciliation verified")


def test_command_runner_execution_capture():
    """Verify execution wrapper captures commands, exit codes, and output."""
    from audit_bridge.runner import run_command_with_audit
    from audit_bridge.inspector import get_activity_log

    code = run_command_with_audit(["python", "-c", "print('RUNNER_TEST_OUTPUT'); exit(0)"])
    assert code == 0

    log = get_activity_log(last_n=3)
    matching = [x for x in log if "RUNNER_TEST_OUTPUT" in (x.get("output") or "")]
    assert len(matching) > 0
    assert matching[0]["exit_status"] == 0

    print("  [PASS] Standalone command runner execution capture passed")


def test_hook_handler_simulation():
    """Verify that hook_handler processes Antigravity lifecycle JSON on stdin and outputs {}."""
    import subprocess

    hook_payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "python test_sim.py", "toolSummary": "Simulated tool step"},
        },
        "stepIdx": 42,
        "conversationId": "sim-conv-123",
        "error": None,
    }

    proc = subprocess.run(
        [sys.executable, "audit_bridge/hook_handler.py"],
        input=json.dumps(hook_payload),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0
    assert proc.stdout.strip() == "{}"

    # Verify activity was recorded
    from audit_bridge.inspector import get_activity_log
    log = get_activity_log(last_n=3)
    matching = [x for x in log if "Simulated tool step" in (x.get("description") or "")]
    assert len(matching) > 0

    print("  [PASS] Native hook handler simulation passed")


def test_mcp_server_protocol():
    """Verify MCP JSON-RPC protocol compliance."""
    from audit_bridge.mcp_server import handle_request

    # Test initialize
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    init_resp = handle_request(init_req)
    assert init_resp["result"]["serverInfo"]["name"] == "trading-bot-audit-bridge"

    # Test tools/list
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    list_resp = handle_request(list_req)
    tool_names = [t["name"] for t in list_resp["result"]["tools"]]
    assert "get_audit_index" in tool_names
    assert "get_latest_backtest" in tool_names
    assert "read_source_file" in tool_names

    # Test tools/call (get_audit_index)
    call_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "get_audit_index", "arguments": {}},
    }
    call_resp = handle_request(call_req)
    parsed = json.loads(call_resp["result"]["content"][0]["text"])
    assert "artifacts" in parsed

    # Test read_source_file blocking secrets
    sec_call = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "read_source_file", "arguments": {"filepath": ".env"}},
    }
    sec_resp = handle_request(sec_call)
    sec_content = json.loads(sec_resp["result"]["content"][0]["text"])
    assert "error" in sec_content

    print("  [PASS] MCP server JSON-RPC protocol compliant & path-restricted")


if __name__ == "__main__":
    print("=" * 60)
    print("  AUDIT BRIDGE V2 COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    tests = [
        test_secret_redaction_all_categories,
        test_embedded_secrets_in_json_and_logs,
        test_publication_safety_gate,
        test_backtest_ledger_metrics_consistency,
        test_99_trade_reconciliation_explanation,
        test_command_runner_execution_capture,
        test_hook_handler_simulation,
        test_mcp_server_protocol,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
