# Audit Bridge V2 — Development Observability & Independent Verification System

A lightweight, strictly read-only audit and observability layer for the ETH trading bot project. Designed to allow an independent external AI auditor (via Model Context Protocol, local Python inspection, or GitHub) to inspect development milestones, code modifications, test execution, backtest results, and engineering decisions.

---

## Architecture Overview

```
trading bot/
├── .agents/
│   └── hooks.json               # Native Antigravity lifecycle hook configuration
├── audit_bridge/
│   ├── __init__.py              # Public audit bridge package API
│   ├── sanitizer.py             # Secret detection, multi-pattern redaction & publication safety gate
│   ├── metrics_validator.py     # Mathematical validation & reconciliation of backtest metrics
│   ├── logger.py                # Activity logging (JSONL) with automated sanitization & publication safety
│   ├── tracker.py               # Git commit history, diffs, and working tree tracking
│   ├── backtest_reporter.py     # Validated backtest reporting with trade ledgers & equity curves
│   ├── test_reporter.py         # Test execution result recording
│   ├── status.py                # Project phase, model checkpoint SHA-256, and bot state
│   ├── inspector.py             # Read-only inspection utilities for external auditors
│   ├── hook_handler.py          # Native hook event handler (stdin/stdout contract)
│   ├── runner.py                # Standalone CLI execution wrapper (`python -m audit_bridge.runner <cmd>`)
│   ├── mcp_server.py            # Standard Model Context Protocol (MCP) JSON-RPC 2.0 stdio server
│   ├── README.md                # This document
│   ├── agent_activity.jsonl     # Append-only activity log
│   ├── latest_status.json       # Latest project status
│   ├── latest_backtest.json     # Most recent validated backtest report
│   ├── latest_tests.json        # Test execution history
│   ├── change_history.json      # Git change snapshots
│   ├── reports/                 # Archived backtest reports, trade ledgers, equity curves
│   └── snapshots/               # Configuration snapshots (binaries excluded)
```

---

## Priority 1: Secret Redaction & Publication Safety

The sanitizer (`audit_bridge/sanitizer.py`) automatically strips sensitive data:

| Category | Pattern Detected | Redaction Result |
|---|---|---|
| **Bearer Tokens** | `Bearer <token>` | `Bearer [REDACTED]` |
| **Authorization Headers** | `Authorization: Bearer/Basic <token>` | `Authorization: Bearer [REDACTED]` |
| **JSON Web Tokens (JWT)** | `eyJ...` (three base64url segments) | `[REDACTED_JWT]` |
| **API Keys & Secrets** | `BINANCE_API_KEY=xxx`, `API_SECRET=xxx` | `BINANCE_API_KEY=[REDACTED]` |
| **Telegram Tokens** | `\d{8,12}:[A-Za-z0-9_-]{30,45}` | `[REDACTED_TELEGRAM_TOKEN]` |
| **Passwords** | `PASSWORD=xxx`, `PASSWD: xxx` | `PASSWORD=[REDACTED]` |
| **Private Keys** | `-----BEGIN ... PRIVATE KEY-----` | `[REDACTED_PRIVATE_KEY]` |
| **Database URIs** | Connection strings (`postgres`, `mysql`, `mongodb`, `redis`) | Password masked as `[REDACTED]` |

### Publication Safety Gate
Before any artifact is written or published, `assert_safe_for_publication()` scans the payload. If an un-redacted secret is detected, it raises `PublicationSafetyError` and hard-blocks the write.

---

## Priority 2: Backtest Metric Validation & Reconciliation

Backtests cannot report fabricated or inconsistent summary numbers. The validator (`audit_bridge/metrics_validator.py`):

1. **Recomputes metrics independently from the trade ledger**:
   - `Gross Profit = sum(positive trade gross PnL)`
   - `Gross Loss = absolute sum(negative trade gross PnL)`
   - `Profit Factor = Gross Profit / Gross Loss`
   - `Win Rate (Gross) = gross wins / total trades`
   - `Win Rate (Net) = net wins after fees / total trades`
   - `Net PnL = Gross Profit - Gross Loss - Fees - Slippage - Funding`
   - `Final Capital = Initial Capital + Net PnL + Deposits/Withdrawals`
   - `Max Drawdown = peak-to-trough drop from equity curve`
2. **Discrepancy Gating**:
   If reported summary numbers do not match the trade ledger or violate mathematical invariants, the report is stamped:
   ```json
   "is_mathematically_valid": false,
   "validation_discrepancies": ["Discrepancy explanation..."]
   ```

### 99-Trade Backtest Reconciliation
In earlier audit reports, a 99-trade backtest reported 55 wins (55.6% WR), Avg Win +$1.24, Avg Loss -$1.03, but Net PnL was -$18.24.
- **Root cause:** `backtest_audit.py` counted `wins` based on **gross outcome before fees**, while `avg_win`, `avg_loss`, and `net_pnl` were calculated after commissions and slippage (0.12% round-trip). Trades with tiny gross gains turned negative after fees.
- **Resolution:** The audit bridge now records both gross and net metrics explicitly from the ledger, preventing misleading win-rate interpretations.

---

## Priority 3: Real Automatic Activity Capture

Activity is captured through two complementary mechanisms:

### 1. Native Antigravity Lifecycle Hook (`.agents/hooks.json`)
Antigravity automatically runs `audit_bridge/hook_handler.py` on tool executions (`run_command`, `write_file`, `edit_file`) and lifecycle completions. It captures command lines, files modified, exit codes, git commit, and execution timestamps without manual logging.

### 2. Standalone Command Runner (`audit_bridge/runner.py`)
For manual or automated CLI workflows outside Antigravity:
```bash
python -m audit_bridge.runner pytest tests/
python -m audit_bridge.runner python backtest_audit.py
```
Captures command, duration, stdout/stderr, working tree diffs, exit codes, and test pass/fail status.

---

## Priority 4: Backtest Provenance Linkage

Every backtest report generated by `log_backtest` cryptographically links:
- `git_commit`: Current commit hash (`git rev-parse HEAD`)
- `model_checkpoint_sha256`: SHA-256 hash of `checkpoints/best_model.pt`
- `configuration_hash`: SHA-256 hash of all active parameters in `config.py`
- `dataset_identifier`: Symbol and timeframe (`ETH/USDT:USDT 1h`)
- `dataset_date_range`: Start and end timestamps
- `command`: Command executed to run the backtest
- `trade_ledger`: Stored in `audit_bridge/reports/<ID>_trades.json`
- `equity_curve`: Stored in `audit_bridge/reports/<ID>_equity.json`

---

## Priority 5: Independent Access Tiers

| Tier | Mechanism | How It Works | Current Status |
|---|---|---|---|
| **Tier 1: Local Python Inspection** | `audit_bridge.inspector` | Direct read-only module imports in Python | ✅ **Active & Tested** |
| **Tier 2: Private GitHub Repository** | Git push | Commit code and audit logs to private repo | ✅ **Ready to push** |
| **Tier 3: Model Context Protocol (MCP)** | `audit_bridge/mcp_server.py` | Read-only JSON-RPC 2.0 stdio server | ✅ **Active & Tested** |
| **Tier 4: ChatGPT Connectivity** | OpenAI Action / Webhook | External HTTPS REST or MCP bridge | ⚠️ **Not yet connected** (requires hosted HTTPS endpoint or desktop MCP bridge) |

> [!IMPORTANT]
> Tier 4 (ChatGPT external access) is NOT yet connected to this local machine. No claim of live ChatGPT remote access is made until an authenticated endpoint or tunnel is explicitly configured and verified.

---

## Model Context Protocol (MCP) Setup

External AI auditors (such as Claude Desktop or other MCP-compatible clients) can connect to the bridge using stdio:

Add to MCP client configuration (e.g. `claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "trading-bot-audit": {
      "command": "python",
      "args": ["C:/Python Projects/trading bot/audit_bridge/mcp_server.py"]
    }
  }
}
```

### Exposed Tools (Strictly Read-Only)
- `get_audit_index`: Overview of all available artifacts, file sizes, and timestamps.
- `get_project_status`: Current phase, model checkpoint SHA-256, active mode.
- `get_latest_backtest`: Latest validated backtest with consistency checks.
- `get_activity_log`: Last N activity log records.
- `get_change_history`: Git commit snapshots and diffs.
- `read_source_file`: Safe, path-restricted reader (strictly blocks `.env`, `.pem`, and credentials).

Zero shell execution and zero trading operations are exposed.

---

## Running the Automated Test Suite

```bash
python tests/test_audit_bridge.py
```
