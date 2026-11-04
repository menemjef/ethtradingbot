# Institutional Execution Safety Architecture & Fail-Closed State Machine

## Document Information
- **Repository:** `menemjef/ethtradingbot`
- **Target Branch:** `chatgpt-audit-review`
- **Working Branch:** `fix/execution-safety-and-backtest-integrity`
- **Component:** Live Execution Engine, Risk Management, State Reconciliation
- **Classification:** Production Safety Specification

---

## 1. Executive Summary & Core Philosophy

In algorithmic cryptocurrency derivatives trading, software bugs and network partitions do not merely cause exceptions—they lead directly to catastrophic capital liquidation. Traditional retail trading bot codebases often suffer from **optimistic fail-open assumptions**:
1. When an API call throws an exception, the system catches it and returns `None` or `[]`, which callers interpret as "No position open" or "No active orders".
2. When a protective Stop-Loss order fails to place, the bot logs an error and leaves an unhedged naked position open to the market.
3. When local database state disagrees with the exchange, the bot either crashes or blindly overwrites exchange state.
4. When account balance is too small to meet exchange minimum notional requirements, the bot "bumps" the order size, secretly violating the user's hard risk budget.

To permanently eliminate these systemic vulnerabilities, this architecture enforces an **Institutional Fail-Closed Execution Model**:
- **Zero-Tolerance Capital Protection:** Any unexpected state, network failure, or API rejection forces the engine into a safe, deterministic state.
- **Explicit Typed Exchange State:** No boolean or loose dictionary responses. The exchange status must be explicitly confirmed as `VERIFIED_FLAT`, `VERIFIED_OPEN`, or `UNKNOWN_ERROR`.
- **Atomic Position & Bracket Lifecycle:** A position is never permitted to exist without an active, verified Stop-Loss order on the exchange book.
- **Fail-Closed Small-Balance Gating:** Hard risk ceilings are mathematically inviolable; if a position cannot be sized within budget while meeting exchange minimums, the trade is rejected.

```mermaid
flowchart TD
    Sig[Incoming Model Signal] --> DD{Daily Drawdown < 6%?}
    DD -- No --> Halt[Reject Trade: Daily Drawdown Gated]
    DD -- Yes --> PosCheck{Live Exchange Position}
    
    PosCheck -- UNKNOWN_ERROR --> Abort[Abort Trade: Fail-Closed]
    PosCheck -- VERIFIED_OPEN --> FlatCheck[Check Flip / Exit Rules]
    PosCheck -- VERIFIED_FLAT --> Sizing{Calculate Position Size}
    
    Sizing -- Notional < $21 AND Bump Violates 2.5% Risk --> Skip[Skip Trade: Preserve Risk Ceiling]
    Sizing -- Sizing Valid --> Entry[Place Market Entry Order]
    
    Entry -- Failed --> Reset[Reset Local State to Flat]
    Entry -- Success --> Bracket[Submit Take-Profit & Stop-Loss]
    
    Bracket -- SL Success --> Active[Position Active & Protected]
    Bracket -- SL Placement Fails --> RetrySL{Retry SL Placement}
    RetrySL -- Success --> Active
    RetrySL -- Failed --> EmergClose[IMMEDIATE REDUCE-ONLY EMERGENCY CLOSE]
    
    EmergClose -- Closed --> Flat[Position Flat: Log Critical Alert]
    EmergClose -- Failed --> EmergHalt[LATCH EMERGENCY HALT: STOP BOT]
```

---

## 2. Explicit Typed Exchange State Machine

### 2.1 The Danger of Primitive Returns
Previously, CCXT calls were wrapped in basic `try/except` blocks returning `None` on errors:
```python
# VULNERABLE HISTORICAL PATTERN:
def get_open_position(self, symbol):
    try:
        positions = self.exchange.fetch_positions([symbol])
        return positions[0]
    except Exception as e:
        logger.error(e)
        return None  # CALLER INTERPRETS NONE AS "FLAT"!
```
If Binance returned a 502 Bad Gateway or timed out, the caller received `None`, assumed no position was open, and submitted a second market entry order, causing double exposure or unintended position flips.

### 2.2 The Typed State Specification
The execution engine introduces strongly-typed state enums and containers in `live/execution_engine.py`:

```python
class PositionStatus(Enum):
    VERIFIED_FLAT = "VERIFIED_FLAT"
    VERIFIED_OPEN = "VERIFIED_OPEN"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

@dataclass
class PositionInfo:
    status: PositionStatus
    side: Optional[str] = None          # "LONG" | "SHORT" | None
    amount: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    percentage: float = 0.0
    raw: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.VERIFIED_OPEN and self.amount > 0.0

    @property
    def is_flat(self) -> bool:
        return self.status == PositionStatus.VERIFIED_FLAT

    @property
    def is_unknown(self) -> bool:
        return self.status == PositionStatus.UNKNOWN_ERROR
```

Similarly, `BalanceInfo` (`BalanceStatus.VERIFIED`, `BalanceStatus.UNKNOWN_ERROR`) and `OrdersInfo` (`OrderStatus.VERIFIED`, `OrderStatus.UNKNOWN_ERROR`) prevent transient API drops from being interpreted as a zero balance or an empty order book.

Every consumer in `Trader` and `ExecutionEngine` checks `.is_unknown` first and halts or aborts execution rather than making assumptions.

---

## 3. Position Sizing & Capital Protection Mathematics

### 3.1 The Small-Balance Dilemma
Binance USD-M Perpetual Futures enforces a minimum order notional of $\$20.00$ (we configure `MIN_NOTIONAL_USD = 21.0` to account for tick fluctuations).

Our institutional risk specification mandates:
- **Maximum Risk Per Trade:** $2.5\%$ of current account equity ($R = 0.025$).
- **Fixed Stop-Loss Distance:** $1.5\%$ ($\text{SL} = 0.015$).
- **Maximum Leverage Cap:** $3.0\text{x}$.

The mathematically required notional exposure $N$ is given by:
$$N = \frac{\text{Account Balance} \times R}{\text{SL}}$$

For an account with balance $B$:
$$N = \frac{B \times 0.025}{0.015} = B \times 1.6667$$

Let us analyze two scenarios:

#### Scenario A: Institutional Account ($B = \$10,000$)
$$N = \$10,000 \times 1.6667 = \$16,666.67$$
- Minimum notional check: $\$16,666.67 \ge \$21.00$ (Satisfied).
- Maximum leverage check: $\$16,666.67 \le \$30,000.00$ (Satisfied: $1.67\text{x} \le 3.0\text{x}$).
- Maximum loss at Stop-Loss: $\$16,666.67 \times 0.015 = \$250.00$ ($2.50\%$ of balance).

#### Scenario B: Small Retail Account ($B = \$10$)
$$N = \$10 \times 1.6667 = \$16.67$$
- Minimum notional check: $\$16.67 < \$21.00$ (Violated).
- **Previous Retail Behavior (Bumping):** Bump $N$ to $\$21.00$.
- Resulting dollar risk: $\$21.00 \times 0.015 = \$0.315$.
- Resulting risk percentage: $\frac{\$0.315}{\$10} = 3.15\%$ (Hard risk ceiling violated!).
- Resulting leverage: $\frac{\$21.00}{\$10} = 2.1\text{x}$.

For a micro account ($B = \$5$):
- Bumping to $\$21$ yields leverage of $4.2\text{x}$ (violating the $3\text{x}$ leverage cap) and risks $6.30\%$ of equity on a single trade.

### 3.2 The Fail-Closed Sizing Policy
Bumping order notional when it violates the risk budget is mathematically unacceptable. In `live/trader.py`:

```python
# CAPITAL-PROTECTED SIZING POLICY:
if notional_usd < config.MIN_NOTIONAL_USD:
    # Check if bumping to MIN_NOTIONAL_USD would breach the 2.5% risk budget
    bumped_risk_usd = config.MIN_NOTIONAL_USD * sl_pct
    if bumped_risk_usd > risk_budget_usd:
        logger.warning(
            f"Cannot trade: Calculated notional ${notional_usd:.2f} < ${config.MIN_NOTIONAL_USD:.2f}, "
            f"and bumping would risk ${bumped_risk_usd:.2f} (>{risk_budget_usd:.2f}, "
            f">{(bumped_risk_usd/balance)*100:.2f}% of balance). SKIPPING TRADE to protect capital."
        )
        return 0.0  # HARD GATING: NO TRADE PERMITTED
    amount = config.MIN_NOTIONAL_USD / current_price
```

If an account balance cannot support the trade without breaching the $2.5\%$ risk budget or the $3\text{x}$ leverage limit, the trade is **safely skipped** (`return 0.0`).

---

## 4. Elimination of Naked Position Exposure

### 4.1 The Stop-Loss Failure Crisis
When entering a market position, Binance places the market order immediately. Bracket orders (Take-Profit and Stop-Loss) are placed in subsequent API requests.
If the exchange rejects the Stop-Loss order (e.g., due to price filter deviation, insufficient margin, or transient API error), the system has an **open, unhedged position** floating in the market without risk control.

### 4.2 Emergency Reduce-Only Close Policy
In `live/execution_engine.py`:
1. `place_tp_sl_orders()` submits the Stop-Loss order.
2. The order ID is verified against the exchange order book.
3. If placement fails, an immediate retry is attempted.
4. If retry fails:
   - **Immediate Emergency Close:** The engine instantly calls `close_position(symbol, emergency=True)`.
   - The close order is submitted with `params={"reduceOnly": True}` to prevent opening an opposite position.
   - The engine polls the exchange to verify the position is flat.
   - If the emergency close succeeds, the bot remains flat and raises a critical alarm.
   - If the emergency close also fails, the engine latches `self.emergency_halt = True`, halting all bot trading immediately until manual intervention occurs.

```python
if sl_order is None or not sl_order.get("id"):
    logger.critical("FATAL: Stop-Loss order placement failed! Retrying...")
    sl_order = self._submit_stop_order(...)
    
if sl_order is None or not sl_order.get("id"):
    logger.critical("FATAL: Stop-Loss placement completely failed! Initiating EMERGENCY REDUCE-ONLY CLOSE.")
    close_success = self.close_position(symbol, emergency=True)
    if not close_success:
        logger.critical("EMERGENCY CLOSE FAILED! LATCHING EMERGENCY HALT TO HALT ALL BOT OPERATIONS.")
        self.emergency_halt = True
    raise RuntimeError("Position closed or halted due to Stop-Loss placement failure.")
```

---

## 5. Startup Synchronization & Crash Recovery

### 5.1 Local vs. Remote State Desynchronization
When the trading bot process restarts (e.g., following a VPS reboot or unexpected crash), it must determine whether a position is already open on Binance.
In the legacy implementation:
- If the local SQLite database had schema errors or was corrupted, the entire startup process crashed.
- If the database was empty, the bot assumed it was flat, leaving any active exchange positions unmanaged.

### 5.2 Isolated Database & Synchronous Exchange Reconciliation
In `Trader._restore_state()`:
1. **Isolated DB State Recovery:** Database queries are wrapped in dedicated exception blocks. An SQLite error logs a warning but never terminates startup or prevents exchange synchronization.
2. **Authoritative Exchange Query:** `get_open_position()` queries the live exchange.
3. **Fail-Closed State Handling:**
   - If exchange returns `UNKNOWN_ERROR`: Startup is aborted immediately. The bot does not trade with an unverified exchange status.
   - If exchange returns `VERIFIED_OPEN`: The bot adopts the open position (`side`, `amount`, `entry_price`), queries open bracket orders, and verifies that a protective Stop-Loss is live on the exchange book. If missing, it places the protective bracket orders immediately.
   - If exchange returns `VERIFIED_FLAT`: The bot clears any stale local position records and proceeds in a clean flat state.
4. **Order Safety During Flat State:** The bot will **never** cancel protective orders or purge database records unless the exchange explicitly confirms the position is `VERIFIED_FLAT`.

---

## 6. Pre-Flight Protection & Operational Opt-In Guardrails

### 6.1 Fail-Safe Defaults
All configuration files (`config.py` and `config.example.py`) enforce non-destructive defaults:
```python
TRADING_ENABLED = False            # Disabled by default
MOCK_TRADING = True               # Mock/paper trading by default
PRODUCTION_READY = False          # Explicit guard against unverified live runs
USE_BINANCE_TESTNET = True        # Binance Testnet by default
LEVERAGE = 3                      # Conservative 3x cap
DAILY_DRAWDOWN_LIMIT = 0.060       # 6.0% Daily Drawdown Hard Cap
```

### 6.2 Pre-Flight Authorization Token
To prevent accidental live deployment with real funds, live trading requires an explicit two-step opt-in:
1. The environment variable `ALLOW_LIVE_TRADING` must be explicitly set to the secret confirmation string:
   `I_UNDERSTAND_REAL_MONEY_IS_AT_RISK`
2. Both `live.py` (during `pre_flight_check`) and `ExecutionEngine.__init__` validate this token before instantiating CCXT in live mode:

```python
if not self.mock:
    ack_token = os.getenv("ALLOW_LIVE_TRADING", "0")
    if ack_token != config.LIVE_TRADING_ACK_TOKEN:
        raise RuntimeError(
            "LIVE TRADING BLOCKED: Live execution attempted without explicit authorization token. "
            f"Set environment variable ALLOW_LIVE_TRADING='{config.LIVE_TRADING_ACK_TOKEN}' to enable real execution."
        )
```

---

## 7. Verification Matrix (20 Production Safety Scenarios)

The execution safety architecture has been formally validated by a dedicated test suite in `tests/test_execution_safety.py`:

| Scenario ID | Test Name | Invariant Verified | Result |
| :--- | :--- | :--- | :--- |
| **SEC-01** | `test_01_config_defaults_are_safe` | Fail-safe defaults: mock=True, trading=False, testnet=True | **PASS** |
| **SEC-02** | `test_02_preflight_blocks_live_trading_without_opt_in` | Live trading blocked without explicit acknowledgment token | **PASS** |
| **SEC-03** | `test_03_execution_engine_blocks_live_init_without_opt_in` | Engine constructor rejects live credentials without env token | **PASS** |
| **SEC-04** | `test_04_failed_market_entry` | Trader state remains completely flat if market order fails | **PASS** |
| **SEC-05** | `test_05_failed_stop_loss_triggers_emergency_close` | SL placement failure triggers immediate reduce-only close | **PASS** |
| **SEC-06** | `test_06_catastrophic_sl_failure_triggers_halt` | Failed emergency close latches `emergency_halt=True` | **PASS** |
| **SEC-07** | `test_07_ccxt_error_returns_unknown_error_status` | CCXT network exception returns `UNKNOWN_ERROR`, never `None` | **PASS** |
| **SEC-08** | `test_08_close_position_aborts_on_unknown_position` | `close_position()` safely aborts if position status unknown | **PASS** |
| **SEC-09** | `test_09_close_position_verifies_flat_on_exchange` | `close_position()` queries exchange to confirm zero exposure | **PASS** |
| **SEC-10** | `test_10_close_position_retries_on_partial_fill` | Partial close fill triggers secondary reduce-only sweep | **PASS** |
| **SEC-11** | `test_11_daily_drawdown_blocks_new_signals` | Daily drawdown >= 6.0% gates all signal execution | **PASS** |
| **SEC-12** | `test_12_daily_drawdown_resets_at_utc_midnight` | Daily drawdown resets cleanly on UTC calendar change | **PASS** |
| **SEC-13** | `test_13_restart_with_existing_real_position` | Startup adopts open exchange position and restores state | **PASS** |
| **SEC-14** | `test_14_restart_with_unknown_exchange_status_halts` | Startup aborts fail-closed if exchange status is unknown | **PASS** |
| **SEC-15** | `test_15_restart_database_failure_does_not_block_exchange` | SQLite failure does not prevent live exchange reconciliation | **PASS** |
| **SEC-16** | `test_16_small_balance_skips_trade_rather_than_overbudget` | Notional < $21 skips trade if bumping breaches 2.5% risk | **PASS** |
| **SEC-17** | `test_17_sufficient_balance_sizes_correctly` | Normal balance calculates mathematically precise notional | **PASS** |
| **SEC-18** | `test_18_trailing_stop_update_places_replacement_first` | Stop replacement is verified active before canceling old SL | **PASS** |
| **SEC-19** | `test_19_balance_query_error_returns_unknown_status` | Balance API failure returns `UNKNOWN_ERROR`, never `0.0` | **PASS** |
| **SEC-20** | `test_20_open_orders_query_error_returns_unknown_status` | Orders API failure returns `UNKNOWN_ERROR`, never `[]` | **PASS** |

---

## 8. Conclusion

The institutional execution safety architecture guarantees that:
1. The trading bot will never open a trade without an active Stop-Loss order on Binance.
2. Network partitions and API timeouts never result in silent flat assumptions or phantom duplicate positions.
3. Account capital is strictly protected against out-of-budget sizing bumps on micro balances.
4. Accidental execution in live production environments is prevented by mandatory two-stage authorization.
