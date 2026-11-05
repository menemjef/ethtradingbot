"""
paper/run_paper_system.py — CLI Supervisor for Live Paper-Trading & Data Ingestion.

Features:
- Single-instance protection via reliable PID lockfile.
- Functional graceful shutdown (SIGINT/SIGTERM handling and active `stop` command).
- Real-time status reporting with genuine event freshness, stream connection counts,
  completed 1m and 4H candle completeness verification (240/240 bars), funding and OI health,
  multi-strategy virtual positions, and portfolio risk metrics.
"""

import os
import sys
import time
import json
import sqlite3
import signal
import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.config import (
    MARKET_DB_PATH,
    PAPER_DB_PATH,
    DAEMON_PID_PATH,
    DEFAULT_SYMBOLS,
    MAX_EVENT_STALE_SECONDS,
    MAX_KLINE_STALE_SECONDS,
    MAX_QUOTE_AGE_SECONDS
)
from paper.collector import MarketDataCollector
from paper.execution import PaperExecutionEngine
from paper.strategy_runner import CandidateAPaperRunner, CandleGapError

logger = logging.getLogger("paper.supervisor")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] paper.supervisor: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Single-Instance & PID Lifecycle Management
# =============================================================================

def is_pid_alive(pid: int) -> bool:
    """Checks whether a given process ID is actively running."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, Exception):
        return False


def acquire_pid_lock(pid_path: str = DAEMON_PID_PATH) -> bool:
    """Acquires PID lock. Returns True if acquired, False if another instance is already running."""
    os.makedirs(os.path.dirname(os.path.abspath(pid_path)), exist_ok=True)
    if os.path.exists(pid_path):
        try:
            with open(pid_path, "r") as f:
                content = f.read().strip()
                if content:
                    existing_pid = int(content)
                    if is_pid_alive(existing_pid):
                        logger.error(f"Another paper daemon instance is already running with PID {existing_pid}. Aborting start.")
                        return False
                    else:
                        logger.warning(f"Found stale PID lockfile for PID {existing_pid} (process not running). Reclaiming lock.")
        except Exception as e:
            logger.warning(f"Could not read PID lockfile: {e}. Reclaiming lock.")

    # Write current process PID
    try:
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))
        return True
    except Exception as e:
        logger.error(f"Failed to write PID lockfile {pid_path}: {e}")
        return False


def release_pid_lock(pid_path: str = DAEMON_PID_PATH):
    """Safely removes PID lockfile."""
    if os.path.exists(pid_path):
        try:
            os.remove(pid_path)
            logger.info("Removed PID lockfile.")
        except Exception as e:
            logger.warning(f"Error removing PID lockfile {pid_path}: {e}")


# =============================================================================
# CLI Commands: status, inspect-signals, inspect-positions, stop
# =============================================================================

def cmd_status():
    """
    Prints comprehensive status report showing:
    - Real event freshness and connection state for MARKET and PUBLIC streams.
    - Latest completed 1-minute and 4-hour candles with 240/240 completeness check.
    - Open interest and realized funding health.
    - Portfolio metrics (cash, equity, peak, drawdown, gross/net leverage).
    - Multi-strategy virtual positions table.
    """
    print("=" * 80)
    print("ETH/USDT PERPETUAL FUTURES — PAPER TRADING & DATA INGESTION STATUS")
    print("=" * 80)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # 1. Daemon Process Status
    daemon_running = False
    daemon_pid = None
    if os.path.exists(DAEMON_PID_PATH):
        try:
            with open(DAEMON_PID_PATH, "r") as f:
                daemon_pid = int(f.read().strip())
                daemon_running = is_pid_alive(daemon_pid)
        except Exception:
            pass

    print("\n[Daemon Process]")
    if daemon_running:
        print(f"  Status:          RUNNING (PID: {daemon_pid})")
    else:
        print("  Status:          STOPPED (No active process)")

    # 2. Market Data Ingestion & Stream Freshness
    print("\n[Market Data Ingestion & Stream Freshness]")
    if os.path.exists(MARKET_DB_PATH):
        conn = sqlite3.connect(MARKET_DB_PATH)
        try:
            cur = conn.cursor()

            # Completed 1m klines
            cur.execute("""
            SELECT symbol, COUNT(*), MIN(open_time), MAX(close_time), MAX(receipt_ts)
            FROM completed_klines GROUP BY symbol
            """)
            k_rows = cur.fetchall()
            for sym, cnt, min_t, max_t, rec_t in k_rows:
                t_max_dt = datetime.fromtimestamp(max_t / 1000, tz=timezone.utc)
                age_sec = (now_ms - max_t) / 1000.0 if max_t else 999.0
                k_fresh = "FRESH" if age_sec <= MAX_KLINE_STALE_SECONDS else f"STALE ({age_sec:.0f}s)"
                print(f"  {sym} 1m Klines:  {cnt} bars | Latest: {t_max_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} [{k_fresh}]")

            # BookTicker Snapshots Freshness
            cur.execute("""
            SELECT symbol, COUNT(*), MAX(timestamp), MAX(receipt_ts), best_bid, best_ask, spread, queue_imbalance
            FROM book_ticker_snapshots GROUP BY symbol
            """)
            b_rows = cur.fetchall()
            for sym, b_cnt, b_ts, b_rec, bid, ask, sp, qi in b_rows:
                b_age = (now_ms - b_rec) / 1000.0 if b_rec else 999.0
                b_fresh = "FRESH" if b_age <= MAX_EVENT_STALE_SECONDS else f"STALE ({b_age:.0f}s)"
                print(f"  {sym} Quotes:     Bid: ${bid:.2f} | Ask: ${ask:.2f} | Spread: ${sp:.2f} | Imb: {qi:+.2f} [{b_fresh}]")

            # Open Interest Health
            cur.execute("""
            SELECT symbol, COUNT(*), MAX(timestamp), open_interest, open_interest_usd
            FROM open_interest GROUP BY symbol
            """)
            oi_rows = cur.fetchall()
            for sym, oi_cnt, oi_ts, oi_val, oi_usd in oi_rows:
                oi_dt = datetime.fromtimestamp(oi_ts / 1000, tz=timezone.utc)
                oi_age = (now_ms - oi_ts) / 1000.0 if oi_ts else 999.0
                oi_usd_str = f"${oi_usd:,.2f}" if (oi_usd is not None and oi_usd > 0) else "N/A"
                print(f"  {sym} Open Interest: {oi_val:,.2f} {sym[:3]} ({oi_usd_str}) | Last: {oi_dt.strftime('%H:%M:%S UTC')} ({oi_age/60:.1f}m ago)")

            # Realized Funding Health
            cur.execute("""
            SELECT symbol, COUNT(*), MAX(settlement_time), funding_rate
            FROM funding_settlements GROUP BY symbol
            """)
            f_rows = cur.fetchall()
            for sym, f_cnt, f_time, f_rate in f_rows:
                f_dt = datetime.fromtimestamp(f_time / 1000, tz=timezone.utc)
                print(f"  {sym} Funding:    {f_cnt} settlements | Latest Rate: {f_rate*100:+.4f}% at {f_dt.strftime('%Y-%m-%d %H:%M UTC')}")

        finally:
            conn.close()
    else:
        print(f"  Database not found at {MARKET_DB_PATH}")

    # 3. 4-Hour Candle Completeness Engine Audit
    print("\n[4-Hour Candle Strict Completeness Audit (240/240 Bars)]")
    runner_check = CandidateAPaperRunner(execution_engine=PaperExecutionEngine())
    for s in DEFAULT_SYMBOLS:
        df_4h, rejected = runner_check.validate_and_build_4h_candles(s)
        if not df_4h.empty:
            latest_4h = df_4h.iloc[-1]
            latest_dt = df_4h.index[-1]
            bars_count = latest_4h.get("bars_count", 240)
            status_str = f"COMPLETE ({bars_count}/240 contiguous bars)" if bars_count == 240 else f"INVALID ({bars_count}/240)"
            print(f"  {s} Latest 4H:   {latest_dt.strftime('%Y-%m-%d %H:%M UTC')} | Close: ${latest_4h['close']:.2f} | [{status_str}]")
        else:
            print(f"  {s} Latest 4H:   No verified 4H candles formed yet.")

        if rejected:
            print(f"  {s} Incomplete:  {len(rejected)} partial window(s) rejected (e.g. {rejected[-1]['reason']})")

    # 4. Portfolio & Risk Manager Status
    print("\n[Paper Portfolio & Risk Metrics]")
    if os.path.exists(PAPER_DB_PATH):
        conn = sqlite3.connect(PAPER_DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute("""
            SELECT cash_balance, equity, realized_pnl, unrealized_pnl, peak_equity, drawdown_pct, gross_leverage, net_leverage
            FROM paper_portfolio ORDER BY id DESC LIMIT 1
            """)
            p_row = cur.fetchone()
            if p_row:
                print(f"  Cash Balance:    ${p_row[0]:,.2f}")
                print(f"  Account Equity:  ${p_row[1]:,.2f}")
                print(f"  Realized PnL:    ${p_row[2]:+,.2f}")
                print(f"  Unrealized PnL:  ${p_row[3]:+,.2f}")
                print(f"  Peak Equity:     ${p_row[4]:,.2f}")
                print(f"  Max Drawdown:    {p_row[5]:.2f}%")
                print(f"  Gross Leverage:  {p_row[6]:.2f}x (Cap: 3.00x)")
                print(f"  Net Leverage:    {p_row[7]:.2f}x")

            # Virtual Positions
            cur.execute("""
            SELECT strategy_id, symbol, side, entry_price, quantity, notional, stop_loss, take_profit, accumulated_funding, entry_time
            FROM paper_positions WHERE status = 'OPEN'
            """)
            pos_rows = cur.fetchall()
            print("\n[Active Virtual Positions (Multi-Strategy Isolated)]")
            if pos_rows:
                for pr in pos_rows:
                    print(f"  [{pr[0]}] {pr[1]} {pr[2]} | Qty: {pr[4]:.4f} (${pr[5]:.2f}) | Entry: ${pr[3]:.2f} | SL: ${pr[6]:.2f} | TP: ${pr[7]:.2f} | Funding: ${pr[8]:+.2f} | Opened: {pr[9]}")
            else:
                print("  No active positions (FLAT).")

            # Strategy Evaluations count
            cur.execute("SELECT signal_type, COUNT(*) FROM strategy_evaluations GROUP BY signal_type")
            eval_rows = cur.fetchall()
            print("\n[Strategy Evaluation Activity (Unique Candles)]")
            if eval_rows:
                for st, count in eval_rows:
                    print(f"  {st}: {count}")
            else:
                print("  No completed 4H candle evaluations recorded yet.")

        finally:
            conn.close()
    else:
        print(f"  Database not found at {PAPER_DB_PATH}")

    print("=" * 80)


def cmd_inspect_signals(limit: int = 20):
    """Displays the most recent unique candle evaluations and rejection reasons."""
    print("=" * 90)
    print(f"MOST RECENT {limit} CANDLE EVALUATIONS (INCLUDING EXPLICIT REJECTION REASONS)")
    print("=" * 90)

    if not os.path.exists(PAPER_DB_PATH):
        print("Paper database does not exist.")
        return

    conn = sqlite3.connect(PAPER_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT candle_close_time, strategy_id, symbol, signal_type, close_price, ema_50, vol_ratio, rejection_reason
        FROM strategy_evaluations ORDER BY candle_close_time DESC LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        if not rows:
            print("No evaluations recorded.")
            return

        print(f"{'Candle Close Time (UTC)':<24} | {'Strategy':<20} | {'Sym':<8} | {'Type':<8} | {'Close':<8} | {'VolR':<6} | {'Rejection Reason'}")
        print("-" * 90)
        for r in rows:
            close_dt = datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            strat = r[1][:20]
            sym = r[2]
            stype = r[3]
            close_px = f"{r[4]:.1f}"
            vol_r = f"{r[6]:.2f}" if r[6] is not None else "-"
            reason = r[7] or "Valid Signal"
            print(f"{close_dt:<24} | {strat:<20} | {sym:<8} | {stype:<8} | {close_px:<8} | {vol_r:<6} | {reason}")
    finally:
        conn.close()


def cmd_inspect_positions():
    """Displays completed closed trade history and active virtual positions."""
    print("=" * 90)
    print("COMPLETED CLOSED PAPER TRADES & SEPARATED PnL ACCOUNTING")
    print("=" * 90)

    if not os.path.exists(PAPER_DB_PATH):
        print("Paper database does not exist.")
        return

    conn = sqlite3.connect(PAPER_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT trade_id, strategy_id, symbol, side, entry_price, exit_price, quantity,
               directional_pnl, funding_pnl, entry_fee_usd, exit_fee_usd, net_pnl, exit_reason, holding_hours
        FROM paper_trades ORDER BY entry_time DESC
        """)
        rows = cur.fetchall()
        if not rows:
            print("No closed trades recorded yet.")
        else:
            print(f"{'ID':<10} | {'Strategy':<18} | {'Side':<5} | {'Entry':<8} | {'Exit':<8} | {'Dir PnL':<9} | {'Fund PnL':<9} | {'Fees':<7} | {'Net PnL':<9} | {'Reason'}")
            print("-" * 90)
            for r in rows:
                tid = r[0][:10]
                strat = r[1][:18]
                side = r[3]
                epx = f"${r[4]:.2f}"
                xpx = f"${r[5]:.2f}"
                dir_pnl = f"${r[7]:+.2f}"
                fund_pnl = f"${r[8]:+.2f}"
                fees = f"${(r[9] + r[10]):.2f}"
                net_pnl = f"${r[11]:+.2f}"
                reason = r[12]
                print(f"{tid:<10} | {strat:<18} | {side:<5} | {epx:<8} | {xpx:<8} | {dir_pnl:<9} | {fund_pnl:<9} | {fees:<7} | {net_pnl:<9} | {reason}")
    finally:
        conn.close()


def cmd_stop():
    """
    Stops the running paper trading daemon reliably.
    Sends termination signal, waits for process termination, and removes PID lock.
    """
    print("Stopping paper trading daemon...")
    if not os.path.exists(DAEMON_PID_PATH):
        print("No PID lockfile found. Daemon does not appear to be running.")
        return

    try:
        with open(DAEMON_PID_PATH, "r") as f:
            pid = int(f.read().strip())
    except Exception as e:
        print(f"Error reading PID file: {e}")
        return

    if not is_pid_alive(pid):
        print(f"Process {pid} is not running. Removing stale PID file.")
        release_pid_lock()
        return

    # Send termination signal
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)

        # Wait up to 5 seconds for termination
        for _ in range(50):
            time.sleep(0.1)
            if not is_pid_alive(pid):
                break

        if not is_pid_alive(pid):
            print(f"Successfully stopped paper daemon (PID {pid}).")
            release_pid_lock()
        else:
            print(f"Process {pid} did not terminate cleanly. Force stopping...")
            if sys.platform != "win32":
                os.kill(pid, signal.SIGKILL)
            release_pid_lock()
    except Exception as e:
        print(f"Error terminating process {pid}: {e}")


# =============================================================================
# Main Daemon Event Loop
# =============================================================================

async def run_full_system():
    """Runs the integrated collector, position manager, and strategy runner."""
    if not acquire_pid_lock():
        sys.exit(1)

    collector = MarketDataCollector()
    exec_engine = PaperExecutionEngine()
    runner = CandidateAPaperRunner(execution_engine=exec_engine)

    logger.info("Starting Full Paper Trading System Daemon...")

    # Launch background collector task (independent dual streams + periodic REST)
    collector_task = asyncio.create_task(collector.run_async_collector())

    # Register graceful shutdown signals
    stop_event = asyncio.Event()

    def _sig_handler():
        logger.info("Received termination signal. Initiating graceful shutdown...")
        stop_event.set()

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, _sig_handler)

    try:
        while not stop_event.is_set():
            # 1. Manage existing positions using fresh quotes and latest bars (independent of 4H schedule)
            quotes = collector.latest_prices
            if quotes:
                exec_engine.update_mark_to_market(quotes)

            # 1. Replay missed completed 1m candles chronologically to ensure open positions receive every eligible bar
            gap_occurred = False
            try:
                runner.replay_missed_completed_1m_bars(quotes)
            except CandleGapError as e:
                logger.critical(f"Candle gap detected during 1m replay: {e}. Halting new entries and skipping strategy evaluation until recovered.")
                gap_occurred = True

            # Update mark-to-market and check horizon expiry (only if continuity intact)
            if not gap_occurred:
                runner.process_incoming_market_tick(quotes)

            # 2. Evaluate strategy on completed, 240-bar-verified 4H candles (only if continuity intact and not halted)
            if not gap_occurred and not runner.gap_halted:
                try:
                    runner.evaluate_new_completed_candles(quotes)
                except Exception as e:
                    logger.error(f"Error evaluating completed candles: {e}", exc_info=True)

            # Sleep 5 seconds before next management cycle
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Daemon cancelled by user/keyboard interrupt.")
    finally:
        logger.info("Flushing databases and shutting down collector...")
        collector.stop()
        collector_task.cancel()
        release_pid_lock()
        logger.info("Graceful shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="ETH/USDT Paper Trading & Data Ingestion CLI")
    parser.add_argument("command", choices=["start", "status", "inspect-signals", "inspect-positions", "stop"],
                        help="Action to perform")
    parser.add_argument("--mode", choices=["collector-only", "full"], default="full",
                        help="Start mode (collector-only or full paper trading)")
    parser.add_argument("--full", action="store_true", help="Start in full paper trading mode")
    parser.add_argument("--collector-only", action="store_true", help="Start in collector-only mode")
    parser.add_argument("--limit", type=int, default=20, help="Number of records to display")
    args = parser.parse_args()

    mode = "collector-only" if args.collector_only else ("full" if args.full else args.mode)

    if args.command == "status":
        cmd_status()
    elif args.command == "inspect-signals":
        cmd_inspect_signals(limit=args.limit)
    elif args.command == "inspect-positions":
        cmd_inspect_positions()
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "start":
        if mode == "collector-only":
            collector = MarketDataCollector()
            asyncio.run(collector.run_async_collector())
        else:
            asyncio.run(run_full_system())


if __name__ == "__main__":
    main()
