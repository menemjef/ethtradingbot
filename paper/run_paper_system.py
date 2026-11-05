"""
paper/run_paper_system.py — CLI Supervisor for Paper Trading & Market Data Ingestion.

Commands:
  python paper/run_paper_system.py start [--collector-only | --full]
  python paper/run_paper_system.py status
  python paper/run_paper_system.py inspect-signals [--limit 20]
  python paper/run_paper_system.py inspect-positions
  python paper/run_paper_system.py stop
"""

import os
import sys
import time
import json
import sqlite3
import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.config import (
    MARKET_DB_PATH,
    PAPER_DB_PATH,
    DEFAULT_SYMBOLS
)
from paper.collector import MarketDataCollector
from paper.execution import PaperExecutionEngine
from paper.strategy_runner import CandidateAPaperRunner

logger = logging.getLogger("paper.supervisor")


def cmd_status():
    """Prints live status of market data database, portfolio, active positions, and recent signals."""
    print("=" * 65)
    print("LIVE PAPER TRADING & MARKET DATA SYSTEM STATUS")
    print("=" * 65)

    # 1. Market Data Status
    if os.path.exists(MARKET_DB_PATH):
        conn = sqlite3.connect(MARKET_DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute("SELECT symbol, COUNT(*), MIN(open_time), MAX(close_time) FROM completed_klines GROUP BY symbol")
            rows = cur.fetchall()
            print("\n[Market Data Ingestion]")
            if rows:
                for sym, cnt, min_t, max_t in rows:
                    t_min_str = datetime.fromtimestamp(min_t / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    t_max_str = datetime.fromtimestamp(max_t / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"  {sym}: {cnt} bars ({t_min_str} -> {t_max_str})")
            else:
                print("  completed_klines table is empty.")
        finally:
            conn.close()
    else:
        print(f"\n[Market Data Ingestion]: Database not found at {MARKET_DB_PATH}")

    # 2. Paper Portfolio Status
    if os.path.exists(PAPER_DB_PATH):
        conn = sqlite3.connect(PAPER_DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute("SELECT cash_balance, equity, realized_pnl, unrealized_pnl, peak_equity, drawdown_pct FROM paper_portfolio ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            print("\n[Paper Portfolio]")
            if row:
                print(f"  Cash Balance:    ${row[0]:.2f}")
                print(f"  Account Equity:  ${row[1]:.2f}")
                print(f"  Realized PnL:    ${row[2]:.2f}")
                print(f"  Unrealized PnL:  ${row[3]:.2f}")
                print(f"  Peak Equity:     ${row[4]:.2f}")
                print(f"  Max Drawdown:    {row[5]:.2f}%")
            else:
                print("  Portfolio table is empty.")

            # Active Positions
            cur.execute("SELECT position_id, symbol, side, entry_price, quantity, stop_loss, take_profit, entry_time FROM paper_positions WHERE status = 'OPEN'")
            pos_rows = cur.fetchall()
            print("\n[Active Positions]")
            if pos_rows:
                for p in pos_rows:
                    print(f"  {p[1]} {p[2]} | Entry: ${p[3]:.2f} | Qty: {p[4]:.4f} | SL: ${p[5]:.2f} | TP: ${p[6]:.2f} | Since: {p[7]}")
            else:
                print("  No open positions (FLAT).")

            # Signal Count
            cur.execute("SELECT signal_type, COUNT(*) FROM paper_signals GROUP BY signal_type")
            sig_rows = cur.fetchall()
            print("\n[Signal Activity]")
            if sig_rows:
                for st, count in sig_rows:
                    print(f"  {st}: {count}")
            else:
                print("  No signals evaluated yet.")
        finally:
            conn.close()
    else:
        print(f"\n[Paper Portfolio]: Database not found at {PAPER_DB_PATH}")
    print("=" * 65)


def cmd_inspect_signals(limit: int = 20):
    """Displays the most recent signals and rejection reasons."""
    print("=" * 75)
    print(f"MOST RECENT {limit} EVALUATED SIGNALS (INCLUDING REJECTIONS)")
    print("=" * 75)

    if not os.path.exists(PAPER_DB_PATH):
        print("Paper database does not exist.")
        return

    conn = sqlite3.connect(PAPER_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT timestamp, symbol, signal_type, rejection_reason, close_price, ema_50, vol_ratio, atr
        FROM paper_signals ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        if not rows:
            print("No signal evaluations recorded.")
            return

        print(f"{'Timestamp':<25} | {'Symbol':<8} | {'Type':<10} | {'Close':<8} | {'VolR':<6} | {'Reason'}")
        print("-" * 75)
        for r in rows:
            ts = r[0][:19] if r[0] else ""
            sym = r[1]
            stype = r[2]
            reason = r[3] or "-"
            close_px = f"{r[4]:.1f}"
            vol_r = f"{r[6]:.2f}" if r[6] is not None else "-"
            print(f"{ts:<25} | {sym:<8} | {stype:<10} | {close_px:<8} | {vol_r:<6} | {reason}")
    finally:
        conn.close()


def cmd_inspect_positions():
    """Displays full history of closed and open paper trades."""
    print("=" * 75)
    print("PAPER TRADING POSITIONS & COMPLETED TRADES")
    print("=" * 75)

    if not os.path.exists(PAPER_DB_PATH):
        print("Paper database does not exist.")
        return

    conn = sqlite3.connect(PAPER_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT trade_id, symbol, side, entry_time, exit_time, entry_price, exit_price,
               net_pnl, exit_reason, holding_hours
        FROM paper_trades ORDER BY entry_time DESC
        """)
        rows = cur.fetchall()
        if not rows:
            print("No closed trades recorded yet.")
        else:
            print(f"{'ID':<12} | {'Side':<5} | {'Entry Px':<9} | {'Exit Px':<9} | {'Net PnL':<9} | {'Reason':<15} | {'Duration'}")
            print("-" * 75)
            for r in rows:
                pnl_str = f"${r[7]:+.2f}"
                dur_str = f"{r[9]:.1f}h"
                print(f"{r[0]:<12} | {r[2]:<5} | ${r[5]:<8.2f} | ${r[6]:<8.2f} | {pnl_str:<9} | {r[8]:<15} | {dur_str}")
    finally:
        conn.close()


async def run_full_system():
    """Runs the integrated collector and Candidate A strategy runner in an event loop."""
    collector = MarketDataCollector()
    exec_engine = PaperExecutionEngine()
    runner = CandidateAPaperRunner(execution_engine=exec_engine)

    logger.info("Initializing Full Paper Trading System...")
    # First recover any missing historical data
    collector.recover_gaps_on_startup()

    # Launch background collector task
    collector_task = asyncio.create_task(collector.run_async_collector())

    try:
        while True:
            await asyncio.sleep(60.0)  # Periodic runner check
            # Check latest completed 1m kline for ETHUSDT
            with collector._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT open, high, low, close, volume FROM completed_klines WHERE symbol = 'ETHUSDT' ORDER BY open_time DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    bar = {"open": row[0], "high": row[1], "low": row[2], "close": row[3], "volume": row[4]}
                    runner.process_candle_close(bar)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down Full Paper Trading System...")
        collector.stop()
        collector_task.cancel()


def main():
    parser = argparse.ArgumentParser(description="ETH/USDT Paper Trading & Data Ingestion CLI")
    parser.add_argument("command", choices=["start", "status", "inspect-signals", "inspect-positions", "stop"],
                        help="Action to perform")
    parser.add_argument("--mode", choices=["collector-only", "full"], default="full",
                        help="Start mode (collector-only or full paper trading)")
    parser.add_argument("--limit", type=int, default=20, help="Number of records to display")
    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "inspect-signals":
        cmd_inspect_signals(limit=args.limit)
    elif args.command == "inspect-positions":
        cmd_inspect_positions()
    elif args.command == "start":
        if args.mode == "collector-only":
            collector = MarketDataCollector()
            asyncio.run(collector.run_async_collector())
        else:
            asyncio.run(run_full_system())
    elif args.command == "stop":
        print("To stop a running paper trading process, send SIGINT (Ctrl+C).")


if __name__ == "__main__":
    main()
