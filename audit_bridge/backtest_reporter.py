"""
audit_bridge/backtest_reporter.py — Rigorous, verifiable backtest result reporting.

Integrates mathematical consistency validation, independent metrics computation from trade ledgers,
and strict cryptographic linkage to git commits, model checkpoints, and datasets.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from audit_bridge.sanitizer import sanitize_data, assert_safe_for_publication
from audit_bridge.metrics_validator import validate_backtest_report, compute_metrics_from_ledger

BRIDGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BRIDGE_DIR.parent
LATEST_BACKTEST = BRIDGE_DIR / "latest_backtest.json"
REPORTS_DIR = BRIDGE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _file_hash(filepath: str) -> str:
    """Compute full SHA-256 of a file."""
    try:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (FileNotFoundError, OSError):
        return "unavailable"


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        from audit_bridge.tracker import get_current_commit
        return get_current_commit()
    except Exception:
        return "unknown"


def _get_model_config() -> dict:
    """Extract current model configuration (no secrets)."""
    try:
        import config
        return {
            "symbol": getattr(config, "SYMBOL", "unknown"),
            "timeframe": getattr(config, "TIMEFRAME", "unknown"),
            "lookback": getattr(config, "LOOKBACK", None),
            "transformer_dim": getattr(config, "TRANSFORMER_DIM", None),
            "transformer_layers": getattr(config, "TRANSFORMER_LAYERS", None),
            "attention_heads": getattr(config, "ATTENTION_HEADS", None),
            "dropout": getattr(config, "DROPOUT", None),
            "attention_dropout": getattr(config, "ATTENTION_DROPOUT", None),
            "batch_size": getattr(config, "BATCH_SIZE", None),
            "learning_rate": getattr(config, "LEARNING_RATE", None),
            "max_epochs": getattr(config, "MAX_EPOCHS", None),
            "patience": getattr(config, "PATIENCE", None),
            "train_split": getattr(config, "TRAIN_SPLIT", None),
            "target_horizon": getattr(config, "TARGET_HORIZON", None),
            "target_type": getattr(config, "TARGET_TYPE", None),
            "tbm_pt_mult": getattr(config, "TBM_PT_MULT", None),
            "tbm_sl_mult": getattr(config, "TBM_SL_MULT", None),
            "noise_injection": getattr(config, "NOISE_INJECTION", None),
            "leverage": getattr(config, "LEVERAGE", None),
            "risk_per_trade_pct": getattr(config, "RISK_PER_TRADE_PERCENT", None),
            "commission_rate": getattr(config, "COMMISSION_RATE", None),
            "slippage_rate": getattr(config, "SLIPPAGE_RATE", None),
            "candle_dim": getattr(config, "CANDLE_DIM", None),
            "market_dim": getattr(config, "MARKET_DIM", None),
        }
    except ImportError:
        return {"error": "config module not importable"}


def _compute_config_hash(cfg: dict) -> str:
    """Compute SHA-256 hash of configuration dictionary."""
    serialized = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def log_backtest(
    *,
    # Dataset
    dataset_date_range: Tuple[str, str] = None,
    dataset_candles: int = None,
    dataset_hash: str = None,
    dataset_identifier: str = None,
    # Model
    model_checkpoint: str = None,
    # Execution
    command: str = None,
    evaluation_method: str = None,
    threshold: float = None,
    train_val_split: str = None,
    execution_status: str = "completed",
    execution_error: str = None,
    duration_seconds: float = None,
    # Raw Data Inputs (preferred over manual metrics)
    trades: List[Dict[str, Any]] = None,
    equity_curve: List[float] = None,
    monthly_breakdown: List[Dict[str, Any]] = None,
    # Manual Metrics (validated against trades if provided, or checked for internal consistency)
    initial_capital: float = None,
    final_capital: float = None,
    num_trades: int = None,
    win_count: int = None,
    loss_count: int = None,
    win_rate: float = None,
    gross_return_pct: float = None,
    net_return_pct: float = None,
    total_fees: float = None,
    total_slippage: float = None,
    total_funding: float = None,
    max_drawdown_pct: float = None,
    sharpe_ratio: float = None,
    sortino_ratio: float = None,
    profit_factor: float = None,
    expectancy: float = None,
    avg_win: float = None,
    avg_loss: float = None,
    # Meta
    notes: str = None,
    is_sample: bool = False,
) -> Dict[str, Any]:
    """
    Save a verified, machine-readable backtest report.

    If `trades` (trade ledger) is supplied, all metrics are independently derived from it.
    If manual metrics are supplied, they are validated against mathematical consistency rules.
    If discrepancies are found, the report is explicitly marked INVALID.
    """
    checkpoint_path = model_checkpoint or str(PROJECT_ROOT / "checkpoints" / "best_model.pt")
    cfg = _get_model_config()
    cfg_hash = _compute_config_hash(cfg)

    # Prepare claimed metrics bundle for validation
    claimed = {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "num_trades": num_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "gross_return_pct": gross_return_pct,
        "net_return_pct": net_return_pct,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_funding": total_funding,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }

    # Run mathematical consistency validation
    validation = validate_backtest_report(claimed, trades=trades, equity_curve=equity_curve)

    # Use reconciled metrics if ledger was provided
    final_results = validation.reconciled_metrics if (trades and len(trades) > 0) else claimed

    report_id = datetime.now(timezone.utc).strftime("BT-%Y%m%d-%H%M%S")

    report = {
        "report_id": report_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_sample_data": is_sample,
        "is_mathematically_valid": validation.is_valid,
        "validation_discrepancies": validation.discrepancies,
        "validation_warnings": validation.warnings,
        "provenance": {
            "git_commit": _get_git_commit(),
            "command": command,
            "model_checkpoint": checkpoint_path,
            "model_checkpoint_sha256": _file_hash(checkpoint_path),
            "configuration_hash": cfg_hash,
            "dataset_identifier": dataset_identifier or "ETH/USDT:USDT 1h",
            "dataset_hash": dataset_hash,
            "dataset_date_range": list(dataset_date_range) if dataset_date_range else None,
            "total_candles": dataset_candles,
        },
        "configuration": cfg,
        "results": final_results,
        "methodology": {
            "evaluation_method": evaluation_method,
            "threshold": threshold,
            "train_val_split": train_val_split,
        },
        "execution": {
            "status": "VALID" if validation.is_valid else "INVALID_METRICS",
            "error": execution_error,
            "duration_seconds": duration_seconds,
        },
        "monthly_breakdown": monthly_breakdown,
        "notes": notes,
    }

    # Verify publication safety (no secrets)
    assert_safe_for_publication(report, context_name=f"Backtest Report {report_id}")

    # Write latest
    LATEST_BACKTEST.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    # Archive report
    archive_path = REPORTS_DIR / f"{report_id}.json"
    archive_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    # Save trade ledger separately if provided
    if trades:
        ledger_path = REPORTS_DIR / f"{report_id}_trades.json"
        assert_safe_for_publication(trades, context_name=f"Trade Ledger {report_id}")
        ledger_path.write_text(
            json.dumps(trades, indent=2, default=str),
            encoding="utf-8",
        )

    # Save equity curve separately if provided
    if equity_curve:
        curve_path = REPORTS_DIR / f"{report_id}_equity.json"
        curve_path.write_text(
            json.dumps(equity_curve, indent=2, default=str),
            encoding="utf-8",
        )

    return report
