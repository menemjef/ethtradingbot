"""
audit_bridge/status.py — Latest project status tracking.

Maintains a single latest_status.json with the current state of the project.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BRIDGE_DIR.parent
LATEST_STATUS = BRIDGE_DIR / "latest_status.json"


def _checkpoint_info() -> dict:
    """Get model checkpoint metadata without loading the model."""
    import hashlib
    ckpt = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    if not ckpt.exists():
        return {"exists": False}

    stat = ckpt.stat()
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return {
        "exists": True,
        "path": str(ckpt),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256_prefix": h.hexdigest()[:16],
    }


def _scaler_info() -> dict:
    """Check scaler files exist."""
    sc = PROJECT_ROOT / "checkpoints" / "scaler_candle.pkl"
    sm = PROJECT_ROOT / "checkpoints" / "scaler_market.pkl"
    return {
        "scaler_candle": {"exists": sc.exists(), "path": str(sc)},
        "scaler_market": {"exists": sm.exists(), "path": str(sm)},
    }


def update_status(
    phase: str,
    description: str,
    *,
    bot_running: bool = None,
    last_trade_time: str = None,
    balance: float = None,
    model_version: str = None,
    active_mode: str = None,
    extras: dict = None,
) -> dict:
    """
    Update the latest project status.

    Args:
        phase: Current development phase (e.g., 'RETRAINING', 'DEPLOYED', 'AUDITING').
        description: Human-readable status summary.
        bot_running: Whether the live bot is currently running.
        last_trade_time: ISO timestamp of last trade.
        balance: Current account balance (only if user has explicitly shared it).
        model_version: Model version identifier.
        active_mode: Trading mode name.
        extras: Additional key-value pairs.

    Returns:
        The status dict.
    """
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "description": description,
        "model_checkpoint": _checkpoint_info(),
        "scalers": _scaler_info(),
        "bot_running": bot_running,
        "last_trade_time": last_trade_time,
        "balance_usd": balance,
        "model_version": model_version,
        "active_mode": active_mode,
    }

    if extras:
        status["extras"] = extras

    LATEST_STATUS.write_text(
        json.dumps(status, indent=2, default=str),
        encoding="utf-8",
    )

    return status
