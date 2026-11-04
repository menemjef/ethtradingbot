"""
utils/logger.py — Structured logging with rotating file handler.
"""

import io
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from config import LOG_DIR


def _utf8_stdout() -> io.TextIOWrapper:
    """Return a UTF-8 wrapper around stdout.buffer (Windows-safe)."""
    # Check if we can wrap the buffer for true UTF-8 output
    if hasattr(sys.stdout, "buffer"):
        try:
            return io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding="utf-8", 
                errors="replace", 
                line_buffering=True
            )
        except Exception:
            pass
            
    # Fallback: try reconfiguring existing stdout
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
            
    return sys.stdout


def supports_unicode() -> bool:
    """Check if the current stdout encoding supports common Unicode decorators."""
    try:
        "🕯".encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, TypeError):
        return False


class SafeRotatingFileHandler(RotatingFileHandler):
    """
    Windows-safe rotating file handler that doesn't crash on PermissionError
    if the log file is temporarily locked by another process or the OS.
    """
    def doRollover(self):
        try:
            super().doRollover()
        except (PermissionError, OSError) as e:
            # On Windows, file locks often prevent rotation. 
            # We ignore this to prevent crashing the entire trading engine.
            import sys
            sys.stderr.write(f"⚠️  Logging Warning: Could not rotate log file (Locked). Continuing... {e}\n")

def setup_logging(level: int = logging.INFO):
    """
    Configure root logger with:
    - Console handler (stdout UTF-8)
    - Safe rotating file handler -> logs/live.log (50 MB, 3 backups)
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    # Console (UTF-8 safe)
    console = logging.StreamHandler(_utf8_stdout())
    console.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(console)

    # Power-User Rotating file (50MB to minimize rotation frequency)
    log_file = os.path.join(LOG_DIR, "live.log")
    file_handler = SafeRotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,   # 50 MB
        backupCount=3,
        encoding="utf-8",
        delay=True  # Don't open file until first log record
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(file_handler)

    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
