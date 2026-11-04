"""
audit_bridge/hook_handler.py inside .agents/ for native Antigravity hook execution.
"""
import sys
from pathlib import Path

# Set project root to parent directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Forward to the main hook handler
from audit_bridge.hook_handler import handle_hook

if __name__ == "__main__":
    handle_hook()
