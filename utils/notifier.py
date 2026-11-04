"""
utils/notifier.py — Handlers for sending notifications (Telegram, etc).
"""

import logging
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

def send_telegram_message(message: str):
    """Send a message via Telegram Bot API."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        # Silently skip if not configured
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

class NotificationService:
    @staticmethod
    def alert_signal(symbol: str, prediction: float, side: str, price: float):
        icon = "📈" if side == "LONG" else "📉"
        msg = (
            f"*🚨 NEW SIGNAL: {symbol}*\n"
            f"Direction: {side} {icon}\n"
            f"Price: ${price:,.2f}\n"
            f"Prediction: {prediction:+.6f}"
        )
        send_telegram_message(msg)

    @staticmethod
    def alert_trade(symbol: str, side: str, price: float, amount: float, status: str):
        icon = "✅" if status == "OPEN" else "⏹"
        msg = (
            f"*{icon} TRADE {status}: {symbol}*\n"
            f"Side: {side}\n"
            f"Price: ${price:,.2f}\n"
            f"Size: {amount:.4f}"
        )
        send_telegram_message(msg)

    @staticmethod
    def alert_error(error_msg: str):
        msg = f"*❌ BOT ERROR*\n`{error_msg}`"
        send_telegram_message(msg)
