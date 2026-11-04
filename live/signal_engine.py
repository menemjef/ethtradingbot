"""
live/signal_engine.py — WebSocket-driven live signal engine.

WHAT THIS FILE DOES:
This is the "Eyes and Ears" of the bot. 
1. It connects to Binance using a WebSocket (a continuous live stream of data).
2. It waits patiently for a candle to "Close" (e.g., at the end of the hour).
3. When the candle closes, it pulls the latest data, runs it through the indicators 
   (the math engine), and asks the AI Model (the brain) for a prediction.
4. If the prediction is strong enough (passes the Threshold), it screams "SIGNAL!"
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import threading
import websocket
import torch

from config import (
    SYMBOL,
    WS_SYMBOL,
    TIMEFRAME,
    WS_BASE,
    LIVE_FETCH_LIMIT,
    MODEL_PATH,
    LOOKBACK,
    BTC_CORRELATION_GUARD,
    TRADING_MODES,
    TARGET_MULTIPLIER,
    get_current_threshold
)
from utils.settings_manager import get_active_mode
from data.fetcher import fetch_latest_candles, fetch_funding_rates, fetch_open_interest, fetch_ohlcv
from data.preprocessor import prepare_live_window, load_scalers
from model.architecture import HybridModel
from utils.logger import supports_unicode

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Signal printing
# ─────────────────────────────────────────────────────────────────────────────

def print_signal(close_time: datetime, prediction: float, current_close: float):
    direction_text = "LONG  [UP]  " if prediction > 0 else "SHORT [DOWN]"
    direction_icon = "📈" if prediction > 0 else "📉"
    
    current_threshold = get_current_threshold()
    active_mode = get_active_mode()
    mode_color = TRADING_MODES.get(active_mode, {}).get("color", "#00f0ff")

    threshold_met = abs(prediction) >= current_threshold
    verdict = f"[{active_mode.upper()} SIGNAL ACTIVE]" if threshold_met else f"[{active_mode} - Below threshold]"
    status_icon = "🚀" if threshold_met else "⏸"

    price_str = f"${current_close:,.2f}"
    pred_str = f"{prediction:+.6f}"
    
    if supports_unicode():
        msg = (
            f"\n╔══════════════════════════════════════════════════════════╗\n"
            f"║  🕯  ETH/USDT {TIMEFRAME.upper()} CANDLE CLOSED                          ║\n"
            f"╠══════════════════════════════════════════════════════════╣\n"
            f"║  Protocol:     {active_mode:<40} ║\n"
            f"║  Time (UTC):   {close_time.strftime('%Y-%m-%d %H:%M:%S UTC'):<40} ║\n"
            f"║  Close price:  {price_str:<40} ║\n"
            f"║  Prediction:   {pred_str:<40} ║\n"
            f"║  Direction:    {direction_text} {direction_icon:<28} ║\n"
            f"║  Threshold:    |{prediction:.6f}| >= {current_threshold:.4f}? → {str(threshold_met):<11} ║\n"
            f"║  Status:       {status_icon} {verdict:<39} ║\n"
            f"╚══════════════════════════════════════════════════════════╝"
        )
    else:
        # Fallback for plain terminals
        msg = (
            f"\n[ ETH/USDT {TIMEFRAME.upper()} CANDLE CLOSED ]\n"
            f"Protocol:     {active_mode}\n"
            f"Time (UTC):   {close_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Close price:  ${current_close:,.2f}\n"
            f"Prediction:   {prediction:+.6f}\n"
            f"Direction:    {direction_text}\n"
            f"Threshold:    |{prediction:.6f}| >= {current_threshold:.4f}? -> {threshold_met}\n"
            f"Status:       {verdict}"
        )

    logger.info(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Signal Engine
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Binance Futures WebSocket signal engine.

    Args:
        on_signal: optional callback called on every candle close with
                   (close_time, prediction, close_price)
    """

    RECONNECT_DELAY = 5      # seconds before first reconnect
    MAX_RECONNECT = 10       # max reconnect attempts before giving up

    def __init__(self, on_signal: Optional[Callable] = None, shutdown_event: Optional[threading.Event] = None):
        self.on_signal = on_signal
        self.shutdown_event = shutdown_event
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading model and scalers ...")
        self.model = self._load_model()
        self.scaler_candle, self.scaler_market = load_scalers()
        logger.info("Model and scalers loaded. Device: %s", self.device)

        self._ws: Optional[websocket.WebSocketApp] = None
        self._reconnect_count = 0
        self._last_signal_time: Optional[datetime] = None  # Dedup guard
        self.last_msg_time = time.time()
        self._keepalive_thread: Optional[threading.Thread] = None

    def _load_model(self) -> HybridModel:
        model = HybridModel()
        state = torch.load(MODEL_PATH, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _infer(self) -> Optional[dict]:
        """
        THE INFERENCE LOOP:
        This is where the magic happens every time a candle closes.
        1. Download latest data.
        2. Calculate indicators.
        3. Scale data.
        4. Ask the Neural Network for a prediction.
        5. Return the prediction to the Trader.
        """
        try:
            # 1. Fetch ETH data (Main Timeframe)
            df_raw = fetch_latest_candles(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIVE_FETCH_LIMIT)
            
            # 2. Fetch Multi-Timeframe and External Data (Phase 5)
            from config import SECONDARY_TIMEFRAMES, MARKET_FEATURES
            from data.fetcher import fetch_funding_rates, fetch_open_interest, fetch_ohlcv
            from data.indicators import integrate_external_data, compute_mtf_features
            
            # Fetch External metrics
            funding_df = fetch_funding_rates(SYMBOL, limit=200)  # Covers >1000h
            oi_df = fetch_open_interest(SYMBOL, limit=500)      # Binance max for Hist
            
            # Fetch MTF candles
            df_15m = fetch_ohlcv(SYMBOL, "15m", limit=2000)     # Covers 500h
            df_4h = fetch_ohlcv(SYMBOL, "4h", limit=500)        # Covers >1000h
            
            # 3. Integrate all into indicators
            # StandardIndicators + External + MTF
            from data.indicators import compute_indicators
            df_feat = compute_indicators(df_raw)
            df_feat = integrate_external_data(df_feat, funding_df, oi_df)
            df_feat = compute_mtf_features(df_feat, df_15m, df_4h)
            
            # 4. BTC Guard
            btc_bullish = True
            if BTC_CORRELATION_GUARD:
                try:
                    from config import BTC_SYMBOL
                    # Standardize symbol to CCXT format if needed
                    df_btc = fetch_latest_candles(BTC_SYMBOL, TIMEFRAME, limit=300)
                    from data.indicators import _ema
                    # EMA 50 needs enough data
                    btc_ema50 = _ema(df_btc["close"], 50).iloc[-1]
                    btc_bullish = df_btc["close"].iloc[-1] > btc_ema50
                except Exception as btc_err:
                    logger.error(f"BTC Guard fetch failed: {btc_err}")

        except Exception as exc:
            logger.error("Data pipeline failed: %s", exc)
            return None

        # Prepare window for model (Phase 5 fix: unified pipeline)
        result = prepare_live_window(
            df_raw, 
            self.scaler_candle, 
            self.scaler_market, 
            LOOKBACK,
            funding_df=funding_df,
            oi_df=oi_df,
            df_15m=df_15m,
            df_4h=df_4h
        )
        
        if result is None:
            logger.warning("Insufficient data for inference — skipping candle.")
            return None

        candle_window, market_window = result
        
        # DEBUG: Check feature health
        c_mean = candle_window.mean()
        m_mean = market_window.mean()
        c_zeros = (candle_window == 0).sum()
        m_zeros = (market_window == 0).sum()
        logger.info(
            f"Feature Health — Candle Mean: {c_mean:.4f} (Zeros: {c_zeros}/{candle_window.size}) | "
            f"Market Mean: {m_mean:.4f} (Zeros: {m_zeros}/{market_window.size})"
        )

        x_c = torch.from_numpy(candle_window).to(self.device)
        x_m = torch.from_numpy(market_window).to(self.device)

        with torch.no_grad():
            reg_pred_tensor, clf_pred_tensor = self.model(x_c, x_m)
            raw_reg = reg_pred_tensor.item()
            raw_clf = clf_pred_tensor.item()
            
            # Regression-driven inference (consistent with backtest and March architecture)
            pred = raw_reg / TARGET_MULTIPLIER
            magnitude = abs(pred)
            direction = 1.0 if pred > 0 else -1.0
            
            logger.info(
                f"Inference — Magnitude: {magnitude:.6f} | "
                f"Direction: {'LONG' if direction > 0 else 'SHORT'} (raw_reg: {raw_reg:.4f}, raw_clf: {raw_clf:.4f}) | "
                f"Prediction: {pred:.6f}"
            )
            
        # Extract ATR for sizing
        atr = df_feat["atr_14"].iloc[-1]
        close_price = float(df_raw["close"].iloc[-1])

        # BTC Correlation Check (Safety) - This logic is already handled above in step 4.
        # The instruction snippet was a placeholder, retaining the more robust existing logic.

        return {
            "prediction": pred,
            "close_price": close_price,
            "atr": atr,
            "btc_bullish": btc_bullish,
            "raw_reg": raw_reg,
            "raw_clf": raw_clf
        }

    # ── WebSocket callbacks ──────────────────────────────────────────────────

    def _on_open(self, ws):
        logger.info("✅ WebSocket connected to Binance Futures stream.")
        self._reconnect_count = 0
        self.last_msg_time = time.time()

    def _on_message(self, ws, message: str):
        try:
            self.last_msg_time = time.time()
            data = json.loads(message)
            kline = data.get("k", {})

            if not kline.get("x", False):
                # 'x' stands for is_closed in Binance WebSockets.
                # If the candle isn't closed yet, we do nothing.
                return  # Candle not yet closed

            close_time_ms = kline.get("T", 0)
            close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)

            # Dedup: skip if we already emitted a signal for this candle
            if self._last_signal_time and close_time <= self._last_signal_time:
                logger.debug("Skipping duplicate candle close at %s", close_time)
                return

            logger.info("Candle closed at %s - running inference ...", close_time)
            res = self._infer()
            if res is None:
                return

            prediction = res["prediction"]
            close_price = res["close_price"]
            
            print_signal(close_time, prediction, close_price)
            self._last_signal_time = close_time

            if self.on_signal:
                self.on_signal(
                    close_time, 
                    prediction, 
                    close_price, 
                    atr=res["atr"], 
                    btc_bullish=res["btc_bullish"]
                )

        except Exception as exc:
            logger.exception("Error processing WebSocket message: %s", exc)

    def _on_error(self, ws, error):
        logger.error("WebSocket error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(
            "WebSocket closed (code=%s, msg=%s). Will reconnect in %ds ...",
            close_status_code, close_msg, self.RECONNECT_DELAY,
        )

    # ── Public interface ─────────────────────────────────────────────────────

    def stop(self):
        """Request graceful socket closure."""
        if self._ws:
            self._ws.close()

    def run(self):
        """Start WebSocket listener with auto-reconnect."""
        self._reconnect_count = 0
        tf_ws = TIMEFRAME.replace("h", "h")   # '4h' as-is
        stream_url = f"{WS_BASE}/ws/{WS_SYMBOL}@kline_{tf_ws}"
        logger.info("Subscribing to stream: %s", stream_url)
        
        # Start keep-alive monitor with REST fallback
        def monitor_connectivity():
            while self.shutdown_event and not self.shutdown_event.is_set():
                time.sleep(60)  # Check every 60 seconds
                idle_time = time.time() - self.last_msg_time

                # --- REST FALLBACK: If we missed a candle close, run inference via REST ---
                if self._last_signal_time:
                    seconds_since_last_signal = (datetime.now(timezone.utc) - self._last_signal_time).total_seconds()
                    # If >70 mins since last signal, we missed at least one hourly candle
                    if seconds_since_last_signal > 4200:  # 70 minutes
                        logger.warning(
                            "🔥 REST FALLBACK: No candle close received for %.0f mins. "
                            "Running inference via REST API...", seconds_since_last_signal / 60
                        )
                        try:
                            res = self._infer()
                            if res and self.on_signal:
                                now = datetime.now(timezone.utc)
                                print_signal(now, res["prediction"], res["close_price"])
                                self._last_signal_time = now
                                self.on_signal(
                                    now,
                                    res["prediction"],
                                    res["close_price"],
                                    atr=res["atr"],
                                    btc_bullish=res["btc_bullish"]
                                )
                                logger.info("✅ REST fallback inference completed successfully.")
                        except Exception as e:
                            logger.error(f"REST fallback inference failed: {e}")

                # --- WebSocket health check ---
                if idle_time > 600:
                    logger.warning("⚠️ WebSocket IDLE for >10 mins. Forcing reconnect...")
                    self.stop()
                elif idle_time > 120:
                    logger.info("💙 WebSocket Heartbeat: Last message received %.1fs ago.", idle_time)
        
        if self._keepalive_thread is None or not self._keepalive_thread.is_alive():
            self._keepalive_thread = threading.Thread(target=monitor_connectivity, daemon=True)
            self._keepalive_thread.start()

        # Initial inference on startup for better UX (only on first connect)
        if self._last_signal_time is None:
            try:
                logger.info("Running initial startup inference...")
                res = self._infer()
                if res:
                    now = datetime.now(timezone.utc)
                    print_signal(now, res["prediction"], res["close_price"])
                    self._last_signal_time = now
                    if self.on_signal:
                        self.on_signal(
                            now, 
                            res["prediction"], 
                            res["close_price"], 
                            atr=res["atr"], 
                            btc_bullish=res["btc_bullish"]
                        )
            except Exception as e:
                logger.error(f"Initial inference failed: {e}")

        while self._reconnect_count <= self.MAX_RECONNECT:
            # Check for shutdown before each connect/reconnect
            if self.shutdown_event and self.shutdown_event.is_set():
                break

            try:
                self._ws = websocket.WebSocketApp(
                    stream_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(
                    ping_interval=30, 
                    ping_timeout=10
                )
            except KeyboardInterrupt:
                logger.info("Shutting down by user request.")
                break
            except Exception as exc:
                if self.shutdown_event and self.shutdown_event.is_set():
                    break
                logger.error("WebSocket run exception: %s", exc)

            self._reconnect_count += 1
            delay = self.RECONNECT_DELAY * (2 ** min(self._reconnect_count, 5))
            
            # Final shutdown check before sleeping 
            if self.shutdown_event and self.shutdown_event.is_set():
                break

            logger.info(
                "Reconnect attempt %d/%d in %.0fs ...",
                self._reconnect_count, self.MAX_RECONNECT, delay,
            )
            time.sleep(delay)

        logger.info("Signal engine stopped.")
