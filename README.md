# ETH/USDT Perpetual Futures Trading Signal System

A production-ready **live crypto trading signal system** for ETHUSDT perpetual futures using a hybrid **CNN + LSTM** deep learning model in PyTorch.

Every research prompt must run run_engine_controls.py and include its table in the report.

---

## Architecture Overview

```
Historical OHLCV Data (4H candles)
          │
          ▼
┌─────────────────────────────────┐
│       Technical Indicators       │
│  RSI, MACD, EMA, ATR, BB,       │
│  Stochastic, ADX, trend flags   │
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
 Branch 1: Candle    Branch 2: Market
 Structure (8 feat)  State (20 feat)
        │                 │
   Conv1D → LSTM     Conv1D → LSTM
        │                 │
        └────────┬─────────┘
                 ▼
          Fusion MLP
                 │
                 ▼
     Scalar log-return prediction
     LONG (>0) / SHORT (<0)
     Signal threshold: |pred| ≥ 0.003
```

---

## Project Structure

```
trading-bot/
├── config.py                  # All hyperparameters & paths
├── train.py                   # Training entry point
├── live.py                    # Live signal entry point
├── requirements.txt
│
├── data/
│   ├── fetcher.py             # CCXT Binance Futures REST fetcher
│   ├── indicators.py          # All technical indicators (pure pandas)
│   └── preprocessor.py        # Feature assembly, scalers, windows
│
├── model/
│   ├── architecture.py        # Hybrid CNN+LSTM PyTorch model
│   ├── dataset.py             # PyTorch Dataset (sliding windows)
│   └── trainer.py             # Training loop w/ early stopping
│
├── live/
│   ├── signal_engine.py       # WebSocket listener + inference
│   └── accuracy_tracker.py    # Win rate tracking + CSV logging
│
├── utils/
│   └── logger.py              # Rotating file logger
│
├── deploy/
│   ├── setup_ec2.sh           # One-command EC2 Ubuntu setup
│   └── run_live.sh            # Screen session launcher
│
├── checkpoints/               # Auto-created — model + scalers saved here
└── logs/                      # Auto-created — signal CSV + log files
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For PyTorch (CPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# For PyTorch (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Train the model

```bash
# Full training (fetches ~2500 candles, up to 100 epochs)
python train.py

# Fast smoke test (500 candles, 5 epochs — just verify the pipeline works)
python train.py --fast
```

Training saves three files:
- `checkpoints/best_model.pt`
- `checkpoints/scaler_candle.pkl`
- `checkpoints/scaler_market.pkl`

### 3. Run the live signal engine

```bash
python live.py
```

No API key required — uses public Binance Futures WebSocket + REST endpoints.

---

## Live Signal Output

At every 4H candle close, the system prints:

```
╔══════════════════════════════════════════════════════════╗
║  🕯  ETH/USDT 4H CANDLE CLOSED                          ║
╠══════════════════════════════════════════════════════════╣
║  Time (UTC):   2024-01-15 08:00:00 UTC                  ║
║  Close price:  $2,345.67                                 ║
║  Prediction:   +0.004231                                 ║
║  Direction:    LONG 📈                                   ║
║  Threshold:    |0.004231| >= 0.003? → True               ║
║  Status:       ✅ SIGNAL ACTIVE                           ║
╚══════════════════════════════════════════════════════════╝
```

After each subsequent candle:
```
┌─ RESULT ───────────────────────────
│  Prev Predicted: +0.004231 (LONG) | Threshold: ✓ YES
│  Actual move:    +0.006812 (LONG)
│  Correct:        ✓ WIN
│  Cumulative win rate: 62.5%  (5/8 signals)
│  Above-threshold win rate: 66.7%  (4/6 signals)
└────────────────────────────────────
```

All signals are also logged to `logs/signals.csv`.

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `SYMBOL` | `ETH/USDT:USDT` | CCXT perpetual futures symbol |
| `TIMEFRAME` | `4h` | Candle size |
| `LOOKBACK` | `100` | Sequence length fed to model |
| `TARGET_HORIZON` | `3` | Forward candles for target |
| `SIGNAL_THRESHOLD` | `0.003` | Minimum predicted move to emit signal |
| `LSTM_HIDDEN` | `128` | LSTM hidden size per branch |
| `CNN_FILTERS` | `64` | Conv1D output channels |
| `BATCH_SIZE` | `64` | Training batch size |
| `LEARNING_RATE` | `1e-3` | Adam initial LR |
| `MAX_EPOCHS` | `100` | Max training epochs |
| `PATIENCE` | `10` | Early stopping patience |

---

## EC2 Deployment

```bash
# 1. Copy project to EC2
scp -r . ubuntu@<EC2-IP>:~/trading-bot/

# 2. SSH in and run setup
ssh ubuntu@<EC2-IP>
cd ~/trading-bot
bash deploy/setup_ec2.sh

# 3. Train on EC2
source venv/bin/activate
python train.py

# 4. Start live engine in background
bash deploy/run_live.sh

# 5. Monitor
screen -r eth-live-signal
tail -f logs/signals.csv
```

---

## Feature Branches

### Branch 1 — Candle Structure (8 features)
`open, high, low, close, body, upper_wick, lower_wick, candle_range`

### Branch 2 — Market State (20 features)
`log_return, rsi_14, macd, macd_signal, macd_hist, ema_ratio_20_50, ema_ratio_50_200, atr_14, bb_upper, bb_middle, bb_lower, bb_width, stoch_k, stoch_d, adx_14, di_plus, di_minus, trend_flag, dist_from_high_20, dist_from_low_20`

---

## Future Upgrades
- [ ] REST fallback to detect missed candle closes
- [ ] Binance Futures API execution (market/limit orders)
- [ ] Risk management: position sizing, SL/TP logic
- [ ] Multi-timeframe ensemble (1H + 4H + 1D)
- [ ] Telegram/Discord alert integration
- [ ] Dashboard UI with signal history charts
