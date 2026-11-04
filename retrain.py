"""
retrain.py — Automated retraining script for adapting to new feature sets.
"""

import logging
import os
import torch
import torch.nn as nn
from data.fetcher import fetch_ohlcv
from data.preprocessor import build_features, transform, build_windows, fit_and_save_scalers
from model.architecture import HybridModel
from config import SYMBOL, TIMEFRAME, LOOKBACK, MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrain")

def run_retrain(limit=2000):
    logger.info(f"Starting auto-retrain for {SYMBOL} with expanded features...")
    
    # 1. Fetch main data
    from data.fetcher import fetch_funding_rates, fetch_open_interest
    df_raw = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    
    # 2. Fetch External and MTF Data (Phase 5 fix)
    logger.info("Fetching funding rates and open interest...")
    funding_df = fetch_funding_rates(SYMBOL, limit=limit)
    oi_df = fetch_open_interest(SYMBOL, limit=limit)
    
    logger.info("Fetching multi-timeframe candles (15m, 4h)...")
    df_15m = fetch_ohlcv(SYMBOL, "15m", limit=limit*4) # More 15m needed for alignment
    df_4h = fetch_ohlcv(SYMBOL, "4h", limit=limit//4)
    
    # 3. Build Features (using the new config-defined MARKET_FEATURES)
    df_feat = build_features(
        df_raw, 
        add_targets=True,
        funding_df=funding_df,
        oi_df=oi_df,
        df_15m=df_15m,
        df_4h=df_4h
    )
    
    # 3. Fit new scalers ONLY on train split
    from config import TRAIN_SPLIT
    split_idx = int(len(df_feat) * TRAIN_SPLIT)
    scaler_candle, scaler_market = fit_and_save_scalers(df_feat.iloc[:split_idx])
    
    # 4. Prepare windows
    candle_arr, market_arr = transform(df_feat, scaler_candle, scaler_market)
    targets = df_feat["target"].values.astype("float32")
    X_candle, X_market, y = build_windows(candle_arr, market_arr, targets)
    
    # 5. Initialize Fresh Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridModel().to(device)
    
    # 6. Mini-Training Loop (Proof of concept)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_reg = nn.MSELoss()
    criterion_clf = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(10): # Quick 10 epoch retrain
        x_c = torch.tensor(X_candle, dtype=torch.float32).to(device)
        x_m = torch.tensor(X_market, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        y_clf = (y_tensor > 0).float().unsqueeze(-1)
        
        reg_out, clf_out = model(x_c, x_m)
        loss = criterion_reg(reg_out.squeeze(), y_tensor) + criterion_clf(clf_out, y_clf)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            logger.info(f"Epoch {epoch} | Loss: {loss.item():.6f}")
            
    # 7. Save updated model
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Retraining complete. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    run_retrain(limit=3000)
