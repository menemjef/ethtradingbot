"""
model/architecture.py — Multi-Task Hybrid LSTM-Transformer for directional trading.

WHAT THIS FILE DOES:
This file defines the "Brain" (Neural Network) of the trading bot using PyTorch.

Architecture:
  - Input: Concatenated Candle + Market features (The data prepared in step 2)
  - Sequential Processing: Bidirectional LSTM to capture temporal dynamics (understanding patterns over time).
  - Attention Processing: Transformer Encoder (understanding which time periods are most important right now).
  - Tasks (Multi-task learning):
      1. Regression Head: Predicted log-return magnitude (How BIG will the move be?)
      2. Classification Head: Predicted direction (Logit for Up/Down) (Which WAY will it go?)
"""

import torch
import torch.nn as nn
from config import (
    CANDLE_DIM,
    MARKET_DIM,
    LOOKBACK,
    DROPOUT,
    ATTENTION_HEADS,
    ATTENTION_DROPOUT,
    TRANSFORMER_LAYERS,
    TRANSFORMER_DIM,
)

class TransformerBackbone(nn.Module):
    """
    Hybrid LSTM-Transformer backbone.
    
    Why this combination?
    - LSTMs are great at reading time-series data sequentially (like reading a book left to right).
    - Transformers are great at "Attention" — looking back at specific moments in the past 
      (like remembering a plot twist from chapter 1 while reading chapter 5).
    Combining them gives us the best of both worlds for crypto prediction.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        lstm_hidden = TRANSFORMER_DIM // 2  # 64

        # LSTM pre-processor for sequential pattern extraction
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.lstm_dropout = nn.Dropout(p=DROPOUT)  # Regularize LSTM output
        # LSTM output is 2*lstm_hidden = TRANSFORMER_DIM (bidirectional)

        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, LOOKBACK, TRANSFORMER_DIM))
        self.pos_dropout = nn.Dropout(p=DROPOUT) # Added positional dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=TRANSFORMER_DIM * 4, # Increased capacity
            dropout=ATTENTION_DROPOUT,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_LAYERS,
            enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(TRANSFORMER_DIM)
        self.final_dropout = nn.Dropout(p=DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        lstm_out, _ = self.lstm(x)              # (B, T, TRANSFORMER_DIM)
        lstm_out = self.lstm_dropout(lstm_out)   # Regularize
        x = lstm_out + self.pos_encoder         # (B, T, TRANSFORMER_DIM)
        x = self.pos_dropout(x)                 # Regularize after pos encoding
        x = self.transformer(x)                 # (B, T, TRANSFORMER_DIM)
        # Use both last timestep and mean pooling for richer representation
        last = x[:, -1, :]                      # (B, TRANSFORMER_DIM)
        mean = x.mean(dim=1)                    # (B, TRANSFORMER_DIM)
        out = last + mean                       # (B, TRANSFORMER_DIM)
        out = self.norm(out)
        return self.final_dropout(out)

class HybridModel(nn.Module):
    """
    Unified Multi-Task LSTM-Transformer model.
    This is the final wrapper around the "Backbone". Once the backbone has processed 
    the data and understands the current market state, it passes that understanding 
    to TWO separate "Heads":
    
    1. Regression Head: Tries to guess the exact numerical price change.
    2. Classification Head: Tries to simply guess UP or DOWN.
    
    Training both together forces the model to learn a deeper representation of the market.
    """
    def __init__(self):
        super().__init__()
        total_dim = CANDLE_DIM + MARKET_DIM
        import logging
        logging.getLogger("model").info(f"HybridModel initialized: Candle({CANDLE_DIM}) + Market({MARKET_DIM}) = {total_dim}")
        self.backbone = TransformerBackbone(total_dim)

        # Output Head 1: Regression (Expected move magnitude)
        self.reg_head = nn.Sequential(
            nn.Linear(TRANSFORMER_DIM, TRANSFORMER_DIM // 2),
            nn.GELU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(TRANSFORMER_DIM // 2, 1)
        )

        # Output Head 2: Classification (Up/Down Direction) — Simplified to prevent overfitting
        self.clf_head = nn.Sequential(
            nn.Linear(TRANSFORMER_DIM, TRANSFORMER_DIM // 2),
            nn.GELU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(TRANSFORMER_DIM // 2, 1)
        )

    def forward(self, candle_seq: torch.Tensor, market_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([candle_seq, market_seq], dim=-1)
        h = self.backbone(x)

        reg_out = self.reg_head(h)
        clf_out = self.clf_head(h)

        return reg_out, clf_out

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = HybridModel()
    print(f"HybridModel parameters: {count_parameters(model):,}")

    B = 4
    x1 = torch.randn(B, LOOKBACK, CANDLE_DIM)
    x2 = torch.randn(B, LOOKBACK, MARKET_DIM)
    reg, clf = model(x1, x2)
    print(f"Reg shape: {reg.shape} | Clf shape: {clf.shape}")
    assert reg.shape == (B, 1) and clf.shape == (B, 1), "Shape mismatch!"
    print("All checks passed.")
