"""
model/dataset.py — PyTorch Dataset for sliding-window OHLCV sequences.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ETHDataset(Dataset):
    """
    Dataset wrapping pre-built sliding windows.

    Args:
        X_candle: (N, lookback, CANDLE_DIM)  float32 array
        X_market: (N, lookback, MARKET_DIM)  float32 array
        y:        (N,)                        float32 target array

    Each __getitem__ returns:
        candle_seq (Tensor), market_seq (Tensor), target (scalar Tensor)
    """

    def __init__(
        self,
        X_candle: np.ndarray,
        X_market: np.ndarray,
        y: np.ndarray,
    ):
        assert len(X_candle) == len(X_market) == len(y), "Array length mismatch"
        self.X_candle = torch.from_numpy(X_candle)
        self.X_market = torch.from_numpy(X_market)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            self.X_candle[idx],    # (lookback, CANDLE_DIM)
            self.X_market[idx],    # (lookback, MARKET_DIM)
            self.y[idx],           # scalar
        )
