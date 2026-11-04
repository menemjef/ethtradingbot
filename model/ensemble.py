"""
model/ensemble.py — Inference with multiple model checkpoints.
Averages predictions from 3 models trained with different seeds to boost accuracy.
"""

import torch
import torch.nn as nn
import os
from model.architecture import HybridModel
from config import CHECKPOINT_DIR

class EnsembleModel(nn.Module):
    def __init__(self, suffixes=["42", "1337", "7"], device="cpu"):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList()
        
        for s in suffixes:
            m_path = os.path.join(CHECKPOINT_DIR, f"best_model_{s}.pt")
            if os.path.exists(m_path):
                m = HybridModel().to(device)
                m.load_state_dict(torch.load(m_path, map_location=device))
                m.eval()
                self.models.append(m)
        
        if len(self.models) == 0:
            raise ValueError("No ensemble components found in checkpoints directory.")
        
        print(f"Ensemble initialized with {len(self.models)} models.")

    def forward(self, x_c, x_m):
        reg_total = 0
        clf_total = 0
        
        with torch.no_grad():
            for m in self.models:
                reg, clf = m(x_c, x_m)
                reg_total += reg
                clf_total += clf
        
        return reg_total / len(self.models), clf_total / len(self.models)
