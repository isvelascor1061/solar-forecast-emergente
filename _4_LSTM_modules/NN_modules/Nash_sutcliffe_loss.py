#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:50:58 2025

@author: leonardmerl
"""

# ----------------------------------------------
# 1)  NSE-Loss definieren (z.B. in losses.py)
# ----------------------------------------------
import torch
import torch.nn as nn

class NashSutcliffeLoss(nn.Module):
    """Minimiert   1 – NSE   (perfektes Modell ⇒ 0)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true_f = y_true.view(y_true.size(0), -1)
        y_pred_f = y_pred.view_as(y_true_f)

        num = torch.sum((y_true_f - y_pred_f) ** 2, dim=1)                 # ∑(obs-sim)^2
        den = torch.sum((y_true_f - y_true_f.mean(dim=1, keepdim=True)) ** 2,
                        dim=1) + self.eps                                   # ∑(obs-mean)^2
        loss = num / den                                                   # = 1 - NSE
        return loss.mean()
    
class MixedLoss(nn.Module):
    def __init__(self, alpha=0.7, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.nse   = NashSutcliffeLoss(eps)
        self.mse   = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.alpha * self.nse(y_pred, y_true) + \
               (1 - self.alpha) * self.mse(y_pred, y_true)

loss_fn = MixedLoss(alpha=0.7)      # 70 % NSE, 30 % MSE
