#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 11:40:15 2025

@author: leonardmerl
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

class MixedHuberMSELoss(nn.Module):
    """
    L = α · MSE  +  (1-α) · Huber(β)

    Args
    ----
    alpha : Gewicht des MSE-Anteils   (0…1)
    beta  : Huber-Schwelle            (entspricht `nn.SmoothL1Loss(beta)`)
    """
    def __init__(self, alpha: float = 0.5, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse   = F.mse_loss(y_pred, y_true)
        huber = F.smooth_l1_loss(y_pred, y_true, beta=self.beta)
        return self.alpha * mse + (1.0 - self.alpha) * huber
    
  

# _4_LSTM_modules/NN_modules/ramp_mse_loss.py


class RampMSELoss(nn.Module):
    """MSE + λ·MSE der 1. Differenz (nur falls vorhanden)"""
    def __init__(self, lam: float = 0.1):
        super().__init__()
        self.lam = lam

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mse_lvl = F.mse_loss(y_pred, y_true)

        # Ramp-Term nur, wenn letztes Dimension-Größer-1
        if y_pred.dim() >= 2 and y_pred.size(-1) > 1:
            diff   = y_pred[..., 1:] - y_pred[..., :-1]
            diff_t = y_true[..., 1:] - y_true[..., :-1]
            mse_ramp = F.mse_loss(diff, diff_t)
            return (1.0 - self.lam) * mse_lvl + self.lam * mse_ramp

        # 1-Step-Forecast → reine MSE
        return mse_lvl
    
class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for 0–1 normalized data: Nachtstunden (y≈0) haben
    geringes Gewicht, Tagsstunden (y≈1) hohes.
    L = mean( w * (ŷ - y)² ), with w = clamp(y, 0, 1)
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # w = 0…1: bei y_true≈0 (Nacht) w≈0, y_true≈1 (Tag) w≈1
        w = torch.clamp(y_true, 0.0, 1.0)
        return torch.mean(w * (y_pred - y_true) ** 2)
    
    
class NashSutcliffeLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true_f = y_true.view(y_true.size(0), -1)
        y_pred_f = y_pred.view_as(y_true_f)

        num = torch.sum((y_true_f - y_pred_f) ** 2, dim=1)
        den = torch.sum((y_true_f - y_true_f.mean(dim=1, keepdim=True)) ** 2,
                        dim=1) + self.eps
        loss = num / den                      # = 1 – NSE
        return loss.mean()

class MixedLoss(nn.Module):
    """
    alpha * NSE  +  (1-alpha) * MSE
    """
    def __init__(self, alpha: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.nse   = NashSutcliffeLoss(eps)
        self.mse   = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.alpha * self.nse(y_pred, y_true) + \
               (1 - self.alpha) * self.mse(y_pred, y_true)