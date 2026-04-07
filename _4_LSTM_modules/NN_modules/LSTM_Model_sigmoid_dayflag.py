#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:03:45 2025

@author: leonardmerl
"""
import torch.nn as nn
import torch

class LSTMRegressorsigmoidday(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, seq_len: int = 25, multi_step: int | None = None,
                 dropout: float = 0.2, num_layers: int = 2):
        super().__init__()
        self.n_feat = n_feat 
        self.seq_len = seq_len
        self.multi_step = multi_step
        
        self.encoder = nn.LSTM(
            input_size=self.n_feat,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, causal: bool = True):
        features = x[:, :, :]  
        out, _ = self.encoder(features)  # Shape: (Batch, Seq_len, Hidden)
        
        if causal:
            # Ausgabe am letzten Zeitschritt (nur Vergangenheit)
            idx = -1
        else:
            # Ausgabe am mittleren Zeitschritt (symmetrische Sequenz)
            idx = (self.seq_len-1) // 2
        
        selected_out = out[:, idx, :]
        pred = self.sigmoid(self.fc(selected_out))
        return pred.squeeze(-1)


