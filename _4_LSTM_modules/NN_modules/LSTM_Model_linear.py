#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 16:41:50 2025

@author: leonardmerl
"""

from __future__ import annotations
import torch
import torch.nn as nn

class LSTMRegressorlinear(nn.Module):
    """Many‑to‑One oder Many‑to‑Many (optional) LSTM‑Regressor.

    Parameters
    ----------
    n_feat : int
        Anzahl Feature‑Kanäle (Input‑Dim pro Zeitschritt).
    hidden : int | tuple[int,int]
        Hidden‑Size(n). Entweder eine Zahl (1 LSTM‑Schicht) oder Tuple für
        (hidden_encoder, hidden_decoder).
    seq_len : int
        Länge des Eingabe‑Fensters (z. B. 25 bei symmetrisch k=12).
    multi_step : int | None, default None
        Wenn `None`: Many‑to‑One (ein Skalar)
        Wenn `m > 1`: Many‑to‑Many (m Outputs)
    dropout : float, default 0.0
        Dropout‑Rate zwischen LSTM‑Layer(s) und Dense‑Layer.
    """

    def __init__(
        self,
        n_feat: int,
        hidden: int | tuple[int, int] = 64,
        seq_len: int = 25,
        multi_step: int | None = None,
        dropout: float = 0.2,
        num_layers: int = 2  # Anzahl der LSTM-Schichten
    ) -> None:
        super().__init__()
        self.n_feat = n_feat
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.multi_step = multi_step  # None = 1‑Schritt

        if isinstance(hidden, int):
            hidden_enc = hidden
            hidden_dec = hidden
        else:
            hidden_enc, hidden_dec = hidden

        # Encoder: gesamte Sequenz → letzter Hidden‑State
        self.encoder = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden_enc,
            num_layers=num_layers,  # Anzahl der LSTM-Schichten
            batch_first=True,
        )

        if multi_step is None or multi_step == 1:
            # Many‑to‑One
            self.fc = nn.Linear(hidden_enc, 1)
        else:
            # Many‑to‑Many  (Repeat + Decoder‑LSTM + TimeDistributed Dense)
            self.repeat = nn.Linear(hidden_enc, hidden_dec)
            self.decoder = nn.LSTM(
                input_size=hidden_dec,
                hidden_size=hidden_dec,
                num_layers=num_layers,  # Anzahl der LSTM-Schichten
                batch_first=True,
            )
            self.out_dense = nn.Linear(hidden_dec, 1)
            self.multi_step = int(multi_step)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = self.dropout(h[-1])

        if self.multi_step is None or self.multi_step == 1:
            out = self.fc(h).squeeze(-1)       # (B,)
            return out                          # Linear statt Sigmoid
        else:
            rep = self.repeat(h).unsqueeze(1).repeat(1, self.multi_step, 1)
            dec_out, _ = self.decoder(rep)
            out = self.out_dense(dec_out)      # (B, m, 1)
            return out.squeeze(-1)             # (B, m)  (Linear, keine Sigmoid)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ----------------- Test ----------------------------------------------------
if __name__ == "__main__":
    B, L, F = 8, 25, 2
    x = torch.randn(B, L, F)

    # Many‑to‑One (Beispiel)
    model1 = LSTMRegressorlinear(n_feat=F, seq_len=L, hidden=64)
    print("One‑step params:", model1.n_parameters)
    yhat = model1(x)
    print("Output shape:", yhat.shape)   # (B,)

    # Many‑to‑Many (Beispiel m=24)
    model2 = LSTMRegressorlinear(n_feat=F, seq_len=L, hidden=(64,32), multi_step=24)
    print("Multi‑step params:", model2.n_parameters)
    yhat2 = model2(x)
    print("Output shape:", yhat2.shape)  # (B,24)
