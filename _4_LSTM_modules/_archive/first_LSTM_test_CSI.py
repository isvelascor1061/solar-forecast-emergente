#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:22:52 2025

@author: leonardmerl

train_lstm_main.py

End-to-end-Script:
    • lädt train/val/test aus .npz
    • baut und trainiert ein LSTM (hyperparameter-steuerbar)
    • speichert bestes Modell + CSVs + Plots + run_info.txt
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from _4_LSTM_modules.NN_modules.first_LSTM_Model import LSTMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import trange

# ===================== USER CONFIG =====================================
RUN_NAME   = "GFS_all_Siata_all_normalized_min_max_49_sym"
SEQ_NPZ_PATH = "_4_LSTM_modules/Prepared_data/seq_sym25_shuffled.npz"

LR         = 1e-3
EPOCHS     = 50
BATCH_SIZE = 32
HIDDEN     = 64
MULTI_STEP = None
EARLY_STOP = 20
DROPOUT    = 0.15
NUM_LAYERS = 1
L2_LAMBDA  = 1e-4

RUN_DIR   = os.path.join(
    "_4_LSTM_modules/_runs",
    f"{RUN_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
MODEL_OUT = os.path.join(RUN_DIR, "best_model.pt")
LOSS_PLOT = os.path.join(RUN_DIR, "loss_curve.png")
PRED_PLOT = os.path.join(RUN_DIR, "pred_vs_true.png")
CSV_PRED  = os.path.join(RUN_DIR, "test_pred.csv")
HIST_PATH = os.path.join(RUN_DIR, "residual_hist.png")
INFO_TXT  = os.path.join(RUN_DIR, "run_info.txt")
# =======================================================================

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def write_run_info(filepath: str, run_name: str, hyperparams: dict, metrics: dict):
        """Schreibt hyperparams + metrics in eine Textdatei."""
        with open(filepath, "w") as f:
            f.write(f"Run: {run_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write("=== Hyperparameters ===\n")
            for k, v in hyperparams.items():
                f.write(f"{k:15}= {v}\n")
            f.write("\n=== Test Metrics ===\n")
            for k, v in metrics.items():
                f.write(f"{k:15}= {v:.4f}\n")

def load_splits(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["X_train"], data["y_train"], data["t_train"],
        data["X_val"],   data["y_val"],   data["t_val"],
        data["X_test"],  data["y_test"],  data["t_test"],
    )

def plot_loss(history, outfile):
    epochs, train_l, val_l = zip(*history)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_l, label="train")
    plt.plot(epochs, val_l,   label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.grid(True)
    plt.title(f"{RUN_NAME} — Loss Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(outfile, dpi=300); plt.close()

def plot_predictions(y_true, y_pred, outfile):
    n = min(len(y_true), 10000)
    x, y = y_true[:n], y_pred[:n]
    plt.figure(figsize=(5,5))
    plt.scatter(x, y, s=10, alpha=0.6)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.plot(lims, lims, "r--", lw=1.2)
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.title(f"{RUN_NAME} — True vs. Pred")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300); plt.close()

def plot_residuals(resid, outfile):
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=40, alpha=0.8, edgecolor="k")
    plt.xlabel("Residual (pred−true)"); plt.ylabel("Count")
    plt.title(f"{RUN_NAME} — Residuals")
    plt.grid(True); plt.tight_layout()
    plt.savefig(outfile, dpi=300); plt.close()

def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    # 1) Daten laden
    X_train, y_train, t_train, X_val, y_val, t_val, X_test, y_test, t_test = load_splits(SEQ_NPZ_PATH)
    train_dl = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(SeqDataset(X_val,   y_val  ), batch_size=BATCH_SIZE)
    test_dl  = DataLoader(SeqDataset(X_test,  y_test ), batch_size=BATCH_SIZE)

    # 2) Modell & Optimizer
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]
    model = LSTMRegressor(n_feat=n_feat, hidden=HIDDEN, seq_len=seq_len,
                          multi_step=MULTI_STEP, dropout=DROPOUT, num_layers=NUM_LAYERS)
    optim   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)
    loss_fn = nn.MSELoss()

    # 3) Training + EarlyStopping
    best_val, bad_epochs, history = float('inf'), 0, []
    for epoch in trange(1, EPOCHS+1, desc="Epochs"):
        model.train(); tl=0.0; cnt=0
        for xb, yb in train_dl:
            optim.zero_grad()
            l = loss_fn(model(xb), yb)
            l.backward(); optim.step()
            tl += l.item()*len(xb); cnt += len(xb)
        tl /= cnt

        model.eval(); vl=0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                vl += loss_fn(model(xb), yb).item()*len(xb)
        vl /= len(val_dl.dataset)

        history.append((epoch, tl, vl))
        if vl < best_val:
            best_val, bad_epochs = vl, 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP:
                print("Early stopping.") 
                break

    # 4) Auswertung + Plots + CSV
    plot_loss(history, LOSS_PLOT)
    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval(); preds=[]
    with torch.no_grad():
        for xb, _ in test_dl:
            preds.append(model(xb).numpy())
    y_pred = np.concatenate(preds)

    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0,1]

    plot_predictions(y_test, y_pred, PRED_PLOT)
    plot_residuals(y_pred - y_test, HIST_PATH)
    pd.DataFrame({"time":pd.to_datetime(t_test),"y_true":y_test,"y_pred":y_pred})\
      .to_csv(CSV_PRED, index=False)

    # 5) Run-Info schreiben
    hyperparams = {
        "LR": LR, "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE,
        "HIDDEN": HIDDEN, "MULTI_STEP": MULTI_STEP,
        "EARLY_STOP": EARLY_STOP, "DROPOUT": DROPOUT,
        "NUM_LAYERS": NUM_LAYERS, "L2_LAMBDA": L2_LAMBDA
    }
    metrics = {"MSE": mse, "MAE": mae, "R2": r2, "Corr": corr}
    SeqDataset.write_run_info(INFO_TXT, RUN_NAME, hyperparams, metrics)
    print(f"Run info saved → {INFO_TXT}")

if __name__ == "__main__":
    main()
