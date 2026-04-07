#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 6 16:45:11 2025

@author: leonardmerl

LSTM with a linear output function.
"""
from __future__ import annotations
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, xarray as xr
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import trange
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from _4_LSTM_modules.NN_modules.LSTM_Model_linear import LSTMRegressorlinear  # Ändere den Import hier
from tqdm.auto import tqdm


# -------------------------------------------------------------------------
RUN_NAME   = "1feature_sym13_descaled_average"
SEQ_NPZ    = "_4_LSTM_modules/Prepared_data/1feat_seq_sym13_average_shuffle.npz"

# descaling-Spezifikation --------------------------------------------------
DESCALER   = (
    "_3_Data_preparation_for_LSTM/Preparation_data/Siata_GHI/Netcdf_Siata_GHI/SIATA_GHI_all.nc",
    "average",                #  | "z_score" | "average" | "none"
    "GHI"             # Name im NetCDF   (None→erstes DataVar)
)

# ---------------- Hyperparameter -----------------------------------------
LR= 1e-3
EPOCHS = 50
BATCH_SIZE= 32 
HIDDEN = 64 
DROPOUT= 0.15
NUM_LAYERS= 1
L2_LAMBDA= 1e-4
EARLY_STOP = 20
MULTI_STEP = None
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
_NOW   = datetime.now().strftime("%Y%m%d_%H%M%S")     # <─ EINMAL erzeugt
RUN_DIR = os.path.join("_4_LSTM_modules/_runs", f"{RUN_NAME}_{_NOW}")
os.makedirs(RUN_DIR, exist_ok=True)

PATHS = dict(
    model      = os.path.join(RUN_DIR, "best_model.pt"),
    csv_n      = os.path.join(RUN_DIR, "pred_norm.csv"),
    csv_r      = os.path.join(RUN_DIR, "pred_real.csv"),
    loss       = os.path.join(RUN_DIR, "loss_curve.png"),
    scatter_n  = os.path.join(RUN_DIR, "scatter_norm.png"),
    scatter_r  = os.path.join(RUN_DIR, "scatter_real.png"),
    hist_n     = os.path.join(RUN_DIR, "hist_norm.png"),
    hist_r     = os.path.join(RUN_DIR, "hist_real.png"),
    hist_n_zeros = os.path.join(RUN_DIR, "hist_norm_without_zeros.png"),  # Neuer Speicherort
    hist_r_zeros = os.path.join(RUN_DIR, "hist_real_without_zeros.png"),  # Neuer Speicherort
    summary    = os.path.join(RUN_DIR, "summary.png"),
    report     = os.path.join(RUN_DIR, "report.txt"),
)


# ----------------------------- Datensätze ---------------------------------
class SeqDS(torch.utils.data.Dataset):
    def __init__(self, X, y): self.X, self.y = map(torch.tensor,(X,y))
    def __len__(self): return len(self.y)
    def __getitem__(s,i): return s.X[i].float(), s.y[i].float()

def load_splits(path):
    d = np.load(path, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["t_train"],
            d["X_val"],   d["y_val"],   d["t_val"],
            d["X_test"],  d["y_test"],  d["t_test"])

# ------------------------- Descale-Utilities ------------------------------
def get_dataarray(path:str, var:str|None):
    ds = xr.open_dataset(path)
    if var is None:     var = list(ds.data_vars)[0]
    return ds[var].squeeze()

def build_stats(da: xr.DataArray, method:str):
    if method=="z_score":
        return {"mu": float(da.mean()), "sigma": float(da.std())}
    if method=="average":
        return {"mean": float(da.mean())}
    return {}

def descale(arr: np.ndarray, times: np.ndarray,
            method:str, da:xr.DataArray, stats:dict):

    if method=="z_score":
        return arr* stats["sigma"] + stats["mu"]
    if method=="average":
        return arr * stats["mean"]
    return arr  # "none"

# ------------------------- Plots ------------------------------------------
def plot_loss(hist, out,RUN_NAME):
    ep, tl, vl = zip(*hist)
    plt.figure(); plt.plot(ep,tl,label="train"); plt.plot(ep,vl,label="val")
    plt.title(f"{RUN_NAME}, Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

def scatter(y_t, y_p, out, ttl,RUN_NAME):
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    plt.figure(figsize=(8, 8))
    plt.scatter(y_t, y_p, s=8, alpha=.5)
    plt.plot(lims, lims, "r--")
    
    # Manuelle Änderung des Titels und der Achsenbeschriftungen
    if ttl == "Real":  # Nur für die echten Werte (descaled)
        plt.title(f"{RUN_NAME}, Predicted vs. Actual Values (Real)")
        plt.xlabel(r'$y_{true}$ (actual values in W/m²)')
        plt.ylabel(r'$y_{pred}$ (predicted values in W/m²)')
    else:  # Für normalisierte Werte
        plt.title(f"{RUN_NAME}, Predicted vs. Actual Values (Normalized)")
        plt.xlabel(r'$y_{true}$ (actual values normalised)')
        plt.ylabel(r'$y_{pred}$ (real values normalised)')

    plt.grid()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def hist(resid, out, ttl,RUN_NAME):
    plt.figure(); plt.hist(resid,bins=50,edgecolor="k",alpha=0.7); 
    if ttl == "Residual-Real":
        plt.title(f"{RUN_NAME}, Residuals Histogram")
        plt.xlabel(" Residual [W/m²]")
    else:
        plt.title(f"{RUN_NAME}, Residuals Histogram")
        plt.xlabel(" Residual")
        
    plt.ylabel("Frequency");plt.grid(True); plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    
def hist_without_zeros(resid, out, ttl, RUN_NAME):
    residuals_nonzero = resid[resid != 0]  # Entferne die Nullen
    plt.figure(figsize=(8, 6))
    plt.hist(residuals_nonzero, bins=50, edgecolor="k", alpha=0.7)
    if ttl =="Residuals-Real (without Zeros)":
        plt.xlabel("Residuals [W/m²]")
    else:
        plt.xlabel("Residuals")
    plt.title(f"{RUN_NAME}, Residuals Histogram (without Zeros)")
    plt.xlabel("Residuals [W/m²]")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def make_summary(paths: dict):
    # sicherstellen, dass alle Einzel-Plots bereits auf Platte liegen
    required = ["loss", "scatter_n", "scatter_r", "hist_n", "hist_r"]
    for k in required:
        if not os.path.isfile(paths[k]):
            raise FileNotFoundError(f"Plot fehlt: {paths[k]}")

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(RUN_NAME, fontsize=14)

    layout = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]   # raster 2×3
    for key, pos in zip(required, layout):
        ax = fig.add_subplot(2, 3, pos[0]*3 + pos[1] + 1)
        ax.imshow(plt.imread(paths[key]))
        ax.set_title(key.replace("_", " "))
        ax.axis("off")

    # leeres Feld unten links (0,0 schon belegt, 1,0 frei)
    ax_empty = fig.add_subplot(2, 3, 4)
    ax_empty.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(paths["summary"], dpi=300)
    plt.close(fig)


# -------------------------------- MAIN ------------------------------------
def main():
    # ---------- Daten ----------------------------------------------------
    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(SEQ_NPZ)
    tr_dl = DataLoader(SeqDS(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(SeqDS(X_va, y_va), BATCH_SIZE)
    te_dl = DataLoader(SeqDS(X_te, y_te), BATCH_SIZE)

    seq_len, n_feat = X_tr.shape[1], X_tr.shape[2]

    # ---------- Modell ---------------------------------------------------
    model = LSTMRegressorlinear(n_feat, HIDDEN, seq_len, MULTI_STEP, DROPOUT, NUM_LAYERS)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)
    loss_fn = nn.MSELoss()

    
    # ---------- Training  (Progress-Bar + feste Log-Zeile) -----------------
    best_val, patience, history = float("inf"), 0, []
    
    bar = trange(1, EPOCHS + 1, desc="Ep", unit="ep", leave=True)
    for ep in bar:
        # ------- Train -----------------------------------------------------
        model.train(); train_loss = 0.0
        for xb, yb in tr_dl:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(tr_dl.dataset)
    
        # ------- Validation ------------------------------------------------
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= len(va_dl.dataset)
    
        # Fortschritts-Anzeige aktualisieren
       
        tqdm.write(f"[{ep:3d}/{EPOCHS}] train {train_loss:.4f} | val {val_loss:.4f}")
    
        # Verlaufs-Log für Plot
        history.append((ep, train_loss, val_loss))
    
        # ------- Early-Stopping -------------------------------------------
        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), PATHS["model"])
        else:
            patience += 1
            if patience >= EARLY_STOP:
                tqdm.write("Early-Stopping ausgelöst.")
                break

    plot_loss(history, PATHS["loss"], RUN_NAME)      
    # ---------- Test (Norm-Raum) ----------------------------------------
    model.load_state_dict(torch.load(PATHS["model"])); model.eval()
    y_pred = np.concatenate([model(xb).detach().numpy() for xb,_ in te_dl])
    

    metrics_n = dict(
        MSE = mean_squared_error(y_te, y_pred),
        MAE = mean_absolute_error(y_te, y_pred),
        R2  = r2_score(y_te, y_pred),
        Corr= np.corrcoef(y_te, y_pred)[0,1]
    )
    
    # ---------- Berechne Varianz der Residuen (Norm-Raum) ---------------
    residuals_n = y_te - y_pred
    residual_variance_n = np.var(residuals_n)  # Varianz der Residuen im normierten Raum
    metrics_n["Residual_Variance"] = residual_variance_n  # Varianz der Residuen im normierten Raum

    pd.DataFrame({"time":pd.to_datetime(t_te), "y_true":y_te, "y_pred":y_pred}
                 ).to_csv(PATHS["csv_n"], index=False)
    scatter(y_te, y_pred, PATHS["scatter_n"], "Norm", RUN_NAME)
    hist(y_pred - y_te, PATHS["hist_n"], "Residual-Norm", RUN_NAME)
    hist_without_zeros(y_pred - y_te, PATHS["hist_n_zeros"], "Residuals-Norm (without Zeros)", RUN_NAME)

    # ---------- Descale --------------------------------------------------
    src_path, method, var = DESCALER
    da = get_dataarray(src_path, var)
    stats = build_stats(da, method)
    y_true_r = descale(y_te, t_te, method, da, stats)
    y_pred_r = descale(y_pred, t_te, method, da, stats)

    # ---------- Berechne Varianz der Residuen (Real-Raum) --------------
    residuals_r = y_true_r - y_pred_r
    residual_variance_r = np.var(residuals_r)  # Varianz der Residuen im realen Raum
    metrics_r = dict(
        MSE = mean_squared_error(y_true_r, y_pred_r),
        RMSE= root_mean_squared_error(y_true_r, y_pred_r),
        MAE = mean_absolute_error(y_true_r, y_pred_r),
        R2  = r2_score(y_true_r, y_pred_r),
        Corr= np.corrcoef(y_true_r, y_pred_r)[0,1],
        Residual_Variance = residual_variance_r  # Varianz der Residuen im realen Raum
    )

    print("\n=== Evaluation (normierter Raum) ===")
    for k, v in metrics_n.items():
        print(f"{k:>5}: {v:8.4f}")
    
    print("\n=== Evaluation (reales Skalen-Raum) ===")
    for k, v in metrics_r.items():
        print(f"{k:>5}: {v:8.4f}")
    print()

    pd.DataFrame({"time":pd.to_datetime(t_te), "y_true":y_true_r, "y_pred":y_pred_r}
                 ).to_csv(PATHS["csv_r"], index=False)
    scatter(y_true_r, y_pred_r, PATHS["scatter_r"], "Real", RUN_NAME)
    hist(y_pred_r - y_true_r, PATHS["hist_r"], "Residual-Real", RUN_NAME)
    hist_without_zeros(y_pred_r - y_true_r, PATHS["hist_r_zeros"], "Residuals-Real (without Zeros)", RUN_NAME)
    
    # ---------- Summary-PNG + Report ------------------------------------
    make_summary(PATHS)
    with open(PATHS["report"], "w") as f:
        f.write(f"# LSTM Run Report — {RUN_NAME}\n\n")
        f.write("## Hyperparameter\n")
        for k, v in dict(LR=LR, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, HIDDEN=HIDDEN,
                         DROPOUT=DROPOUT, NUM_LAYERS=NUM_LAYERS, L2_LAMBDA=L2_LAMBDA,
                         EARLY_STOP=EARLY_STOP).items():
            f.write(f"{k}: {v}\n")
        f.write("\n## Metrics (Norm)\n")
        for k, v in metrics_n.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n## Metrics (Real)\n")
        for k, v in metrics_r.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n## Descale\n")
        f.write(json.dumps({"method": method, "file": src_path, "variable": var}, indent=2))
    print("Alle Artefakte →", RUN_DIR)

if __name__ == "__main__":
    main()
