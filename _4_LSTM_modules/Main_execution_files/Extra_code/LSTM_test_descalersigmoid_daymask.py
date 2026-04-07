#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:25:16 2025

@author: leonardmerl
"""
from __future__ import annotations
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, xarray as xr
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from tqdm import trange
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from _4_LSTM_modules.NN_modules.LSTM_Model_sigmoid_daymask import LSTMRegressorsigmoidday
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Settings ---------------------------------------------
RUN_NAME   = "1feat_sym13_CSI_Numl_8_hidden_128_filter"
SEQ_NPZ    = "_4_LSTM_modules/Prepared_data/1feat_seq_sym13_CSI_shuffle.npz"
DESCALER   = (
    "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/CSI_GHI_grid25_avg_with_horizon_and_enhancement.nc",
    "clear_sky",
    "clear_sky_ghi"
)

DAY_START, DAY_END = 6, 19  # Definiere Tageszeiten

LR_INIT   = 1e-3
MIN_LR    = 1e-6   
EPOCHS = 75
BATCH_SIZE= 32 
HIDDEN = 128 
DROPOUT= 0.15
NUM_LAYERS= 8
L2_LAMBDA= 1e-4
EARLY_STOP = 20
MULTI_STEP = None
LR_FACTOR = 0.5
LR_PATIENCE = 5

_NOW   = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("_4_LSTM_modules/_runs/1feat_sym13_CSI_MSEloss_daymask", f"{RUN_NAME}_{_NOW}")
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
    hist_n_zeros = os.path.join(RUN_DIR, "hist_norm_without_zeros.png"),
    hist_r_zeros = os.path.join(RUN_DIR, "hist_real_without_zeros.png"),
    summary    = os.path.join(RUN_DIR, "summary.png"),
    report     = os.path.join(RUN_DIR, "report.txt"),
)

# ---------------- DataSet mit Maske -------------------------------------
class SeqDS(torch.utils.data.Dataset):
    def __init__(self, X, y, m):
        self.X, self.y, self.m = map(torch.tensor, (X, y, m))
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i].float(), self.y[i].float(), self.m[i].float()

def load_splits(path):
    d = np.load(path, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["t_train"],
            d["X_val"], d["y_val"], d["t_val"],
            d["X_test"], d["y_test"], d["t_test"])



def fmt_de(val, prec=4):
    """
    Formatiert Zahl im Stil 1.234.567,8910.
    prec = Nachkommastellen (nur für float).
    """
    if isinstance(val, (float, np.floating)):
        s = f"{val:,.{prec}f}"
    else:                       # int, np.integer
        s = f"{val:,}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def pprint_de(title: str, d: dict):
    print(f"\n=== {title} ===")
    for k, v in d.items():
        prec = 4 if isinstance(v, (float, np.floating)) else 0
        print(f"{k:>5}: {fmt_de(v, prec)}")




# ----------- Andere Utilities (formatieren, descale, plots, etc.) ---------
def get_dataarray(path:str, var:str|None):
    ds = xr.open_dataset(path)
    if var is None: var = list(ds.data_vars)[0]
    return ds[var].squeeze()

def build_stats(da: xr.DataArray, method:str):
    if method=="z_score":
        return {"mu": float(da.mean()), "sigma": float(da.std())}
    if method=="average":
        return {"mean": float(da.mean())}
    return {}

def descale(arr, times, method, da, stats=None):
    if method == "clear_sky":
        return arr * da.sel(observation_time=times).values
    elif method == "minmax":
        return arr * (stats["max"] - stats["min"]) + stats["min"]
   
    elif method == "none":
        return arr
    else:
        raise ValueError(f"Unbekannte Deskalierungsmethode: {method}")

# ... (plot_loss, scatter, hist, hist_without_zeros, make_summary gleich wie vorher)
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


def make_summary_daymask(paths: dict, metrics_n: dict,
                          metrics_r: dict,
                         hparams: dict):
    """
    3 × 3 Canvas Layout:
       ┌───────────────┬───────────────┬───────────────┐
       │ Scatter Norm  │ Scatter Real  │ Loss-Curve    │  (Zeile 0)
       ├───────────────┼───────────────┼───────────────┤
       │ Hist Norm     │ Hist Real     │ Hyperparams   │  (Zeile 1)
       ├───────────────┼───────────────┼───────────────┤
       │ Hist Norm wo0 │ Hist Real wo0 │ Metrics Real  │  (Zeile 2)
       └───────────────┴───────────────┴───────────────┘
    """
    

    fig = plt.figure(figsize=(16, 15))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])

    # Erste Zeile: Scatter Norm, Scatter Real, Loss Curve
    axes_files = [
        (gs[0, 0], paths["scatter_n"], "Scatter Norm"),
        (gs[0, 1], paths["scatter_r"], "Scatter Real"),
        (gs[0, 2], paths["loss"],      "Loss Curve"),
        # Zweite Zeile: Hist Norm, Hist Real, Hyperparams (Hyperparams als Tabelle)
        (gs[1, 0], paths["hist_n"],   "Hist Norm"),
        (gs[1, 1], paths["hist_r"],   "Hist Real"),
    ]

    # Zeichne Bild-Plots (Scatter, Loss, Hist)
    for cell, fname, ttl in axes_files:
        ax = fig.add_subplot(cell)
        img = plt.imread(fname)
        ax.imshow(img)
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")

    # Zweite Zeile, dritte Spalte: Hyperparameter Tabelle
    ax_hp = fig.add_subplot(gs[1, 2])
    ax_hp.axis("off")
    hp_tab = [[k, str(v)] for k, v in hparams.items()]
    tbl_hp = ax_hp.table(cellText=hp_tab,
                         colLabels=["Hyperparameter", "Value"],
                         loc="center", cellLoc="center")
    tbl_hp.auto_set_font_size(False)
    tbl_hp.set_fontsize(9)
    tbl_hp.scale(1.2, 1.4)
    ax_hp.set_title("Hyperparameters", fontsize=11, pad=6)

    # Dritte Zeile: Hist Norm ohne Nullwerte, Hist Real ohne Nullwerte, Metrics Real
    ax_hist_n_zeros = fig.add_subplot(gs[2, 0])
    img_hist_n_zeros = plt.imread(paths["hist_n_zeros"])
    ax_hist_n_zeros.imshow(img_hist_n_zeros)
    ax_hist_n_zeros.set_title("Hist Norm wo 0", fontsize=11)
    ax_hist_n_zeros.axis("off")

    ax_hist_r_zeros = fig.add_subplot(gs[2, 1])
    img_hist_r_zeros = plt.imread(paths["hist_r_zeros"])
    ax_hist_r_zeros.imshow(img_hist_r_zeros)
    ax_hist_r_zeros.set_title("Hist Real wo 0", fontsize=11)
    ax_hist_r_zeros.axis("off")

    # Dritte Zeile, dritte Spalte: Metrics Tabelle (nur Real)
    ax_met = fig.add_subplot(gs[2, 2])
    ax_met.axis("off")
    rows = ["MSE", "RMSE", "MAE", "R2", "Corr", "Residual_Variance"]
    tab_met = [[r, f"{metrics_r[r]:.3f}"] for r in rows if r in metrics_r]
    tbl_met = ax_met.table(cellText=tab_met,
                           colLabels=["Metric", "Real"],
                           loc="center", cellLoc="center")
    tbl_met.auto_set_font_size(False)
    tbl_met.set_fontsize(9)
    tbl_met.scale(1.1, 1.5)
    ax_met.set_title("Evaluation Metrics (Real)", fontsize=11, pad=6)

    fig.suptitle(RUN_NAME, fontsize=17, y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(paths["summary"], dpi=300)
    plt.close(fig)


# -------------------------------- MAIN ------------------------------------
def main():
    # ---------- Daten laden --------------------------------------------
    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(SEQ_NPZ)
    
    # ---------- Sequenzlänge ermitteln --------------------------------
    seq_len = X_tr.shape[1]  # z.B. 13 für ±6 Stunden
    
    # ---------- Hilfsfunktionen ----------------------------------------
    def process_time_data(times, seq_len):
        """Verarbeitet Zeitstempel zu Tageszeit-Features und trimmt auf volle Sequenzen"""
        # 1. Zeitstempel zu Stunden konvertieren
        hours = pd.to_datetime(times).hour.values
        
        # 2. Auf volle Sequenzen trimmen
        n_total = len(hours)
        n_sequences = n_total // seq_len
        trimmed_length = n_sequences * seq_len
        hours_trimmed = hours[:trimmed_length]
        
        # 3. Tageszeit-Feature erstellen (1=Tag, 0=Nacht)
        day_feature = ((hours_trimmed >= DAY_START) & (hours_trimmed < DAY_END)).astype(np.float32)
        
        # 4. In Sequenzform bringen (n_sequences, seq_len, 1)
        return day_feature.reshape(n_sequences, seq_len, 1)

    # ---------- Tageszeit-Feature hinzufügen --------------------------
    def add_day_feature(X, times):
        day_feature = process_time_data(times, seq_len)
        # Trimme X auf dieselbe Anzahl an Sequenzen
        X_trimmed = X[:day_feature.shape[0]]
        return np.concatenate([X_trimmed, day_feature], axis=-1)
    
    X_tr = add_day_feature(X_tr, t_tr)
    X_va = add_day_feature(X_va, t_va)
    X_te = add_day_feature(X_te, t_te)
    
    # ---------- Masken erstellen --------------------------------------
    def create_masks(times, seq_len):
        day_feature = process_time_data(times, seq_len)
        return day_feature.squeeze(-1)  # Shape: (n_sequences, seq_len)
    
    m_tr = create_masks(t_tr, seq_len)
    m_va = create_masks(t_va, seq_len)
    m_te = create_masks(t_te, seq_len)
    
    # ---------- Targets trimmen & maskieren --------------------------
    y_tr = y_tr[:m_tr.shape[0]] * m_tr
    y_va = y_va[:m_va.shape[0]] * m_va
    y_te = y_te[:m_te.shape[0]] * m_te
    
    # ---------- Konsistenzprüfung ------------------------------------
    assert X_tr.shape[0] == m_tr.shape[0], "Anzahl der Sequenzen stimmt nicht überein!"
    assert X_tr.shape[1] == seq_len, "Sequenzlänge inkonsistent!"
    
    # ... Rest des Codes (Datensätze, DataLoader, Training) unverändert ...
    # ---------- Datensätze erstellen ----------------------------------
    tr_ds = SeqDS(X_tr, y_tr, m_tr)
    va_ds = SeqDS(X_va, y_va, m_va)
    te_ds = SeqDS(X_te, y_te, m_te)

    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(va_ds, BATCH_SIZE)
    te_dl = DataLoader(te_ds, BATCH_SIZE)
    assert len(t_tr) == X_tr.shape[0] * X_tr.shape[1]  # n_samples * seq_len
    # ---------- Modell initialisieren --------------------------------
    seq_len, n_feat = X_tr.shape[1], X_tr.shape[2]  # Wichtig: mit Day-Feature!
    
    model = LSTMRegressorsigmoidday(
        n_feat=n_feat,
        hidden=HIDDEN,
        seq_len=seq_len,
        multi_step=MULTI_STEP,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS
    )
    
    optim = torch.optim.AdamW(model.parameters(), 
                             lr=LR_INIT, 
                             weight_decay=L2_LAMBDA)
    loss_fn = nn.MSELoss(reduction="none")
    
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=LR_FACTOR,
                                  patience=LR_PATIENCE, threshold=1e-4,
                                  min_lr=MIN_LR, verbose=True)

    # ---------- Training -------------------------------------------------
    best_val, patience, history = float("inf"), 0, []
    bar = trange(1, EPOCHS+1, desc="Ep", unit="ep", leave=True)

    for ep in bar:
        # ------- Train ---------------------------------------------------
        model.train()
        train_loss = 0.0
        for xb, yb, mb in tr_dl:
            optim.zero_grad()
            preds = model(xb)
            # Maske im Loss: nur Tageswerte berücksichtigen
            loss = (loss_fn(preds, yb) * mb.unsqueeze(1)).mean()
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(tr_dl.dataset)

        # ------- Validation ---------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, mb in va_dl:
                loss = (loss_fn(model(xb), yb) * mb.unsqueeze(1)).mean()
                val_loss += loss.item() * len(xb)
        val_loss /= len(va_dl.dataset)

        scheduler.step(val_loss)

        tqdm.write(f"[{ep:3d}/{EPOCHS}] train {train_loss:.6f} | val {val_loss:.6f} | lr {optim.param_groups[0]['lr']:.1e}")
        history.append((ep, train_loss, val_loss))

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), PATHS["model"])
        else:
            patience += 1
            if patience >= EARLY_STOP:
                tqdm.write("Early-Stopping ausgelöst."); break

    plot_loss(history, PATHS["loss"],RUN_NAME)

    # ---------- Test-Vorhersagen (normiert, ohne Maske) -----------------------------
    model.load_state_dict(torch.load(PATHS["model"]))
    model.eval()

    y_pred = np.concatenate([model(xb).detach().numpy() for xb, _, _ in te_dl])

    # Evaluation **ohne** Maske, also alle Zeitpunkte auswerten
    metrics_n = dict(
        MSE=mean_squared_error(y_te, y_pred),
        MAE=mean_absolute_error(y_te, y_pred),
        R2=r2_score(y_te, y_pred),
        Corr=np.corrcoef(y_te, y_pred)[0, 1]
    )

    residuals_n = y_te - y_pred
    residual_variance_n = np.var(residuals_n)
    metrics_n["Residual_Variance"] = residual_variance_n

    print("\n=== Evaluation (normierter Raum, alle Zeiten) ===")
    for k, v in metrics_n.items():
        print(f"{k:>5}: {v:.4f}")
    # Optional: Tagesmetriken separat
    y_te_flat = y_te.flatten()
    y_pred_flat = y_pred.flatten()
    m_te_flat = m_te.flatten()
    yt_day = y_te_flat[m_te_flat == 1]
    yp_day = y_pred_flat[m_te_flat == 1]

    metrics_day = dict(
        MSE=mean_squared_error(yt_day, yp_day),
        MAE=mean_absolute_error(yt_day, yp_day),
        R2=r2_score(yt_day, yp_day),
        Corr=np.corrcoef(yt_day, yp_day)[0, 1]
    )

    print("\n=== Evaluation (normierter Raum, alle Zeiten) ===")
    for k, v in metrics_n.items():
        print(f"{k:>15}: {fmt_de(v, prec=4)}")


    print("\n=== Evaluation (normierter Raum, nur Tageswerte) ===")
    for k, v in metrics_day.items():
        print(f"{k:>15}: {fmt_de(v, prec=4)}")
    # Residuen und CSV
    residuals_n = y_te - y_pred
    residual_variance_n = np.var(residuals_n)
    metrics_n["Residual_Variance"] = residual_variance_n

    pd.DataFrame({"time": pd.to_datetime(t_te), "y_true": y_te.flatten(), "y_pred": y_pred.flatten()}).to_csv(PATHS["csv_n"], index=False)
    scatter(y_te.flatten(), y_pred.flatten(), PATHS["scatter_n"], "Norm", RUN_NAME)
    hist(y_pred.flatten() - y_te.flatten(), PATHS["hist_n"], "Residual-Norm", RUN_NAME)
    hist_without_zeros(y_pred.flatten() - y_te.flatten(), PATHS["hist_n_zeros"], "Residuals-Norm (without Zeros)", RUN_NAME)

    # Deskalierung
    src_path, method, var = DESCALER
    da = get_dataarray(src_path, var)
    stats = build_stats(da, method)
    y_true_r = descale(y_te.flatten(), t_te, method, da, stats)
    y_pred_r = descale(y_pred.flatten(), t_te, method, da, stats)
    
    yt_day_r = y_true_r[m_te_flat == 1]  # m_te_flat ist bereits aus vorherigem Code
    yp_day_r = y_pred_r[m_te_flat == 1]

    metrics_day_r = dict(
        MSE=mean_squared_error(yt_day_r, yp_day_r),
        RMSE=np.sqrt(mean_squared_error(yt_day_r, yp_day_r)),
        MAE=mean_absolute_error(yt_day_r, yp_day_r),
        R2=r2_score(yt_day_r, yp_day_r),
        Corr=np.corrcoef(yt_day_r, yp_day_r)[0, 1]
    )

    print("\n=== Evaluation (realer Raum, nur Tageswerte) ===")
    for k, v in metrics_day_r.items():
        print(f"{k:>15}: {fmt_de(v, prec=4)}")

    residuals_r = y_true_r - y_pred_r
    residual_variance_r = np.var(residuals_r)
    metrics_r = dict(
        MSE=mean_squared_error(y_true_r, y_pred_r),
        RMSE=np.sqrt(mean_squared_error(y_true_r, y_pred_r)),
        MAE=mean_absolute_error(y_true_r, y_pred_r),
        R2=r2_score(y_true_r, y_pred_r),
        Corr=np.corrcoef(y_true_r, y_pred_r)[0, 1],
        Residual_Variance=residual_variance_r
    )
    
    print("\n=== Evaluation (realer Raum, alle Zeiten) ===")
    for k, v in metrics_r.items():
        print(f"{k:>15}: {fmt_de(v, prec=4)}")

    pd.DataFrame({"time": pd.to_datetime(t_te), "y_true": y_true_r, "y_pred": y_pred_r}).to_csv(PATHS["csv_r"], index=False)
    scatter(y_true_r, y_pred_r, PATHS["scatter_r"], "Real", RUN_NAME)
    hist(y_pred_r - y_true_r, PATHS["hist_r"], "Residual-Real", RUN_NAME)
    hist_without_zeros(y_pred_r - y_true_r, PATHS["hist_r_zeros"], "Residuals-Real (without Zeros)", RUN_NAME)
    
    hparams = {
        "LR": LR_INIT,
        "Epochs": EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Hidden": HIDDEN,
        "Dropout": DROPOUT,
        "Num Layers": NUM_LAYERS,
        "L2 Lambda": L2_LAMBDA,
        "Early Stop": EARLY_STOP,
        "LR Factor": LR_FACTOR,
        "LR Patience": LR_PATIENCE,
    }
    make_summary_daymask(PATHS,metrics_n, metrics_r, hparams)
    with open(PATHS["report"], "w") as f:
        f.write(f"# LSTM Run Report — {RUN_NAME}\n\n")
        f.write("## Hyperparameter\n")
        for k, v in hparams.items():
            if isinstance(v, float):
                f.write(f"{k}: {fmt_de(v, prec=4)}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write("\n## Metrics (Norm)\n")
        for k, v in metrics_n.items():
            f.write(f"{k}: {fmt_de(v, prec=4)}\n")
        f.write("\n## Metrics (Real)\n")
        for k, v in metrics_r.items():
            f.write(f"{k}: {fmt_de(v, prec=4)}\n")
        f.write("\n## Descale\n")
        f.write(json.dumps({"method": method, "file": src_path, "variable": var}, indent=2))



if __name__ == "__main__":
    main()
