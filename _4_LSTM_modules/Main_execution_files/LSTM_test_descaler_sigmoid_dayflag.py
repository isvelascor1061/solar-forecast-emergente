#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:31:25 2025
Author: Leonard Merl

DESCALER tuples:
    • Choose the file that contains the physical/normalisation reference
    • Choose the way it should be “descaled” later on (“physical”, “minmax”, …)
    • Provide the variable name inside the NetCDF  (None → first data-var)
"""
from __future__ import annotations
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, xarray as xr
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from tqdm import trange
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from _4_LSTM_modules.NN_modules.BiLSTMRegressor import BiLSTMRegressor # Ändere den Import hier
from tqdm.auto import tqdm
from torchviz import make_dot 
from torch.utils.tensorboard import SummaryWriter

# -------------------------------------------------------------------------
RUN_NAME   = "4launch_Multfeat_sym24_numl3_hidden96_3"
SEQ_NPZ    = "_4_LSTM_modules/Prepared_data/4launch_multfeat_sym24.npz"

# descaling-Specification --------------------------------------------------
DESCALER   = (
    "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct2.nc",
    "physical",                #  | "physical for clearness and clear sky, min_max
    "clear_sky_ghi"             # Name in NetCDF   (None→first DataVar) clearsky_ghi, extraterrestrial_ghi
)

"""
How to execute:
    1. Initialize the hyperparametes 
    2. Input name of run and the path to the npz file 
    3. choose descale method (keep as it is with CSI Siata as target)
    4.Input MSE baseline (from Comparison script)
    5. Rund dir -> where run should be safed (if you want different folders for different tests)
    6. Modell, loss, optimizer etc can be changed in the main block if needed
    7. launch test
"""

# ---------------- Hyperparameters -----------------------------------------
LR_INIT   = 1e-3
MIN_LR    = 1e-6   
EPOCHS =    50
BATCH_SIZE= 128
HIDDEN = 96
DROPOUT= 0.3
NUM_LAYERS= 3
L2_LAMBDA= 10e-4
EARLY_STOP = 25
MULTI_STEP = None
LR_FACTOR = 0.5        
LR_PATIENCE = 4  
causal_mode = False
# ---------------------------Evaluation---------------------------------------
MSE_BASELINE_R= 22322.349260 #this is the mse you calculate on the _02_Comparison_before_NN_all_launch_times.py script

# -------------------------------------------------------------------------
_NOW   = datetime.now().strftime("%Y%m%d_%H%M%S")    
RUN_DIR = os.path.join("_4_LSTM_modules/_runs/4launch_multfeat_sym", f"{RUN_NAME}_{_NOW}")
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
    graph      = os.path.join(RUN_DIR, "computational_graph")
)


# ----------------------------- Data ---------------------------------
class SeqDS(torch.utils.data.Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.indices = indices

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]


def load_splits(path):
    d = np.load(path, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["t_train"],
            d["X_val"],   d["y_val"],   d["t_val"],
            d["X_test"],  d["y_test"],  d["t_test"])

def compute_day_mask(timestamps, flag_start=5, flag_end=20, tz_offset=0):
    """
    Parameters
    ----------
    timestamps  : DatetimeIndex | list[pd.Timestamp]
    flag_start / flag_end : hour boundaries (local time)
    tz_offset   : hours to add (if data are UTC but “day” is local)

    Returns
    -------
    torch.Tensor of floats (1 = day, 0 = night)
    """
    timestamps = pd.to_datetime(timestamps)
    local_times = timestamps + pd.Timedelta(hours=tz_offset)
    hours = local_times.hour
    day_mask = ((hours > flag_start) & (hours < flag_end)).astype(float)
    return torch.tensor(day_mask.values if hasattr(day_mask, "values") else day_mask, dtype=torch.float32)




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

# ------------------------- Descale-Utilities ------------------------------
def get_dataarray(path:str, var:str|None):
    ds = xr.open_dataset(path)
    if var is None:     var = list(ds.data_vars)[0]
    return ds[var].squeeze()

def build_stats(da: xr.DataArray, method:str):
    if method == "z_score":
        return {"mu": float(da.mean()), "sigma": float(da.std())}
    if method == "average":
        return {"mean": float(da.mean())}
    if method == "minmax":
        return {"min": float(da.min()), "max": float(da.max())}
    return {}


def descale(
    arr: np.ndarray,
    times: np.ndarray,
    method: str,
    da: xr.DataArray,
    stats: dict = None,
) -> np.ndarray:
    """"funciton that descales the Values from the normalized space to real space/"
     eversing the normalization process and securing a readable output"""
    if method == "physical":
        return arr * da.sel(observation_time=times).values
    elif method == "minmax":  # Beispiel für alternative Methoden
        return arr * (stats["max"] - stats["min"]) + stats["min"]
    else:
        raise ValueError(f"Unknown descale method: {method}")

# ------------------------- Plots ------------------------------------------
def plot_loss(hist, out,RUN_NAME):
    """This function plots the train and validation loss during training to visualize convergence"""
    ep, tl, vl = zip(*hist)
    plt.figure(); plt.plot(ep,tl,label="train"); plt.plot(ep,vl,label="val")
    plt.title(f"{RUN_NAME}, Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()

def scatter(y_t, y_p, out, ttl,RUN_NAME):
    """ This funtion produces a scatter plot in normalized space and real space"""
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    plt.figure(figsize=(8, 8))
    plt.scatter(y_t, y_p, s=8, alpha=.5)
    plt.plot(lims, lims, "r--")
    
    #
    if ttl == "Real":  
        plt.title(f"{RUN_NAME}, Predicted vs. Actual Values (Real)")
        plt.xlabel(r'$y_{true}$ (actual values in W/m²)')
        plt.ylabel(r'$y_{pred}$ (predicted values in W/m²)')
    else: 
        plt.title(f"{RUN_NAME}, Predicted vs. Actual Values (Normalized)")
        plt.xlabel(r'$y_{true}$ (actual values normalised)')
        plt.ylabel(r'$y_{pred}$ (real values normalised)')

    plt.grid()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def hist(resid, out, ttl,RUN_NAME):
    """ This funciton is used to plot a histogramm of the residual values (y_pred-y_true)"""
    plt.figure(); plt.hist(resid,bins=50,edgecolor="k",alpha=0.7); 
    if ttl == "Residual-Real":
        plt.title(f"{RUN_NAME}, Residuals Histogram")
        plt.xlabel(" Residual [W/m²]")
    else:
        plt.title(f"{RUN_NAME}, Residuals Histogram")
        plt.xlabel(" Residual")
        
    plt.ylabel("Frequency");plt.grid(True); plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    
def hist_without_zeros(resid, out, ttl, RUN_NAME):
    """This funtion plots the residuals without zeros -> Zeros make up the largest amount of residuals
    -> better overview of the residuals on a smaller scale"""
    residuals_nonzero = resid[resid != 0]  
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


def make_summary(paths: dict,metrics_n: dict,
                          metrics_r: dict,
                         hparams: dict):
    """This function procuces a png summary with all plots, hyperparameters and statistical metrics"""
    """
    3 × 3 Canvas Layout:
       ┌───────────────┬───────────────┬───────────────┐
       │ Scatter Norm  │ Scatter Real  │ Loss-Curve    │  (Row 0)
       ├───────────────┼───────────────┼───────────────┤
       │ Hist Norm     │ Hist Real     │ Hyperparams   │  (Row 1)
       ├───────────────┼───────────────┼───────────────┤
       │ Hist Norm wo0 │ Hist Real wo0 │ Metrics Real  │  (Row 2)
       └───────────────┴───────────────┴───────────────┘
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 15))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])

   
    axes_files = [
        (gs[0, 0], paths["scatter_n"], "Scatter Norm"),
        (gs[0, 1], paths["scatter_r"], "Scatter Real"),
        (gs[0, 2], paths["loss"],      "Loss Curve"),
        
        (gs[1, 0], paths["hist_n"],   "Histogram Normalized"),
        (gs[1, 1], paths["hist_r"],   "Histogram Real"),
    ]

    
    for cell, fname, ttl in axes_files:
        ax = fig.add_subplot(cell)
        img = plt.imread(fname)
        ax.imshow(img)
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")

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

    
    ax_hist_n_zeros = fig.add_subplot(gs[2, 0])
    img_hist_n_zeros = plt.imread(paths["hist_n_zeros"])
    ax_hist_n_zeros.imshow(img_hist_n_zeros)
    ax_hist_n_zeros.set_title("Histogram Normalized wo 0", fontsize=11)
    ax_hist_n_zeros.axis("off")

    ax_hist_r_zeros = fig.add_subplot(gs[2, 1])
    img_hist_r_zeros = plt.imread(paths["hist_r_zeros"])
    ax_hist_r_zeros.imshow(img_hist_r_zeros)
    ax_hist_r_zeros.set_title("Histogram Real wo 0", fontsize=11)
    ax_hist_r_zeros.axis("off")

    
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

   
def debug_predictions(t_te, y_true_r, y_pred_r, dayflag_target, n=20, time_idx=-1):
    
    """This is a helper function to debug the daymask-> just use if u want to make sure that the daymask is applied properly"""
    indices = np.random.choice(len(y_true_r), size=n, replace=False)

    rows = []
    for i in indices:
        mask_applied = dayflag_target[i] == 1
        timestamp = t_te[i]
        if hasattr(timestamp, "__len__") and len(timestamp) > 1:
            timestamp = timestamp[time_idx]
        rows.append({
            "time": pd.to_datetime(timestamp),
            "dayflag": int(dayflag_target[i]),
            "mask_applied": mask_applied,
            "y_true": y_true_r[i],
            "y_pred": y_pred_r[i],
        })

    df_debug = pd.DataFrame(rows).sort_values(by="time")
    return df_debug



# -------------------------------- MAIN ------------------------------------
def main():
    # ---------- Data ----------------------------------------------------
    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(SEQ_NPZ)
    tr_ds = SeqDS(X_tr, y_tr, np.arange(len(y_tr)))
    va_ds = SeqDS(X_va, y_va, np.arange(len(y_va)))
    te_ds = SeqDS(X_te, y_te, np.arange(len(y_te)))
    
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(va_ds, BATCH_SIZE)
    te_dl = DataLoader(te_ds, BATCH_SIZE)


    seq_len, n_feat = X_tr.shape[1], X_tr.shape[2]
    
    daymask_train = compute_day_mask(t_tr)
    daymask_val= compute_day_mask(t_va)
    
    
    print("Input Shape (train):", X_tr.shape)  #displays shape of Train vektor
    

   # 

    # ---------- Modell -------------------------------------------------
    """ The Modell is imported from the NN_modules file -> new ones can be produced an tested"""
    model = BiLSTMRegressor(n_feat, hidden=HIDDEN,seq_len=seq_len,
                        num_layers=NUM_LAYERS, dropout=DROPOUT)
    
    
    # --- TensorBoard ----------------------------------------------------------
    """ This produces the tensor board you can access trough the console"""
    writer = SummaryWriter(log_dir=RUN_DIR)       
    dummy_inp = torch.randn(1, seq_len, n_feat)    
    writer.add_graph(model, dummy_inp)             
    
   
    

    optim = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=L2_LAMBDA)
    #adamW as optimizer
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=LR_FACTOR,
                                  patience=LR_PATIENCE, threshold=1e-4,
                                  min_lr=MIN_LR, verbose=True)
    #The scheduler gradually decreases the learning rate based on patience and validation loss
    
    
    loss_fn = nn.MSELoss(reduction="none")
    #MSE as loss funtion 


    
    # ---------- Training  -----------------
    best_val, patience, history = float("inf"), 0, []
    
    bar = trange(1, EPOCHS + 1, desc="Ep", unit="ep", leave=True)
    
    
    
    for ep in bar:
        # ---------- Train --------------------------------------------------
        model.train(); train_loss = 0.0
        for xb, yb, idx in tr_dl:
            optim.zero_grad()
    
            # Prediction
            preds = model(xb)                      
    
            #day mask creation
            day_mask = daymask_train[idx] 
    
            # --- Hard-Clip: Night values → 0 --------------------------------
            preds = preds * day_mask               # Night = 0
    
            #MSE is also clipped during the night
            loss_unmasked = loss_fn(preds, yb)      # (Batch,)
            loss = (loss_unmasked*day_mask).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   
            optim.step()
    
            train_loss += loss.item() * len(xb)
        train_loss /= len(tr_dl.dataset)
    
        # ---------- Validation --------------------------------------------
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb, idx in va_dl:
                preds = model(xb)
                #same clip here
                day_mask = daymask_val[idx] 
                preds = preds * day_mask           
    
                loss_unmasked = loss_fn(preds, yb)
                loss = (loss_unmasked*day_mask).mean()
                val_loss += loss.item() * len(xb)
            val_loss /= len(va_dl.dataset)

        
        
        
        scheduler.step(val_loss)
        tqdm.write(f"[{ep:3d}/{EPOCHS}] train {train_loss:.5f} | val {val_loss:.5f} /r {optim.param_groups[0]['lr']:.1e}")
        
        
      
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
    # ---------- Test (Norm-Space) ----------------------------------------
    model.load_state_dict(torch.load(PATHS["model"]))
    model.eval()
    daymask_test = compute_day_mask(t_te)                 # Länge = n_test
    daymask_test = torch.tensor(daymask_test, dtype=torch.float32)

    
    # ---------- Test (Norm-Space) ----------------------------------------
    model.eval()
    y_preds = []
    with torch.no_grad():
        for xb, _, idx in te_dl:
            preds = model(xb)                            
            day_mask = daymask_test[idx]
            #LSTM never learned night values so hard clip here aswell                 
            preds *= day_mask                             
            y_preds.append(preds.numpy())
    
    y_pred = np.concatenate(y_preds)                      # (n_test,)

    
    

    metrics_n = dict(
        MSE = mean_squared_error(y_te, y_pred),
        MAE = mean_absolute_error(y_te, y_pred),
        R2  = r2_score(y_te, y_pred),
        Corr = np.corrcoef(y_te.ravel(), y_pred.ravel())[0, 1],   # <- .ravel()

    )
    
    
    # ---------- (Norm-Space) ---------------
    residuals_n = y_te - y_pred
    residual_variance_n = np.var(residuals_n) 
    metrics_n["Residual_Variance"] = residual_variance_n 

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
    
  
    
     
    dayflag_test = compute_day_mask(t_te)

    
    df_debug = debug_predictions(
        t_te, y_true_r, y_pred_r, dayflag_test, n=50, time_idx=-1
    )
    #print(df_debug) #-> if u want to see where the daymask is applied remove the # before the print


    # ---------- (Real-Space) --------------
    residuals_r = y_true_r - y_pred_r
    residual_variance_r = np.var(residuals_r)  
    


    
    
    y_true_flat = y_true_r.ravel()  
    y_pred_flat = y_pred_r.ravel()   


    metrics_r = dict(
        MSE = mean_squared_error(y_true_r, y_pred_r),
        RMSE= root_mean_squared_error(y_true_r, y_pred_r),
        MAE = mean_absolute_error(y_true_r, y_pred_r),
        R2  = r2_score(y_true_r, y_pred_r),
        Corr = np.corrcoef(y_true_flat, y_pred_flat)[0, 1],
        Residual_Variance = residual_variance_r  
    )
    
    mse_baseline_r = MSE_BASELINE_R #Baseline MSE of the GFS -> from Comparison script before NN
    metrics_r["SkillScore"] = 1 - metrics_r["MSE"] / mse_baseline_r
    print("\n=== Evaluation (Normalised) ===")
    for k, v in metrics_n.items():
        print(f"{k:>15}: {fmt_de(v, prec=4)}")


    print("\n=== Evaluation (Real) ===")
    for k, v in metrics_r.items():
        print(f"{k:>15}: {fmt_de(v, prec=4)}")

    pd.DataFrame({"time":pd.to_datetime(t_te), "y_true":y_true_r, "y_pred":y_pred_r}
                 ).to_csv(PATHS["csv_r"], index=False)
    scatter(y_true_r, y_pred_r, PATHS["scatter_r"], "Real", RUN_NAME)
    hist(y_pred_r - y_true_r, PATHS["hist_r"], "Residual-Real", RUN_NAME)
    hist_without_zeros(y_pred_r - y_true_r, PATHS["hist_r_zeros"], "Residuals-Real (without Zeros)", RUN_NAME)
    
    # ---------- Summary-PNG + Report ------------------------------------
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
    make_summary(PATHS, metrics_n, metrics_r, hparams)
    
    with open(PATHS["report"], "w") as f:
        f.write(f"# LSTM Run Report — {RUN_NAME}\n\n")
        f.write("## Hyperparameter\n")
        for k, v in dict(LR=LR_INIT, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, HIDDEN=HIDDEN,
                         DROPOUT=DROPOUT, NUM_LAYERS=NUM_LAYERS, L2_LAMBDA=L2_LAMBDA,
                         EARLY_STOP=EARLY_STOP).items():
            f.write(f"{k}: {v}\n")
        f.write("\n## Metrics (Norm)\n")
        for k, v in metrics_n.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n## Metrics (Real)\n")
        for k, v in metrics_r.items():
            f.write(f"{k}: {v:.4f}\n")
        for k, v in hparams.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write("\n## Descale\n")
        f.write(json.dumps({"method": method, "file": src_path, "variable": var}, indent=2))
    writer.close()

    print("All Artifacts →", RUN_DIR)

if __name__ == "__main__":
    main()
