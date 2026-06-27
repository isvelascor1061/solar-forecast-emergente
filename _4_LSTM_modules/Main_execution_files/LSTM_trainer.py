#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM_trainer.py
===============
Unified training script for the Bi-LSTM network for solar irradiance
prediction in Medellín (Proyecto Emergente).

Merges the 5 previous scripts:
    • first_LSTM_test_CSI.py
    • LSTM_test_descaler_linear.py
    • LSTM_test_descaler_sigmoid.py
    • LSTM_test_descalersigmoid_daymask.py
    • LSTM_test_descaler_sigmoid_dayflag.py

Behaviour is configurable from config.py:
    • ACTIVATION      : "sigmoid" or "linear" — output activation function
    • DESCALER_METHOD : "physical" | "minmax" | "z_score" | "average"
    • USE_DAYMASK     : True/False — night mask during training

Main workflow:
    1. Load train/val/test splits from the .npz file
    2. Build the Bi-LSTM with hyperparameters from config.py
    3. Train with AdamW + ReduceLROnPlateau + early stopping
       (if USE_DAYMASK=True: night predictions → 0, night loss = 0)
    4. Evaluate in normalised and real space (de-scaled)
    5. Generate plots: loss curve, scatter, residual histograms
    6. Save CSV with predictions, 3×3 summary image and report.txt
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from tqdm import trange
from tqdm.auto import tqdm
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
)

# Main architecture: bidirectional Bi-LSTM
from _4_LSTM_modules.NN_modules.BiLSTMRegressor import BiLSTMRegressor

# TensorBoard (optional — only if installed)
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD = True
except ImportError:
    _TENSORBOARD = False

# -----------------------------------------------------------------------
# Import all hyperparameters and paths from config.py
# -----------------------------------------------------------------------
from config import (
    SEQ_NPZ_FILE, RUNS_DIR,
    CSI_GHI_FILE, CSI_VAR_NAME,
    LR_INIT, MIN_LR, EPOCHS, BATCH_SIZE, HIDDEN, NUM_LAYERS, DROPOUT,
    L2_LAMBDA, EARLY_STOP, LR_FACTOR, LR_PATIENCE,
    ACTIVATION, DESCALER_METHOD, USE_DAYMASK,
    FLAG_START, FLAG_END, UTC_OFFSET,
    MSE_BASELINE_R,
)

# -----------------------------------------------------------------------
# RUN CONFIGURATION — the only block the user needs to edit
# -----------------------------------------------------------------------
# Descriptive name for the run (a timestamp is appended automatically)
RUN_NAME = "4launch_Multfeat_sym18_clim79_BiLSTM_attn"

# Path to the .npz file with the prepared sequences
SEQ_NPZ = SEQ_NPZ_FILE

# De-scaling specification: (NetCDF path, variable)
# The method is controlled by DESCALER_METHOD in config.py
DESCALER_FILE = CSI_GHI_FILE
DESCALER_VAR  = CSI_VAR_NAME

# -----------------------------------------------------------------------
# Create the run directory with a timestamp
# -----------------------------------------------------------------------
_NOW    = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUNS_DIR, f"{RUN_NAME}_{_NOW}")
os.makedirs(RUN_DIR, exist_ok=True)

# Dictionary with all run artefact paths
PATHS = dict(
    model        = os.path.join(RUN_DIR, "best_model.pt"),
    csv_n        = os.path.join(RUN_DIR, "pred_norm.csv"),
    csv_r        = os.path.join(RUN_DIR, "pred_real.csv"),
    loss         = os.path.join(RUN_DIR, "loss_curve.png"),
    scatter_n    = os.path.join(RUN_DIR, "scatter_norm.png"),
    scatter_r    = os.path.join(RUN_DIR, "scatter_real.png"),
    hist_n       = os.path.join(RUN_DIR, "hist_norm.png"),
    hist_r       = os.path.join(RUN_DIR, "hist_real.png"),
    hist_n_zeros = os.path.join(RUN_DIR, "hist_norm_without_zeros.png"),
    hist_r_zeros = os.path.join(RUN_DIR, "hist_real_without_zeros.png"),
    summary      = os.path.join(RUN_DIR, "summary.png"),
    report       = os.path.join(RUN_DIR, "report.txt"),
    # Attention weights for the test split — shape (n_test, seq_len)
    # Can be loaded later for visualisation with np.load()
    attn_weights = os.path.join(RUN_DIR, "attn_weights_test.npy"),
)


# =======================================================================
# DATASET
# =======================================================================

class SeqDS(Dataset):
    """3D sequence dataset (samples, time_steps, features)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx   # returns the index for masking


def load_splits(path: str):
    """Load train/val/test splits from the .npz file."""
    d = np.load(path, allow_pickle=True)
    return (
        d["X_train"], d["y_train"], d["t_train"],
        d["X_val"],   d["y_val"],   d["t_val"],
        d["X_test"],  d["y_test"],  d["t_test"],
    )


# =======================================================================
# DAY/NIGHT MASK
# =======================================================================

def compute_day_mask(
    timestamps,
    flag_start: int = FLAG_START,
    flag_end: int = FLAG_END,
    tz_offset: int = UTC_OFFSET,
) -> torch.Tensor:
    """
    Generates a binary tensor (1 = day, 0 = night) for each timestamp.

    Parameters
    ----------
    timestamps  : array of timestamps (UTC) corresponding to the target.
    flag_start  : local hour at which night begins (after this hour = night).
    flag_end    : local hour at which night ends (before this hour = night).
    tz_offset   : UTC → local time offset (hours).
    """
    ts_local = pd.to_datetime(timestamps) + pd.Timedelta(hours=tz_offset)
    hours    = ts_local.hour
    mask     = ((hours >= flag_end) & (hours <= flag_start)).astype(float)
    vals     = mask.values if hasattr(mask, "values") else mask
    return torch.tensor(vals, dtype=torch.float32)


# =======================================================================
# DE-SCALING UTILITIES
# =======================================================================

def get_dataarray(path: str, var: str | None) -> xr.DataArray:
    """Open the reference NetCDF and return the DataArray for the specified variable."""
    ds = xr.open_dataset(path)
    if var is None:
        var = list(ds.data_vars)[0]
    return ds[var].squeeze()


def build_stats(da: xr.DataArray, method: str) -> dict:
    """Pre-compute statistics needed for de-scaling."""
    if method == "z_score":
        return {"mu": float(da.mean()), "sigma": float(da.std())}
    if method == "average":
        return {"mean": float(da.mean())}
    if method == "minmax":
        return {"min": float(da.min()), "max": float(da.max())}
    return {}   # "physical" and "none" require no pre-computed statistics


def descale(
    arr: np.ndarray,
    times: np.ndarray,
    method: str,
    da: xr.DataArray,
    stats: dict,
) -> np.ndarray:
    """
    Converts predictions from normalised space to real space (W/m²).

    Supported methods
    -----------------
    physical : multiplies by the reference value (GHI_cs) at each timestamp.
    minmax   : inverts min-max normalisation.
    z_score  : inverts standardisation (mean=0, std=1).
    average  : inverts mean normalisation.
    none     : no transformation (returns arr unchanged).
    """
    if method == "physical":
        # Select the GHI_cs value at each observation timestamp
        return arr * da.sel(observation_time=times).values
    elif method == "minmax":
        return arr * (stats["max"] - stats["min"]) + stats["min"]
    elif method == "z_score":
        return arr * stats["sigma"] + stats["mu"]
    elif method == "average":
        return arr * stats["mean"]
    elif method == "none":
        return arr
    else:
        raise ValueError(f"Unknown de-scaling method: '{method}'")


# =======================================================================
# NUMBER FORMATTING (European style with thousands dot and decimal comma)
# =======================================================================

def fmt_de(val, prec: int = 4) -> str:
    """Formats a number as '1.234.567,8910' (European style)."""
    if isinstance(val, (float, np.floating)):
        s = f"{val:,.{prec}f}"
    else:
        s = f"{val:,}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# =======================================================================
# VISUALISATION FUNCTIONS
# =======================================================================

def plot_loss(history: list, out: str) -> None:
    """Saves the loss curve (train vs val) per epoch."""
    ep, tl, vl = zip(*history)
    plt.figure()
    plt.plot(ep, tl, label="train")
    plt.plot(ep, vl, label="val")
    plt.title(f"{RUN_NAME} — Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def scatter(y_t: np.ndarray, y_p: np.ndarray, out: str, space: str) -> None:
    """
    Generates a scatter plot of true vs predicted values.
    space : "Norm" for normalised space, "Real" for W/m².
    """
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    plt.figure(figsize=(8, 8))
    plt.scatter(y_t, y_p, s=8, alpha=0.5)
    plt.plot(lims, lims, "r--")
    if space == "Real":
        plt.title(f"{RUN_NAME} — Predicted vs True (W/m²)")
        plt.xlabel(r"$y_{true}$ (W/m²)")
        plt.ylabel(r"$y_{pred}$ (W/m²)")
    else:
        plt.title(f"{RUN_NAME} — Predicted vs True (normalised)")
        plt.xlabel(r"$y_{true}$ (normalised)")
        plt.ylabel(r"$y_{pred}$ (normalised)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def hist_residuos(resid: np.ndarray, out: str, space: str) -> None:
    """Histogram of residuals (predicted − true)."""
    plt.figure()
    plt.hist(resid, bins=50, edgecolor="k", alpha=0.7)
    plt.title(f"{RUN_NAME} — Residual histogram")
    plt.xlabel("Residual [W/m²]" if space == "Real" else "Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def hist_sin_ceros(resid: np.ndarray, out: str, space: str) -> None:
    """
    Histogram of residuals excluding zeros (night-time hours produce
    many residuals = 0 that distort the histogram scale).
    """
    resid_nz = resid[resid != 0]
    plt.figure(figsize=(8, 6))
    plt.hist(resid_nz, bins=50, edgecolor="k", alpha=0.7)
    plt.title(f"{RUN_NAME} — Residuals (without zeros)")
    plt.xlabel("Residual [W/m²]" if space == "Real" else "Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def make_summary(paths: dict, metrics_n: dict, metrics_r: dict, hparams: dict) -> None:
    """
    Generates a summary image with a 3×3 layout:
    ┌──────────────┬──────────────┬──────────────┐
    │ Scatter Norm │ Scatter Real │  Loss Curve  │ row 0
    ├──────────────┼──────────────┼──────────────┤
    │  Hist Norm   │  Hist Real   │ Hyperparams  │ row 1
    ├──────────────┼──────────────┼──────────────┤
    │ Hist Norm ∅0 │ Hist Real ∅0 │   Metrics    │ row 2
    └──────────────┴──────────────┴──────────────┘
    """
    fig = plt.figure(figsize=(16, 15))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])

    # Rows 0 and part of row 1: scatter, loss, and histogram images
    images = [
        (gs[0, 0], paths["scatter_n"],    "Scatter Norm"),
        (gs[0, 1], paths["scatter_r"],    "Scatter Real"),
        (gs[0, 2], paths["loss"],         "Loss curve"),
        (gs[1, 0], paths["hist_n"],       "Histogram Norm"),
        (gs[1, 1], paths["hist_r"],       "Histogram Real"),
    ]
    for cell, fname, title in images:
        ax = fig.add_subplot(cell)
        ax.imshow(plt.imread(fname))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Row 1, column 2: hyperparameter table
    ax_hp = fig.add_subplot(gs[1, 2])
    ax_hp.axis("off")
    tbl = ax_hp.table(
        cellText=[[k, str(v)] for k, v in hparams.items()],
        colLabels=["Hyperparameter", "Value"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    ax_hp.set_title("Hyperparameters", fontsize=11, pad=6)

    # Row 2: histograms without zeros and metrics table
    for cell, key, title in [
        (gs[2, 0], "hist_n_zeros", "Hist Norm ∅0"),
        (gs[2, 1], "hist_r_zeros", "Hist Real ∅0"),
    ]:
        ax = fig.add_subplot(cell)
        ax.imshow(plt.imread(paths[key]))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    ax_met = fig.add_subplot(gs[2, 2])
    ax_met.axis("off")
    metric_rows = ["MSE", "RMSE", "MAE", "R2", "Corr", "Residual_Variance", "SkillScore"]
    metric_data = [[r, f"{metrics_r[r]:.4f}"] for r in metric_rows if r in metrics_r]
    tbl_met = ax_met.table(
        cellText=metric_data,
        colLabels=["Metric", "Real value"],
        loc="center", cellLoc="center",
    )
    tbl_met.auto_set_font_size(False)
    tbl_met.set_fontsize(9)
    tbl_met.scale(1.1, 1.5)
    ax_met.set_title("Evaluation metrics (real space)", fontsize=11, pad=6)

    fig.suptitle(RUN_NAME, fontsize=17, y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(paths["summary"], dpi=300)
    plt.close(fig)


# =======================================================================
# MAIN TRAINING FUNCTION
# =======================================================================

def main():
    # -------------------------------------------------------------------
    # 1) Load train / val / test splits from the .npz file
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Run: {RUN_NAME}")
    print(f"  Data:   {SEQ_NPZ}")
    print(f"  Activation:  {ACTIVATION} | De-scaling: {DESCALER_METHOD} | Night mask: {USE_DAYMASK}")
    print(f"{'='*60}\n")

    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(SEQ_NPZ)
    print(f"Input shape (train): {X_tr.shape}")

    tr_dl = DataLoader(SeqDS(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(SeqDS(X_va, y_va), BATCH_SIZE)
    te_dl = DataLoader(SeqDS(X_te, y_te), BATCH_SIZE)

    seq_len, n_feat = X_tr.shape[1], X_tr.shape[2]

    # -------------------------------------------------------------------
    # 2) Pre-compute day/night masks (used when USE_DAYMASK=True)
    # -------------------------------------------------------------------
    # Masks have one value per sample (1=day, 0=night)
    mask_train = compute_day_mask(t_tr)
    mask_val   = compute_day_mask(t_va)
    mask_test  = compute_day_mask(t_te)

    # -------------------------------------------------------------------
    # 3) Build the Bi-LSTM model with hyperparameters from config.py
    # -------------------------------------------------------------------
    model = BiLSTMRegressor(
        n_feat     = n_feat,
        hidden     = HIDDEN,
        seq_len    = seq_len,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
        activation = ACTIVATION,   # "sigmoid" or "linear" — from config.py
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # AdamW optimiser with L2 regularisation
    optim     = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=L2_LAMBDA)
    # Scheduler that reduces the LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optim, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
        threshold=1e-4, min_lr=MIN_LR, verbose=True,
    )
    # MSE loss without reduction (to allow weighting with the mask)
    loss_fn = nn.MSELoss(reduction="none")

    # TensorBoard: logs the model graph if available
    if _TENSORBOARD:
        writer    = SummaryWriter(log_dir=RUN_DIR)
        dummy_inp = torch.randn(1, seq_len, n_feat)
        writer.add_graph(model, dummy_inp)

    # -------------------------------------------------------------------
    # 4) Training loop with early stopping
    # -------------------------------------------------------------------
    best_val, patience, history = float("inf"), 0, []

    pbar = trange(1, EPOCHS + 1, desc="Epochs", unit="ep", leave=True)
    for ep in pbar:
        # ---- Training phase -------------------------------------------
        model.train()
        train_loss = 0.0
        for xb, yb, idx in tr_dl:
            optim.zero_grad()
            preds = model(xb)              # (B,)

            if USE_DAYMASK:
                # Apply mask: night predictions → 0
                mask  = mask_train[idx]
                preds = preds * mask
                # Loss is not accumulated for night-time hours
                loss  = (loss_fn(preds, yb) * mask).mean()
            else:
                loss = loss_fn(preds, yb).mean()

            loss.backward()
            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(tr_dl.dataset)

        # ---- Validation phase -----------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, idx in va_dl:
                preds = model(xb)
                if USE_DAYMASK:
                    mask  = mask_val[idx]
                    preds = preds * mask
                    loss  = (loss_fn(preds, yb) * mask).mean()
                else:
                    loss = loss_fn(preds, yb).mean()
                val_loss += loss.item() * len(xb)
        val_loss /= len(va_dl.dataset)

        scheduler.step(val_loss)
        tqdm.write(
            f"[{ep:3d}/{EPOCHS}] train {train_loss:.5f} | "
            f"val {val_loss:.5f} | lr {optim.param_groups[0]['lr']:.1e}"
        )
        history.append((ep, train_loss, val_loss))

        # ---- Early stopping -------------------------------------------
        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), PATHS["model"])
        else:
            patience += 1
            if patience >= EARLY_STOP:
                tqdm.write("Early stopping triggered.")
                break

    plot_loss(history, PATHS["loss"])

    # -------------------------------------------------------------------
    # 5) Evaluation in normalised space (test set)
    # -------------------------------------------------------------------
    model.load_state_dict(torch.load(PATHS["model"]))
    model.eval()

    preds_list  = []
    attn_list   = []   # attention weights per batch, shape (B, seq_len)
    with torch.no_grad():
        for xb, _, idx in te_dl:
            # Request attention weights from the model alongside predictions
            preds, attn = model(xb, return_attention=True)
            if USE_DAYMASK:
                # Hard-clip is also applied during inference for night hours
                preds = preds * mask_test[idx]
            preds_list.append(preds.numpy())
            attn_list.append(attn.numpy())

    y_pred       = np.concatenate(preds_list)   # (n_test,)
    attn_weights = np.concatenate(attn_list)    # (n_test, seq_len)

    # Persist attention weights so they can be loaded for visualisation
    np.save(PATHS["attn_weights"], attn_weights)
    print(f"Attention weights saved -> shape {attn_weights.shape}")

    # Compute metrics in normalised space
    residuals_n          = y_te - y_pred
    residual_variance_n  = float(np.var(residuals_n))
    metrics_n = dict(
        MSE              = mean_squared_error(y_te, y_pred),
        MAE              = mean_absolute_error(y_te, y_pred),
        R2               = r2_score(y_te, y_pred),
        Corr             = float(np.corrcoef(y_te.ravel(), y_pred.ravel())[0, 1]),
        Residual_Variance= residual_variance_n,
    )
    print("\n=== Evaluation — normalised space ===")
    for k, v in metrics_n.items():
        print(f"{k:>20}: {fmt_de(v)}")

    # Save normalised CSV and generate plots
    pd.DataFrame({"time": pd.to_datetime(t_te), "y_true": y_te, "y_pred": y_pred})\
        .to_csv(PATHS["csv_n"], index=False)
    scatter(y_te,    y_pred,           PATHS["scatter_n"],    "Norm")
    hist_residuos(y_pred - y_te,       PATHS["hist_n"],        "Norm")
    hist_sin_ceros(y_pred - y_te,      PATHS["hist_n_zeros"],  "Norm")

    # -------------------------------------------------------------------
    # 6) De-scaling: convert normalised predictions to W/m²
    # -------------------------------------------------------------------
    da_ref    = get_dataarray(DESCALER_FILE, DESCALER_VAR)
    stats_ref = build_stats(da_ref, DESCALER_METHOD)
    y_true_r  = descale(y_te,   t_te, DESCALER_METHOD, da_ref, stats_ref)
    y_pred_r  = descale(y_pred, t_te, DESCALER_METHOD, da_ref, stats_ref)

    # -------------------------------------------------------------------
    # 7) Evaluation in real space (W/m²)
    # -------------------------------------------------------------------
    residuals_r          = y_true_r - y_pred_r
    residual_variance_r  = float(np.var(residuals_r))
    metrics_r = dict(
        MSE              = mean_squared_error(y_true_r, y_pred_r),
        RMSE             = root_mean_squared_error(y_true_r, y_pred_r),
        MAE              = mean_absolute_error(y_true_r, y_pred_r),
        R2               = r2_score(y_true_r, y_pred_r),
        Corr             = float(np.corrcoef(y_true_r.ravel(), y_pred_r.ravel())[0, 1]),
        Residual_Variance= residual_variance_r,
    )

    # Daytime-only evaluation (clear_sky_ghi > 0).
    # da_ref is already loaded and squeezed (observation_time dim) for descaling.
    # MSE_BASELINE_R is the daytime-only GFS baseline, so both sides must be daytime-only.
    csi_at_test  = da_ref.sel(observation_time=t_te).values
    day_mask     = csi_at_test > 0
    y_true_day   = y_true_r[day_mask]
    y_pred_day   = y_pred_r[day_mask]
    mse_day      = float(mean_squared_error(y_true_day, y_pred_day))
    metrics_r["RMSE_daytime"] = float(np.sqrt(mse_day))
    metrics_r["MAE_daytime"]  = float(mean_absolute_error(y_true_day, y_pred_day))
    # SkillScore: daytime-only model MSE vs daytime-only GFS baseline
    metrics_r["SkillScore"]   = 1.0 - mse_day / MSE_BASELINE_R

    print("\n=== Evaluation — real space (W/m²) ===")
    for k, v in metrics_r.items():
        print(f"{k:>20}: {fmt_de(v)}")

    # Save real-space CSV and generate plots
    pd.DataFrame({"time": pd.to_datetime(t_te), "y_true": y_true_r, "y_pred": y_pred_r})\
        .to_csv(PATHS["csv_r"], index=False)
    scatter(y_true_r, y_pred_r,            PATHS["scatter_r"],    "Real")
    hist_residuos(y_pred_r - y_true_r,     PATHS["hist_r"],        "Real")
    hist_sin_ceros(y_pred_r - y_true_r,    PATHS["hist_r_zeros"],  "Real")

    # -------------------------------------------------------------------
    # 8) 3×3 summary image and report.txt
    # -------------------------------------------------------------------
    hparams = {
        "LR":          LR_INIT,
        "Epochs":      EPOCHS,
        "Batch Size":  BATCH_SIZE,
        "Hidden":      HIDDEN,
        "Dropout":     DROPOUT,
        "Num Layers":  NUM_LAYERS,
        "L2 Lambda":   L2_LAMBDA,
        "Early Stop":  EARLY_STOP,
        "LR Factor":   LR_FACTOR,
        "LR Patience": LR_PATIENCE,
        "Activation":  ACTIVATION,
        "De-scaling":  DESCALER_METHOD,
        "Night mask":  USE_DAYMASK,
    }
    make_summary(PATHS, metrics_n, metrics_r, hparams)

    # Write text report with all run details
    with open(PATHS["report"], "w", encoding="utf-8") as f:
        f.write(f"# Bi-LSTM run report — {RUN_NAME}\n")
        f.write(f"Timestamp: {_NOW}\n\n")

        f.write("## Hyperparameters\n")
        for k, v in hparams.items():
            f.write(f"{k}: {v}\n")

        f.write("\n## Run configuration\n")
        f.write(f"NPZ file: {SEQ_NPZ}\n")
        f.write(f"Descaler: {json.dumps({'method': DESCALER_METHOD, 'file': DESCALER_FILE, 'variable': DESCALER_VAR}, indent=2)}\n")

        f.write("\n## Metrics — normalised space\n")
        for k, v in metrics_n.items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\n## Metrics — real space (W/m²)\n")
        for k, v in metrics_r.items():
            f.write(f"{k}: {v:.6f}\n")

    if _TENSORBOARD:
        writer.close()

    print(f"\nAll artefacts saved to: {RUN_DIR}")


# =======================================================================
if __name__ == "__main__":
    main()
