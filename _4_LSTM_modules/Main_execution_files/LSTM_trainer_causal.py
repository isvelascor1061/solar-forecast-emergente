#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM_trainer_causal.py
======================
Training script for the causal single-launch-time experiment.

Uses a UNIDIRECTIONAL LSTM (UniLSTMRegressor) with Bahdanau attention,
trained on 24-hour causal windows from the 0100 GFS launch time only.

Key differences from LSTM_trainer.py
--------------------------------------
  - Model      : UniLSTMRegressor  (bidirectional=False)
  - Sequences  : 1launch_causal_lt0100_clim.npz  (N, 24, 28)
  - HIDDEN     : 128  (local override — config.py has 96 for BiLSTM)
  - RUN_NAME   : 1launch_causal_lt0100_clim_UniLSTM_attn

Everything else — loss, descaling, daytime SkillScore, plots, report —
is identical to LSTM_trainer.py.
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

# Unidirectional LSTM architecture for this experiment
from _4_LSTM_modules.NN_modules.UniLSTMRegressor import UniLSTMRegressor

# Asymmetric loss (kept for consistency — disabled by default)
from _4_LSTM_modules.NN_modules.AsymmetricLoss import AsymmetricHourWeightedLoss

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD = True
except ImportError:
    _TENSORBOARD = False

from config import (
    RUNS_DIR,
    CSI_GHI_FILE, CSI_VAR_NAME,
    LR_INIT, MIN_LR, EPOCHS, BATCH_SIZE, DROPOUT,
    L2_LAMBDA, EARLY_STOP, LR_FACTOR, LR_PATIENCE,
    ACTIVATION, DESCALER_METHOD, USE_DAYMASK, USE_DAYTIME_ONLY_LOSS,
    FLAG_START, FLAG_END, UTC_OFFSET,
    MSE_BASELINE_R,
    SEQ_NPZ_CAUSAL_FILE,
)

# -----------------------------------------------------------------------
# RUN CONFIGURATION
# -----------------------------------------------------------------------
RUN_NAME = "1launch_causal_lt0100_clim_UniLSTM_attn"

SEQ_NPZ = SEQ_NPZ_CAUSAL_FILE

DESCALER_FILE = CSI_GHI_FILE
DESCALER_VAR  = CSI_VAR_NAME

# Causal-specific hyperparameters (local overrides of config defaults)
HIDDEN_CAUSAL     = 128   # wider than BiLSTM-96 to compensate for fewer features
NUM_LAYERS_CAUSAL = 3

# Asymmetric loss — disabled for a clean baseline comparison
USE_ASYMMETRIC_LOSS = False
ALPHA               = 3.0
HOUR_WEIGHTS        = {8: 1.5, 9: 2.5, 10: 2.5}

# -----------------------------------------------------------------------
_NOW    = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUNS_DIR, f"{RUN_NAME}_{_NOW}")
os.makedirs(RUN_DIR, exist_ok=True)

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
    attn_weights = os.path.join(RUN_DIR, "attn_weights_test.npy"),
)


# =======================================================================
# DATASET
# =======================================================================

class SeqDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


def load_splits(path: str):
    d = np.load(path, allow_pickle=True)
    return (
        d["X_train"], d["y_train"], d["t_train"],
        d["X_val"],   d["y_val"],   d["t_val"],
        d["X_test"],  d["y_test"],  d["t_test"],
    )


# =======================================================================
# DAY/NIGHT MASK
# =======================================================================

def compute_day_mask(timestamps, flag_start=FLAG_START, flag_end=FLAG_END,
                     tz_offset=UTC_OFFSET) -> torch.Tensor:
    ts_local = pd.to_datetime(timestamps) + pd.Timedelta(hours=tz_offset)
    hours    = ts_local.hour
    mask     = ((hours >= flag_end) & (hours <= flag_start)).astype(float)
    vals     = mask.values if hasattr(mask, "values") else mask
    return torch.tensor(vals, dtype=torch.float32)


def compute_hours(timestamps, tz_offset=UTC_OFFSET) -> torch.Tensor:
    ts_local = pd.to_datetime(timestamps) + pd.Timedelta(hours=tz_offset)
    hrs  = ts_local.hour
    vals = hrs.values if hasattr(hrs, "values") else hrs
    return torch.tensor(vals, dtype=torch.long)


# =======================================================================
# DE-SCALING
# =======================================================================

def get_dataarray(path: str, var: str | None) -> xr.DataArray:
    ds = xr.open_dataset(path, engine="h5netcdf")
    if var is None:
        var = list(ds.data_vars)[0]
    return ds[var].squeeze()


def build_stats(da: xr.DataArray, method: str) -> dict:
    if method == "z_score":
        return {"mu": float(da.mean()), "sigma": float(da.std())}
    if method == "average":
        return {"mean": float(da.mean())}
    if method == "minmax":
        return {"min": float(da.min()), "max": float(da.max())}
    return {}


def descale(arr, times, method, da, stats) -> np.ndarray:
    if method == "physical":
        return arr * da.sel(observation_time=times).values
    elif method == "minmax":
        return arr * (stats["max"] - stats["min"]) + stats["min"]
    elif method == "z_score":
        return arr * stats["sigma"] + stats["mu"]
    elif method == "average":
        return arr * stats["mean"]
    return arr


# =======================================================================
# NUMBER FORMATTING
# =======================================================================

def fmt_de(val, prec: int = 4) -> str:
    if isinstance(val, (float, np.floating)):
        s = f"{val:,.{prec}f}"
    else:
        s = f"{val:,}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# =======================================================================
# VISUALISATION
# =======================================================================

def plot_loss(history, out):
    ep, tl, vl = zip(*history)
    plt.figure()
    plt.plot(ep, tl, label="train")
    plt.plot(ep, vl, label="val")
    plt.title(f"{RUN_NAME} — Loss curve")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=300); plt.close()


def scatter(y_t, y_p, out, space):
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    plt.figure(figsize=(8, 8))
    plt.scatter(y_t, y_p, s=8, alpha=0.5)
    plt.plot(lims, lims, "r--")
    label = "W/m²" if space == "Real" else "normalised"
    plt.title(f"{RUN_NAME} — Predicted vs True ({label})")
    plt.xlabel(f"y_true ({label})"); plt.ylabel(f"y_pred ({label})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out, dpi=300); plt.close()


def hist_residuos(resid, out, space):
    plt.figure()
    plt.hist(resid, bins=50, edgecolor="k", alpha=0.7)
    plt.title(f"{RUN_NAME} — Residual histogram")
    plt.xlabel("Residual [W/m²]" if space == "Real" else "Residual")
    plt.ylabel("Frequency"); plt.grid(True); plt.tight_layout()
    plt.savefig(out, dpi=300); plt.close()


def hist_sin_ceros(resid, out, space):
    resid_nz = resid[resid != 0]
    plt.figure(figsize=(8, 6))
    plt.hist(resid_nz, bins=50, edgecolor="k", alpha=0.7)
    plt.title(f"{RUN_NAME} — Residuals (without zeros)")
    plt.xlabel("Residual [W/m²]" if space == "Real" else "Residual")
    plt.ylabel("Frequency"); plt.grid(True); plt.tight_layout()
    plt.savefig(out, dpi=300); plt.close()


def make_summary(paths, metrics_n, metrics_r, hparams):
    fig = plt.figure(figsize=(16, 15))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])
    images = [
        (gs[0, 0], paths["scatter_n"],    "Scatter Norm"),
        (gs[0, 1], paths["scatter_r"],    "Scatter Real"),
        (gs[0, 2], paths["loss"],         "Loss curve"),
        (gs[1, 0], paths["hist_n"],       "Histogram Norm"),
        (gs[1, 1], paths["hist_r"],       "Histogram Real"),
    ]
    for cell, fname, title in images:
        ax = fig.add_subplot(cell)
        ax.imshow(plt.imread(fname)); ax.set_title(title, fontsize=11); ax.axis("off")

    ax_hp = fig.add_subplot(gs[1, 2])
    ax_hp.axis("off")
    tbl = ax_hp.table(
        cellText=[[k, str(v)] for k, v in hparams.items()],
        colLabels=["Hyperparameter", "Value"], loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.4)
    ax_hp.set_title("Hyperparameters", fontsize=11, pad=6)

    for cell, key, title in [
        (gs[2, 0], "hist_n_zeros", "Hist Norm ∅0"),
        (gs[2, 1], "hist_r_zeros", "Hist Real ∅0"),
    ]:
        ax = fig.add_subplot(cell)
        ax.imshow(plt.imread(paths[key])); ax.set_title(title, fontsize=11); ax.axis("off")

    ax_met = fig.add_subplot(gs[2, 2])
    ax_met.axis("off")
    metric_rows = ["MSE", "RMSE", "MAE", "R2", "Corr", "Residual_Variance", "SkillScore"]
    metric_data = [[r, f"{metrics_r[r]:.4f}"] for r in metric_rows if r in metrics_r]
    tbl_met = ax_met.table(
        cellText=metric_data, colLabels=["Metric", "Real value"],
        loc="center", cellLoc="center",
    )
    tbl_met.auto_set_font_size(False); tbl_met.set_fontsize(9); tbl_met.scale(1.1, 1.5)
    ax_met.set_title("Evaluation metrics (real space)", fontsize=11, pad=6)

    fig.suptitle(RUN_NAME, fontsize=17, y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(paths["summary"], dpi=300); plt.close(fig)


# =======================================================================
# MAIN
# =======================================================================

def main():
    print(f"\n{'='*62}")
    print(f"  Run: {RUN_NAME}")
    print(f"  Data: {SEQ_NPZ}")
    print(f"  Activation: {ACTIVATION} | De-scaling: {DESCALER_METHOD} | Night mask: {USE_DAYMASK}")
    print(f"{'='*62}\n")

    # ── 1. Load splits ────────────────────────────────────────────────────────
    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(SEQ_NPZ)
    print(f"Input shape (train): {X_tr.shape}  — expected (N, 24, 28)")

    seq_len, n_feat = X_tr.shape[1], X_tr.shape[2]

    tr_dl = DataLoader(SeqDS(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(SeqDS(X_va, y_va), BATCH_SIZE)
    te_dl = DataLoader(SeqDS(X_te, y_te), BATCH_SIZE)

    # ── 2. Day/night masks ────────────────────────────────────────────────────
    mask_train  = compute_day_mask(t_tr)
    mask_val    = compute_day_mask(t_va)
    mask_test   = compute_day_mask(t_te)
    hours_train = compute_hours(t_tr)
    hours_val   = compute_hours(t_va)

    # ── 3. Build UniLSTM model ────────────────────────────────────────────────
    model = UniLSTMRegressor(
        n_feat     = n_feat,
        hidden     = HIDDEN_CAUSAL,
        seq_len    = seq_len,
        num_layers = NUM_LAYERS_CAUSAL,
        dropout    = DROPOUT,
        activation = ACTIVATION,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (BiLSTM-128 reference: ~1,005,216)")
    assert n_params < 1_005_216, \
        f"UniLSTM has {n_params:,} params — expected < 1,005,216"

    # First-batch sanity check: loss must be finite before training begins
    xb_check, yb_check, _ = next(iter(tr_dl))
    with torch.no_grad():
        out_check = model(xb_check)
    loss_check = nn.MSELoss()(out_check, yb_check)
    assert torch.isfinite(loss_check), "First batch loss is NaN/Inf — check input data"
    print(f"First batch loss : {loss_check.item():.5f}  (finite ✓)")

    optim     = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=L2_LAMBDA)
    scheduler = ReduceLROnPlateau(
        optim, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
        threshold=1e-4, min_lr=MIN_LR, verbose=True,
    )
    loss_fn = nn.MSELoss(reduction="none")

    if USE_ASYMMETRIC_LOSS:
        asym_loss_fn = AsymmetricHourWeightedLoss(
            alpha=ALPHA, hour_weights=HOUR_WEIGHTS, use_daymask=USE_DAYMASK,
        )
    else:
        asym_loss_fn = None

    if _TENSORBOARD:
        writer    = SummaryWriter(log_dir=RUN_DIR)
        dummy_inp = torch.randn(1, seq_len, n_feat)
        writer.add_graph(model, dummy_inp)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    best_val, patience, history = float("inf"), 0, []
    pbar = trange(1, EPOCHS + 1, desc="Epochs", unit="ep", leave=True)
    for ep in pbar:
        model.train()
        train_loss = 0.0
        for xb, yb, idx in tr_dl:
            optim.zero_grad()
            preds = model(xb)

            if USE_DAYMASK:
                day_b = mask_train[idx]
                preds = preds * day_b
            else:
                day_b = torch.ones(len(yb))

            if USE_ASYMMETRIC_LOSS:
                loss = asym_loss_fn(preds, yb, hours_train[idx], day_b)
            elif USE_DAYTIME_ONLY_LOSS:
                day_idx = day_b > 0
                if day_idx.any():
                    loss = loss_fn(preds[day_idx], yb[day_idx]).mean()
                else:
                    loss = torch.tensor(0.0, requires_grad=True)
            elif USE_DAYMASK:
                loss = (loss_fn(preds, yb) * day_b).mean()
            else:
                loss = loss_fn(preds, yb).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(tr_dl.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, idx in va_dl:
                preds = model(xb)
                if USE_DAYMASK:
                    day_b = mask_val[idx]
                    preds = preds * day_b
                else:
                    day_b = torch.ones(len(yb))

                if USE_ASYMMETRIC_LOSS:
                    loss = asym_loss_fn(preds, yb, hours_val[idx], day_b)
                elif USE_DAYTIME_ONLY_LOSS:
                    day_idx = day_b > 0
                    if day_idx.any():
                        loss = loss_fn(preds[day_idx], yb[day_idx]).mean()
                    else:
                        loss = torch.tensor(0.0)
                elif USE_DAYMASK:
                    loss = (loss_fn(preds, yb) * day_b).mean()
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

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save(model.state_dict(), PATHS["model"])
        else:
            patience += 1
            if patience >= EARLY_STOP:
                tqdm.write("Early stopping triggered.")
                break

    plot_loss(history, PATHS["loss"])

    # ── 5. Evaluation — normalised space ─────────────────────────────────────
    model.load_state_dict(torch.load(PATHS["model"]))
    model.eval()

    preds_list, attn_list = [], []
    with torch.no_grad():
        for xb, _, idx in te_dl:
            preds, attn = model(xb, return_attention=True)
            if USE_DAYMASK:
                preds = preds * mask_test[idx]
            preds_list.append(preds.numpy())
            attn_list.append(attn.numpy())

    y_pred       = np.concatenate(preds_list)
    attn_weights = np.concatenate(attn_list)
    np.save(PATHS["attn_weights"], attn_weights)
    print(f"Attention weights saved → shape {attn_weights.shape}")

    residuals_n         = y_te - y_pred
    residual_variance_n = float(np.var(residuals_n))
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

    pd.DataFrame({"time": pd.to_datetime(t_te), "y_true": y_te, "y_pred": y_pred})\
        .to_csv(PATHS["csv_n"], index=False)
    scatter(y_te, y_pred, PATHS["scatter_n"], "Norm")
    hist_residuos(y_pred - y_te, PATHS["hist_n"], "Norm")
    hist_sin_ceros(y_pred - y_te, PATHS["hist_n_zeros"], "Norm")

    # ── 6. De-scaling → W/m² ─────────────────────────────────────────────────
    da_ref    = get_dataarray(DESCALER_FILE, DESCALER_VAR)
    stats_ref = build_stats(da_ref, DESCALER_METHOD)
    y_true_r  = descale(y_te,   t_te, DESCALER_METHOD, da_ref, stats_ref)
    y_pred_r  = descale(y_pred, t_te, DESCALER_METHOD, da_ref, stats_ref)

    # ── 7. Evaluation — real space ────────────────────────────────────────────
    residuals_r         = y_true_r - y_pred_r
    residual_variance_r = float(np.var(residuals_r))
    metrics_r = dict(
        MSE              = mean_squared_error(y_true_r, y_pred_r),
        RMSE             = root_mean_squared_error(y_true_r, y_pred_r),
        MAE              = mean_absolute_error(y_true_r, y_pred_r),
        R2               = r2_score(y_true_r, y_pred_r),
        Corr             = float(np.corrcoef(y_true_r.ravel(), y_pred_r.ravel())[0, 1]),
        Residual_Variance= residual_variance_r,
    )

    csi_at_test = da_ref.sel(observation_time=t_te).values
    day_mask    = csi_at_test > 0
    y_true_day  = y_true_r[day_mask]
    y_pred_day  = y_pred_r[day_mask]
    mse_day     = float(mean_squared_error(y_true_day, y_pred_day))
    metrics_r["RMSE_daytime"] = float(np.sqrt(mse_day))
    metrics_r["MAE_daytime"]  = float(mean_absolute_error(y_true_day, y_pred_day))
    metrics_r["SkillScore"]   = 1.0 - mse_day / MSE_BASELINE_R

    print("\n=== Evaluation — real space (W/m²) ===")
    for k, v in metrics_r.items():
        print(f"{k:>20}: {fmt_de(v)}")

    pd.DataFrame({"time": pd.to_datetime(t_te), "y_true": y_true_r, "y_pred": y_pred_r})\
        .to_csv(PATHS["csv_r"], index=False)
    scatter(y_true_r, y_pred_r, PATHS["scatter_r"], "Real")
    hist_residuos(y_pred_r - y_true_r, PATHS["hist_r"], "Real")
    hist_sin_ceros(y_pred_r - y_true_r, PATHS["hist_r_zeros"], "Real")

    # ── 8. Summary image + report ─────────────────────────────────────────────
    hparams = {
        "LR":               LR_INIT,
        "Epochs":           EPOCHS,
        "Batch Size":       BATCH_SIZE,
        "Hidden":           HIDDEN_CAUSAL,
        "Dropout":          DROPOUT,
        "Num Layers":       NUM_LAYERS_CAUSAL,
        "L2 Lambda":        L2_LAMBDA,
        "Early Stop":       EARLY_STOP,
        "LR Factor":        LR_FACTOR,
        "LR Patience":      LR_PATIENCE,
        "Activation":       ACTIVATION,
        "De-scaling":       DESCALER_METHOD,
        "Night mask":       USE_DAYMASK,
        "Daytime-only loss":USE_DAYTIME_ONLY_LOSS,
        "Asymmetric loss":  USE_ASYMMETRIC_LOSS,
        "Architecture":     "UniLSTM (bidirectional=False)",
        "Launch time":      "0100 only",
        "Seq mode":         "causal next-step (k=24)",
        "n_features":       n_feat,
        "n_params":         f"{n_params:,}",
    }
    make_summary(PATHS, metrics_n, metrics_r, hparams)

    with open(PATHS["report"], "w", encoding="utf-8") as f:
        f.write(f"# UniLSTM run report — {RUN_NAME}\n")
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


if __name__ == "__main__":
    main()
