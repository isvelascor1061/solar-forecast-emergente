#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity_analysis.py
=======================
Architecture sensitivity grid search for the Bi-LSTM solar irradiance model.

Tests all 15 combinations of:
    num_layers  : [3, 5, 7, 8, 10]
    hidden_size : [64, 128, 256]

Fixed hyperparameters (from config.py):
    LR          = 0.001   (LR_INIT)
    BATCH_SIZE  = 128
    DROPOUT     = 0.25
    EARLY_STOP  = 15
    ACTIVATION  = "sigmoid"
    USE_DAYMASK = True
    DESCALER    = "physical"
    MSE_BASELINE_R imported from config.py (daytime-only GFS baseline)

Each combination trains for at most 35 epochs (early stopping patience = 15).

Resume capability
-----------------
If sensitivity_results.csv already exists, combinations already present
(matched by num_layers + hidden_size) are skipped automatically.

Metrics reported on the test set
---------------------------------
    RMSE_all   : RMSE over ALL hours (W/m²)
    RMSE_day   : RMSE over DAYTIME hours only (clear_sky_ghi > 0)
    R2_day     : R² over daytime hours only
    SkillScore : 1 − MSE_model_day / MSE_BASELINE_R  (daytime W/m² space)
    n_params   : number of trainable parameters
    epochs_run : epochs actually trained (may be < 35 if early stop fired)

Outputs (written to sensitivity_results/ next to this script)
--------------------------------------------------------------
    sensitivity_results.csv
    heatmap_rmse_day.png
    heatmap_skillscore.png
    lineplot_rmse_vs_hidden.png
    lineplot_rmse_vs_layers.png

Usage
-----
    python _4_LSTM_modules/Main_execution_files/sensitivity_analysis.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score

# ── Project root (two levels up from this file) ────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import (
    SEQ_NPZ_FILE,
    CSI_GHI_FILE, CSI_VAR_NAME,
    LR_INIT, BATCH_SIZE, DROPOUT,
    L2_LAMBDA, LR_FACTOR, LR_PATIENCE, MIN_LR,
    ACTIVATION, DESCALER_METHOD, USE_DAYMASK,
    FLAG_START, FLAG_END, UTC_OFFSET,
    MSE_BASELINE_R,
)
from _4_LSTM_modules.NN_modules.BiLSTMRegressor import BiLSTMRegressor


# ===========================================================================
# SENSITIVITY GRID
# ===========================================================================
NUM_LAYERS_LIST  = [3, 5, 7, 8, 10]
HIDDEN_SIZE_LIST = [64, 128, 256]

# Training limits for the sensitivity sweep
SWEEP_EPOCHS     = 35   # maximum epochs per combination
SWEEP_EARLY_STOP = 15   # early stopping patience

# Output directory (relative to this script)
OUT_DIR = Path(__file__).parent / "sensitivity_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "sensitivity_results.csv"


# ===========================================================================
# UTILITIES
# ===========================================================================

class SeqDS(Dataset):
    """3-D sequence dataset that also yields the sample index."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


def load_splits(path: str):
    """Load train/val/test splits from the .npz prepared-sequence file."""
    d = np.load(path, allow_pickle=True)
    return (
        d["X_train"], d["y_train"], d["t_train"],
        d["X_val"],   d["y_val"],   d["t_val"],
        d["X_test"],  d["y_test"],  d["t_test"],
    )


def compute_day_mask(timestamps) -> torch.Tensor:
    """Binary mask: 1 = daytime, 0 = night (FLAG_END <= local_hour <= FLAG_START)."""
    ts_local = pd.to_datetime(timestamps) + pd.Timedelta(hours=UTC_OFFSET)
    hours    = ts_local.hour
    mask     = ((hours >= FLAG_END) & (hours <= FLAG_START)).astype(float)
    vals     = mask.values if hasattr(mask, "values") else mask
    return torch.tensor(vals, dtype=torch.float32)


def load_clearsky_series(path: str, var: str) -> pd.Series:
    """Load Ineichen clear-sky GHI as a Series indexed by observation_time."""
    ds = xr.open_dataset(path, engine="h5netcdf")
    da = ds[var]
    if {"lat", "lon"}.issubset(da.dims):
        da = da.mean(dim=("lat", "lon"))
    for dim in list(da.dims):
        if dim != "observation_time":
            da = da.isel({dim: 0})
    ser = da.to_series()
    ds.close()
    return ser


def descale_physical(csi: np.ndarray, times: np.ndarray, clearsky: pd.Series) -> np.ndarray:
    """Multiply normalised CSI by clear-sky GHI to obtain W/m²."""
    csg_vals = clearsky.reindex(pd.DatetimeIndex(times)).values
    return csi * csg_vals


def load_existing_results() -> set[tuple[int, int]]:
    """
    Return a set of (num_layers, hidden_size) tuples already present in
    sensitivity_results.csv so they can be skipped.
    """
    if not CSV_PATH.exists():
        return set()
    df = pd.read_csv(CSV_PATH)
    done = set(
        zip(df["num_layers"].astype(int), df["hidden_size"].astype(int))
    )
    return done


def append_result(record: dict) -> None:
    """Append one result row to sensitivity_results.csv (create if needed)."""
    row_df = pd.DataFrame([record])
    if CSV_PATH.exists():
        row_df.to_csv(CSV_PATH, mode="a", header=False, index=False, float_format="%.6f")
    else:
        row_df.to_csv(CSV_PATH, index=False, float_format="%.6f")


# ===========================================================================
# TRAINING FUNCTION
# ===========================================================================

def train_one_combo(
    num_layers: int,
    hidden_size: int,
    X_tr, y_tr, t_tr,
    X_va, y_va, t_va,
    X_te, y_te, t_te,
    clearsky: pd.Series,
) -> dict:
    """Train one BiLSTM combo and return a dict of test-set metrics."""
    seq_len = X_tr.shape[1]
    n_feat  = X_tr.shape[2]

    # ---- Data loaders ------------------------------------------------
    tr_dl = DataLoader(SeqDS(X_tr, y_tr), BATCH_SIZE, shuffle=True,  drop_last=False)
    va_dl = DataLoader(SeqDS(X_va, y_va), BATCH_SIZE, shuffle=False, drop_last=False)
    te_dl = DataLoader(SeqDS(X_te, y_te), BATCH_SIZE, shuffle=False, drop_last=False)

    # ---- Day masks ---------------------------------------------------
    mask_train = compute_day_mask(t_tr)
    mask_val   = compute_day_mask(t_va)
    mask_test  = compute_day_mask(t_te)

    # ---- Model -------------------------------------------------------
    model = BiLSTMRegressor(
        n_feat     = n_feat,
        hidden     = hidden_size,
        seq_len    = seq_len,
        num_layers = num_layers,
        dropout    = DROPOUT,
        activation = ACTIVATION,
    )
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_INIT, weight_decay=L2_LAMBDA
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
        threshold=1e-4, min_lr=MIN_LR,
    )
    loss_fn = nn.MSELoss(reduction="none")

    # ---- Training loop -----------------------------------------------
    best_val   = float("inf")
    patience   = 0
    best_state = None
    epochs_run = 0

    for ep in range(1, SWEEP_EPOCHS + 1):
        epochs_run = ep

        model.train()
        train_loss = 0.0
        for xb, yb, idx in tr_dl:
            optimizer.zero_grad()
            preds = model(xb)
            if USE_DAYMASK:
                mask  = mask_train[idx]
                preds = preds * mask
                loss  = (loss_fn(preds, yb) * mask).mean()
            else:
                loss = loss_fn(preds, yb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(tr_dl.dataset)

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

        if val_loss < best_val:
            best_val   = val_loss
            patience   = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= SWEEP_EARLY_STOP:
                break

    # ---- Inference on test set ---------------------------------------
    model.load_state_dict(best_state)
    model.eval()

    preds_list = []
    with torch.no_grad():
        for xb, _, idx in te_dl:
            preds = model(xb)
            if USE_DAYMASK:
                preds = preds * mask_test[idx]
            preds_list.append(preds.numpy())

    y_pred_csi = np.concatenate(preds_list)

    # ---- Descale to W/m² --------------------------------------------
    y_true_wm2 = descale_physical(y_te, t_te, clearsky)
    y_pred_wm2 = descale_physical(y_pred_csi, t_te, clearsky)

    # ---- RMSE all hours ---------------------------------------------
    rmse_all = float(np.sqrt(mean_squared_error(y_true_wm2, y_pred_wm2)))

    # ---- Daytime-only metrics (clear_sky_ghi > 0) -------------------
    csg_te     = clearsky.reindex(pd.DatetimeIndex(t_te)).values
    day_mask   = csg_te > 0
    mse_day    = float(mean_squared_error(y_true_wm2[day_mask], y_pred_wm2[day_mask]))
    rmse_day   = float(np.sqrt(mse_day))
    r2_day     = float(r2_score(y_true_wm2[day_mask], y_pred_wm2[day_mask]))
    skill_day  = 1.0 - mse_day / MSE_BASELINE_R

    return {
        "num_layers":    num_layers,
        "hidden_size":   hidden_size,
        "n_params":      n_params,
        "epochs_run":    epochs_run,
        "best_val_loss": best_val,
        "RMSE_all":      rmse_all,
        "RMSE_day":      rmse_day,
        "R2_day":        r2_day,
        "SkillScore":    skill_day,
    }


# ===========================================================================
# PLOT HELPERS
# ===========================================================================

def save_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
    cmap: str = "viridis_r",
    fmt: str = ".2f",
) -> None:
    """Save a heatmap (rows = num_layers, columns = hidden_size)."""
    pivot = (
        df.pivot(index="num_layers", columns="hidden_size", values=metric)
        .sort_index()
        .reindex(sorted(df["hidden_size"].unique()), axis=1)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("hidden_size")
    ax.set_ylabel("num_layers")
    ax.set_title(title)

    for r_idx, row in enumerate(pivot.index):
        for c_idx, col in enumerate(pivot.columns):
            val = pivot.loc[row, col]
            if pd.isna(val):
                continue
            text_color = "white" if im.norm(val) < 0.5 else "black"
            ax.text(c_idx, r_idx, format(val, fmt),
                    ha="center", va="center", fontsize=9, color=text_color)

    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"  Saved: {Path(filename).name}")


def save_lineplot_vs_hidden(df: pd.DataFrame, filename: str) -> None:
    """Line plot: RMSE_day vs hidden_size, one line per num_layers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for nl in sorted(df["num_layers"].unique()):
        sub = df[df["num_layers"] == nl].sort_values("hidden_size")
        ax.plot(sub["hidden_size"], sub["RMSE_day"], marker="o", label=f"layers={nl}")
    ax.set_xlabel("hidden_size")
    ax.set_ylabel("RMSE daytime [W/m²]")
    ax.set_title("Daytime RMSE vs hidden_size  (lower is better)")
    ax.legend(title="num_layers")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"  Saved: {Path(filename).name}")


def save_lineplot_vs_layers(df: pd.DataFrame, filename: str) -> None:
    """Line plot: RMSE_day vs num_layers, one line per hidden_size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for hs in sorted(df["hidden_size"].unique()):
        sub = df[df["hidden_size"] == hs].sort_values("num_layers")
        ax.plot(sub["num_layers"], sub["RMSE_day"], marker="o", label=f"hidden={hs}")
    ax.set_xlabel("num_layers")
    ax.set_ylabel("RMSE daytime [W/m²]")
    ax.set_title("Daytime RMSE vs num_layers  (lower is better)")
    ax.legend(title="hidden_size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"  Saved: {Path(filename).name}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 65)
    print("  BiLSTM Sensitivity Analysis — num_layers × hidden_size")
    print(f"  Grid: {NUM_LAYERS_LIST} × {HIDDEN_SIZE_LIST}  ({len(NUM_LAYERS_LIST) * len(HIDDEN_SIZE_LIST)} combos)")
    print(f"  Epochs/combo: {SWEEP_EPOCHS}  |  Early-stop patience: {SWEEP_EARLY_STOP}")
    print(f"  Fixed — LR: {LR_INIT}  BATCH: {BATCH_SIZE}  DROPOUT: {DROPOUT}")
    print(f"          ACTIVATION: {ACTIVATION}  DAYMASK: {USE_DAYMASK}  DESCALER: {DESCALER_METHOD}")
    print(f"  MSE_BASELINE_R (GFS daytime): {MSE_BASELINE_R:.3f}")
    print("=" * 65)

    # ---- Resume: find already-completed combinations -----------------
    done = load_existing_results()
    if done:
        print(f"\nResume mode — skipping {len(done)} already-completed combo(s):")
        for nl, hs in sorted(done):
            print(f"  layers={nl}  hidden={hs}")

    # ---- Load data once ----------------------------------------------
    npz_path = ROOT / SEQ_NPZ_FILE
    print(f"\nLoading sequences from:\n  {npz_path}")
    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(str(npz_path))
    print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    # ---- Load Ineichen clear-sky reference ---------------------------
    csg_path = ROOT / CSI_GHI_FILE
    print(f"\nLoading clear-sky reference from:\n  {csg_path}")
    clearsky = load_clearsky_series(str(csg_path), CSI_VAR_NAME)
    print(f"  Clear-sky series: {len(clearsky):,} hourly values")

    # ---- Grid search -------------------------------------------------
    combos   = list(product(NUM_LAYERS_LIST, HIDDEN_SIZE_LIST))
    n_combos = len(combos)
    pending  = [(nl, hs) for nl, hs in combos if (nl, hs) not in done]

    print(f"\nStarting grid search — {len(pending)}/{n_combos} combinations to run\n")

    for combo_idx, (nl, hs) in enumerate(pending, start=len(done) + 1):
        t0 = time.time()
        print(f"[{combo_idx:02d}/{n_combos}]  num_layers={nl}  hidden_size={hs}", end="  ", flush=True)

        try:
            metrics = train_one_combo(
                num_layers=nl, hidden_size=hs,
                X_tr=X_tr, y_tr=y_tr, t_tr=t_tr,
                X_va=X_va, y_va=y_va, t_va=t_va,
                X_te=X_te, y_te=y_te, t_te=t_te,
                clearsky=clearsky,
            )
            elapsed = time.time() - t0
            metrics["elapsed_s"] = round(elapsed, 1)
            append_result(metrics)

            print(
                f"params={metrics['n_params']:,}  "
                f"epochs={metrics['epochs_run']}  "
                f"RMSE_day={metrics['RMSE_day']:.2f}  "
                f"SS={metrics['SkillScore']:.4f}  "
                f"R2={metrics['R2_day']:.4f}  "
                f"({elapsed:.0f}s)"
            )
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.0f}s): {exc}")
            append_result({
                "num_layers": nl, "hidden_size": hs,
                "n_params": None, "epochs_run": None,
                "best_val_loss": None,
                "RMSE_all": None, "RMSE_day": None,
                "R2_day": None, "SkillScore": None,
                "elapsed_s": round(elapsed, 1),
            })

    # ---- Load full results (including previously saved rows) ---------
    df = pd.read_csv(CSV_PATH).sort_values(["num_layers", "hidden_size"]).reset_index(drop=True)
    df_valid = df.dropna(subset=["RMSE_day", "SkillScore"])

    # ---- Print ranked results table (sorted by daytime RMSE) ---------
    rank_cols = ["num_layers", "hidden_size", "n_params", "epochs_run",
                 "RMSE_all", "RMSE_day", "R2_day", "SkillScore"]
    ranked = df_valid.sort_values("RMSE_day")[rank_cols].reset_index(drop=True)
    ranked.index += 1   # 1-based rank

    print("\n" + "=" * 75)
    print("  RESULTS RANKED BY DAYTIME RMSE (ascending — lower is better)")
    print("=" * 75)
    print(ranked.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 75)

    best = ranked.iloc[0]
    print(
        f"\n  Best: num_layers={int(best['num_layers'])}  "
        f"hidden_size={int(best['hidden_size'])}  "
        f"RMSE_day={best['RMSE_day']:.2f} W/m²  "
        f"SkillScore={best['SkillScore']:.4f}  "
        f"R2={best['R2_day']:.4f}"
    )

    # ---- Plots -------------------------------------------------------
    print("\nGenerating plots…")

    save_heatmap(
        df_valid, "RMSE_day",
        title="RMSE daytime [W/m²]  — lower is better",
        filename=str(OUT_DIR / "heatmap_rmse_day.png"),
        cmap="YlOrRd", fmt=".1f",
    )
    save_heatmap(
        df_valid, "SkillScore",
        title="Skill Score (daytime)  — higher is better",
        filename=str(OUT_DIR / "heatmap_skillscore.png"),
        cmap="RdYlGn", fmt=".3f",
    )
    save_lineplot_vs_hidden(
        df_valid,
        filename=str(OUT_DIR / "lineplot_rmse_vs_hidden.png"),
    )
    save_lineplot_vs_layers(
        df_valid,
        filename=str(OUT_DIR / "lineplot_rmse_vs_layers.png"),
    )

    print(f"\nAll outputs in: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
