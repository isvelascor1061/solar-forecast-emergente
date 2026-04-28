#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optuna_search.py
================
Hyperparameter search for the BiLSTM solar radiation forecasting model
using the Optuna framework.

Search space
------------
  lr          : log-uniform in [1e-4, 1e-2]
  hidden      : categorical [32, 64, 96, 128, 192]
  num_layers  : categorical [1, 2, 3]
  dropout     : uniform in [0.1, 0.5]
  batch_size  : categorical [32, 64, 128, 256]
  early_stop  : categorical [10, 15, 20]

Protocol per trial
------------------
  - Up to 20 training epochs
  - Validation MSE reported after every epoch (for Optuna pruning)
  - MedianPruner cuts trials that perform worse than the median of
    completed trials at the same epoch step
  - Objective: minimise final validation MSE

Data
----
  Sym18 sequences: 4launch_multfeat_sym18.npz (37 steps, 69 features)

Output
------
  _4_LSTM_modules/Prepared_data/optuna_results.csv
  Columns: trial, rmse, r2, lr, hidden, num_layers, dropout,
           batch_size, early_stop
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.pruners import MedianPruner

# Project imports
from _4_LSTM_modules.NN_modules.BiLSTMRegressor import BiLSTMRegressor
from config import (
    FLAG_START, FLAG_END, UTC_OFFSET,
    CSI_GHI_FILE, CSI_VAR_NAME,
    MSE_BASELINE_R,
)
import xarray as xr

# ============================================================
# FIXED CONFIGURATION (not part of the search space)
# ============================================================

# Sequence file to use for all trials
SEQ_NPZ = "_4_LSTM_modules/Prepared_data/4launch_multfeat_sym18.npz"

# Maximum epochs per trial
MAX_EPOCHS = 20

# Total number of Optuna trials
N_TRIALS = 30

# Minimum learning rate for the ReduceLROnPlateau scheduler
MIN_LR = 1e-6

# LR scheduler settings (fixed, same as baseline training)
LR_FACTOR   = 0.5
LR_PATIENCE = 4

# Output CSV path
OUT_CSV = "_4_LSTM_modules/Prepared_data/optuna_results.csv"

# De-scaling uses the physical method (CSI × clear-sky GHI)
DESCALER_FILE = CSI_GHI_FILE
DESCALER_VAR  = CSI_VAR_NAME

# Activation and descaling method (fixed, matching baseline)
ACTIVATION       = "sigmoid"
DESCALER_METHOD  = "physical"

# Night mask applied during training and inference
USE_DAYMASK = True

# ============================================================
# DATASET
# ============================================================

class SeqDS(Dataset):
    """Simple 3-D sequence dataset returning (X, y, original_index)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


def load_splits(path: str):
    """Load train / val / test arrays from the .npz file."""
    d = np.load(path, allow_pickle=True)
    return (
        d["X_train"], d["y_train"], d["t_train"],
        d["X_val"],   d["y_val"],   d["t_val"],
        d["X_test"],  d["y_test"],  d["t_test"],
    )


# ============================================================
# DAY / NIGHT MASK
# ============================================================

def compute_day_mask(timestamps) -> torch.Tensor:
    """
    Returns a float tensor (1 = daytime, 0 = night-time) for every
    timestamp.  Night is defined as local hours outside [FLAG_END, FLAG_START].
    """
    ts_local = pd.to_datetime(timestamps) + pd.Timedelta(hours=UTC_OFFSET)
    hours = ts_local.hour
    mask  = ((hours >= FLAG_END) & (hours <= FLAG_START)).astype(float)
    vals  = mask.values if hasattr(mask, "values") else mask
    return torch.tensor(vals, dtype=torch.float32)


# ============================================================
# DE-SCALING (normalised CSI -> W/m²)
# ============================================================

def get_clearsky_da():
    """Load the clear-sky GHI DataArray used for physical de-scaling."""
    ds = xr.open_dataset(DESCALER_FILE)
    return ds[DESCALER_VAR].squeeze()


def descale_physical(arr: np.ndarray, times: np.ndarray,
                     da: xr.DataArray) -> np.ndarray:
    """Multiply normalised predictions by the clear-sky reference value."""
    return arr * da.sel(observation_time=times).values


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def make_objective(X_tr, y_tr, t_tr,
                   X_va, y_va, t_va,
                   X_te, y_te, t_te,
                   mask_tr, mask_va, mask_te,
                   da_ref):
    """
    Returns a closure that Optuna calls for each trial.
    Closing over the pre-loaded data avoids reloading the .npz on
    every trial.
    """
    n_feat  = X_tr.shape[2]
    seq_len = X_tr.shape[1]

    def objective(trial: optuna.Trial) -> float:
        # ---- Sample hyperparameters ----------------------------------
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden     = trial.suggest_categorical("hidden", [32, 64, 96, 128, 192])
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        dropout    = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        early_stop = trial.suggest_categorical("early_stop", [10, 15, 20])

        # ---- DataLoaders --------------------------------------------
        tr_dl = DataLoader(SeqDS(X_tr, y_tr), batch_size, shuffle=True)
        va_dl = DataLoader(SeqDS(X_va, y_va), batch_size)

        # ---- Model --------------------------------------------------
        model = BiLSTMRegressor(
            n_feat=n_feat, hidden=hidden, seq_len=seq_len,
            num_layers=num_layers, dropout=dropout,
            activation=ACTIVATION,
        )

        optim_obj = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(
            optim_obj, mode="min", factor=LR_FACTOR,
            patience=LR_PATIENCE, min_lr=MIN_LR,
        )
        loss_fn = nn.MSELoss(reduction="none")

        best_val  = float("inf")
        patience  = 0

        # ---- Training loop (max MAX_EPOCHS) -------------------------
        for ep in range(1, MAX_EPOCHS + 1):
            # Train
            model.train()
            for xb, yb, idx in tr_dl:
                optim_obj.zero_grad()
                preds = model(xb)
                if USE_DAYMASK:
                    m     = mask_tr[idx]
                    preds = preds * m
                    loss  = (loss_fn(preds, yb) * m).mean()
                else:
                    loss = loss_fn(preds, yb).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim_obj.step()

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb, idx in va_dl:
                    preds = model(xb)
                    if USE_DAYMASK:
                        m     = mask_va[idx]
                        preds = preds * m
                        loss  = (loss_fn(preds, yb) * m).mean()
                    else:
                        loss = loss_fn(preds, yb).mean()
                    val_loss += loss.item() * len(xb)
            val_loss /= len(va_dl.dataset)

            scheduler.step(val_loss)

            # Report to Optuna (enables pruning)
            trial.report(val_loss, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Early stopping within the trial
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                # Save best weights in memory (state_dict copy)
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}
            else:
                patience += 1
                if patience >= early_stop:
                    break

        # ---- Evaluate best checkpoint on test set -------------------
        model.load_state_dict(best_state)
        model.eval()
        te_dl = DataLoader(SeqDS(X_te, y_te), batch_size=256)

        preds_list = []
        with torch.no_grad():
            for xb, _, idx in te_dl:
                preds = model(xb)
                if USE_DAYMASK:
                    preds = preds * mask_te[idx]
                preds_list.append(preds.numpy())
        y_pred = np.concatenate(preds_list)

        # De-scale to W/m²
        y_true_r = descale_physical(y_te,   t_te, da_ref)
        y_pred_r = descale_physical(y_pred, t_te, da_ref)

        rmse = float(np.sqrt(mean_squared_error(y_true_r, y_pred_r)))
        r2   = float(r2_score(y_true_r, y_pred_r))

        # Store extra metrics as user attributes on the trial
        trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("r2",   r2)

        return best_val   # objective to minimise = best validation MSE

    return objective


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading sequences from:", SEQ_NPZ)
    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(SEQ_NPZ)
    print(f"  Train: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")

    # Pre-compute masks once so every trial reuses them
    mask_tr = compute_day_mask(t_tr)
    mask_va = compute_day_mask(t_va)
    mask_te = compute_day_mask(t_te)

    # Load clear-sky DataArray once
    da_ref = get_clearsky_da()

    # Build the objective closure
    objective = make_objective(
        X_tr, y_tr, t_tr,
        X_va, y_va, t_va,
        X_te, y_te, t_te,
        mask_tr, mask_va, mask_te,
        da_ref,
    )

    # ---- Create Optuna study ----------------------------------------
    # MedianPruner: prune a trial if its intermediate value is worse
    # than the median of all completed trials at the same epoch step.
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study  = optuna.create_study(direction="minimize", pruner=pruner)

    print(f"\nStarting Optuna search: {N_TRIALS} trials, "
          f"up to {MAX_EPOCHS} epochs each.\n")
    study.optimize(objective, n_trials=N_TRIALS,
                   catch=(RuntimeError,))   # catch CUDA OOM if GPU is used

    # ---- Collect results into a DataFrame ---------------------------
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue   # skip pruned / failed trials
        p = t.params
        rows.append({
            "trial":      t.number,
            "rmse":       t.user_attrs.get("rmse", float("nan")),
            "r2":         t.user_attrs.get("r2",   float("nan")),
            "lr":         p["lr"],
            "hidden":     p["hidden"],
            "num_layers": p["num_layers"],
            "dropout":    p["dropout"],
            "batch_size": p["batch_size"],
            "early_stop": p["early_stop"],
        })

    df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResults saved to: {OUT_CSV}")
    print(df.to_string(index=False))

    # ---- Print best trial -------------------------------------------
    best = study.best_trial
    print("\n" + "="*55)
    print("BEST TRIAL")
    print("="*55)
    print(f"  Trial number : {best.number}")
    print(f"  Val MSE      : {best.value:.6f}")
    print(f"  RMSE (W/m2)  : {best.user_attrs.get('rmse', 'n/a'):.4f}")
    print(f"  R2           : {best.user_attrs.get('r2',   'n/a'):.4f}")
    print("  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<14}: {v}")
    print("="*55)


if __name__ == "__main__":
    main()
