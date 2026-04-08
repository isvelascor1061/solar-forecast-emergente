#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 26 Jun 2025 – 10:11

@author: leonardmerl
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the *best* Bi-LSTM on **all** data splits, descale the predictions and
export everything to a single CSV file.

Workflow
--------
1.  Load the trained Bi-LSTM (`MODEL_FILE`).
2.  Read the prepared splits from the NPZ file (`NPZ_FILE`).
3.  Generate predictions for *train*, *val* and *test*.
4.  Descale the normalised output back to physical units (e.g. W m-²).
5.  Save a tidy DataFrame with columns  
       `time | split | y_true | y_pred`  
    to `CSV_OUT`.
"""

import numpy as np
import pandas as pd
import torch
import xarray as xr
from pathlib import Path

# -----------------------------------------------------------------------
# File paths & runtime settings
# -----------------------------------------------------------------------
from config import (
    SEQ_NPZ_FILE, BEST_MODEL_FILE, CSV_OUT,
    CSI_GHI_FILE, CSI_VAR_NAME,
    N_FEAT, HIDDEN, NUM_LAYERS, DROPOUT, BATCH_SIZE,
)

NPZ_FILE   = SEQ_NPZ_FILE
MODEL_FILE = BEST_MODEL_FILE

# Descaling specification: (NetCDF-path, method, variable)
DESCALER   = (
    CSI_GHI_FILE,
    "physical",              # "physical"   or "minmax"
    CSI_VAR_NAME,            # variable name inside the NetCDF
)

# ---------------- Hyper-parameters (identical to training) -----------------
SEQ_LEN     = None          # auto-detect from first batch
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper functions: load DataArray & apply inverse normalisation
# ---------------------------------------------------------------------------
def get_dataarray(path: str, var: str | None):
    """Open NetCDF and return the specified DataArray (1st variable if None)."""
    ds = xr.open_dataset(path)
    if var is None:
        var = list(ds.data_vars)[0]
    return ds[var]

def build_stats(da: xr.DataArray, method: str):
    """Pre-compute min/max for *minmax* rescaling."""
    if method == "minmax":
        return {"min": float(da.min()), "max": float(da.max())}
    return {}

def descale(
    arr: np.ndarray,
    times: np.ndarray,
    method: str,
    da: xr.DataArray,
    stats: dict | None = None,
) -> np.ndarray:
    """
    Convert normalised values back to physical space.
    Always returns a 1-D array with the same length as `arr`.
    """
    if method == "physical":
        # Multiplicative factor: select by time, average over lat/lon if present
        factor = (
            da.sel(observation_time=times)
              .mean(dim=[d for d in ("lat", "lon") if d in da.dims])
              .values
        )
        return arr * factor.squeeze()
    elif method == "minmax":
        return arr * (stats["max"] - stats["min"]) + stats["min"]
    else:
        raise ValueError(f"Unknown descaling method: {method}")

# ---------------------------------------------------------------------------
# 1)  Load the trained Bi-LSTM
# ---------------------------------------------------------------------------
from _4_LSTM_modules.NN_modules.BiLSTMRegressor import BiLSTMRegressor

model = BiLSTMRegressor(
    n_feat     = N_FEAT,
    hidden     = HIDDEN,
    seq_len    = SEQ_LEN,
    num_layers = NUM_LAYERS,
    dropout    = DROPOUT,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

# ---------------------------------------------------------------------------
# 2)  Prepare the descaler (reference DataArray + stats)
# ---------------------------------------------------------------------------
src_path, method, var = DESCALER
da_ref  = get_dataarray(src_path, var)
stats   = build_stats(da_ref, method)

# ---------------------------------------------------------------------------
# 3)  Load NPZ splits
# ---------------------------------------------------------------------------
data = np.load(NPZ_FILE, allow_pickle=True)
splits = [
    ("train", data["X_train"], data["y_train"], data["t_train"]),
    ("val",   data["X_val"],   data["y_val"],   data["t_val"]),
    ("test",  data["X_test"],  data["y_test"],  data["t_test"]),
]

frames = []
with torch.no_grad():
    for tag, X_np, y_np, t_np in splits:
        X_torch = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)

        preds = []
        for i in range(0, len(X_torch), BATCH_SIZE):
            preds.append(model(X_torch[i:i+BATCH_SIZE]).cpu().numpy())
        y_pred = np.concatenate(preds)               # (N,)

        # ----------------- Descale ------------------------------------
        times    = pd.to_datetime(t_np)              # (N,) DatetimeIndex
        y_true_r = descale(y_np,   times, method, da_ref, stats)
        y_pred_r = descale(y_pred, times, method, da_ref, stats)

        frames.append(pd.DataFrame({
            "time":   times,
            "split":  tag,
            "y_true": y_true_r,
            "y_pred": y_pred_r,
        }))

# ---------------------------------------------------------------------------
# 4)  Write CSV
# ---------------------------------------------------------------------------
df_out = pd.concat(frames).sort_values("time").reset_index(drop=True)
Path(CSV_OUT).parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(CSV_OUT, index=False)

print(f"✅  {len(df_out):,} rows written to {CSV_OUT}")
