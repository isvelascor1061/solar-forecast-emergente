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
NPZ_FILE   = "_4_LSTM_modules/Prepared_data/4launch_multfeat_sym24.npz"
MODEL_FILE = "_4_LSTM_modules/_runs/4launch_multfeat_sym/4launch_Multfeat_sym24_numl3_hidden96_20250701_153709/best_model.pt"
CSV_OUT    = "_4_LSTM_modules/Evaluation/Sym24_predictions_full.csv"

# Descaling specification: (NetCDF-path, method, variable)
DESCALER   = (
    "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/"
    "CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct2.nc",
    "physical",              # "physical"   or "minmax"
    "clear_sky_ghi",         # variable name inside the NetCDF
)

# ---------------- Hyper-parameters (identical to training) -----------------
N_FEAT      = 69
HIDDEN      = 96
NUM_LAYERS  = 3
DROPOUT     = 0.25
SEQ_LEN     = None          # auto-detect from first batch
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 128
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
