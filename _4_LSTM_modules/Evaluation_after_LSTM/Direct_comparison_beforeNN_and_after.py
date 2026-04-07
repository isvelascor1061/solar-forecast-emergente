#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline check + (optional) NN-adjusted forecasts from a CSV file.

This script

    • builds a **scatter plot**        (Before-NN = blue, After-NN = red)
    • builds a **histogram**           (Before = blue, After = orange //)
    • builds a **histogram w/o zeros** (same colours)

and prints the evaluation metrics for the *Before-NN* variant.  
If you pass a CSV containing the neural-network output, the *After-NN*
results are plotted and their residual variance is printed as well.
"""

# ---------------- 0)  OPTIONAL NN CSV  ------------------------------------
CSV_PATH      = "_4_LSTM_modules/_runs/4launch_multfeat_sym/4launch_Multfeat_sym24_numl3_hidden96_20250701_153709/pred_real.csv"
CSV_IDX_COL   = "time"
CSV_TRUE_COL  = "y_true"
CSV_PRED_COL  = "y_pred"

# ---------------- 1)  Imports & constants  --------------------------------
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# --- File paths -----------------------------------------------------------
PAIRS = [
    (
        [
            "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped/dswrf1_CSI_0100.nc",
            "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped/dswrf1_CSI_0700.nc",
            "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped/dswrf1_CSI_1300.nc",
            "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped/dswrf1_CSI_1900.nc",
        ],
        ["dswrf1_0100", "dswrf1_0700", "dswrf1_1300", "dswrf1_1900"],   # same var in every file
        "_3_Data_preparation_for_LSTM/Preparation_data/_03_Siata_GHI/"
        "Netcdf_Siata_GHI/GHI_CSI_clipped.nc",
        "GHI_clean",
        "Best-lead GFS vs. SIATA"
    )
]

TEST_IDX_FILE = Path("_4_LSTM_modules/test_indices/test_indices_4launch_multfeat_sym24.npy")
SUBSET_COORDS = None          # e.g. {"lat": 6.25, "lon": -75.5}
USE_AREA_MEAN = True          # average over grid?

# ---------------- 2)  Load test indices -----------------------------------
test_idx = pd.to_datetime(np.load(TEST_IDX_FILE))

# ---------------- 3)  Helper functions ------------------------------------
def to_timeseries(da, subset=None, area_mean=False):
    """Arbitrary 3-D/4-D DataArray → 1-D time series."""
    time_dim = next(d for d in da.dims if np.issubdtype(da[d].dtype, np.datetime64))
    if subset is not None:
        da = da.sel(subset, method="nearest")
    elif area_mean and {"lat", "lon"} <= set(da.dims):
        da = da.mean(("lat", "lon"))
    for d in list(da.dims):
        if d != time_dim:
            da = da.isel({d: 0})
    return da.to_series()

def build_best_lead_series(
    gfs_paths,
    gfs_vars,
    subset_coords=None,
    use_area_mean=True,
    name="gfs",
):
    """
    Take several GFS launch files and, for every observation_time, return
    the value with the *shortest positive* lead time.
    """
    if isinstance(gfs_paths, (str, Path)):
        gfs_paths = [str(gfs_paths)]
    if isinstance(gfs_vars, str):
        gfs_vars = [gfs_vars] * len(gfs_paths)
    if len(gfs_paths) != len(gfs_vars):
        raise ValueError("gfs_paths and gfs_vars must be the same length.")

    val_frames, lead_frames = [], []

    for pth, var in zip(gfs_paths, gfs_vars):
        ds = xr.open_dataset(pth)
        da = ds[var]

        # optional subset / area mean
        if subset_coords is not None:
            da = da.sel(subset_coords, method="nearest")
        elif use_area_mean and {"lat", "lon"} <= set(da.dims):
            da = da.mean(dim=("lat", "lon"))

        # squeeze to 1-D
        for d in list(da.dims):
            if d != "observation_time":
                da = da.isel({d: 0})

        ser_val = da.to_series()

        # compute lead time
        launch = pd.to_datetime(ds["launch_time"].to_series())
        launch.index = ser_val.index
        lead = ser_val.index.to_series() - launch
        lead[lead <= pd.Timedelta(0)] = pd.NaT   # drop negative / zero leads

        val_frames.append(ser_val.to_frame(name=pth))
        lead_frames.append(lead.to_frame(name=pth))

    val_df  = pd.concat(val_frames, axis=1)
    lead_df = pd.concat(lead_frames, axis=1)

    idx_min = lead_df.idxmin(axis=1)                     # column labels
    col_pos = val_df.columns.get_indexer(idx_min)        # integer positions
    row_pos = np.arange(len(val_df))

    best_vals = val_df.to_numpy()[row_pos, col_pos]

    return (
        pd.Series(best_vals, index=val_df.index, name=name)
          .dropna()
          .sort_index()
    )

def load_csv(path):
    """Load the optional NN output CSV."""
    if not path:
        return None
    df = (
        pd.read_csv(path, parse_dates=[CSV_IDX_COL])
          [[CSV_IDX_COL, CSV_TRUE_COL, CSV_PRED_COL]]
          .dropna()
          .rename(columns={CSV_TRUE_COL: "siata_csv", CSV_PRED_COL: "nn"})
          .set_index(CSV_IDX_COL)
    )
    return df

df_csv = load_csv(CSV_PATH)

# ---------------- 4)  Evaluation & plots ----------------------------------
def evaluate_pair(gfs_paths, gfs_vars, siata_path, siata_var, label):
    # ---------- load GFS --------------------------------------------------
    if isinstance(gfs_paths, (list, tuple)):
        gfs_ser = build_best_lead_series(
            gfs_paths, gfs_vars,
            subset_coords=SUBSET_COORDS,
            use_area_mean=USE_AREA_MEAN,
            name="gfs",
        )
    else:
        gfs_ser = to_timeseries(
            xr.open_dataset(gfs_paths)[gfs_vars],
            SUBSET_COORDS, USE_AREA_MEAN
        )

    # ---------- load SIATA ------------------------------------------------
    siata_ser = to_timeseries(
        xr.open_dataset(siata_path)[siata_var],
        SUBSET_COORDS, USE_AREA_MEAN
    )

    # ---------- Join & test split ----------------------------------------
    joined = (
        pd.concat([gfs_ser, siata_ser.rename("siata")], axis=1, join="inner")
          .dropna(subset=["siata"])
    )
    joined = joined[joined.index.isin(test_idx)].dropna()

    # optional merge of NN CSV
    if df_csv is not None:
        joined = joined.merge(df_csv, left_index=True, right_index=True, how="left")

    # residuals
    joined["res_before"] = joined["siata"] - joined["gfs"]
    if df_csv is not None:
        joined["res_after"] = joined["siata_csv"] - joined["nn"]

    # ---------- Scatter ---------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(joined["siata"], joined["gfs"],
                s=10, alpha=.5, label="Before NN", color="blue")
    if df_csv is not None:
        plt.scatter(joined["siata_csv"], joined["nn"],
                    s=10, alpha=.5, label="After NN", color="red")
    lims = [joined["siata"].min(), joined["siata"].max()]
    plt.plot(lims, lims, "k--", lw=1)
    plt.title(f"Scatter – {label}")
    plt.xlabel("SIATA [W m⁻²]"); plt.ylabel("GFS / NN [W m⁻²]")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

    # ---------- Histogram (incl. zeros) ----------------------------------
    if df_csv is not None:
        bins = np.linspace(joined[["res_before", "res_after"]].min().min(),
                           joined[["res_before", "res_after"]].max().max(), 50)
    else:
        bins = np.linspace(joined["res_before"].min(),
                           joined["res_before"].max(), 50)

    plt.figure(figsize=(7, 6))
    plt.hist(joined["res_before"], bins=bins, color="blue", alpha=.6,
             label="Before NN")
    if df_csv is not None:
        n, _, patches = plt.hist(joined["res_after"], bins=bins,
                                 color="orange", alpha=.35,
                                 edgecolor="orange", label="After NN")
        for p in patches:
            p.set_hatch("//")
    plt.title(f"Residual histogram (incl. 0) – {label}")
    plt.xlabel("Residuals [W m⁻²]"); plt.ylabel("Count")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

    # ---------- Histogram (without zeros) --------------------------------
    res_before_nz = joined["res_before"][joined["res_before"] != 0]
    if df_csv is not None:
        res_after_nz = joined["res_after"][joined["res_after"] != 0]
        all_nz = pd.concat([res_before_nz, res_after_nz])
    else:
        all_nz = res_before_nz

    bins_nz = np.linspace(all_nz.min(), all_nz.max(), 50)

    plt.figure(figsize=(7, 6))
    plt.hist(res_before_nz, bins=bins_nz, color="blue", alpha=.6,
             label="Before NN")
    if df_csv is not None:
        n, _, patches = plt.hist(res_after_nz, bins=bins_nz,
                                 color="orange", alpha=.35,
                                 edgecolor="orange", label="After NN")
        for p in patches:
            p.set_hatch("//")
    plt.title(f"Residual histogram (without 0) – {label}")
    plt.xlabel("Residuals [W m⁻²]"); plt.ylabel("Count")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

    # ---------- Metrics (Before-NN) --------------------------------------
    y_true = joined["siata"].values
    y_pred = joined["gfs"].values
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    corr, pval = pearsonr(y_true, y_pred)

    print("------------------------------------------------------------")
    print(label)
    print(f"Test hours        : {len(joined):,}")
    print(f"MSE   [W m⁻²]     : {mse:,.3f}")
    print(f"RMSE  [W m⁻²]     : {rmse:,.3f}")
    print(f"MAE   [W m⁻²]     : {mae:,.3f}")
    print(f"R²                : {r2:,.4f}")
    print(f"Pearson-r         : {corr:,.4f}  (p = {pval:.1e})")
    print(f"Var(res_before)   : {np.var(joined['res_before']):,.3f}")
    if df_csv is not None:
        print(f"Var(res_after)    : {np.var(joined['res_after']):,.3f}")

# ---------------- 5)  Run --------------------------------------------------
for gfs_p, gfs_v, siata_p, siata_v, lbl in PAIRS:
    try:
        evaluate_pair(gfs_p, gfs_v, siata_p, siata_v, lbl)
    except Exception as e:
        print(f"{lbl} – ERROR: {e}")
