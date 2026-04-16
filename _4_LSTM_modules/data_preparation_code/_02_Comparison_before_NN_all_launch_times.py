#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline check for arbitrary NetCDF-pairs on the **identical** test split.
Calculates MSE, MAE, RMSE, R², Pearson-r and the variance of the residuals
and prints all numbers in German number format (1.234.567,89).

The script also supports a *set* of GFS launch-time files:  
for every observation_time it automatically chooses the value
with the **smallest positive lead-time** ("best lead" approach).
"""

from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from config import (
    VAR_GFS_DSWRF_0100, VAR_GFS_DSWRF_0700, VAR_GFS_DSWRF_1300, VAR_GFS_DSWRF_1900,
    VAR_SIATA_GHI,
    DSWRF1_CSI_0100_FILE, DSWRF1_CSI_0700_FILE, DSWRF1_CSI_1300_FILE, DSWRF1_CSI_1900_FILE,
    SIATA_GHI_FILE,
)

# ────────────────────────────────────────────────────────────────
# 0)  German number formatting  (thousands = ".", decimals = ", ")
# ────────────────────────────────────────────────────────────────
def fmt_de(value, prec: int = 5) -> str:
    """Return *value* as a German-style formatted string."""
    s = f"{value:,.{prec}f}"               # 1,234,567.89000
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# ────────────────────────────────────────────────────────────────
# 1)  Files to compare  ― extend as you like
# ────────────────────────────────────────────────────────────────
PAIRS = [
    (
        [  # ← four launch-time files (rutas desde config)
            DSWRF1_CSI_0100_FILE,
            DSWRF1_CSI_0700_FILE,
            DSWRF1_CSI_1300_FILE,
            DSWRF1_CSI_1900_FILE,
        ],
        # nombres de variable dentro de cada .nc (gfs_dswrf_<LT>)
        [VAR_GFS_DSWRF_0100, VAR_GFS_DSWRF_0700, VAR_GFS_DSWRF_1300, VAR_GFS_DSWRF_1900],
        SIATA_GHI_FILE,      # ruta al .nc de SIATA (desde config)
        VAR_SIATA_GHI,       # siata_ghi → "GHI_clean"
        "Best-lead GFS-DSWRF1  vs  SIATA-GHI",
    )
]

TEST_IDX_FILE = Path("_4_LSTM_modules/test_indices/test_indices_4launch_multfeat_sym24.npy")
SUBSET_COORDS = None          # e.g. {"lat": 6.25, "lon": -75.5}
USE_AREA_MEAN = True          # average over lat/lon grid if present

# ────────────────────────────────────────────────────────────────
# 2)  Load test timestamps
# ────────────────────────────────────────────────────────────────
test_idx = pd.to_datetime(np.load(TEST_IDX_FILE))
print(f"{len(test_idx):,} timestamps from {TEST_IDX_FILE.name} will be used.\n")


# ────────────────────────────────────────────────────────────────
# 3)  Generic helper → DataArray ➜ 1-D time series
# ────────────────────────────────────────────────────────────────
def to_timeseries(da: xr.DataArray,
                  subset: dict | None = None,
                  area_mean: bool = False) -> pd.Series:
    """Reduce *da* to a 1-D Series indexed by its time coordinate."""
    time_dim = next((d for d in da.dims if np.issubdtype(da[d].dtype, np.datetime64)), None)
    if time_dim is None:
        raise ValueError("No time axis found.")

    if subset:
        da = da.sel(subset, method="nearest")
    elif area_mean and {"lat", "lon"} <= set(da.dims):
        da = da.mean(dim=("lat", "lon"))

    # drop every remaining non-time dimension
    for d in list(da.dims):
        if d != time_dim:
            da = da.isel({d: 0})

    return da.to_series()


# ────────────────────────────────────────────────────────────────
# 4)  Build "best-lead" GFS series from several launch files
# ────────────────────────────────────────────────────────────────
def build_best_lead_series(
    gfs_paths,
    gfs_vars,
    subset_coords=None,
    use_area_mean=True,
    name="gfs",
) -> pd.Series:
    """
    For every observation_time pick the value with the **smallest
    positive** lead-time across all provided launch-time files.
    """
    # normalise input
    if isinstance(gfs_paths, (str, Path)):
        gfs_paths = [str(gfs_paths)]
    if isinstance(gfs_vars, str):
        gfs_vars = [gfs_vars] * len(gfs_paths)
    if len(gfs_paths) != len(gfs_vars):
        raise ValueError("gfs_paths and gfs_vars must be the same length.")

    value_frames, lead_frames = [], []

    for path, var in zip(gfs_paths, gfs_vars):
        ds = xr.open_dataset(path)
        da = ds[var]

        if subset_coords is not None:
            da = da.sel(subset_coords, method="nearest")
        elif use_area_mean and {"lat", "lon"} <= set(da.dims):
            da = da.mean(dim=("lat", "lon"))

        for d in da.dims:
            if d != "observation_time":
                da = da.isel({d: 0})

        ser_val = da.to_series()

        # compute lead-time
        lt = pd.to_datetime(ds["launch_time"].to_series())
        lt.index = ser_val.index
        lead = ser_val.index.to_series() - lt
        lead[lead <= pd.Timedelta(0)] = pd.NaT      # discard negative/zero leads

        value_frames.append(ser_val.to_frame(name=path))
        lead_frames.append(lead.to_frame(name=path))

    # choose the column with the minimum positive lead
    val_df  = pd.concat(value_frames, axis=1)
    lead_df = pd.concat(lead_frames,  axis=1)

    idx_min  = lead_df.idxmin(axis=1)                       # column label
    col_pos  = val_df.columns.get_indexer(idx_min)          # → integer positions
    row_pos  = np.arange(len(val_df))
    best_val = val_df.to_numpy()[row_pos, col_pos]

    return (
        pd.Series(best_val, index=val_df.index, name=name)
          .dropna()
          .sort_index()
    )


# ────────────────────────────────────────────────────────────────
# 5)  Evaluation routine
# ────────────────────────────────────────────────────────────────
def evaluate_pair(gfs_paths, gfs_vars, siata_path, siata_var):
    """Load data, restrict to test split, compute metrics & plots."""
    # --- GFS ---------------------------------------------------------
    if isinstance(gfs_paths, (list, tuple)):
        gfs_ser = build_best_lead_series(
            gfs_paths, gfs_vars,
            subset_coords=SUBSET_COORDS,
            use_area_mean=USE_AREA_MEAN,
        )
    else:  # single file
        gfs_ser = build_best_lead_series(
            gfs_paths, gfs_vars,
            subset_coords=SUBSET_COORDS,
            use_area_mean=USE_AREA_MEAN,
        )

    # --- SIATA -------------------------------------------------------
    siata_da  = xr.open_dataset(siata_path)[siata_var]
    siata_ser = to_timeseries(siata_da, SUBSET_COORDS, USE_AREA_MEAN)

    # --- join & restrict to test split ------------------------------
    joined_all  = pd.concat([gfs_ser.rename("gfs"),
                             siata_ser.rename("siata")], axis=1, join="inner")
    joined_test = joined_all.loc[test_idx].dropna()

    # report missing timestamps (optional)
    missing = test_idx.difference(joined_test.index)
    if not missing.empty:
        print(f"⚠️  {len(missing)} timestamps from test split are missing "
              f"(no positive lead or NaNs) and are skipped.\n")

    print(f"Number of test hours after filtering: {len(joined_test):,}\n")

    # --- metrics ----------------------------------------------------
    y_true = joined_test["siata"].values
    y_pred = joined_test["gfs"].values
    residuals = y_true - y_pred

    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    corr, pval = pearsonr(y_true, y_pred)

    # baseline skill vs. persistence
    siata = joined_test["siata"]
    baseline_df = pd.concat([siata, siata.shift(1)], axis=1).dropna()
    baseline_df.columns = ["y_true", "y_base"]
    mse_base   = mean_squared_error(baseline_df["y_true"], baseline_df["y_base"])
    gfs_aligned = joined_test["gfs"].loc[baseline_df.index]
    mse_gfs     = mean_squared_error(baseline_df["y_true"], gfs_aligned)
    skill_gfs   = np.nan if mse_base == 0 else 1 - mse_gfs / mse_base

    # --- scatter ----------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=8, alpha=.7)
    lims = [y_true.min(), y_true.max()]
    plt.plot(lims, lims, "r--")
    plt.title(f"Predicted vs Actual  |  {len(joined_test)} test hours")
    plt.xlabel("SIATA [W m⁻²]"); plt.ylabel("GFS [W m⁻²]")
    plt.grid(True); plt.tight_layout(); plt.show()

    # --- residual histogram ----------------------------------------
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, edgecolor="k", alpha=.7)
    plt.title("Residual histogram"); plt.xlabel("Residual [W m⁻²]")
    plt.grid(True); plt.tight_layout(); plt.show()

    return len(joined_test), mse, rmse, mae, r2, corr, pval, residuals, mse_base, skill_gfs


# ────────────────────────────────────────────────────────────────
# 6)  Run all comparisons
# ────────────────────────────────────────────────────────────────
for gfs_p, gfs_v, siata_p, siata_v, lbl in PAIRS:
    try:
        n, mse, rmse, mae, r2, corr, pval, residuals, mse_base, skill = (
            evaluate_pair(gfs_p, gfs_v, siata_p, siata_v)
        )
        print("------------------------------------------------------------")
        print(lbl)
        print(f"Test hours                : {fmt_de(n, 0)}")
        print(f"MSE   [W m⁻²]             : {fmt_de(mse)}")
        print(f"RMSE  [W m⁻²]             : {fmt_de(rmse)}")
        print(f"MAE   [W m⁻²]             : {fmt_de(mae)}")
        print(f"R²                        : {fmt_de(r2)}")
        print(f"Pearson-r                 : {fmt_de(corr)} (p = {pval:.1e})")
        print(f"Var(residuals) [W m⁻²]    : {fmt_de(np.var(residuals))}")
        print(f"Baseline-MSE    [W m⁻²]    : {fmt_de(mse_base)}")
        print(f"Skill score (GFS vs base) : {fmt_de(skill, 4)}\n")

    except Exception as exc:
        print("------------------------------------------------------------")
        print(f"{lbl}  ➜  ERROR: {exc}")

print("------------------------------------------------------------")
