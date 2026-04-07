#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual comparison of

    • SIATA ground-measurement (reference)
    • Raw GFS dswrf1 forecast (best-lead, i.e. shortest **positive** lead-time
      chosen out of 4 launch times)
    • LSTM-adjusted forecast (y_pred taken from a CSV)

for an arbitrarily selected period.

The script

1. reads the full-set CSV produced by *run_best_bilstm_save_csv.py*,
2. loads the SIATA reference series,
3. builds a new “best lead” GFS series by scanning all four launch-time
   NetCDF files and, for each observation_time, selecting the value with
   the shortest **positive** lead-time (using the **step** coordinate),
4. merges the three series, restricts them to START–END and
5. plots them together.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1)  File paths & user parameters
# ----------------------------------------------------------------------
CSV_FILE = "_4_LSTM_modules/Evaluation_after_LSTM/Sym12_predictions_full_descaled.csv"

# Four launch times (0100 / 0700 / 1300 / 1900) -------------------------
GFS_PATHS = [
    "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped/dswrf1_CSI_0100.nc",
    "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped/dswrf1_CSI_0700.nc",
    "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped/dswrf1_CSI_1300.nc",
    "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped/dswrf1_CSI_1900.nc",
]
GFS_VARS  = ["dswrf1_0100", "dswrf1_0700", "dswrf1_1300", "dswrf1_1900"]

NC_SIATA  = (
    "_3_Data_preparation_for_LSTM/Preparation_data/_03_Siata_GHI/"
    "Netcdf_Siata_GHI/GHI_CSI_clipped.nc"
)
VAR_SIATA = "GHI_clean"

START = "2025-05-20 00:00"
END   = "2025-05-20 23:59"

# optional: extract a single grid-point or take the spatial mean ----------
SUBSET_COORDS = None          # e.g. {"lat": 6.25, "lon": -75.5}
USE_AREA_MEAN = True
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 2)  Helper functions
# ----------------------------------------------------------------------
def load_series(nc_path: str, var: str) -> pd.Series:
    """
    Open a NetCDF file, optionally extract a point / area mean and
    return a tidy 1-D pandas Series indexed by `observation_time`.
    """
    da = xr.open_dataset(nc_path)[var]

    # point selection or area mean
    if SUBSET_COORDS:
        da = da.sel(SUBSET_COORDS, method="nearest")
    elif USE_AREA_MEAN and {"lat", "lon"} <= set(da.dims):
        da = da.mean(dim=("lat", "lon"))

    # keep only the time dimension
    for d in da.dims:
        if d != "observation_time":
            da = da.isel({d: 0})

    return (
        da.to_series()
          .tz_localize(None)     # make naive
          .sort_index()
    )


# ----------------------------------------------------------------------
# 2a)  *Robust* build_best_lead_series  (handles explicit **step** coord)
# ----------------------------------------------------------------------
def build_best_lead_series(gfs_paths, gfs_vars):
    """
    Combine several GFS launch files and, for every observation_time,
    pick the value with the shortest strictly positive lead-time
    (derived from the **step** coordinate).

    Returns
    -------
    pandas.Series
        Index : observation_time
        Name  : "gfs_raw"
        Data  : raw GFS dswrf1 radiation [W m⁻²]
    """
    if isinstance(gfs_vars, str):
        gfs_vars = [gfs_vars] * len(gfs_paths)

    val_frames  = []   # actual values
    lead_frames = []   # corresponding lead-times (Timedelta)

    for path, var in zip(gfs_paths, gfs_vars):
        ds = xr.open_dataset(path)

        # ----- values ----------------------------------------------------
        da = ds[var]

        if SUBSET_COORDS:
            da = da.sel(SUBSET_COORDS, method="nearest")
        elif USE_AREA_MEAN and {"lat", "lon"} <= set(da.dims):
            da = da.mean(dim=("lat", "lon"))

        for d in da.dims:
            if d != "observation_time":
                da = da.isel({d: 0})

        ser_val = da.to_series().rename(path).astype(float)

        # ----- lead-time via step ---------------------------------------
        step_da = ds["step"]
        for d in step_da.dims:
            if d != "observation_time":
                step_da = step_da.isel({d: 0})

        ser_lead = step_da.to_series().rename(path)
        ser_lead[ser_lead < pd.Timedelta(0)] = pd.NaT   # drop negative

        val_frames.append(ser_val)
        lead_frames.append(ser_lead)

    # ----- concatenate & select best lead --------------------------------
    val_df  = pd.concat(val_frames,  axis=1)
    lead_df = pd.concat(lead_frames, axis=1)

    min_lead  = lead_df.min(axis=1)
    best_mask = lead_df.eq(min_lead, axis=0)            # True for “winner”

    best_vals = val_df.where(best_mask).bfill(axis=1).iloc[:, 0]
    best_vals.name = "gfs_raw"

    return best_vals.dropna().sort_index()


# ----------------------------------------------------------------------
# 3)  Main routine
# ----------------------------------------------------------------------
def main():
    # --- LSTM CSV -------------------------------------------------------
    df = (
        pd.read_csv(CSV_FILE, parse_dates=["time"])
          .assign(time=lambda d: d["time"].dt.tz_localize(None))
          .set_index("time")
    )

    # --- SIATA ----------------------------------------------------------
    siata = load_series(NC_SIATA, VAR_SIATA).rename("siata")

    # --- Raw GFS (best positive lead-time) ------------------------------
    gfs_best = build_best_lead_series(GFS_PATHS, GFS_VARS)

    # --- Merge & time-slice --------------------------------------------
    merged = df.join([siata, gfs_best], how="outer").loc[START:END]

    if merged[["y_true", "y_pred", "siata", "gfs_raw"]].dropna(how="all").empty:
        raise ValueError("No overlapping data in the selected period.")

    # --- Plot -----------------------------------------------------------
    plt.figure(figsize=(11, 5))
    plt.plot(merged.index, merged["siata"],   label="SIATA GHI",            lw=1.0)
    plt.plot(merged.index, merged["gfs_raw"], label="GFS dswrf1 (best LT)", lw=1.0, ls="--")
    plt.plot(merged.index, merged["y_pred"],  label="LSTM adjusted",        lw=1.4)

    plt.title(f"GHI – Raw GFS vs. LSTM Adjusted  |  {START[:10]}")
    plt.ylabel("W m$^{-2}$");  plt.xlabel("Time")
    plt.grid(True, ls="--", alpha=.5)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    out_png = f"plot_{START[:10]}_{END[:10]}.png".replace(" ", "_")
    Path(out_png).parent.mkdir(exist_ok=True, parents=True)
    # plt.savefig(out_png, dpi=150)   # activate if you want to save
    print(f"✅  Plot completed – {out_png}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
