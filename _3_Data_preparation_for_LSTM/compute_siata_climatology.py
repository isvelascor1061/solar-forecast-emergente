#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_siata_climatology.py
============================
Compute climatological statistics from SIATA clear-sky index (CSI) data.

Daytime filter: only hours with clear_sky_ghi > 0 are included.

Statistics saved to siata_climatology.nc:
  Per (month, hour): mean_csi, std_csi, prob_cloudy, prob_clear,
                     prob_partial, bimodality
  Per (hour only):   std_by_hour, transition_risk

Usage:
    set PYTHONPATH=C:\\Users\\isabe\\Projects\\codigors\\carpetasdetrabajo
    python _3_Data_preparation_for_LSTM/compute_siata_climatology.py
"""

import numpy as np
import pandas as pd
import xarray as xr

from config import (
    SIATA_CSI_FILE, SIATA_CSI_VAR,
    CSI_GHI_FILE, CSI_VAR_NAME,
    PREP_DATA_DIR,
)

# Output path for the climatology NetCDF
CLIM_OUTPUT_FILE = f"{PREP_DATA_DIR}/siata_climatology.nc"


def main() -> None:

    # ---- 1) Load SIATA CSI -----------------------------------------------
    print("Loading SIATA CSI ...")
    ds_siata = xr.open_dataset(SIATA_CSI_FILE, engine="h5netcdf")
    da_csi = ds_siata[SIATA_CSI_VAR].squeeze()

    # ---- 2) Load clear-sky GHI to define the daytime mask ---------------
    print("Loading clear-sky GHI (daytime filter) ...")
    ds_ghi = xr.open_dataset(CSI_GHI_FILE, engine="h5netcdf")
    da_ghi = ds_ghi[CSI_VAR_NAME].squeeze()

    # Align both arrays to their common time axis
    da_csi_aligned, da_ghi_aligned = xr.align(da_csi, da_ghi, join="inner")

    # Daytime only: mask CSI values where clear_sky_ghi == 0 (night)
    da_csi_day = da_csi_aligned.where(da_ghi_aligned > 0)

    # ---- 3) Build a flat DataFrame with month/hour columns ---------------
    series_csi = da_csi_day.to_series().rename("csi")
    df = series_csi.to_frame()
    df["month"] = df.index.month   # 1–12
    df["hour"]  = df.index.hour    # 0–23

    df_valid = df.dropna(subset=["csi"])
    print(f"Total daytime observations (clear_sky_ghi > 0): {len(df_valid):,}")

    # ---- 4) Statistics per (month, hour) ---------------------------------
    grouped = df_valid.groupby(["month", "hour"])["csi"]

    mean_csi     = grouped.mean()
    std_csi      = grouped.std()
    prob_cloudy  = grouped.apply(lambda x: (x < 0.4).mean())   # CSI < 0.4
    prob_clear   = grouped.apply(lambda x: (x > 0.8).mean())   # CSI > 0.8
    prob_partial = grouped.apply(lambda x: ((x >= 0.4) & (x <= 0.8)).mean())  # in between
    # Bimodality: high when both overcast AND clear sky are frequent (morning transition)
    bimodality   = prob_cloudy * prob_clear

    # ---- 5) Statistics per hour only ------------------------------------
    # std_by_hour: CSI variability aggregated across all months
    std_by_hour = df_valid.groupby("hour")["csi"].std()

    # transition_risk: fraction of consecutive hours where |ΔCSI| > 0.4
    # Only pairs that are truly 1 h apart are counted (avoids day-boundary artifacts)
    time_gap       = series_csi.index.to_series().diff()
    valid_consec   = time_gap == pd.Timedelta("1h")
    diff_abs       = series_csi.diff().abs()
    is_transition  = (
        (diff_abs > 0.4)
        & valid_consec
        & series_csi.notna()
        & series_csi.shift(1).notna()
    )
    df_trans = pd.DataFrame(
        {"is_transition": is_transition, "hour": series_csi.index.hour}
    )
    df_trans = df_trans[valid_consec]
    transition_risk = df_trans.groupby("hour")["is_transition"].mean()

    # ---- 6) Pack into full (month × hour) grids -------------------------
    months_arr = np.arange(1, 13)   # 1–12
    hours_arr  = np.arange(0, 24)   # 0–23

    def to_2d(stat: pd.Series) -> np.ndarray:
        """Unstack a (month, hour)-indexed Series into a (12, 24) array."""
        arr = np.full((12, 24), np.nan, dtype=np.float32)
        for (m, h), val in stat.items():
            arr[m - 1, h] = float(val)
        return arr

    def to_1d(stat: pd.Series) -> np.ndarray:
        """Align a per-hour Series to a fixed 24-element array."""
        arr = np.full(24, np.nan, dtype=np.float32)
        for h, val in stat.items():
            arr[int(h)] = float(val)
        return arr

    # ---- 7) Build and save xarray Dataset --------------------------------
    ds_out = xr.Dataset(
        data_vars={
            "mean_csi":        (["month", "hour"], to_2d(mean_csi)),
            "std_csi":         (["month", "hour"], to_2d(std_csi)),
            "prob_cloudy":     (["month", "hour"], to_2d(prob_cloudy)),
            "prob_clear":      (["month", "hour"], to_2d(prob_clear)),
            "prob_partial":    (["month", "hour"], to_2d(prob_partial)),
            "bimodality":      (["month", "hour"], to_2d(bimodality)),
            "std_by_hour":     (["hour"],          to_1d(std_by_hour)),
            "transition_risk": (["hour"],          to_1d(transition_risk)),
        },
        coords={
            "month": months_arr,
            "hour":  hours_arr,
        },
        attrs={
            "description": "SIATA CSI climatological statistics for Medellin",
            "source_csi":  SIATA_CSI_FILE,
            "source_ghi":  CSI_GHI_FILE,
            "daytime_filter": "clear_sky_ghi > 0",
            "created_by":  "compute_siata_climatology.py",
        },
    )

    ds_out.to_netcdf(CLIM_OUTPUT_FILE, engine="h5netcdf")
    print(f"\nClimatology saved to: {CLIM_OUTPUT_FILE}")

    # ---- 8) Print top-5 most bimodal (month, hour) combinations ---------
    bimod_df = bimodality.reset_index()
    bimod_df.columns = ["month", "hour", "bimodality"]
    top5 = bimod_df.nlargest(5, "bimodality").reset_index(drop=True)

    print("\nTop 5 most bimodal (month, hour) combinations:")
    print(f"{'Rank':>4} {'Month':>6} {'Hour':>5} {'Bimodality':>12} "
          f"{'prob_cloudy':>12} {'prob_clear':>11}")
    print("-" * 55)
    for rank, row in top5.iterrows():
        m, h = int(row["month"]), int(row["hour"])
        bi   = row["bimodality"]
        pc   = prob_cloudy.get((m, h), np.nan)
        pcl  = prob_clear.get((m, h), np.nan)
        print(f"{rank + 1:>4} {m:>6} {h:>5} {bi:>12.4f} {pc:>12.4f} {pcl:>11.4f}")


if __name__ == "__main__":
    main()
