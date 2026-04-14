#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:38:27 2025

@author: leonardmerl
"""

import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------
def exceedance_report(gfs_path: str,
                      gfs_var: str,
                      clear_path: str,
                      clear_var: str,
                      clear_threshold: float = 1.0,
                      diff_threshold: float  = 1.0):
    """
    Returns a list of all exceedances (GFS - ClearSky) > diff_threshold
    during daylight hours (Clear >= clear_threshold), sorted descending by Diff.
    """
    # 1) Load data and align to a common time axis
    da_gfs   = xr.open_dataset(gfs_path)[gfs_var].squeeze()
    da_clear = xr.open_dataset(clear_path)[clear_var].squeeze()
    da_gfs, da_clear = xr.align(da_gfs, da_clear, join="inner")

    # --- Average if 2-D (Lat/Lon) -----------------------------------
    def to_series(da):
        if da.ndim > 1:
            other = [d for d in da.dims if d != "observation_time"]
            da = da.mean(dim=other)
        return da.to_series()

    s_gfs   = to_series(da_gfs)
    s_clear = to_series(da_clear)

    # 2) Combine and apply daylight filter
    df = pd.DataFrame({"GHI": s_gfs, "Clear": s_clear}).dropna()
    df_day = df[df["Clear"] >= clear_threshold]
    df_day["Diff"] = df_day["GHI"] - df_day["Clear"]

    # 3) Filter exceedances
    df_exceed = df_day[df_day["Diff"] > diff_threshold]\
                    .sort_values("Diff", ascending=False)

    # 4) Summary
    count = len(df_exceed)
    print(f"{count} exceedances > {diff_threshold} W/m² "
          f"(out of {len(df_day)} daylight timestamps)")
    print("\nLargest exceedances:")
    print(df_exceed.head(50).to_string(formatters={"Diff": "{:.1f}".format}))

    return df_exceed, count
# -------------------------------------------------------------------

# --- Example call ------------------------------------------------
if __name__ == "__main__":
    GFS  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Unclipped_merged_dswrf1/dswrf1_0100.nc"
    GVAR = "dswrf1_0100"
    CL   = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct.nc"
    CLVAR= "clear_sky_ghi"

    df_exceed, n_exc = exceedance_report(
        GFS, GVAR, CL, CLVAR,
        clear_threshold=0.0,
        diff_threshold=0.0
    )

    # df_exceed contains ALL exceedances, sorted descending
