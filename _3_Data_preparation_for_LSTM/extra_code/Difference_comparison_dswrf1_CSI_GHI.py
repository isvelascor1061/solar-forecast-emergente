#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-04-21

List and count all daytime exceedances of GFS dswrf1 over Clear‑Sky GHI
where the difference exceeds a threshold, sorted descending by magnitude.
Includes hourly exceedance statistics.
"""
import xarray as xr
import pandas as pd

def list_daytime_exceedances(
    gfs_path: str,
    gfs_var: str,
    clear_path: str,
    clear_var: str,
    clear_threshold: float = 1.0,
    diff_threshold: float = 1.0
):
    # Load and align datasets
    da_gfs = xr.open_dataset(gfs_path)[gfs_var].squeeze()
    da_clear = xr.open_dataset(clear_path)[clear_var].squeeze()
    da_gfs, da_clear = xr.align(da_gfs, da_clear, join="inner")

    # Convert to pandas Series
    def to_series(da):
        if da.ndim > 1:
            other_dims = [d for d in da.dims if d != "observation_time"]
            da = da.mean(dim=other_dims)
        return da.to_series()

    s_gfs = to_series(da_gfs)
    s_clear = to_series(da_clear)

    # Create combined DataFrame
    df = pd.DataFrame({"GHI": s_gfs, "Clear": s_clear}).dropna()

    # Filter daytime values
    df_day = df[df["Clear"] >= clear_threshold].copy()
    df_day["Diff"] = df_day["GHI"] - df_day["Clear"]

    # Find exceedances
    mask = df_day["Diff"] > diff_threshold
    df_exceed = df_day.loc[mask, ["GHI", "Clear", "Diff"]]
    df_exceed = df_exceed.sort_values("Diff", ascending=False)

    # Calculate statistics
    count_exceed = len(df_exceed)
    total_day = len(df_day)
    pct_exceed = count_exceed / total_day if total_day else float('nan')


    # Calculate hourly breakdown
    def get_hourly_stats(df_exceedances, df_daytime):
        hours = pd.DataFrame(index=range(24), columns=["Count", "Total"])

        # Exceedance counts per hour
        ex_hours = df_exceedances.index.hour.value_counts()
        hours["Count"] = ex_hours.reindex(hours.index, fill_value=0)

        # Total daytime points per hour
        day_hours = df_daytime.index.hour.value_counts()
        hours["Total"] = day_hours.reindex(hours.index, fill_value=0)

        hours["Percentage"] = hours["Count"] / hours["Total"]
        hours.index.name = "Hour"
        return hours.reset_index()

    df_hourly = get_hourly_stats(df_exceed, df_day)

    return df_exceed, count_exceed, pct_exceed, df_hourly, df_day

if __name__ == "__main__":
    # === User configuration ===
    GFS_PATH = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Unclipped_merged_dswrf1/dswrf1_0100.nc"
    GFS_VAR = "dswrf1_0100"
    CLEAR_PATH = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct2.nc"
    CLEAR_VAR = "clear_sky_ghi"

    # Thresholds
    CLEAR_THRESHOLD = 0.0
    DIFF_THRESHOLD  = 20   # lower bound
    HIGH_THRESHOLD  = 0.0  # upper bound

    # Run analysis
    results = list_daytime_exceedances(
        GFS_PATH, GFS_VAR,
        CLEAR_PATH, CLEAR_VAR,
        clear_threshold=CLEAR_THRESHOLD,
        diff_threshold=DIFF_THRESHOLD
    )
    df_exceed, count_exceed, pct_exceed, df_hourly, df_day = results

    # --- Share of exceedances above HIGH_THRESHOLD -----------------
    high_count = (df_exceed["Diff"] > HIGH_THRESHOLD).sum()
    pct_high   = high_count / len(df_exceed) if len(df_exceed) else float('nan')

    # Print results
    print(f"Total daytime points: {len(df_day)}")
    print(f"Exceedances (> {DIFF_THRESHOLD} W/m²): {count_exceed}")
    print(f"Exceedance percentage: {pct_exceed:.2%}\n")
    print(f"Exceedances (> {DIFF_THRESHOLD} W/m²): {count_exceed}")
    print(f"  of which > {HIGH_THRESHOLD} W/m²: {high_count}  "
          f"({pct_high:.2%} of exceedances)")
    print(f"Exceedance percentage (total): {pct_exceed:.2%}\n")


    print("Hourly statistics:")
    print(df_hourly.to_string(
        index=False,
        formatters={
            "Percentage": "{:.1%}".format,
            "Count": "{:d}".format,
            "Total": "{:d}".format
        })
    )
