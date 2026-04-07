#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 14:49:22 2025

@author: leonardmerl
"""

import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def plot_monthly_files(
    gfs_path: str,
    gfs_var: str,
    clear_path: str,
    clear_var: str,
    output_dir: str,
    figsize: tuple = (10, 5)
):
    # Load & squeeze
    da_gfs   = xr.open_dataset(gfs_path)[gfs_var].squeeze()
    da_clear = xr.open_dataset(clear_path)[clear_var].squeeze()

    # Align on time
    da_gfs, da_clear = xr.align(da_gfs, da_clear, join="inner")

    # Average spatial dims if any
    def to_series(da):
        if da.ndim > 1:
            dims = [d for d in da.dims if d != "observation_time"]
            da = da.mean(dim=dims)
        return da.to_series()

    s_gfs   = to_series(da_gfs)
    s_clear = to_series(da_clear)

    # Combine into single DataFrame
    df = pd.DataFrame({
        "GFS_dswrf1":   s_gfs,
        "ClearSky_GHI": s_clear
    }).dropna()

    # Add Year-Month period
    df["Period"] = df.index.to_period("M")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each year-month and plot
    for period, grp in df.groupby("Period"):
        fig, ax = plt.subplots(figsize=figsize)
        # plot both series
        ax.plot(grp.index, grp["ClearSky_GHI"], label="Clear‑Sky GHI", linewidth=2)
        ax.plot(grp.index, grp["GFS_dswrf1"],   label="GFS dswrf1",   linestyle="--")

        # highlight where GFS > ClearSky
        ax.fill_between(
            grp.index,
            grp["ClearSky_GHI"],
            grp["GFS_dswrf1"],
            where=(grp["GFS_dswrf1"] > grp["ClearSky_GHI"]),
            color="red",
            alpha=0.4,
            label="GFS > Clear‑Sky"
        )

        # formatting
        ax.set_title(f"GFS vs Clear‑Sky — {period.strftime('%Y-%m')}")
        ax.set_xlabel("Time")
        ax.set_ylabel("GHI (W/m²)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        # save
        fname = os.path.join(output_dir, f"{period.strftime('%Y_%m')}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"Saved plot for {period} → {fname}")


if __name__ == "__main__":
    # === User configuration ===
    GFS_PATH   = (
        "_3_Data_preparation_for_LSTM/Preparation_data/"
        "GFS_dswrf1/merged_all/GFS_merged_dswrf1_all_0100.nc"
    )
    GFS_VAR    = "dswrf1"

    CLEAR_PATH = (
        "_3_Data_preparation_for_LSTM/Preparation_data/"
        "Clear_sky_GHI/Ineichen_GHI/CSI_GHI_all.nc"
    )
    CLEAR_VAR  = "clear_sky_ghi"

    OUTPUT_DIR = (
        "_3_Data_preparation_for_LSTM/Preparation_data/Merged_Plots/Merged_monthly_plots/GFS_vs_CSI"
    )

    plot_monthly_files(
        gfs_path=GFS_PATH,
        gfs_var=GFS_VAR,
        clear_path=CLEAR_PATH,
        clear_var=CLEAR_VAR,
        output_dir=OUTPUT_DIR
    )
