#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-04-21

Script to plot one or up to three time series along the 'observation_time' dimension
from specified NetCDF files and variables, limited to a given date range.
"""
import os
import xarray as xr
import matplotlib.pyplot as plt


def plot_multiple_timeseries(
    file_paths,
    var_names,
    start_date=None,
    end_date=None,
    output_fig=None,
    title=None,
    ylabel=None,
    figsize=(12, 6)
):
    """
    Plot up to three time series from NetCDF files over a specified date range.

    Parameters
    ----------
    file_paths : list of str or None
        List of up to 3 NetCDF file paths. Use None for unused slots.
    var_names : list of str or None
        Corresponding variable names to plot for each file. None to skip.
    start_date : str, optional
        Start date for slicing (inclusive) in 'YYYY-MM-DD' or full ISO format.
    end_date : str, optional
        End date for slicing (inclusive) in 'YYYY-MM-DD' or full ISO format.
    output_fig : str, optional
        Path to save the figure. If None, shows interactively.
    title : str, optional
        Plot title. Defaults to blank.
    ylabel : str, optional
        Y-axis label. Defaults to variable name(s).
    figsize : tuple, optional
        Figure size in inches.
    """
    plt.figure(figsize=figsize)
    legend_labels = []

    for fp, var in zip(file_paths, var_names):
        if fp is None or var is None:
            continue
        ds = xr.open_dataset(fp)
        da = ds[var].squeeze()
        # slice time
        da = da.sel(observation_time=slice(start_date, end_date))
        # collapse spatial dims
        if da.ndim > 1:
            dims = [d for d in da.dims if d != 'observation_time']
            da = da.mean(dim=dims)
        # plot
        label = var
        plt.plot(da['observation_time'], da.values)
        legend_labels.append(label)

    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel(ylabel or ', '.join([v for v in var_names if v]))
    plt.title(title or '')
    if legend_labels:
        plt.legend(legend_labels)

    if output_fig:
        out_dir = os.path.dirname(output_fig)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_fig, dpi=300)
        print(f"Plot saved to: {output_fig}")
    else:
        plt.show()


if __name__ == '__main__':
    # ===== User configuration =====
    # Specify up to three file paths and corresponding variable names
    FILE_PATHS = [
        "_3_Data_preparation_for_LSTM/Preparation_data/_03_Siata_GHI/Netcdf_Siata_GHI/SIATA_GHI_all.nc",
        "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Unclipped_merged_dswrf1/dswrf1_0100.nc",
        ]
    VAR_NAMES = [
        'GHI',"dswrf1_0100"
       
    ]

    # Optional date range slice (set to None to plot full range)
    START_DATE ="20210410"
    END_DATE = "20210410"

    # Optional: output figure path (set to None to display)
    OUTPUT_FIG = None

    # Optional: custom title and y-axis label
    TITLE = "GFS and InSitu Radiation after Postprocessing 10-04-2021"
    YLABEL = 'Radiation [W/m^2]'

    plot_multiple_timeseries(
        FILE_PATHS,
        VAR_NAMES,
        start_date=START_DATE,
        end_date=END_DATE,
        output_fig=OUTPUT_FIG,
        title=TITLE,
        ylabel=YLABEL
    )
