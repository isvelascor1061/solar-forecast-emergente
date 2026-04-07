#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:21:33 2025

@author: leonardmerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare Siata data with GFS data and Clear-Sky data.

Assumptions:
    - The Siata file contains a variable "GHI" with dimension (observation_time) 
      and scalar coordinates 'lat' and 'lon'.
    - The GFS file contains a variable "dswrf1". If lat and lon are dimensions, the first grid point is used.
    - The Clear-Sky file contains a variable "clear_sky_ghi" with dimensions 
      (observation_time, surface, lat, lon); for plotting, the first surface, lat, lon values are selected.
"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

def plot_siata_vs_gfs(file_siata, file_gfs):
    """
    Loads a Siata NetCDF file and a GFS NetCDF file, restricts both to their overlapping time range,
    extracts a time series from a common grid point (if necessary) and plots the two time series.
    
    Parameters
    ----------
    file_siata : str
        Full path to the Siata NetCDF file (should contain variable "GHI").
    file_gfs : str
        Full path to the GFS NetCDF file (should contain variable "dswrf1").
    """
    # Load datasets
    ds_siata = xr.open_dataset(file_siata)
    ds_gfs   = xr.open_dataset(file_gfs)
    
    # Determine common observation time range
    common_start = max(pd.to_datetime(ds_siata.observation_time.values[0]),
                       pd.to_datetime(ds_gfs.observation_time.values[0]))
    common_end   = min(pd.to_datetime(ds_siata.observation_time.values[-1]),
                       pd.to_datetime(ds_gfs.observation_time.values[-1]))
    
    ds_siata_common = ds_siata.sel(observation_time=slice(common_start, common_end))
    ds_gfs_common   = ds_gfs.sel(observation_time=slice(common_start, common_end))
    
    # Extract time series
    ts_siata = ds_siata_common["GHI"]
    
    if "lat" in ds_gfs_common["dswrf1"].dims or "lon" in ds_gfs_common["dswrf1"].dims:
        ts_gfs = ds_gfs_common["dswrf1"].isel(lat=0, lon=0)
    else:
        ts_gfs = ds_gfs_common["dswrf1"]
    
    # Plot the two time series: Siata in orange (foreground) and GFS in blue.
    plt.figure(figsize=(12, 6))
    plt.plot(ts_siata.observation_time.values, ts_siata.values, marker='o', linestyle='-', color='orange', label="Siata GHI", zorder=2)
    plt.plot(ts_gfs.observation_time.values, ts_gfs.values, marker='x', linestyle='-', color='blue', label="GFS dswrf1", zorder=1)
    plt.xlabel("Observation Time")
    plt.ylabel("GHI (W/m²)")
    plt.title("Comparison: Siata vs. GFS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    ds_siata.close()
    ds_gfs.close()

def plot_siata_vs_clear_sky(file_siata, file_clear_sky):
    """
    Loads a Siata NetCDF file and a Clear-Sky NetCDF file, restricts both to their overlapping time range,
    extracts a time series from a common grid point (for clear-sky, the first surface, lat, and lon),
    and plots the two time series such that the Clear-Sky time series is drawn in the background.
    
    Parameters
    ----------
    file_siata : str
        Full path to the Siata NetCDF file (should contain variable "GHI").
    file_clear_sky : str
        Full path to the Clear-Sky NetCDF file (should contain variable "clear_sky_ghi").
    """
    # Load datasets
    ds_siata = xr.open_dataset(file_siata)
    ds_clear_sky = xr.open_dataset(file_clear_sky)
    
    # Determine common observation time range
    common_start = max(pd.to_datetime(ds_siata.observation_time.values[0]),
                       pd.to_datetime(ds_clear_sky.observation_time.values[0]))
    common_end   = min(pd.to_datetime(ds_siata.observation_time.values[-1]),
                       pd.to_datetime(ds_clear_sky.observation_time.values[-1]))
    
    ds_siata_common = ds_siata.sel(observation_time=slice(common_start, common_end))
    ds_clear_common = ds_clear_sky.sel(observation_time=slice(common_start, common_end))
    
    # Extract the time series:
    ts_siata = ds_siata_common["GHI"]
    ts_clear = ds_clear_common["clear_sky_ghi"].isel(surface=0, lat=0, lon=0)
    
    # Plot: Clear-Sky GHI in the background (blue, transparent, lower zorder), then Siata GHI on top.
    plt.figure(figsize=(12, 6))
    plt.plot(ts_clear.observation_time.values, ts_clear.values,
             marker='None', linestyle='-', color='blue', alpha=0.3, label="Clear-Sky GHI", zorder=1)
    plt.plot(ts_siata.observation_time.values, ts_siata.values,
             marker='o', linestyle='-', color='orange', label="Siata GHI", zorder=2)
    plt.xlabel("Observation Time")
    plt.ylabel("GHI (W/m²)")
    plt.title("Comparison: Siata vs. Clearness ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    ds_siata.close()
    ds_clear_sky.close()
    
if __name__ == "__main__":
    # Define full paths to the NetCDF files:
    
    # Siata file (e.g., created from the Siata CSV processing, with variable "GHI").
    file_siata = ""
    
    # GFS file: should contain variable "dswrf1".
    file_gfs = "_3_Data_preparation_for_LSTM/Preparation_data/GFS_dswrf1/merged_all/GFS_merged_dswrf1_all_0100.nc"
    
    # Clear-Sky file: should contain variable "clear_sky_ghi". For example, CSI data.
    file_clear_sky = "_3_Data_preparation_for_LSTM/Preparation_data/CSI_EXT_radiation/Extraterrestrial_GHI/EXT_GHI_all.nc"
    
    # Plot Siata vs. GFS
    plot_siata_vs_gfs(file_siata, file_gfs)
    
    # Plot Siata vs. Clear-Sky
    plot_siata_vs_clear_sky(file_siata, file_clear_sky)
