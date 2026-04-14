#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_wind_sunsd_unified.py
=========================
Merges fix_wind_sunsd.py and fix_wind_sunsd_and_rename.py into a single
script configurable by launch time.

Steps performed:
  1. Load the MultiGFS NetCDF file for the specified launch time.
  2. Compute wind speed as sqrt(U² + V²) from the UGRD_10m and VGRD_10m
     components.
  3. Check for NaNs before computing: if Wind10m already exists, only fill
     the NaN positions; if it does not exist, create the variable from scratch.
  4. Convert SUNSD from seconds to minutes using the same NaN logic.
  5. Drop the UGRD, VGRD and SUNSD_surface variables from the dataset.
  6. Save the result as a new NetCDF file.
"""

import xarray as xr
import numpy as np
import os
from config import MERGED_MULTGFS_DIR, LAUNCH_TIME_DEFAULT


# -----------------------------------------------------------------------
# Execution parameter — adjust according to the launch time to process
# -----------------------------------------------------------------------
LAUNCH_TIME = LAUNCH_TIME_DEFAULT   # e.g. "0100", "0700", "1300", "1900"


def process_netcdf(input_file: str, output_folder: str, output_filename: str,
                   launch_time: str) -> None:
    """
    Process a MultiGFS NetCDF file: compute vector wind speed, convert
    SUNSD to minutes, and drop the component variables.

    Parameters
    ----------
    input_file     : Path to the input NetCDF file.
    output_folder  : Folder where the processed file will be saved.
    output_filename: Name of the output file.
    launch_time    : Launch time string (e.g. "0100").
    """
    # --- Variable names that depend on the launch time -----------------
    var_wind   = f"Wind10m_{launch_time}"
    var_ugrd   = f"UGRD_10m_{launch_time}"
    var_vgrd   = f"VGRD_10m_{launch_time}"
    var_sunsd_min = f"SUNSD_minutes_{launch_time}"
    var_sunsd_sec = f"SUNSD_surface_{launch_time}"

    # --- Load the input dataset ----------------------------------------
    ds = xr.open_dataset(input_file)

    # --- Wind speed: sqrt(U² + V²) -------------------------------------
    # Always computed from the vector components to guarantee consistency.
    # If Wind10m already exists in the dataset (possibly with NaNs), only
    # fill the missing positions; otherwise create the variable from scratch.
    viento_calculado = np.sqrt(ds[var_ugrd] ** 2 + ds[var_vgrd] ** 2)

    if var_wind in ds.data_vars:
        # Variable exists: check for NaNs and fill only those positions
        mascara_nan = ds[var_wind].isnull()
        ds[var_wind] = ds[var_wind].where(~mascara_nan, viento_calculado)
    else:
        # Variable does not exist: create it directly
        ds[var_wind] = viento_calculado

    # --- Sunshine duration: convert from seconds to minutes ------------
    # SUNSD_surface stores the duration in seconds; dividing by 60 gives
    # SUNSD_minutes. Same NaN logic as for wind speed.
    sunsd_minutos = ds[var_sunsd_sec] / 60

    if var_sunsd_min in ds.data_vars:
        # Variable exists: fill only NaN positions
        mascara_nan_sun = ds[var_sunsd_min].isnull()
        ds[var_sunsd_min] = ds[var_sunsd_min].where(~mascara_nan_sun, sunsd_minutos)
    else:
        # Variable does not exist: create it directly
        ds[var_sunsd_min] = sunsd_minutos

    # --- Drop component variables that are no longer needed ------------
    # The individual U and V components are redundant after computing the
    # vector magnitude; SUNSD_surface is redundant after conversion.
    ds = ds.drop_vars([var_ugrd, var_vgrd, var_sunsd_sec])

    # --- Save the processed dataset ------------------------------------
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, output_filename)
    ds.to_netcdf(output_file)
    ds.close()

    print(f"Processed file saved to: {output_file}")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Input path: merged MultiGFS file for the chosen launch time
    input_file = os.path.join(MERGED_MULTGFS_DIR, f"MultGFS_{LAUNCH_TIME}.nc")

    # Output path: same folder, filename with "_fixed" suffix
    output_folder   = MERGED_MULTGFS_DIR
    output_filename = f"MultGFS_{LAUNCH_TIME}_fixed.nc"

    process_netcdf(input_file, output_folder, output_filename, LAUNCH_TIME)
