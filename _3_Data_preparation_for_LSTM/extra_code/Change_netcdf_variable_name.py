#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 17:50:36 2025

@author: leonardmerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename dswrf1 → dswrf1_0100 and re-save the file.
"""

import xarray as xr
import os

RAW_PATH  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Raw_merged/dswrf1_0100.nc"      # input file
OUT_PATH  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/dswrf1_0100.nc"              # output file
VAR_OLD   = "dswrf1"                            # current variable name
VAR_NEW   = "dswrf1_0100"                       # new name

# ------------------------------------------------------------------
ds = xr.open_dataset(RAW_PATH)

if VAR_OLD not in ds:
    raise KeyError(f"Variable '{VAR_OLD}' not found in {RAW_PATH}")

# Rename variable
ds = ds.rename({VAR_OLD: VAR_NEW})

# Set compression (optional)
encoding = {VAR_NEW: {"zlib": True, "complevel": 4}}

# Create directory if it does not exist
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Save
ds.to_netcdf(OUT_PATH, encoding=encoding)
print(f"NetCDF saved with new variable name: {OUT_PATH}")
