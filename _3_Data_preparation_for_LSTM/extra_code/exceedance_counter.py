#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 14:33:24 2025

@author: leonardmerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-04-21

Script to count how many timestamps Siata GHI or GFS dswrf1 exceed the modeled clear-sky GHI.
"""
import xarray as xr

def count_exceedances(siata_path, siata_var,
                      gfs_path, gfs_var,
                      clear_path, clear_var):
    """
    Loads three NetCDF files (Siata, GFS, Clear‑Sky), aligns them on 'observation_time',
    and returns the counts of timestamps where Siata or GFS values exceed the clear‑sky values.
    """
    # Open datasets
    ds_siata = xr.open_dataset(siata_path)
    ds_gfs   = xr.open_dataset(gfs_path)
    ds_clear = xr.open_dataset(clear_path)

    da_siata = ds_siata[siata_var].squeeze()
    da_gfs   = ds_gfs[gfs_var].squeeze()
    da_clear = ds_clear[clear_var].squeeze()

    # Align all three on common observation_time (inner join)
    da_siata, da_gfs, da_clear = xr.align(
        da_siata, da_gfs, da_clear, join="inner"
    )

    # Count exceedances
    siata_exceed = int((da_siata > da_clear).sum().item())
    gfs_exceed   = int((da_gfs   > da_clear).sum().item())
    total_points = da_clear.sizes.get("observation_time", len(da_clear))

    return siata_exceed, gfs_exceed, total_points

if __name__ == "__main__":
    # === User configuration ===
    SIATA_PATH   = "_3_Data_preparation_for_LSTM/Preparation_data/_03_Siata_GHI/Netcdf_Siata_GHI/SIATA_GHI_all.nc"
    SIATA_VAR    = "GHI"

    GFS_PATH     = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Unclipped_merged_dswrf1/dswrf1_1900.nc"
    GFS_VAR      = "dswrf1_1900"

    CLEAR_PATH   = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/CSI_GHI_grid25_avg_with_horizon_and_enhancement.nc"
    CLEAR_VAR    = "clear_sky_ghi"

    siata_ex, gfs_ex, nt = count_exceedances(
        SIATA_PATH, SIATA_VAR,
        GFS_PATH,   GFS_VAR,
        CLEAR_PATH, CLEAR_VAR
    )

    print(f"Total aligned timestamps: {nt}")
    print(f"Siata GHI > clear-sky GHI at {siata_ex} timestamps "
          f"({siata_ex/nt:.1%})")
    print(f"GFS dswrf1 > clear-sky GHI at {gfs_ex} timestamps "
          f"({gfs_ex/nt:.1%})")
