#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 10:45:45 2025

@author: leonardmerl
"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-04-21

Script to extract and combine time series from multiple NetCDF files into a pandas DataFrame.
The data is limited to a given date range and the clear-sky index (quotient) is calculated.
"""
import os
import xarray as xr
import pandas as pd

def extract_timeseries_to_dataframe(
    file_paths,
    var_names,
    start_date=None,
    end_date=None
):
    """
    Extract time series from up to three NetCDF files and combine them into a pandas DataFrame.
    Also calculates the clear sky index as the quotient of 'dswrf1' over 'clear_sky_ghi'.

    Parameters
    ----------
    file_paths : list of str or None
        List of up to 3 NetCDF file paths. Use None for unused slots.
    var_names : list of str or None
        Corresponding variable names to extract from each file. None to skip.
    start_date : str, optional
        Start date for slicing (inclusive) in 'YYYY-MM-DD' or full ISO format.
    end_date : str, optional
        End date for slicing (inclusive) in 'YYYY-MM-DD' or full ISO format.

    Returns
    -------
    pd.DataFrame
        DataFrame with time as the index and variables as columns, including the clear sky index.
    """
    data_dict = {}  # Dictionary to hold the data for each variable

    for fp, var in zip(file_paths, var_names):
        if fp is None or var is None:
            continue
        ds = xr.open_dataset(fp)
        da = ds[var].squeeze()

        # Slice the time range
        da = da.sel(observation_time=slice(start_date, end_date))

        # Collapse spatial dimensions if any
        if da.ndim > 1:
            dims = [d for d in da.dims if d != 'observation_time']
            da = da.mean(dim=dims)

        # Convert to pandas Series and store in dictionary
        data_dict[var] = da.to_series()

    # Combine the data into a DataFrame
    df = pd.concat(data_dict, axis=1)

    # Calculate the clear sky index as the quotient of dswrf1 over clear_sky_ghi
    if 'dswrf1' in df.columns and 'clear_sky_ghi' in df.columns:
        df['clear_sky_index'] = df['dswrf1'] / df['clear_sky_ghi']
        df['clear_sky_index'] = df['clear_sky_index'].clip(0, 1)  # Ensure values are within 0-1

    return df

if __name__ == '__main__':
    # ===== User configuration =====
    # Specify up to three file paths and corresponding variable names
    FILE_PATHS = [
        "_3_Data_preparation_for_LSTM/Preparation_data/_03_Siata_GHI/Netcdf_Siata_GHI/SIATA_GHI_all.nc",
      
        
     
    ]
    VAR_NAMES = [
        "GHI"
        
    ]

    # Optional date range slice (set to None to include full range)
    START_DATE = "20210615"
    END_DATE = "20210615"

    # Extract the data and convert to a DataFrame
    df = extract_timeseries_to_dataframe(
        FILE_PATHS,
        VAR_NAMES,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # Print the DataFrame
    print(df.head(50))  # Display the first few rows of the DataFrame
