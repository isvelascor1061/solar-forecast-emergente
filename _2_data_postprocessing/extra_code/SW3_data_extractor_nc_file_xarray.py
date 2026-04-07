#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:03:12 2025

@author: leonardmerl
"""

import xarray as xr 
import os

def extract_data_from_nc(file_path, start_index=None, end_index=None ):
    """
    Extract data from NetCDF files and save it to self.dict_all_together.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.
    start_index : int, optional
        Index of the first file to process. The default is None.
    end_index : int, optional
        Index of the last file to process. The default is None.

    Returns
    -------
    lat_values : numpy.ndarray
        Array containing latitude values.
    lon_values : numpy.ndarray
        Array containing longitude values.
    launch_time_local : numpy.ndarray
        Array containing local GFS launch time.
    forecast_time_local: numpy.ndarray
        Array containing local GFS forecast time
    timestep: numpy.ndarray
        Array containig time difference between launch time and forecast time in hours
    data_values : numpy.ndarray
        Array containing data values for each variable in the dataset.
    """
    with xr.open_dataset(file_path, engine="netcdf4") as ds:
        
        lat_values = ds.coords["latitude"].values
        lon_values = ds.coords["longitude"].values
        lon_values = (lon_values+180) %360 -180
        
        if "time" in ds.coords:
            launch_time_local = ds.coords["time"].values
            
        if "valid_time" in ds.coords:
            forecast_time_local= ds.coords["valid_time"].values
        
        if "step" in ds.coords: 
            timestep = ds.coords["step"].values
        
        for var_name in ds.data_vars:
            data_values= ds[var_name].values
    
    return lat_values, lon_values, launch_time_local,forecast_time_local,timestep,data_values


if __name__ =="__main__":
    file_path="raw_data/2025-03-18_01_13_DSWRF1_20250318_0100_0600_2025-03-18_1400.nc"
    
    lat_data, lon_data, launch_time_local, forecast_time_local, timestep, rad_data = extract_data_from_nc(file_path)
    
            
        
    
        
        
    
    
        
        