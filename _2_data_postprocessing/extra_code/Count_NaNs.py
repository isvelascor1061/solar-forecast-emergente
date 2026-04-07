#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:33:06 2025

@author: leonardmerl
"""

import xarray as xr
import numpy as np

def count_nans_in_netcdf(file_path: str):
    # Lade das NetCDF4-Dataset
    ds = xr.open_dataset(file_path)
    
    # Schleife durch alle Variablen in den 'data_vars'
    for var_name, var_data in ds.data_vars.items():
        # Überprüfe, wie viele NaN-Werte in der Variable existieren
        nan_count = var_data.isnull().sum().item()  # Summe der NaN-Werte
        
        print(f"Variable '{var_name}': {nan_count} NaN-Werte")

    # Schließe das Dataset
    ds.close()

# Beispiel: Pfad zur NetCDF4-Datei
file_path = "_3_Data_preparation_for_LSTM/Preparation_data/_03_Siata_GHI/Netcdf_Siata_GHI/GHI_CSI_clipped.nc"
count_nans_in_netcdf(file_path)
