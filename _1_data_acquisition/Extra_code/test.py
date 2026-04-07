#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:05:23 2025

@author: leonardmerl
"""

import xarray as xr
LT="1300"
# Dataset öffnen
path = f"_1_data_acquisition/_02_raw_MultGFS_data/raw_MultGFS_0100/2021-04-01_01_01_MultGFS_2021-04-01_0200_UTCSTART:2021-04-01_0600.nc"
ds   = xr.open_dataset(path, engine="netcdf4")

# 1) Anzahl an Launch‐Times
#n_launches = ds['launch_time'].size
#    oder: n_launches = len(ds.launch_time)

# 2) Anzahl an Observation‐Times
#    (falls dein Datensatz die Koordinate so heißt)
n_observations = ds['observation_time'].size
#    oder: n_observations = len(ds.observation_time)
Max = ds.to_array().max().item()
Min= ds.to_array().min().item()
#print(f"Anzahl launch_time-Einträge:      {n_launches}")
print(f"Anzahl observation_time-Einträge: {n_observations}")
print(ds.data_vars.values)

#print(ds.observation_time.values)
#print(Max)
#print(Min)
#print(ds)
