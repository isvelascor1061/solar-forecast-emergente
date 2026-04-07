#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:26:36 2025

@author: leonardmerl
"""
import xarray as xr

path = "_3_Data_preparation_for_LSTM/Preparation_data/_12_DLWRF/dlwrf1_0100.nc"

var_name = "dlwrf1_0100"
threshold=400

ds = xr.open_dataset(path, engine="netcdf4")

if var_name not in ds:
    print("var isnt in dataset")

data=ds[var_name]
mask = data> threshold
Anzahl = mask.sum().item()

print(f"Anzahl der Werte > {threshold}: {Anzahl}")
if Anzahl > 0:
    print("Es gibt Werte über dem Schwellenwert.")
else:
    print("Keine Werte über dem Schwellenwert gefunden.")
