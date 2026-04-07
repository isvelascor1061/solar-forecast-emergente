#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:59:53 2025

@author: leonardmerl
"""

import xarray as xr

path = "_1_data_acquisition/_01_raw_rad_data/dswrf/raw_dswrf_0100/2021-04-02_01_01_DSWRF1_2021-04-02_0200_UTCSTART:2021-04-02_0600.nc"

ds= xr.open_dataset(path,engine="netcdf4")

print(ds.sdswrf.values)