#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 17:50:36 2025

@author: leonardmerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dswrf1 → dswrf1_0100 umbenennen und Datei neu speichern
"""

import xarray as xr
import os

RAW_PATH  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Raw_merged/dswrf1_0100.nc"      # Eingabedatei
OUT_PATH  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/dswrf1_0100.nc"              # Ausgabedatei
VAR_OLD   = "dswrf1"                            # aktueller Variablenname
VAR_NEW   = "dswrf1_0100"                       # neuer Name

# ------------------------------------------------------------------
ds = xr.open_dataset(RAW_PATH)

if VAR_OLD not in ds:
    raise KeyError(f"Variable '{VAR_OLD}' nicht in {RAW_PATH}")

# Variable umbenennen
ds = ds.rename({VAR_OLD: VAR_NEW})

# Komprimierung einstellen (optional)
encoding = {VAR_NEW: {"zlib": True, "complevel": 4}}

# Verzeichnis anlegen, falls nötig
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Speichern
ds.to_netcdf(OUT_PATH, encoding=encoding)
print(f"NetCDF mit neuem Variablennamen gespeichert: {OUT_PATH}")
