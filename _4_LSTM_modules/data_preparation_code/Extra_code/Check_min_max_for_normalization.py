#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:02:43 2025

@author: leonardmerl
"""

import xarray as xr
import numpy as np
LT ="0100"
ds = xr.open_dataset(f"_3_Data_preparation_for_LSTM/Preparation_data/_13_PWAT_ent/PWAT_ent_{LT}.nc")       # Pfad anpassen
da = ds[f"PWAT_ent_{LT}"].squeeze()                  # 1-D: observation_time

# ------------------------------------------------------------
# (a) Absoluter Max-Wert + zugehörige Zeit
# ------------------------------------------------------------
max_val  = da.max().item()
max_time = da.idxmax(dim="observation_time").item()

print(f"Max: {max_val:.2f}  bei  observation_time = {max_time}")

# ------------------------------------------------------------
# (b) Alle Zeiten, an denen DLWRF > 500 (oder ein anderer Schwellenwert) ist
# ------------------------------------------------------------
thresh =34

mask   = da >thresh
times_over = da["observation_time"].where(mask, drop=True).values
vals_over  = da.where(mask, drop=True)             # DataArray mit nur den Peaks
vals_count   = int(vals_over.count())              # xarray.count() → 0-D DataArray ⇒ int
total_vals   = int(da["observation_time"].size)    # Gesamtzahl der Zeitpunkte
percentage   = 100 * vals_count / total_vals       # in Prozent


# Ausgeben jeder Überschreitung
for t, v in zip(times_over, vals_over.values):
    print(f"{np.datetime_as_string(t, unit='s', timezone='UTC')}  →  {v:.2f}")

print(f"{vals_count} von {total_vals} Zeitpunkten (≈ {percentage:.3f} %) liegen über {thresh}")