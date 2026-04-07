#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:18:59 2025

@author: leonardmerl
"""

import xarray as xr
import numpy as np

# ------------------------------------------------------------
# 1) Datei laden  (Pfad anpassen!)
path = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct2.nc"
ds   = xr.open_dataset(path)

# ------------------------------------------------------------
# 2) Beobachtungs-Zeitkoordinate extrahieren  (Name ggf. anpassen)
times = np.sort(ds['observation_time'].values)

# ------------------------------------------------------------
# 3) Schrittweite (nominales Intervall) bestimmen
#    → kleinste positive Differenz als Referenz
diffs    = np.diff(times).astype('timedelta64[s]')
nominal  = diffs[diffs > np.timedelta64(0, 's')].min()

# ------------------------------------------------------------
# 4) Lücken finden: alle Abstände, die größer als 1× nominal sind
gap_idxs = np.where(diffs > nominal)[0]

if len(gap_idxs) == 0:
    print("✅ Keine Lücken – Intervall durchgehend", nominal)
else:
    print(f"⚠️  Gefundene Lücken (nominal {nominal}):")
    for idx in gap_idxs:
        start = times[idx]
        end   = times[idx + 1]
        missing_steps = diffs[idx] // nominal - 1
        print(f"  • {start} → {end}  (fehlen {missing_steps} Zeit­schritt(e))")
