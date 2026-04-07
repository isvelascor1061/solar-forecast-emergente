#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vergleicht zwei Strahlungs-Files (NetCDF4) und plottet sie
gegenüber der Tageszeit (UTC oder gewünschte Zeitzone).
"""

from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# --------------------------------------------------------------------------- #
# 1) Dateipfade anpassen
FILE_1 = Path("_2_data_postprocessing/_01_merged_Radrawdata/dswrf/merged_raw_dswrf_0100/sdswrf_20210919_0100.nc")     # enthält Variable 'sdswrf'
FILE_2 = Path("_2_data_postprocessing/_02_merged_Rad1data/dswrf/merged_raw_dswrf1_0100/dswrf1_20210919_0100.nc")    # enthält Variable 'dswrf1'

# --------------------------------------------------------------------------- #
# 2) Daten einlesen
ds1 = xr.open_dataset(FILE_1)
ds2 = xr.open_dataset(FILE_2)

# 3) Zeit- und Werte-Arrays extrahieren
time1 = ds1["observation_time"].to_pandas()
time2 = ds2["observation_time"].to_pandas()

rad1  = ds1["sdswrf"].to_pandas()
rad2  = ds2["dswrf1"].to_pandas()

# Falls beide Dateien denselben Zeitstempel-Satz haben, reicht ein Plot-Befehl.
# Ansonsten per merge/join auf gemeinsamen Index bringen:
if not time1.equals(time2):
    # auf gemeinsamen Zeitindex bringen; keine Werte gehen verloren
    df = pd.concat({"sdswrf": rad1, "dswrf1": rad2}, axis=1)
    time = df.index
    rad1, rad2 = df["sdswrf"], df["dswrf1"]
else:
    time = time1

# --------------------------------------------------------------------------- #
# 4) Plot erstellen
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(time, rad1, marker="o", label="sdswrf")
ax.plot(time, rad2, marker="s", label="dswrf1")

# Achsen & Formatierung
ax.set_xlabel("Uhrzeit")
ax.set_ylabel("Strahlung [W m⁻²]")
ax.set_title("Intervallstrahlung vs. 1-Stunden Intervalle, [6.25/-75.5], 2021-09-19")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.3)

# Nur Stunden anzeigen („02:00“, „06:00“, …)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.autofmt_xdate()

plt.tight_layout()
plt.show()
