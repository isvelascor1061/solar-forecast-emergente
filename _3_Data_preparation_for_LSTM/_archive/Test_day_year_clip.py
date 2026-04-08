#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:42:00 2025

@author: leonardmerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFS-GHI Cleaning  •  Ersatz durch (Monat, Tag, Stunde)-Mittelwert anderer Jahre
Autor: leonardmerl   (15-05-2025)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------------------------------------
GFS_PATH = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Raw_merged/dswrf1_0100.nc"
CSI_PATH = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/CSI_GHI_grid25_avg_with_horizon_and_enhancement.nc"
OUT_DIR  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped"
OUT_FILE = "GFS_GHI_dayhour_clipped.nc"

GFS_VAR  = "dswrf1"
CSI_VAR  = "clear_sky_ghi"
# -----------------------------------------------------------

def dayhour_mean_clean():
    # 1) Laden & align
    gfs = xr.open_dataset(GFS_PATH)[GFS_VAR]
    csi = xr.open_dataset(CSI_PATH)[CSI_VAR].squeeze()
    gfs, csi = xr.align(gfs, csi, join="inner")

    exceed = gfs > csi
    n_exc  = int(exceed.sum())
    print(f"Exceedances gesamt: {n_exc:,}")

    ser_gfs = gfs.to_series()
    ser_csi = csi.to_series()
    idx     = ser_gfs.index

    # DataFrame mit Zeitmerkmalen
    df = pd.DataFrame({
        "ghi": ser_gfs,
        "csi": ser_csi,
        "month":  idx.month,
        "day":    idx.day,
        "hour":   idx.hour,
        "year":   idx.year
    })

    # (Monat, Tag, Stunde)-Mittelwerte anderer Jahre
    ref = (
        df.loc[~exceed.to_series(), ["month", "day", "hour", "ghi"]]
          .groupby(["month", "day", "hour"])["ghi"]
          .mean()
    )

    def replace(row):
        if row["ghi"] <= row["csi"]:
            return row["ghi"]                # kein Exceedance
        # Mittelwert gleicher Kalendertag+Stunde anderer Jahre
        mean_val = ref.get((row["month"], row["day"], row["hour"]), np.nan)
        if pd.isna(mean_val):
            mean_val = row["csi"]            # Fallback
        # falls immer noch ≥ CSI  → direkt auf CSI setzen
        return mean_val if mean_val < row["csi"] else row["csi"]

    df["ghi_clean"] = df.apply(replace, axis=1)

    # Stats
    n_replaced  = int((exceed & (df["ghi_clean"] < df["csi"])).sum())
    n_set_csi   = int((exceed & (df["ghi_clean"] == df["csi"])).sum())
    print(f"   mit Tag-Stunde-Mittel ersetzt : {n_replaced:,}")
    print(f"   direkt auf CSI gesetzt       : {n_set_csi:,}")

    # Rückwandeln & speichern
    clean_da = xr.DataArray(
        df["ghi_clean"].values,
        coords={"observation_time": idx},
        dims=["observation_time"],
        name=GFS_VAR
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILE)
    clean_da.to_dataset().to_netcdf(out_path)
    print("Gespeichert:", out_path)

if __name__ == "__main__":
    dayhour_mean_clean()
