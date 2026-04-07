#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:43:30 2025

@author: leonardmerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:56:23 2025

@author: leonardmerl

prepare_sequences.py

Lädt Feature- und Target-NetCDFs über NetCDFTimeSeriesLoader (mit 
string-basierten Normalisierungs-Methoden), aligned auf 'observation_time',
baut symmetrische oder kausale Sequenzen, filtert NaNs aus den Sequenzen,
mischt (shuffle) und splittet in Train/Val/Test,
und speichert alles als .npz.
"""

"""
("_3_Data_preparation_for_LSTM/Preparation_data/Clear-sky-indices/clear_sky_index_GFS_0100.nc",
 "clear_sky_index_GFS",
 "none")

"_3_Data_preparation_for_LSTM/Preparation_data/Clear-sky-indices/clear_sky_index_Siata.nc",
"clear_sky_index_Siata",
"none" 

"_3_Data_preparation_for_LSTM/Preparation_data/GFS_dswrf1/merged_all/GFS_merged_dswrf1_all_0100.nc",
 "dswrf1",
 "min_max")

"_3_Data_preparation_for_LSTM/Preparation_data/Siata_GHI/Netcdf_Siata_GHI/SIATA_GHI_all.nc",
"GHI",
"min_max"

"""

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from _4_LSTM_modules.data_preparation_code.multi_feature_target_converter import MultiLaunchTimeLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# ============ USER CONFIG =============================
launch_times = ["0100","0700","1300","1900"]

feat_templates = [
        # --- Indices, already normalized 0-1 -----------------------------------------------
       ("kc",  "_3_Data_preparation_for_LSTM/Preparation_data/_04_indices/clear_sky_indices/clearsky_index_GFS_{LT}.nc",  "clearsky_index_GFS_{LT}",  "none"),
       ("ks",  "_3_Data_preparation_for_LSTM/Preparation_data/_04_indices/clearness_indices/clearness_index_GFS_{LT}.nc", "clearness_index_GFS_{LT}",  "none"),
       # --- Radiation -----------------------------------------------------------
       ("dswrf1",  "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Unclipped_merged_dswrf1/dswrf1_{LT}.nc",              "dswrf1_{LT}",              "min_max"),
       ("dlwrf1",  "_3_Data_preparation_for_LSTM/Preparation_data/_12_DLWRF/dlwrf1_{LT}.nc",                               "dlwrf1_{LT}",              "auto"),
       # --- Atmosphere ----------------------------------------------------------
       ("TMP_surface",  "_3_Data_preparation_for_LSTM/Preparation_data/_05_Temp_surface/TMP_surface_{LT}.nc",              "TMP_surface_{LT}",            "auto"),
       ("RH_2m",  "_3_Data_preparation_for_LSTM/Preparation_data/_06_RH_2m/RH_2m_{LT}.nc",                                  "RH_2m_{LT}",                  "auto"),
       ("CAPE_surface",  "_3_Data_preparation_for_LSTM/Preparation_data/_10_CAPE_surface/CAPE_surface_{LT}.nc",             "CAPE_surface_{LT}",            "auto"),
       ("HPBL_surface",  "_3_Data_preparation_for_LSTM/Preparation_data/_11_HPBL/HPBL_surface_{LT}.nc",                     "HPBL_surface_{LT}",            "auto"),
       ("PWAT_ent",      "_3_Data_preparation_for_LSTM/Preparation_data/_13_PWAT_ent/PWAT_ent_{LT}.nc",                     "PWAT_ent_{LT}",            "min_max"),
       # --- Clouds & Visibility ------------------------------------------------------
       ("TCDC_ent",  "_3_Data_preparation_for_LSTM/Preparation_data/_07_CDC_ent/_01_TCDC/TCDC_ent_{LT}.nc",              "TCDC_ent_{LT}",            "min_max"),
       ("HCDC_ent",  "_3_Data_preparation_for_LSTM/Preparation_data/_07_CDC_ent/_02_HCDC/HCDC_high_{LT}.nc",              "HCDC_high_{LT}",            "min_max"),
       ("MCDC_ent",  "_3_Data_preparation_for_LSTM/Preparation_data/_07_CDC_ent/_03_MCDC/MCDC_mid_{LT}.nc",              "MCDC_mid_{LT}",            "min_max"),
       ("LCDC_ent",  "_3_Data_preparation_for_LSTM/Preparation_data/_07_CDC_ent/_04_LCDC/LCDC_low_{LT}.nc",              "LCDC_low_{LT}",            "min_max"),
       ("HGT_cloud_ceiling",  "_3_Data_preparation_for_LSTM/Preparation_data/_08_HGT_cloud_ceiling/HGT_cloud_ceiling_{LT}.nc",     "HGT_cloud_ceiling_{LT}",  "auto"),
       # --- Windspeed -----------------------------------------------------------------
       ("Wind10m",  "_3_Data_preparation_for_LSTM/Preparation_data/_09_Wind10m/Wind10m_{LT}.nc",              "Wind10m_{LT}",            "min_max"),
       # --- Duration of sunshine ---------------------------------------------------
       ("SUNSD_minutes",  "_3_Data_preparation_for_LSTM/Preparation_data/_11_SUNSD/SUNSD_minutes_{LT}.nc",              "SUNSD_minutes_{LT}",            "min_max"),
       ]

target_path = "_3_Data_preparation_for_LSTM/Preparation_data/_04_indices/clear_sky_indices/clearsky_index_Siata.nc"
target_var = "clearsky_index_Siata"
target_norm = "none"

SEQ_MODE  = "leadstack"
M         = len(launch_times)                     # seq_len
VAL_SPLIT = 0.15
TEST_SPLIT= 0.15

SHUFFLE_SEED = 42
INCLUDE_STEP=True
add_hod=False
add_doy=False
add_zenith=True
LAT=6.25          # Medellín-Koordinate
LON=-75.5


OUT_NPZ      = "_4_LSTM_modules/Prepared_data/4launch_multfeat_leadstack"
indice_filepath = "_4_LSTM_modules/test_indices/test_indices_4launch_leadstack"
# =======================================================

def build_seq_leadstack(df: pd.DataFrame, launch_times):
    suffixes   = tuple(f"_{lt}" for lt in launch_times)
    feat_roots = sorted({c.rsplit("_", 1)[0] for c in df.columns
                         if c.endswith(suffixes)})

    M, F = len(launch_times), len(feat_roots)
    X_seq = np.empty((len(df), M, F), dtype=np.float32)

    for j, lt in enumerate(launch_times):
        cols = [f"{root}_{lt}" for root in feat_roots]
        X_seq[:, j, :] = df[cols].values.astype(np.float32)

    times = df.index
    return X_seq, times







def main(indice_filepath):
    dfs = []
    target_global = None        # ← hier landet das Ziel einmalig

    for lt in launch_times:
        lt_loader = MultiLaunchTimeLoader(
            launch_times=[lt],
            feature_templates=feat_templates,
            target_path=target_path,
            target_var=target_var,
            normalize_target=target_norm,
            tz_offset=0,
            include_step=INCLUDE_STEP,
            add_zenith=add_zenith,
            add_hod=add_hod,
            add_doy=add_doy,
            include_dayflag=False,
            lat=LAT, lon=LON,
        )
        lt_loader.load()
        df_lt = lt_loader.to_dataframe(dropna=True)

        # ---------- (A) Target EINMAL sichern --------------------------
        if target_global is None:
            target_global = df_lt[target_var].copy()   # Series
        # danach aus dem Lauf-DF entfernen
        df_lt = df_lt.drop(columns=[target_var])

        # ---------- (B) day_flag entfernen -----------------------------
        df_lt = df_lt.drop(columns=[c for c in df_lt.columns
                                    if c == "day_flag"])

        # ---------- (C) zenith umbenennen ------------------------------
        if "zenith" in df_lt.columns:
            df_lt = df_lt.rename(columns={"zenith": f"zenith_{lt}"})

        dfs.append(df_lt)

    # ------------------------------------------------------------------
    # 2) Join aller Launch-DFs  (keine Überschneidung mehr!)
    # ------------------------------------------------------------------
    df = dfs[0].join(dfs[1:], how="inner")
    print("Merged shape:", df.shape)          # z. B. (35589, 72)

    # ------------------------------------------------------------------
    # 3) Lead-Stack bauen  (Target separat)
    # ------------------------------------------------------------------
    X_seq, times = build_seq_leadstack(df, launch_times)
    y_seq = target_global.loc[times].values.astype(np.float32)   # aligns

    print("X_seq shape:", X_seq.shape)       # (N, 4, 18)  ← 17 + zenith
    
    # ---------- Shuffle / Split  (unverändert) ------------------------------
    rng  = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(len(y_seq))
    

    n_test = int(len(y_seq) * TEST_SPLIT)
    n_val  = int(len(y_seq) * VAL_SPLIT)
    
    idx_tr = perm[:-n_val - n_test]
    idx_val = perm[-n_val - n_test:-n_test]
    idx_te = perm[-n_test:]
    
    
    # ------------------------------------------------------------------
    # 5a) **Test-Indizes sichern** – genau wie bisher
    # ------------------------------------------------------------------
    
    t_test = times[idx_te]                     # DatetimeIndex
    np.save(indice_filepath, t_test.values)    # oder  t_test.to_numpy()
    print(f"Test-Indizes gespeichert → {len(t_test):,} Zeitpunkte: {indice_filepath}")
       
 
    
    
   

    np.savez_compressed(
        OUT_NPZ,
        X_train=X_seq[idx_tr], y_train=y_seq[idx_tr], t_train=times[idx_tr].values,
        X_val  =X_seq[idx_val], y_val  =y_seq[idx_val], t_val  =times[idx_val].values,
        X_test =X_seq[idx_te],  y_test =y_seq[idx_te],  t_test =times[idx_te].values,
        feature_vars=np.array(lt_loader.feature_vars),
        target_var=np.array([lt_loader.target_var]),
        mode=np.array(["leadstack"]),
        k=np.array([M], dtype=np.int32),
        splits=np.array([VAL_SPLIT, TEST_SPLIT]),
    )
    print("Lead-Stack NPZ gespeichert:", OUT_NPZ)

if __name__ == "__main__":
    main(indice_filepath)