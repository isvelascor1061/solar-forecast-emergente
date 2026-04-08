#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:56:23 2025

@author: leonardmerl

prepare_sequences.py

Lädt Feature- und Target-NetCDFs über den NetCDFTimeSeriesLoader,
richtet sie an 'observation_time' aus, baut symmetrische oder kausale
Sequenzen und speichert sie als .npz – **ohne zusätzliche Skalierung**,
weil die Indizes bereits im Bereich [0,1] liegen.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:56:23 2025

@author: leonardmerl

prepare_sequences.py

Lädt Feature- und Target-NetCDFs über den NetCDFTimeSeriesLoader,
richtet sie an 'observation_time' aus, baut symmetrische oder kausale
Sequenzen und speichert sie als .npz – **ohne zusätzliche Skalierung**,
außer wenn explizit angefordert (über Flags).
"""

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from _4_LSTM_modules.data_preparation_code.Feature_target_converter_normalizer import NetCDFTimeSeriesLoader
from sklearn.preprocessing import MinMaxScaler

# ============ USER CONFIG =============================
FEATURE_FILES: List[Tuple[str, str, bool]] = [
    # (Pfad, Variable-Name, normalize_flag)
    
    ("_3_Data_preparation_for_LSTM/Preparation_data/GFS_dswrf1/merged_all/GFS_merged_dswrf1_all_0100.nc",
     "dswrf1", True),
]

# Jetzt mit drittem Element: normalize_flag für das Target
TARGET_FILE: Tuple[str, str, bool] = (
    "_3_Data_preparation_for_LSTM/Preparation_data/Siata_GHI/Netcdf_Siata_GHI/SIATA_GHI_all.nc",
    "GHI",
    True  # <— hier auf True setzen, wenn das Target auch normalisiert werden soll
)

SEQ_MODE = "symmetric"  # "symmetric" oder "causal"
K        = 24           # symmetrisches Fenster: 2*K+1, bei causal: Past-Length
OUT_NPZ  = "_4_LSTM_modules/Prepared_data/sym49_radiation_data_normalized.npz"
# =======================================================

def build_sequences(X: np.ndarray, y: np.ndarray, mode: str, k: int):
    N, n_feat = X.shape
    if mode == "symmetric":
        L = 2 * k + 1
        M = N - 2 * k
        X_seq = np.empty((M, L, n_feat), np.float32)
        y_seq = np.empty(M, np.float32)
        idx_seq = np.empty(M, np.int64)
        for i, t in enumerate(range(k, N - k)):
            X_seq[i]   = X[t - k : t + k + 1]
            y_seq[i]   = y[t]
            idx_seq[i] = t
    elif mode == "causal":
        L = k
        M = N - L
        X_seq = np.empty((M, L, n_feat), np.float32)
        y_seq = np.empty(M, np.float32)
        idx_seq = np.empty(M, np.int64)
        for i, t in enumerate(range(L, N)):
            X_seq[i]   = X[t - L : t]
            y_seq[i]   = y[t]
            idx_seq[i] = t
    else:
        raise ValueError("mode must be 'symmetric' or 'causal'")
    return X_seq, y_seq, idx_seq

def normalize_features(df: pd.DataFrame, feature_vars: List[str], normalize_flags: List[bool]):
    for var, do_norm in zip(feature_vars, normalize_flags):
        if do_norm:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[var] = scaler.fit_transform(df[[var]])
    return df

def main():
    # 1) Loader initialisieren ---------------------------------
    feature_paths, feature_vars, normalize_flags = zip(*FEATURE_FILES)
    target_path, target_var, normalize_target = TARGET_FILE

    loader = NetCDFTimeSeriesLoader(
        list(feature_paths),
        list(feature_vars),
        list(normalize_flags),  # Feature-Flags
        target_path,
        target_var
    )
    loader.load()
    loader.align()
    df = loader.to_dataframe()

    # 2) Feature-Normalisierung --------------------------------
    df = normalize_features(df, list(feature_vars), list(normalize_flags))

    # 3) Target-Normalisierung (optional) ----------------------
    if normalize_target:
        scaler_t = MinMaxScaler(feature_range=(0, 1))
        df[target_var] = scaler_t.fit_transform(df[[target_var]])
        print(f"Target '{target_var}' normalized to [0,1].")

    # 4) Alle Zeitpunkte mit NaN im Target rauswerfen -----------
    before = len(df)
    df = df[df[target_var].notna()]
    after = len(df)
    print(f"Zeitpunkte gesamt: {before}, nach Entfernen von Target-NaNs: {after}")

    # 5) Zu NumPy-Arrays ---------------------------------------
    X_all = df[list(feature_vars)].values.astype(np.float32)
    y_all = df[target_var].values.astype(np.float32)

    # 6) Sequenzen bauen ---------------------------------------
    X_seq, y_seq, idx_seq = build_sequences(X_all, y_all, mode=SEQ_MODE, k=K)
    time_seq = df.index[idx_seq]
    print("Sequenzen:", X_seq.shape, y_seq.shape)

    # 7) Speichern ---------------------------------------------
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        X_seq=X_seq,
        y_seq=y_seq,
        time_seq=time_seq.values,   # datetime64[ns]
        feature_vars=np.array(feature_vars),
        target_var=np.array([target_var]),
        mode=np.array([SEQ_MODE]),
        k=np.array([K], dtype=np.int32)
    )
    print(f"Saved sequences to {OUT_NPZ}")

if __name__ == "__main__":
    main()
