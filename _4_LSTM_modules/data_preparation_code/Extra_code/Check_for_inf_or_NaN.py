#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:48:06 2025

@author: leonardmerl
"""
import numpy as np

SEQ_NPZ = "_4_LSTM_modules/Prepared_data/1launch_seq_caus24_CSI_shuffle.npz"
data = np.load(SEQ_NPZ, allow_pickle=True)

def check_nan_inf(name, arr):
    """Zählt NaN & Inf für numerische Arrays – überspringt andere dtypes."""
    if np.issubdtype(arr.dtype, np.number):
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        print(f"{name:<10}: NaN = {n_nan:,} | Inf = {n_inf:,}")
    else:
        print(f"{name:<10}: dtype = {arr.dtype}  ➜  übersprungen")

for key in data.files:
    check_nan_inf(key, data[key])



#%%


import numpy as np

def inspect_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        print(f"Inhalt der Datei '{npz_path}':")
        for key in data.files:
            arr = data[key]
            print(f"- Schlüssel: '{key}' | Typ: {type(arr)} | Form: {getattr(arr, 'shape', 'keine Form')} | Datentyp: {getattr(arr, 'dtype', 'keiner')}")

# Beispiel-Aufruf
inspect_npz("_4_LSTM_modules/Prepared_data/1launch_Multfeat_CSI_0100_caus24.npz")
