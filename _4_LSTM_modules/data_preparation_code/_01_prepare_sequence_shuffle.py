#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:56:23 2025

@author: leonardmerl

Created on Tue Apr 22 14:56:23 2025

@author: leonardmerl

This script assembles multi-launch-time input sequences and a matching
target series for training, validating and testing an LSTM that predicts
a clear-sky index derived from SIATA ground-station data.

The workflow is:

    1) Load GFS-based predictors for the specified launch times.
    2) Load the SIATA clear-sky index as the prediction target.
    3) Optional: normalise / rescale features or target if requested.
    4) Merge everything into a single time-indexed DataFrame.
    5) Build sliding-window input sequences (symmetric or causal).
    6) Filter out sequences containing NaNs.
    7) Shuffle, split into train/val/test and save to .npz.
    8) Persist the timestamps belonging to the test split so you can
       recreate identical test sets later.
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


# Sequence parameters -------------------------------------------------
SEQ_MODE = "symmetric"          # "symmetric" or "causal"
K_LEFT = 24                      # hours back in time
K_RIGHT = 24                     # hours into the future (ignored if causal)
OFF = None                       # target offset inside symmetric window
K = 24                           # history length if causal
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SHUFFLE_SEED = 16               

# Extra “meta” channels ----------------------------------------------
INCLUDE_STEP = True
INCLUDE_DAYFLAG = False
add_hod = False
add_doy = False
add_zenith = True
LAT = 6.25                       # Medellín important for zenith calculation 
LON = -75.5
FLAG_START = 19                  # local civil night start hour not important if dayflag is set to false
FLAG_END = 6                     # local civil night end hour not important if dayflag is set to false

# Output files --------------------------------------------------------
OUT_NPZ = "_4_LSTM_modules/Prepared_data/4launch_multfeat_test"
indice_filepath = "_4_LSTM_modules/test_indices/test_indices_4launch_multfeat_test"

# =======================================================


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    mode: str,
    k_left: int,
    k_right: int,
    k: int | None = None,
    off: int | None = None,
):
    """
    Assemble sliding-window sequences.

    Parameters
    ----------
    X : array shape (N, n_feat)
        Feature matrix.
    y : array shape (N,)
        Target vector.
    mode : {"symmetric", "causal"}
        Window strategy.
    k_left : int
        Hours of look-back (only used for symmetric).
    k_right : int
        Hours of look-ahead (only used for symmetric).
    k : int | None
        Look-back length for causal mode.
    off : int | None
        Position of the prediction hour inside a symmetric window
        (0 … k_left + k_right).  If None → off = k_left.

    Returns
    -------
    Xs : array (M, L, n_feat)
        Sequence tensor.
    ys : array (M,)
        Targets per sequence.
    idx : array (M,)
        Indices of the prediction hours in the original series.
    """
    N, n_feat = X.shape

    if mode == "symmetric":
        if off is None:
            off = k_left
        assert 0 <= off <= k_left + k_right, "Offset lies outside the window."

        L = k_left + k_right + 1         # total window length
        M = N - (k_left + k_right)       # number of possible windows

        Xs = np.empty((M, L, n_feat), np.float32)
        ys = np.empty(M, np.float32)
        idx = np.empty(M, np.int64)

        # Slide window centre from k_left … N-k_right-1
        for i, t in enumerate(range(k_left, N - k_right)):
            start = t - k_left
            Xs[i] = X[start : start + L]
            ys[i] = y[start + off]
            idx[i] = start + off

    elif mode == "causal":
        if k is None:
            raise ValueError("For 'causal', k (look-back) must be given.")
        L = k
        M = N - L

        Xs = np.empty((M, L, n_feat), np.float32)
        ys = np.empty(M, np.float32)
        idx = np.empty(M, np.int64)

        # Window covers t-k … t-1, target = t-1
        for i, t in enumerate(range(L, N)):
            Xs[i] = X[t - L : t]
            ys[i] = y[t - 1]
            idx[i] = t - 1
    else:
        raise ValueError("mode must be 'symmetric' or 'causal'")

    return Xs, ys, idx


def main(indice_filepath: str):
    """Load data, build sequences, split and save."""
    flag_start = FLAG_START
    flag_end = FLAG_END

    # For causal windows the target lies one hour earlier → shift flags back
    if SEQ_MODE == "causal":
        flag_start = (flag_start - 1) % 24
        flag_end = (flag_end - 1) % 24

    # ---- 1) Load predictors & target -------------------------------------
    loader = MultiLaunchTimeLoader(
        launch_times=launch_times,
        feature_templates=feat_templates,
        target_path=target_path,
        target_var=target_var,
        normalize_target=target_norm,
        tz_offset=0,
        dayflag_tz="local",
        flag_start=flag_start,
        flag_end=flag_end,
        include_step=INCLUDE_STEP,
        include_dayflag=INCLUDE_DAYFLAG,
        add_zenith=add_zenith,
        lat=LAT,
        lon=LON,
        add_hod=add_hod,
        add_doy=add_doy,
    )

    loader.load()
    df = loader.to_dataframe(dropna=True)
    print(loader.feature_vars)

    X_all = df[loader.feature_vars].values.astype(np.float32)
    y_all = df[loader.target_var].values.astype(np.float32)

    # ---- 2) Build sequences ----------------------------------------------
    X_seq, y_seq, idx_seq = build_sequences(
        X_all, y_all, SEQ_MODE, K_LEFT, K_RIGHT, K, off=OFF
    )
    times = df.index[idx_seq]

    # ---- 3) Remove sequences containing NaNs ------------------------------
    valid = (~np.isnan(y_seq)) & (~np.isnan(X_seq).any(axis=(1, 2)))
    X_seq = X_seq[valid]
    y_seq = y_seq[valid]
    times = times[valid]
    M = len(y_seq)
    print(f"{M} valid sequences after NaN filtering.")

    # ---- 4) Shuffle & split ------------------------------------------------
    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(M)

    n_test = int(M * TEST_SPLIT)
    n_val = int(M * VAL_SPLIT)
    n_train = M - n_val - n_test

    idx_tr = perm[:n_train]
    idx_va = perm[n_train : n_train + n_val]
    idx_te = perm[n_train + n_val :]

    X_tr, y_tr, t_tr = X_seq[idx_tr], y_seq[idx_tr], times[idx_tr]
    X_va, y_va, t_va = X_seq[idx_va], y_seq[idx_va], times[idx_va]
    X_te, y_te, t_te = X_seq[idx_te], y_seq[idx_te], times[idx_te]
    print(f"Split → train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")

    # ---- 5) Persist test timestamps ---------------------------------------
    np.save(indice_filepath, t_te.values)
    print(f"Test indices saved → {len(t_te):,} timestamps")

    # ---- 6) Save everything to .npz ---------------------------------------
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        X_train=X_tr,
        y_train=y_tr,
        t_train=t_tr.values,
        X_val=X_va,
        y_val=y_va,
        t_val=t_va.values,
        X_test=X_te,
        y_test=y_te,
        t_test=t_te.values,
        feature_vars=np.array(loader.feature_vars),
        target_var=np.array([loader.target_var]),
        mode=np.array([SEQ_MODE]),
        k=np.array([K], dtype=np.int32),
        splits=np.array([VAL_SPLIT, TEST_SPLIT]),
    )
    print(f"Saved sequences to {OUT_NPZ}")


if __name__ == "__main__":
    main(indice_filepath)

    """
    This script uses the multi feature target converter class to prepare sequences used by the LSTM
    1. It iterates over the launch times and different files with the Data to prepare feature target pairs
    it then builds sequences based on the useres settings -> causal or symmetric 
    2. dayflag is a setting that adds an extra feature to the vektor -> one during the day, 0 during the night
    -> the lstm then multiplies the loss with this feature -> SET TO NONE WITH THE CURRENT LSTM 
    flag start and end is then also irrelevant 
    3. add hoy and doy add time of day and day of year as a feature -> i havent tested this yet if it improves the output
    4. add zenith adds the Sun elevation angle as a feature
    5. step adds the difference between observation time and launch time as a feature
    6. the script saves the constructed sequences as an .npz file
    7. the indices (timestamps) are also saved in a .npy file -> to test the performance before the LSTM
    """







