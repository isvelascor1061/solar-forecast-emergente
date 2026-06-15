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
from config import (
    LAUNCH_TIMES,
    FEAT_KC_TEMPLATE, FEAT_KS_TEMPLATE, FEAT_DSWRF1_TEMPLATE, FEAT_DLWRF_TEMPLATE,
    FEAT_TMP_TEMPLATE, FEAT_RH_TEMPLATE, FEAT_CAPE_TEMPLATE, FEAT_HPBL_TEMPLATE,
    FEAT_PWAT_TEMPLATE, FEAT_TCDC_TEMPLATE, FEAT_HCDC_TEMPLATE, FEAT_MCDC_TEMPLATE,
    FEAT_LCDC_TEMPLATE, FEAT_HGT_TEMPLATE, FEAT_WIND10M_TEMPLATE, FEAT_SUNSD_TEMPLATE,
    SIATA_CSI_FILE, SIATA_CSI_VAR,
    SIATA_CLIM_FILE, N_CLIM_FEATURES,
    SEQ_NPZ_CLIM_FILE, TEST_INDICES_CLIM_FILE,
    SEQ_MODE, K_LEFT, K_RIGHT, OFF, K, VAL_SPLIT, TEST_SPLIT, SHUFFLE_SEED,
    INCLUDE_STEP, INCLUDE_DAYFLAG, ADD_HOD, ADD_DOY, ADD_ZENITH,
    LAT, LON, FLAG_START, FLAG_END,
    SEQ_NPZ_TEST_FILE, TEST_INDICES_FILE,
)
import xarray as xr

# ============ USER CONFIG =============================
launch_times = LAUNCH_TIMES

feat_templates = [
        # --- Indices, already normalized 0-1 -----------------------------------------------
       ("kc",       FEAT_KC_TEMPLATE,     "clearsky_index_GFS_{LT}",  "none"),
       ("ks",       FEAT_KS_TEMPLATE,     "clearness_index_GFS_{LT}", "none"),
       # --- Radiation -----------------------------------------------------------
       ("dswrf1",   FEAT_DSWRF1_TEMPLATE, "dswrf1_{LT}",              "min_max"),
       ("dlwrf1",   FEAT_DLWRF_TEMPLATE,  "dlwrf1_{LT}",              "auto"),
       # --- Atmosphere ----------------------------------------------------------
       ("TMP_surface",      FEAT_TMP_TEMPLATE,  "TMP_surface_{LT}",      "auto"),
       ("RH_2m",            FEAT_RH_TEMPLATE,   "RH_2m_{LT}",            "auto"),
       ("CAPE_surface",     FEAT_CAPE_TEMPLATE,  "CAPE_surface_{LT}",     "auto"),
       ("HPBL_surface",     FEAT_HPBL_TEMPLATE,  "HPBL_surface_{LT}",     "auto"),
       ("PWAT_ent",         FEAT_PWAT_TEMPLATE,  "PWAT_ent_{LT}",         "min_max"),
       # --- Clouds & Visibility ------------------------------------------------------
       ("TCDC_ent",         FEAT_TCDC_TEMPLATE,  "TCDC_ent_{LT}",         "min_max"),
       ("HCDC_ent",         FEAT_HCDC_TEMPLATE,  "HCDC_high_{LT}",        "min_max"),
       ("MCDC_ent",         FEAT_MCDC_TEMPLATE,  "MCDC_mid_{LT}",         "min_max"),
       ("LCDC_ent",         FEAT_LCDC_TEMPLATE,  "LCDC_low_{LT}",         "min_max"),
       ("HGT_cloud_ceiling",FEAT_HGT_TEMPLATE,   "HGT_cloud_ceiling_{LT}","auto"),
       # --- Windspeed -----------------------------------------------------------------
       ("Wind10m",          FEAT_WIND10M_TEMPLATE,"Wind10m_{LT}",          "min_max"),
       # --- Duration of sunshine ---------------------------------------------------
       ("SUNSD_minutes",    FEAT_SUNSD_TEMPLATE,  "SUNSD_minutes_{LT}",    "min_max"),
       ]

target_path = SIATA_CSI_FILE
target_var  = SIATA_CSI_VAR
target_norm = "none"


# Sequence parameters -------------------------------------------------
add_hod    = ADD_HOD
add_doy    = ADD_DOY
add_zenith = ADD_ZENITH

# Output files — climatology-enriched sequences (sym18 + 10 clim features)
OUT_NPZ         = SEQ_NPZ_CLIM_FILE
indice_filepath = TEST_INDICES_CLIM_FILE

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

    if mode == "symmetric" or mode.startswith("sym"):
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
        raise ValueError("mode must be 'symmetric', 'sym<N>' or 'causal'")

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

    # ---- 1b) Inject climatological features from siata_climatology.nc ----
    # Load the pre-computed climatology (produced by compute_siata_climatology.py)
    ds_clim = xr.open_dataset(SIATA_CLIM_FILE, engine="h5netcdf")

    # Extract numpy arrays for fast vectorised lookup:
    #   month_hour stats shape: (12, 24) — index by [month-1, hour]
    #   hour_only   stats shape: (24,)   — index by [hour]
    clim_mean_csi      = ds_clim["mean_csi"].values.astype(np.float32)      # (12, 24)
    clim_std_csi       = ds_clim["std_csi"].values.astype(np.float32)       # (12, 24)
    clim_prob_cloudy   = ds_clim["prob_cloudy"].values.astype(np.float32)   # (12, 24)
    clim_prob_clear    = ds_clim["prob_clear"].values.astype(np.float32)    # (12, 24)
    clim_prob_partial  = ds_clim["prob_partial"].values.astype(np.float32)  # (12, 24)
    clim_bimodality    = ds_clim["bimodality"].values.astype(np.float32)    # (12, 24)
    clim_std_hour      = ds_clim["std_by_hour"].values.astype(np.float32)   # (24,)
    clim_trans_risk    = ds_clim["transition_risk"].values.astype(np.float32)  # (24,)

    # Vectorised lookup using the DataFrame's DatetimeIndex
    m_idx = df.index.month - 1   # 0-indexed (0–11)
    h_idx = df.index.hour        # 0-indexed (0–23)

    # (month, hour) lookups — NaN for night hours (stats computed daytime only)
    df["clim_mean_csi"]      = clim_mean_csi[m_idx, h_idx]
    df["clim_std_csi"]       = clim_std_csi[m_idx, h_idx]
    df["clim_prob_cloudy"]   = clim_prob_cloudy[m_idx, h_idx]
    df["clim_prob_clear"]    = clim_prob_clear[m_idx, h_idx]
    df["clim_prob_partial"]  = clim_prob_partial[m_idx, h_idx]
    df["clim_bimodality"]    = clim_bimodality[m_idx, h_idx]

    # (hour) lookups
    df["clim_std_hour"]         = clim_std_hour[h_idx]
    df["clim_transition_risk"]  = clim_trans_risk[h_idx]

    # Cyclic month encodings (always defined, no NaN)
    month_vals = df.index.month
    df["clim_month_sin"] = np.sin(2 * np.pi * month_vals / 12).astype(np.float32)
    df["clim_month_cos"] = np.cos(2 * np.pi * month_vals / 12).astype(np.float32)

    # Fill nighttime NaN values with 0.0 — the model already has the zenith
    # angle to distinguish day from night; 0 means "no climatological signal"
    CLIM_FEAT_NAMES = [
        "clim_mean_csi", "clim_std_csi", "clim_prob_cloudy", "clim_prob_clear",
        "clim_prob_partial", "clim_bimodality", "clim_std_hour",
        "clim_transition_risk", "clim_month_sin", "clim_month_cos",
    ]
    df[CLIM_FEAT_NAMES] = df[CLIM_FEAT_NAMES].fillna(0.0)

    # Combine original + new feature names for the full feature matrix
    all_feature_vars = loader.feature_vars + CLIM_FEAT_NAMES
    n_original = len(loader.feature_vars)   # 69
    print(f"Original features: {n_original}  |  Clim features: {N_CLIM_FEATURES}  "
          f"|  Total: {len(all_feature_vars)}")

    X_all = df[all_feature_vars].values.astype(np.float32)
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
    print(f"Split -> train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")

    # ---- 4b) Min-max normalise the 10 climatological features -----------
    # Scaler is fitted on training data ONLY, then applied to val and test.
    clim_start = n_original          # first clim channel index (= 69)
    clim_end   = clim_start + N_CLIM_FEATURES   # last + 1  (= 79)

    # Flatten (n_sequences, L, 10) → (n_sequences * L, 10) for fitting
    train_clim_flat = X_tr[:, :, clim_start:clim_end].reshape(-1, N_CLIM_FEATURES)
    feat_min   = train_clim_flat.min(axis=0)   # shape (10,)
    feat_max   = train_clim_flat.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0          # avoid division by zero for constant features

    for split in (X_tr, X_va, X_te):
        raw = split[:, :, clim_start:clim_end]
        split[:, :, clim_start:clim_end] = np.clip(
            (raw - feat_min) / feat_range, 0.0, 1.0
        )

    print(f"Clim features normalised with training-set min-max "
          f"(cols {clim_start}–{clim_end - 1})")

    # Verify: no NaN in the new features
    nan_clim = (
        np.isnan(X_tr[:, :, clim_start:clim_end]).any()
        or np.isnan(X_va[:, :, clim_start:clim_end]).any()
        or np.isnan(X_te[:, :, clim_start:clim_end]).any()
    )
    print(f"NaN in clim features: {nan_clim}")
    print(f"Final shapes -> X_train: {X_tr.shape}, X_val: {X_va.shape}, X_test: {X_te.shape}")

    # ---- 5) Persist test timestamps --------------------------------------
    np.save(indice_filepath, t_te.values)
    print(f"Test indices saved -> {len(t_te):,} timestamps")

    # ---- 6) Save everything to .npz -------------------------------------
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
        feature_vars=np.array(all_feature_vars),
        target_var=np.array([loader.target_var]),
        mode=np.array([SEQ_MODE]),
        k=np.array([K], dtype=np.int32),
        splits=np.array([VAL_SPLIT, TEST_SPLIT]),
        clim_feat_min=feat_min,
        clim_feat_max=feat_max,
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







