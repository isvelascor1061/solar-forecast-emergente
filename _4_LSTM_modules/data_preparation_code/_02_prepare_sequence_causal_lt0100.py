#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_02_prepare_sequence_causal_lt0100.py
======================================
Build causal (next-step) sequences using only the GFS 0100 launch time.

The sequence window covers the 24 hours BEFORE the target hour, and the
target is the step AFTER the window ends — a true one-step-ahead forecast:
    Input  : X[t-24 : t]  →  features at hours t-24, t-23, ..., t-1
    Target : y[t]          →  SIATA CSI at hour t

Features (28 total)
-------------------
  16 GFS variables, 0100 launch time:
      kc, ks, dswrf1, dlwrf1, TMP_surface, RH_2m, CAPE_surface,
      HPBL_surface, PWAT_ent, TCDC_ent, HCDC_ent, MCDC_ent, LCDC_ent,
      HGT_cloud_ceiling, Wind10m, SUNSD_minutes
   1 step_0100  (lead time in days)
   1 zenith     (solar zenith angle, normalised 0-1)
  10 SIATA climatological features (siata_climatology.nc):
      clim_mean_csi, clim_std_csi, clim_prob_cloudy, clim_prob_clear,
      clim_prob_partial, clim_bimodality, clim_std_hour,
      clim_transition_risk, clim_month_sin, clim_month_cos

Output
------
  Prepared_data/1launch_causal_lt0100_clim.npz   (N, 24, 28)
  test_indices/test_indices_1launch_causal_lt0100_clim.npy

Usage
-----
    set PYTHONPATH=C:\\Users\\isabe\\Projects\\codigors\\carpetasdetrabajo
    python _4_LSTM_modules/data_preparation_code/_02_prepare_sequence_causal_lt0100.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Resolve project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from _4_LSTM_modules.data_preparation_code.multi_feature_target_converter import MultiLaunchTimeLoader
from config import (
    FEAT_KC_TEMPLATE, FEAT_KS_TEMPLATE, FEAT_DSWRF1_TEMPLATE, FEAT_DLWRF_TEMPLATE,
    FEAT_TMP_TEMPLATE, FEAT_RH_TEMPLATE, FEAT_CAPE_TEMPLATE, FEAT_HPBL_TEMPLATE,
    FEAT_PWAT_TEMPLATE, FEAT_TCDC_TEMPLATE, FEAT_HCDC_TEMPLATE, FEAT_MCDC_TEMPLATE,
    FEAT_LCDC_TEMPLATE, FEAT_HGT_TEMPLATE, FEAT_WIND10M_TEMPLATE, FEAT_SUNSD_TEMPLATE,
    SIATA_CSI_FILE, SIATA_CSI_VAR,
    SIATA_CLIM_FILE, N_CLIM_FEATURES,
    SEQ_NPZ_CAUSAL_FILE, TEST_INDICES_CAUSAL,
    VAL_SPLIT, TEST_SPLIT, SHUFFLE_SEED,
    INCLUDE_STEP, ADD_ZENITH,
    LAT, LON, FLAG_START, FLAG_END,
)

# ── Configuration ─────────────────────────────────────────────────────────────
LAUNCH_TIME = "0100"          # single launch time for this experiment
K           = 24              # causal look-back window (hours)
N_FEAT_EXPECTED = 28          # 16 GFS + 1 step + 1 zenith + 10 clim

feat_templates = [
    # Indices (already normalised 0-1)
    ("kc",                 FEAT_KC_TEMPLATE,      "clearsky_index_GFS_{LT}",  "none"),
    ("ks",                 FEAT_KS_TEMPLATE,      "clearness_index_GFS_{LT}", "none"),
    # Radiation
    ("dswrf1",             FEAT_DSWRF1_TEMPLATE,  "dswrf1_{LT}",              "min_max"),
    ("dlwrf1",             FEAT_DLWRF_TEMPLATE,   "dlwrf1_{LT}",              "auto"),
    # Atmosphere
    ("TMP_surface",        FEAT_TMP_TEMPLATE,     "TMP_surface_{LT}",         "auto"),
    ("RH_2m",              FEAT_RH_TEMPLATE,      "RH_2m_{LT}",               "auto"),
    ("CAPE_surface",       FEAT_CAPE_TEMPLATE,    "CAPE_surface_{LT}",        "auto"),
    ("HPBL_surface",       FEAT_HPBL_TEMPLATE,    "HPBL_surface_{LT}",        "auto"),
    ("PWAT_ent",           FEAT_PWAT_TEMPLATE,    "PWAT_ent_{LT}",            "min_max"),
    # Clouds
    ("TCDC_ent",           FEAT_TCDC_TEMPLATE,    "TCDC_ent_{LT}",            "min_max"),
    ("HCDC_ent",           FEAT_HCDC_TEMPLATE,    "HCDC_high_{LT}",           "min_max"),
    ("MCDC_ent",           FEAT_MCDC_TEMPLATE,    "MCDC_mid_{LT}",            "min_max"),
    ("LCDC_ent",           FEAT_LCDC_TEMPLATE,    "LCDC_low_{LT}",            "min_max"),
    ("HGT_cloud_ceiling",  FEAT_HGT_TEMPLATE,     "HGT_cloud_ceiling_{LT}",   "auto"),
    # Wind and sunshine
    ("Wind10m",            FEAT_WIND10M_TEMPLATE, "Wind10m_{LT}",             "min_max"),
    ("SUNSD_minutes",      FEAT_SUNSD_TEMPLATE,   "SUNSD_minutes_{LT}",       "min_max"),
]


# ── Sequence builder (true next-step causal) ──────────────────────────────────

def build_causal_next_step(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 24,
):
    """
    Build causal next-step sequences.

    Window: X[t-k : t]  — k past hours, NOT including the target hour
    Target: y[t]         — the hour immediately after the window

    This is a true one-step-ahead forecast: the model never sees the
    current or future hours, only the preceding k GFS forecasts.

    Parameters
    ----------
    X : (N, n_feat) feature matrix
    y : (N,) target vector
    k : look-back window length in hours

    Returns
    -------
    Xs  : (M, k, n_feat)  sequence tensor
    ys  : (M,)            target per sequence
    idx : (M,)            index of the target hour in the original series
    """
    N = len(y)
    M = N - k

    Xs  = np.empty((M, k, X.shape[1]), dtype=np.float32)
    ys  = np.empty(M, dtype=np.float32)
    idx = np.empty(M, dtype=np.int64)

    for i, t in enumerate(range(k, N)):
        Xs[i]  = X[t - k : t]   # past k steps (t-k inclusive, t exclusive)
        ys[i]  = y[t]            # next step (true one-step-ahead target)
        idx[i] = t

    return Xs, ys, idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Causal sequence builder — 0100 LT only, 28 features")
    print("=" * 62)

    # ── 1. Load GFS features + target via MultiLaunchTimeLoader ──────────────
    print("\nSection 1 — Loading GFS features (0100 launch time only)")
    loader = MultiLaunchTimeLoader(
        launch_times     = [LAUNCH_TIME],
        feature_templates= feat_templates,
        target_path      = str(ROOT / SIATA_CSI_FILE),
        target_var       = SIATA_CSI_VAR,
        normalize_target = "none",
        tz_offset        = 0,
        dayflag_tz       = "local",
        flag_start       = FLAG_START,
        flag_end         = FLAG_END,
        include_step     = INCLUDE_STEP,    # True → adds step_0100
        include_dayflag  = False,
        add_zenith       = ADD_ZENITH,      # True → adds zenith
        lat              = LAT,
        lon              = LON,
        add_hod          = False,
        add_doy          = False,
    )
    loader.load()
    df = loader.to_dataframe(dropna=True)
    print(f"  Loader features  : {len(loader.feature_vars)}")
    print(f"  DataFrame shape  : {df.shape}")
    print(f"  Time range       : {df.index.min()} -> {df.index.max()}")

    # ── 2. Inject SIATA climatological features ───────────────────────────────
    print("\nSection 2 — Injecting SIATA climatological features")
    ds_clim = xr.open_dataset(str(ROOT / SIATA_CLIM_FILE), engine="h5netcdf")

    clim_mean_csi     = ds_clim["mean_csi"].values.astype(np.float32)      # (12, 24)
    clim_std_csi      = ds_clim["std_csi"].values.astype(np.float32)
    clim_prob_cloudy  = ds_clim["prob_cloudy"].values.astype(np.float32)
    clim_prob_clear   = ds_clim["prob_clear"].values.astype(np.float32)
    clim_prob_partial = ds_clim["prob_partial"].values.astype(np.float32)
    clim_bimodality   = ds_clim["bimodality"].values.astype(np.float32)
    clim_std_hour     = ds_clim["std_by_hour"].values.astype(np.float32)   # (24,)
    clim_trans_risk   = ds_clim["transition_risk"].values.astype(np.float32)
    ds_clim.close()

    m_idx = df.index.month - 1   # 0-indexed month
    h_idx = df.index.hour        # 0-indexed hour

    df["clim_mean_csi"]     = clim_mean_csi[m_idx, h_idx]
    df["clim_std_csi"]      = clim_std_csi[m_idx, h_idx]
    df["clim_prob_cloudy"]  = clim_prob_cloudy[m_idx, h_idx]
    df["clim_prob_clear"]   = clim_prob_clear[m_idx, h_idx]
    df["clim_prob_partial"] = clim_prob_partial[m_idx, h_idx]
    df["clim_bimodality"]   = clim_bimodality[m_idx, h_idx]
    df["clim_std_hour"]     = clim_std_hour[h_idx]
    df["clim_transition_risk"] = clim_trans_risk[h_idx]

    month_vals = df.index.month
    df["clim_month_sin"] = np.sin(2 * np.pi * month_vals / 12).astype(np.float32)
    df["clim_month_cos"] = np.cos(2 * np.pi * month_vals / 12).astype(np.float32)

    CLIM_FEAT_NAMES = [
        "clim_mean_csi", "clim_std_csi", "clim_prob_cloudy", "clim_prob_clear",
        "clim_prob_partial", "clim_bimodality", "clim_std_hour",
        "clim_transition_risk", "clim_month_sin", "clim_month_cos",
    ]
    # Fill night NaN with 0 — zenith already tells the model it's dark
    df[CLIM_FEAT_NAMES] = df[CLIM_FEAT_NAMES].fillna(0.0)

    all_feature_vars = loader.feature_vars + CLIM_FEAT_NAMES
    n_feat = len(all_feature_vars)
    print(f"  GFS + meta features: {len(loader.feature_vars)}")
    print(f"  Clim features      : {N_CLIM_FEATURES}")
    print(f"  Total features     : {n_feat}  (expected {N_FEAT_EXPECTED})")
    assert n_feat == N_FEAT_EXPECTED, \
        f"Feature count mismatch: got {n_feat}, expected {N_FEAT_EXPECTED}"

    # ── 3. Build causal next-step sequences ──────────────────────────────────
    print(f"\nSection 3 — Building causal sequences (k={K}, next-step target)")
    X_all = df[all_feature_vars].values.astype(np.float32)
    y_all = df[loader.target_var].values.astype(np.float32)

    X_seq, y_seq, idx_seq = build_causal_next_step(X_all, y_all, k=K)
    times = df.index[idx_seq]

    # ── 4. Remove sequences containing NaNs ──────────────────────────────────
    valid  = (~np.isnan(y_seq)) & (~np.isnan(X_seq).any(axis=(1, 2)))
    X_seq  = X_seq[valid]
    y_seq  = y_seq[valid]
    times  = times[valid]
    M      = len(y_seq)
    print(f"  Total sequences (after NaN filter): {M:,}")
    print(f"  X shape: {X_seq.shape}   y shape: {y_seq.shape}")

    # ── 5. Shuffle and split ──────────────────────────────────────────────────
    print(f"\nSection 4 — Shuffle and split (seed={SHUFFLE_SEED})")
    rng    = np.random.default_rng(SHUFFLE_SEED)
    perm   = rng.permutation(M)

    n_test  = int(M * TEST_SPLIT)
    n_val   = int(M * VAL_SPLIT)
    n_train = M - n_val - n_test

    idx_tr = perm[:n_train]
    idx_va = perm[n_train : n_train + n_val]
    idx_te = perm[n_train + n_val :]

    X_tr, y_tr, t_tr = X_seq[idx_tr], y_seq[idx_tr], times[idx_tr]
    X_va, y_va, t_va = X_seq[idx_va], y_seq[idx_va], times[idx_va]
    X_te, y_te, t_te = X_seq[idx_te], y_seq[idx_te], times[idx_te]
    print(f"  Split: train={len(idx_tr):,} | val={len(idx_va):,} | test={len(idx_te):,}")

    # ── 6. Min-max normalise the 10 clim features (training set only) ────────
    print("\nSection 5 — Normalising clim features (training min-max)")
    clim_start = len(loader.feature_vars)           # = 18
    clim_end   = clim_start + N_CLIM_FEATURES       # = 28

    train_clim_flat = X_tr[:, :, clim_start:clim_end].reshape(-1, N_CLIM_FEATURES)
    feat_min   = train_clim_flat.min(axis=0)
    feat_max   = train_clim_flat.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0   # avoid division by zero for constant features

    for split in (X_tr, X_va, X_te):
        raw = split[:, :, clim_start:clim_end]
        split[:, :, clim_start:clim_end] = np.clip(
            (raw - feat_min) / feat_range, 0.0, 1.0
        )

    nan_clim = (
        np.isnan(X_tr[:, :, clim_start:clim_end]).any()
        or np.isnan(X_va[:, :, clim_start:clim_end]).any()
        or np.isnan(X_te[:, :, clim_start:clim_end]).any()
    )
    print(f"  Clim features normalised (cols {clim_start}–{clim_end - 1})")
    print(f"  NaN in clim features: {nan_clim}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    print("\nSection 6 — Saving")
    out_npz  = str(ROOT / SEQ_NPZ_CAUSAL_FILE)
    out_idx  = str(ROOT / TEST_INDICES_CAUSAL)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    os.makedirs(os.path.dirname(out_idx),  exist_ok=True)

    np.save(out_idx, t_te.values)
    print(f"  Test indices saved: {len(t_te):,} timestamps -> {out_idx}.npy")

    np.savez_compressed(
        out_npz,
        X_train=X_tr,  y_train=y_tr,  t_train=t_tr.values,
        X_val=X_va,    y_val=y_va,    t_val=t_va.values,
        X_test=X_te,   y_test=y_te,   t_test=t_te.values,
        feature_vars=np.array(all_feature_vars),
        target_var=np.array([loader.target_var]),
        clim_feat_min=feat_min,
        clim_feat_max=feat_max,
        k=np.array([K], dtype=np.int32),
        splits=np.array([VAL_SPLIT, TEST_SPLIT]),
    )
    print(f"  NPZ saved: {out_npz}")

    # ── 8. Final verification ─────────────────────────────────────────────────
    print("\n-- Verification -------------------------------------------------")
    d = np.load(out_npz, allow_pickle=True)
    print(f"  X_train shape : {d['X_train'].shape}   (expected (N, {K}, {N_FEAT_EXPECTED}))")
    print(f"  X_val   shape : {d['X_val'].shape}")
    print(f"  X_test  shape : {d['X_test'].shape}")
    print(f"  NaN in X_train: {np.isnan(d['X_train']).any()}")
    print(f"  NaN in y_train: {np.isnan(d['y_train']).any()}")
    print(f"  Features 18-27: {list(d['feature_vars'][18:])}")
    assert d['X_train'].shape[1] == K,              f"seq_len mismatch: {d['X_train'].shape[1]}"
    assert d['X_train'].shape[2] == N_FEAT_EXPECTED, f"n_feat mismatch: {d['X_train'].shape[2]}"
    print("\nDone.")


if __name__ == "__main__":
    main()
