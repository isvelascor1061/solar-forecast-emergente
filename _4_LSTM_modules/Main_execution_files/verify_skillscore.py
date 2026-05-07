#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_skillscore.py
====================
Verifies the SkillScore reported during training by computing the true GFS
raw baseline MSE on exactly the same test set used by the model.

The current MSE_BASELINE_R = 22322.349260 in config.py was calculated under
possibly different conditions (different test period, single launch time, or
different data subset). This script re-derives it consistently.

Workflow
--------
1.  Load test timestamps for sym18 from the saved .npy file.
2.  Load model predictions (W/m²) from pred_real.csv.
3.  Load Ineichen clear-sky reference for descaling and the daytime mask.
4.  Load all 4 GFS raw CSI files; for each test timestamp pick the
    forecast with the shortest positive lead time ("best-lead" strategy).
5.  Descale GFS CSI → W/m² by multiplying with the Ineichen reference.
6.  Apply daytime mask (clear_sky_ghi > 0) to exclude night-time hours.
7.  Compute MSE / RMSE / MAE / R² for GFS raw and our model on the same
    daytime test subset.
8.  Derive the correct SkillScore = 1 − MSE_model / MSE_GFS_raw.
9.  Save 4 diagnostic plots to _4_LSTM_modules/Evaluation_after_LSTM/.
10. Print a comparison table and recommend the updated MSE_BASELINE_R.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Add project root to sys.path so config.py can be imported ──────────────
# This file lives at _4_LSTM_modules/Main_execution_files/verify_skillscore.py
# so two .parents steps reach the project root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import (
    LAUNCH_TIMES,             # ["0100", "0700", "1300", "1900"]
    VAR_GFS_CSI_TEMPLATE,     # "clearsky_index_GFS_{LT}"
    VAR_REF_CLEARSKY_GHI,     # "clear_sky_ghi"
    CSI_INDEX_DIR,            # relative path to clear_sky_indices/
    CSI_GHI_FILE,             # relative path to the Ineichen NetCDF
    MSE_BASELINE_R,           # current value in config (to compare against)
)

# ===========================================================================
# USER-CONFIGURABLE PATHS
# (edit these if the directory structure or experiment changes)
# ===========================================================================

# Test split timestamps saved by the sequence-builder for sym18
TEST_IDX_FILE = ROOT / "_4_LSTM_modules/test_indices/test_indices_4launch_multfeat_sym18.npy"

# Predictions CSV for sym18 + Bahdanau attention (our best model)
# Columns expected: time, y_true, y_pred  — both in W/m² (already descaled)
PRED_CSV_FILE = ROOT / (
    "_4_LSTM_modules/_runs/4launch_multfeat_sym/"
    "4launch_Multfeat_sym18_BiLSTM_attn_20260428_121958/pred_real.csv"
)

# Directory that contains clearsky_index_GFS_0100/0700/1300/1900.nc
CSI_DIR = ROOT / CSI_INDEX_DIR

# Ineichen clear-sky NetCDF (used for GFS CSI → W/m² and for daytime mask)
INEICHEN_FILE = ROOT / CSI_GHI_FILE

# Output directory for plots and the text report
OUT_DIR = ROOT / "_4_LSTM_modules/Evaluation_after_LSTM"


# ===========================================================================
# SECTION 1 — Load test timestamps
# ===========================================================================
print("=" * 65)
print("SECTION 1 — Loading test timestamps")
print("=" * 65)

test_timestamps = pd.DatetimeIndex(np.load(TEST_IDX_FILE))
print(f"  File            : {TEST_IDX_FILE.name}")
print(f"  Test timestamps : {len(test_timestamps):,}")
print(f"  Range           : {test_timestamps.min()} → {test_timestamps.max()}")


# ===========================================================================
# SECTION 2 — Load model predictions (already in W/m²)
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 2 — Loading model predictions")
print("=" * 65)

if not PRED_CSV_FILE.exists():
    sys.exit(
        f"\n[ERROR] Predictions CSV not found:\n  {PRED_CSV_FILE}\n"
        "Run Forward_prop_with_bestmodel.py first to generate it."
    )

df_pred = (
    pd.read_csv(PRED_CSV_FILE, parse_dates=["time"])
    .set_index("time")
    .rename(columns={"y_true": "siata_wm2", "y_pred": "model_wm2"})
)

# Restrict to the test timestamps only (pred_real.csv is already test-only,
# but we intersect to be safe in case it contains other splits)
df_pred = df_pred.loc[df_pred.index.isin(test_timestamps)].sort_index()
print(f"  CSV rows after test-set intersection : {len(df_pred):,}")

# Sanity check: confirm values look like W/m² (not normalised 0-1)
max_pred = df_pred["model_wm2"].max()
if max_pred <= 2.0:
    print(
        f"  [WARNING] max(y_pred) = {max_pred:.4f} — looks like CSI, not W/m².\n"
        "  Check that Forward_prop_with_bestmodel.py used DESCALER_METHOD='physical'."
    )
else:
    print(f"  max(y_pred) = {max_pred:.1f} W/m²  ✓  (confirmed as physical units)")


# ===========================================================================
# SECTION 3 — Load Ineichen clear-sky (descaling reference + daytime mask)
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 3 — Loading Ineichen clear-sky reference")
print("=" * 65)

ds_ineichen = xr.open_dataset(INEICHEN_FILE, engine="h5netcdf")
da_csg = ds_ineichen[VAR_REF_CLEARSKY_GHI]

# Reduce to a 1-D time series: area-mean if lat/lon present
if {"lat", "lon"}.issubset(da_csg.dims):
    da_csg = da_csg.mean(dim=("lat", "lon"))

# Squeeze any remaining singleton dimensions (e.g. surface)
for dim in list(da_csg.dims):
    if dim != "observation_time":
        da_csg = da_csg.isel({dim: 0})

csg_series = da_csg.to_series().rename("clear_sky_ghi")
ds_ineichen.close()

print(f"  Ineichen series : {len(csg_series):,} hourly values")
print(f"  Range           : {csg_series.index.min()} → {csg_series.index.max()}")
print(f"  Max clear-sky   : {csg_series.max():.1f} W/m²")


# ===========================================================================
# SECTION 4 — Load GFS raw CSI and build the best-lead series
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 4 — Building GFS raw best-lead CSI series")
print("=" * 65)
print(
    "  Strategy: for each observation_time, select the GFS CSI value\n"
    "  from the launch time with the shortest *positive* lead time."
)

val_frames  = []   # CSI value per launch time, indexed by observation_time
lead_frames = []   # Corresponding lead duration

for lt in LAUNCH_TIMES:
    nc_path  = CSI_DIR / f"clearsky_index_GFS_{lt}.nc"
    var_name = VAR_GFS_CSI_TEMPLATE.format(LT=lt)

    ds_gfs = xr.open_dataset(nc_path, engine="h5netcdf")
    da = ds_gfs[var_name]

    # Area mean over spatial grid if lat/lon dimensions are present
    if {"lat", "lon"}.issubset(da.dims):
        da = da.mean(dim=("lat", "lon"))

    # Squeeze all dims except the time axis
    for dim in list(da.dims):
        if dim != "observation_time":
            da = da.isel({dim: 0})

    ser_val = da.to_series()   # index = observation_time

    # Lead time = observation_time − launch_time (per row)
    # Negative or zero lead means the observation pre-dates the model run
    launch_ser = pd.to_datetime(ds_gfs["launch_time"].to_series())
    launch_ser.index = ser_val.index
    lead = ser_val.index.to_series() - launch_ser
    lead[lead <= pd.Timedelta(0)] = pd.NaT   # mark invalid leads as missing

    val_frames.append(ser_val.rename(lt))
    lead_frames.append(lead.rename(lt))
    ds_gfs.close()
    print(f"  LT {lt}: {len(ser_val):,} rows  |  valid leads: {(~lead.isna()).sum():,}")

# Stack into DataFrames and pick the column with the minimum positive lead
val_df  = pd.concat(val_frames,  axis=1)
lead_df = pd.concat(lead_frames, axis=1)

idx_min   = lead_df.idxmin(axis=1)                  # launch-time column that wins
col_pos   = val_df.columns.get_indexer(idx_min)     # integer column positions
row_pos   = np.arange(len(val_df))
best_vals = val_df.to_numpy()[row_pos, col_pos]

gfs_csi_best = (
    pd.Series(best_vals, index=val_df.index, name="gfs_csi")
    .dropna()
    .sort_index()
)
print(f"\n  Best-lead GFS CSI series: {len(gfs_csi_best):,} valid rows")


# ===========================================================================
# SECTION 5 — Descale GFS CSI → W/m²
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 5 — Descaling GFS CSI → W/m²")
print("=" * 65)

# GFS W/m² = CSI × Ineichen clear-sky GHI at the same timestamp
gfs_wm2 = (gfs_csi_best * csg_series).dropna().rename("gfs_wm2")
print(f"  GFS W/m² series after join: {len(gfs_wm2):,} values")
print(f"  Max GFS W/m²              : {gfs_wm2.max():.1f}")


# ===========================================================================
# SECTION 6 — Align all series to the test set and apply daytime mask
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 6 — Aligning to test set and applying daytime mask")
print("=" * 65)

df_all = pd.concat(
    [df_pred["siata_wm2"], df_pred["model_wm2"], gfs_wm2, csg_series],
    axis=1,
    join="inner",
)

# Restrict strictly to the test timestamps
df_all = df_all.loc[df_all.index.isin(test_timestamps)].sort_index()

# Daytime mask: clear_sky_ghi > 0
# Night-time is excluded because: (a) clear_sky = 0 makes CSI undefined,
# and (b) both GFS and the model are forced to 0 at night anyway.
daytime_mask = df_all["clear_sky_ghi"] > 0
df_day = df_all[daytime_mask].copy()

print(f"  Test rows (all hours)   : {len(df_all):,}")
print(f"  Test rows (daytime)     : {len(df_day):,}")
print(f"  Night rows excluded     : {(~daytime_mask).sum():,}")

# Check for NaNs in the aligned daytime frame
nan_counts = df_day[["siata_wm2", "model_wm2", "gfs_wm2"]].isna().sum()
if nan_counts.any():
    print(f"  [WARNING] NaNs present: {nan_counts.to_dict()}")
    df_day = df_day.dropna(subset=["siata_wm2", "model_wm2", "gfs_wm2"])
    print(f"  Test rows after NaN drop: {len(df_day):,}")


# ===========================================================================
# SECTION 7 — Compute metrics
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 7 — Computing metrics")
print("=" * 65)

def compute_metrics(y_true, y_pred):
    """Return a dict of regression metrics."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

y_siata = df_day["siata_wm2"].values
y_gfs   = df_day["gfs_wm2"].values
y_model = df_day["model_wm2"].values

# W/m² space — this is what the training loss operates on
m_gfs   = compute_metrics(y_siata, y_gfs)
m_model = compute_metrics(y_siata, y_model)

# CSI space — dividing by clear_sky_ghi (safe: only daytime rows used)
csg_vals   = df_day["clear_sky_ghi"].values
csi_siata  = y_siata / csg_vals
csi_gfs    = y_gfs   / csg_vals
csi_model  = y_model / csg_vals

m_gfs_csi   = compute_metrics(csi_siata, csi_gfs)
m_model_csi = compute_metrics(csi_siata, csi_model)

# Skill Score in W/m² space (consistent with how training reports it)
MSE_GFS_NEW   = m_gfs["MSE"]
MSE_MODEL_NEW = m_model["MSE"]
skill_score   = 1.0 - MSE_MODEL_NEW / MSE_GFS_NEW

print(f"  GFS  raw   MSE  : {MSE_GFS_NEW:,.3f} W²/m⁴")
print(f"  Model      MSE  : {MSE_MODEL_NEW:,.3f} W²/m⁴")
print(f"  Skill Score     : {skill_score:.6f}")


# ===========================================================================
# SECTION 8 — Diagnostic plots
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 8 — Generating diagnostic plots")
print("=" * 65)

OUT_DIR.mkdir(parents=True, exist_ok=True)

SIATA_s  = df_day["siata_wm2"]
GFS_s    = df_day["gfs_wm2"]
MODEL_s  = df_day["model_wm2"]
RES_GFS  = GFS_s   - SIATA_s   # GFS residuals
RES_MDL  = MODEL_s - SIATA_s   # Model residuals

# ── Plot 1: 2-week time series sample ──────────────────────────────────────
# Pick a 14-day window starting from the middle of the test period
t_mid   = df_day.index[len(df_day) // 2]
t_end   = t_mid + pd.Timedelta(days=14)
sample  = df_day.loc[t_mid:t_end]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(sample.index, sample["siata_wm2"], color="black",     lw=1.5, label="SIATA (observed)")
ax.plot(sample.index, sample["gfs_wm2"],   color="darkorange", lw=1.0, alpha=0.85, label="GFS raw")
ax.plot(sample.index, sample["model_wm2"], color="steelblue",  lw=1.0, alpha=0.85, label="Bi-LSTM + Bahdanau attention")
ax.set_xlabel("Date")
ax.set_ylabel("GHI [W/m²]")
ax.set_title(
    f"Time series sample — 2-week window  ({t_mid.date()} → {t_end.date()})\n"
    "Test set · Daytime hours only"
)
ax.legend(loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout()
p1 = OUT_DIR / "skillscore_01_timeseries_sample.png"
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"  Saved: {p1.name}")

# ── Plot 2: Scatter — GFS raw vs SIATA ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(SIATA_s, GFS_s, s=3, alpha=0.2, color="steelblue", rasterized=True)
vmax = max(float(SIATA_s.max()), float(GFS_s.max())) * 1.05
ax.plot([0, vmax], [0, vmax], "k--", lw=1, label="1:1 line")
ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
ax.set_xlabel("SIATA GHI [W/m²]")
ax.set_ylabel("GFS raw GHI [W/m²]")
ax.set_title(
    f"GFS raw vs SIATA — test set (daytime)\n"
    f"MSE = {MSE_GFS_NEW:,.1f}  |  RMSE = {m_gfs['RMSE']:,.1f} W/m²  |  R² = {m_gfs['R2']:.3f}"
)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
p2 = OUT_DIR / "skillscore_02_scatter_gfs_vs_siata.png"
fig.savefig(p2, dpi=150)
plt.close(fig)
print(f"  Saved: {p2.name}")

# ── Plot 3: Side-by-side residual histograms ────────────────────────────────
# Exclude exactly-zero residuals (overcast/zero-GHI events that both sources
# agree on; including them would artificially sharpen the histogram spike)
res_gfs_nz = RES_GFS[RES_GFS   != 0].values
res_mdl_nz = RES_MDL[RES_MDL   != 0].values

bins = np.linspace(
    min(res_gfs_nz.min(), res_mdl_nz.min()),
    max(res_gfs_nz.max(), res_mdl_nz.max()),
    60,
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

axes[0].hist(res_gfs_nz, bins=bins, color="darkorange", alpha=0.75, edgecolor="saddlebrown")
axes[0].axvline(0, color="black", lw=1.2, ls="--")
axes[0].set_title(
    f"GFS raw\nRMSE = {m_gfs['RMSE']:,.1f} W/m²  |  MAE = {m_gfs['MAE']:,.1f} W/m²"
)
axes[0].set_xlabel("Residual (GFS − SIATA) [W/m²]")
axes[0].set_ylabel("Count")
axes[0].grid(alpha=0.3)

axes[1].hist(res_mdl_nz, bins=bins, color="steelblue", alpha=0.75, edgecolor="navy")
axes[1].axvline(0, color="black", lw=1.2, ls="--")
axes[1].set_title(
    f"Bi-LSTM + Bahdanau attention\nRMSE = {m_model['RMSE']:,.1f} W/m²  |  MAE = {m_model['MAE']:,.1f} W/m²"
)
axes[1].set_xlabel("Residual (model − SIATA) [W/m²]")
axes[1].grid(alpha=0.3)

fig.suptitle("Error histograms — test set (daytime, non-zero residuals)", fontsize=12)
fig.tight_layout()
p3 = OUT_DIR / "skillscore_03_error_histograms.png"
fig.savefig(p3, dpi=150)
plt.close(fig)
print(f"  Saved: {p3.name}")

# ── Plot 4: Monthly MSE comparison ─────────────────────────────────────────
df_day["err_gfs2"]   = RES_GFS  ** 2
df_day["err_model2"] = RES_MDL  ** 2

monthly = (
    df_day[["err_gfs2", "err_model2"]]
    .groupby(df_day.index.to_period("M"))
    .mean()
)
monthly.index = monthly.index.astype(str)   # e.g. "2024-06"

x     = np.arange(len(monthly))
width = 0.35

fig, ax = plt.subplots(figsize=(max(10, len(monthly) * 0.65), 5))
ax.bar(x - width / 2, monthly["err_gfs2"],   width, label="GFS raw",                color="darkorange", alpha=0.85)
ax.bar(x + width / 2, monthly["err_model2"], width, label="Bi-LSTM + Bahdanau attn", color="steelblue",  alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(monthly.index, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Month")
ax.set_ylabel("Mean Squared Error [W²/m⁴]")
ax.set_title("Monthly MSE — GFS raw vs our model  (test set, daytime hours)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
p4 = OUT_DIR / "skillscore_04_monthly_mse.png"
fig.savefig(p4, dpi=150)
plt.close(fig)
print(f"  Saved: {p4.name}")


# ===========================================================================
# SECTION 9 — Comparison table
# ===========================================================================
def pct_change(a, b):
    """Percentage change from baseline a to value b."""
    return (b - a) / a * 100.0

print()
print("=" * 65)
print("  SKILL SCORE VERIFICATION  –  sym18 + Bahdanau attention")
print("=" * 65)
print(f"  Test set size (all hours)  : {len(df_all):,}")
print(f"  Test set size (daytime)    : {len(df_day):,}")
print()
print(
    f"  {'Metric':<22} {'GFS raw':>14} {'Our model':>14} {'Change':>10}"
)
print(f"  {'─' * 22} {'─' * 14} {'─' * 14} {'─' * 10}")
print(
    f"  {'MSE  [W²/m⁴]':<22}"
    f" {m_gfs['MSE']:>14,.1f}"
    f" {m_model['MSE']:>14,.1f}"
    f" {pct_change(m_gfs['MSE'],  m_model['MSE']):>9.1f}%"
)
print(
    f"  {'RMSE [W/m²]':<22}"
    f" {m_gfs['RMSE']:>14,.2f}"
    f" {m_model['RMSE']:>14,.2f}"
    f" {pct_change(m_gfs['RMSE'], m_model['RMSE']):>9.1f}%"
)
print(
    f"  {'MAE  [W/m²]':<22}"
    f" {m_gfs['MAE']:>14,.2f}"
    f" {m_model['MAE']:>14,.2f}"
    f" {pct_change(m_gfs['MAE'],  m_model['MAE']):>9.1f}%"
)
print(
    f"  {'R²':<22}"
    f" {m_gfs['R2']:>14.4f}"
    f" {m_model['R2']:>14.4f}"
    f" {pct_change(m_gfs['R2'],   m_model['R2']):>9.1f}%"
)
print()
print(
    f"  {'─' * 62}\n"
    f"  Skill Score  SS = 1 − MSE_model / MSE_GFS  =  {skill_score:.6f}\n"
    f"  {'─' * 62}"
)
print()
print(f"  MSE in CSI space (dimensionless, daytime only):")
print(f"    GFS raw   : {m_gfs_csi['MSE']:.6f}")
print(f"    Our model : {m_model_csi['MSE']:.6f}")
print()
print(f"  {'─' * 62}")
print(f"  OLD MSE_BASELINE_R (config.py)  : {MSE_BASELINE_R:.6f}")
print(f"  NEW MSE_BASELINE_R (this run)   : {MSE_GFS_NEW:.6f}")
diff_pct = pct_change(MSE_BASELINE_R, MSE_GFS_NEW)
print(f"  Difference                      : {diff_pct:+.2f}%")
print()
print(f"  ► To update config.py line 294, replace with:")
print(f'    MSE_BASELINE_R = {MSE_GFS_NEW:.6f}')
print("=" * 65)


# ===========================================================================
# SECTION 10 — Save text report with the recommended update
# ===========================================================================
report_path = OUT_DIR / "skillscore_baseline_update.txt"
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("SKILL SCORE VERIFICATION — sym18 + Bahdanau attention\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Predictions CSV  : {PRED_CSV_FILE}\n")
    fh.write(f"Test index file  : {TEST_IDX_FILE}\n\n")
    fh.write(f"Test rows (all hours)   : {len(df_all):,}\n")
    fh.write(f"Test rows (daytime)     : {len(df_day):,}\n\n")
    fh.write(f"{'Metric':<24} {'GFS raw':>14} {'Our model':>14}\n")
    fh.write(f"{'─'*24} {'─'*14} {'─'*14}\n")
    fh.write(f"{'MSE  [W2/m4]':<24} {m_gfs['MSE']:>14,.3f} {m_model['MSE']:>14,.3f}\n")
    fh.write(f"{'RMSE [W/m2]':<24} {m_gfs['RMSE']:>14,.3f} {m_model['RMSE']:>14,.3f}\n")
    fh.write(f"{'MAE  [W/m2]':<24} {m_gfs['MAE']:>14,.3f} {m_model['MAE']:>14,.3f}\n")
    fh.write(f"{'R2':<24} {m_gfs['R2']:>14.4f} {m_model['R2']:>14.4f}\n\n")
    fh.write(f"Skill Score  SS = 1 - MSE_model / MSE_GFS  =  {skill_score:.6f}\n\n")
    fh.write(f"OLD MSE_BASELINE_R (config.py) : {MSE_BASELINE_R:.6f}\n")
    fh.write(f"NEW MSE_BASELINE_R (this run)  : {MSE_GFS_NEW:.6f}\n")
    fh.write(f"Difference                     : {diff_pct:+.2f}%\n\n")
    fh.write("To update config.py, replace the MSE_BASELINE_R line with:\n")
    fh.write(f"  MSE_BASELINE_R = {MSE_GFS_NEW:.6f}\n")

print(f"\nText report saved to: {report_path}")
print("Done.")
