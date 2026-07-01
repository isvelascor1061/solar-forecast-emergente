#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_postprocessor.py
=======================
Hybrid prediction post-processor: fixes the ~423 zero-prediction failures
at transition hours 8-10 by substituting GFS directly when the BiLSTM
collapses to near-zero but GFS CSI indicates clear sky.

Correction rule (applied per test-set timestamp):
    local_hour = UTC timestamp - 5 h   (Colombia = UTC-5)
    if local_hour in {8, 9, 10}
    AND y_pred < 10 W/m^2
    AND gfs_csi > 0.5:
        y_pred_hybrid = gfs_csi * clear_sky_ghi    (use GFS)
    else:
        y_pred_hybrid = y_pred                      (keep BiLSTM)

Outputs
-------
  Evaluation_after_LSTM/hybrid/plot1_scatter_3panel.png
  Evaluation_after_LSTM/hybrid/plot2_mae_by_hour.png
  Console comparison table (GFS raw | BiLSTM original | Hybrid)

Usage
-----
    set PYTHONPATH=C:\\Users\\isabe\\Projects\\codigors\\carpetasdetrabajo
    python _4_LSTM_modules/Main_execution_files/hybrid_postprocessor.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import (
    LAUNCH_TIMES,
    FEAT_KC_TEMPLATE,
    VAR_GFS_CSI_TEMPLATE,
    CSI_GHI_FILE,
    CSI_VAR_NAME,
    MSE_BASELINE_R,
)

# ── Fixed model run (specified by user) ───────────────────────────────────────
PRED_CSV = (
    ROOT
    / "_4_LSTM_modules/_runs/4launch_multfeat_sym"
    / "4launch_Multfeat_sym18_clim79_BiLSTM_attn_20260627_081750"
    / "pred_real.csv"
)

OUT_DIR = ROOT / "_4_LSTM_modules/Evaluation_after_LSTM/hybrid"

# ── Correction thresholds ─────────────────────────────────────────────────────
HOUR_WINDOW  = {8, 9, 10}   # local (UTC-5) hours where failures cluster
PRED_THR     = 10.0         # W/m^2 — "near-zero" model prediction
GFS_CSI_THR  = 0.5          # GFS CSI above this = clear sky

# Timestamps in pred_real.csv are already in Colombia local time (UTC-5),
# consistent with the GFS observation_time index.  No offset needed.


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, R^2, SkillScore (daytime W/m^2 space)."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "RMSE":       float(np.sqrt(mse)),
        "MAE":        float(mean_absolute_error(y_true, y_pred)),
        "R2":         float(r2_score(y_true, y_pred)),
        "SkillScore": float(1.0 - mse / MSE_BASELINE_R),
    }


# ── Best-lead GFS CSI series ──────────────────────────────────────────────────

def load_best_lead_series(
    lt_list: list,
    file_template: str,
    var_template: str,
) -> pd.Series:
    """
    For each observation_time, select the GFS value from the launch time
    with the shortest strictly positive lead time.

    Parameters
    ----------
    lt_list       : e.g. ["0100", "0700", "1300", "1900"]
    file_template : path string with {LT} placeholder
    var_template  : variable name string with {LT} placeholder

    Returns
    -------
    pd.Series indexed by observation_time.
    """
    val_frames  = []
    lead_frames = []

    for lt in lt_list:
        nc_path  = ROOT / file_template.format(LT=lt)
        var_name = var_template.format(LT=lt)

        ds = xr.open_dataset(nc_path, engine="h5netcdf")
        da = ds[var_name]

        # Squeeze all singleton spatial dims
        for dim in list(da.dims):
            if dim != "observation_time":
                da = da.isel({dim: 0})

        ser_val = da.to_series()

        launch_ser       = pd.to_datetime(ds["launch_time"].to_series())
        launch_ser.index = ser_val.index
        lead             = ser_val.index.to_series() - launch_ser
        lead[lead <= pd.Timedelta(0)] = pd.NaT   # discard non-positive leads

        val_frames.append(ser_val.rename(lt))
        lead_frames.append(lead.rename(lt))
        ds.close()

        n_valid = int((~lead.isna()).sum())
        print(f"  LT {lt}: {len(ser_val):,} rows | positive leads: {n_valid:,}")

    val_df  = pd.concat(val_frames,  axis=1)
    lead_df = pd.concat(lead_frames, axis=1)

    idx_min = lead_df.idxmin(axis=1)
    col_pos = val_df.columns.get_indexer(idx_min)
    row_pos = np.arange(len(val_df))
    best    = val_df.to_numpy()[row_pos, col_pos]

    series = pd.Series(best, index=val_df.index).dropna().sort_index()
    print(f"  Best-lead series: {len(series):,} valid hours")
    return series


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  HYBRID POST-PROCESSOR — BiLSTM + GFS Correction")
print("=" * 62)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Load BiLSTM predictions
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 1 — Loading BiLSTM predictions")
if not PRED_CSV.exists():
    sys.exit(f"[ERROR] pred_real.csv not found:\n  {PRED_CSV}")

df_pred = (
    pd.read_csv(PRED_CSV, parse_dates=["time"])
    .set_index("time")
    .sort_index()
)
print(f"  File        : {PRED_CSV.parent.name}")
print(f"  Rows loaded : {len(df_pred):,}")
print(f"  Columns     : {list(df_pred.columns)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Build GFS CSI best-lead series
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 2 — Building GFS CSI best-lead series")
gfs_csi_full = load_best_lead_series(
    lt_list       = LAUNCH_TIMES,
    file_template = FEAT_KC_TEMPLATE,
    var_template  = VAR_GFS_CSI_TEMPLATE,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Load clear-sky GHI
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 3 — Loading clear-sky GHI")
ds_csg  = xr.open_dataset(ROOT / CSI_GHI_FILE, engine="h5netcdf")
da_csg  = ds_csg[CSI_VAR_NAME].squeeze()
for dim in list(da_csg.dims):
    if dim not in ("observation_time", "time"):
        da_csg = da_csg.isel({dim: 0})
csg_full = da_csg.to_series().rename("clear_sky_ghi")
ds_csg.close()
print(f"  Clear-sky GHI series: {len(csg_full):,} hourly values")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Align on test timestamps
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 4 — Aligning GFS and clear-sky onto test timestamps")

df = df_pred.copy()
df["gfs_csi"]       = gfs_csi_full.reindex(df.index)
df["clear_sky_ghi"] = csg_full.reindex(df.index)
df["gfs_ghi"]       = df["gfs_csi"] * df["clear_sky_ghi"]

n_total    = len(df)
n_gfs_nan  = int(df["gfs_csi"].isna().sum())
n_csg_nan  = int(df["clear_sky_ghi"].isna().sum())
print(f"  Test rows           : {n_total:,}")
print(f"  GFS CSI join gaps   : {n_gfs_nan}  ({100*n_gfs_nan/n_total:.1f}%)")
print(f"  Clear-sky join gaps : {n_csg_nan}  ({100*n_csg_nan/n_total:.1f}%)")
if n_gfs_nan / n_total > 0.05:
    print("  WARNING: >5% GFS join gaps — check FEAT_KC_TEMPLATE alignment")

# Drop rows where any key column is NaN (ensures fair 3-way comparison)
df.dropna(subset=["y_true", "y_pred", "gfs_csi", "clear_sky_ghi"], inplace=True)
print(f"  Rows after dropna   : {len(df):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Apply hybrid correction rule
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 5 — Applying hybrid correction rule")

# Timestamps are already in local time — use hour directly
local_hours = df.index.hour

correction_mask = (
    local_hours.isin(HOUR_WINDOW)
    & (df["y_pred"] < PRED_THR)
    & (df["gfs_csi"] > GFS_CSI_THR)
)

df["y_pred_hybrid"] = df["y_pred"].copy()
df.loc[correction_mask, "y_pred_hybrid"] = df.loc[correction_mask, "gfs_ghi"]

n_corrected = int(correction_mask.sum())
print(f"  Correction rule     : hour in {sorted(HOUR_WINDOW)}, "
      f"y_pred < {PRED_THR}, gfs_csi > {GFS_CSI_THR}")
print(f"  Corrections applied : {n_corrected}")
if n_corrected > 0:
    print(f"  Mean y_true (corrected cases)   : "
          f"{df.loc[correction_mask, 'y_true'].mean():.1f} W/m^2")
    print(f"  Mean gfs_ghi used as replacement: "
          f"{df.loc[correction_mask, 'gfs_ghi'].mean():.1f} W/m^2")
    hour_dist = local_hours[correction_mask].value_counts().sort_index()
    print(f"  Hour breakdown:")
    for h, c in hour_dist.items():
        print(f"    {h:02d}h : {c}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Daytime mask and metrics
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 6 — Computing metrics (daytime only, clear_sky_ghi > 0)")

daytime = df["clear_sky_ghi"] > 0
df_day  = df[daytime].copy()
print(f"  Daytime rows : {len(df_day):,}   (night excluded: {(~daytime).sum():,})")

y_true       = df_day["y_true"].values
y_gfs        = df_day["gfs_ghi"].values
y_lstm       = df_day["y_pred"].values
y_hybrid     = df_day["y_pred_hybrid"].values

m_gfs    = compute_metrics(y_true, y_gfs)
m_lstm   = compute_metrics(y_true, y_lstm)
m_hybrid = compute_metrics(y_true, y_hybrid)

print(f"  MSE_BASELINE_R : {MSE_BASELINE_R:,.3f} W^2/m^4")
print()
print(f"  {'Model':<18} {'RMSE_day':>10}  {'MAE_day':>8}  {'R2':>8}  {'SkillScore':>11}")
print(f"  {'-'*18} {'-'*10}  {'-'*8}  {'-'*8}  {'-'*11}")
for label, m in [("GFS raw", m_gfs), ("BiLSTM original", m_lstm), ("Hybrid", m_hybrid)]:
    print(f"  {label:<18} {m['RMSE']:>10.1f}  {m['MAE']:>8.1f}  "
          f"{m['R2']:>8.3f}  {m['SkillScore']:>+11.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Plot 1: 3-panel scatter
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 7 — Generating plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VMAX = 1000.0

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

panels = [
    ("GFS raw",          y_gfs,    m_gfs,    "darkorange"),
    ("BiLSTM original",  y_lstm,   m_lstm,   "steelblue"),
    ("Hybrid",           y_hybrid, m_hybrid, "forestgreen"),
]

for ax, (label, y_hat, m, color) in zip(axes, panels):
    ax.scatter(y_true, y_hat, s=1.5, alpha=0.25, color=color, rasterized=True)
    ax.plot([0, VMAX], [0, VMAX], "r--", lw=1.2, label="1:1 line")
    ax.set_xlim(0, VMAX)
    ax.set_ylim(0, VMAX)
    ax.set_xlabel("SIATA observed GHI [W/m^2]")
    ax.set_ylabel("Predicted GHI [W/m^2]")
    ax.set_title(
        f"{label}\nRMSE = {m['RMSE']:.1f} W/m^2   "
        f"SkillScore = {m['SkillScore']:+.3f}",
        fontsize=10,
    )
    ax.grid(alpha=0.3)

fig.suptitle(
    f"GFS vs BiLSTM vs Hybrid — daytime test set   "
    f"({n_corrected} corrections applied at hours {sorted(HOUR_WINDOW)})",
    fontsize=11,
)
fig.tight_layout()
p1 = OUT_DIR / "plot1_scatter_3panel.png"
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"  Saved: {p1}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Plot 2: MAE by local hour (6-19)
# ══════════════════════════════════════════════════════════════════════════════

# Timestamps already in local time — use hour directly
local_hours_day = df_day.index.hour
df_day = df_day.copy()
df_day["local_hour"] = local_hours_day

hour_range = list(range(6, 20))

mae_gfs    = {}
mae_lstm   = {}
mae_hybrid = {}

for h in hour_range:
    mask = df_day["local_hour"] == h
    if mask.sum() == 0:
        mae_gfs[h]    = np.nan
        mae_lstm[h]   = np.nan
        mae_hybrid[h] = np.nan
        continue
    yt = df_day.loc[mask, "y_true"].values
    mae_gfs[h]    = float(np.mean(np.abs(yt - df_day.loc[mask, "gfs_ghi"].values)))
    mae_lstm[h]   = float(np.mean(np.abs(yt - df_day.loc[mask, "y_pred"].values)))
    mae_hybrid[h] = float(np.mean(np.abs(yt - df_day.loc[mask, "y_pred_hybrid"].values)))

hours_arr   = np.array(hour_range)
gfs_arr     = np.array([mae_gfs[h]    for h in hour_range])
lstm_arr    = np.array([mae_lstm[h]   for h in hour_range])
hybrid_arr  = np.array([mae_hybrid[h] for h in hour_range])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hours_arr, gfs_arr,    "o-", color="darkorange",  lw=2,   label="GFS raw",         markersize=5)
ax.plot(hours_arr, lstm_arr,   "s-", color="steelblue",   lw=2,   label="BiLSTM original", markersize=5)
ax.plot(hours_arr, hybrid_arr, "^-", color="forestgreen", lw=2.5, label="Hybrid",          markersize=6)

for h in HOUR_WINDOW:
    ax.axvline(h, ls="--", lw=1.2, color="gray", alpha=0.6)
    ax.text(h, max(np.nanmax(gfs_arr), np.nanmax(lstm_arr)) * 0.98,
            f"{h}h", ha="center", va="top", fontsize=8, color="gray")

ax.set_xlabel("Local hour (UTC-5, Colombia)")
ax.set_ylabel("MAE [W/m^2]")
ax.set_title("Mean Absolute Error by Hour — Hybrid correction")
ax.set_xticks(hours_arr)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()

p2 = OUT_DIR / "plot2_mae_by_hour.png"
fig.savefig(p2, dpi=150)
plt.close(fig)
print(f"  Saved: {p2}")

print("\nDone.")
