#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upper_bound_analysis.py
=======================
Upper-bound analysis: how much would SkillScore improve if the
zero-prediction failures (hours 8-10) were perfectly corrected?

The script takes the best model's predictions, identifies the 424 failure
cases (y_pred < 10 W/m² but y_true > 200 W/m²), replaces them with the
ground truth, and recomputes all metrics.  Everything else — the remaining
~5,000 test predictions — stays exactly as the model produced them.

Model used
----------
  4launch_Multfeat_sym18_clim79_BiLSTM_attn_*  (79-feature, standard MSE)
  The most recently modified folder matching that pattern is selected.
  SkillScore in report.txt shows +0.292 (computed on all hours including
  night — misleading). The honest daytime-only SkillScore is -0.292, which
  is what this script reproduces in the "Original" column.

Outputs
-------
  Evaluation_after_LSTM/upper_bound/plot1_scatter_comparison.png
  Evaluation_after_LSTM/upper_bound/plot2_metrics_bar_chart.png
  Console summary table

Usage
-----
    set PYTHONPATH=C:\\Users\\isabe\\Projects\\codigors\\carpetasdetrabajo
    python _4_LSTM_modules/Main_execution_files/upper_bound_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Project root (two levels up from this file) ───────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import (
    CSI_GHI_FILE,       # Ineichen clear-sky NetCDF → daytime mask
    CSI_VAR_NAME,       # variable name inside that file
    MSE_BASELINE_R,     # GFS raw MSE used as SkillScore denominator
)

# ── Paths ─────────────────────────────────────────────────────────────────────
RUNS_DIR = ROOT / "_4_LSTM_modules/_runs/4launch_multfeat_sym"
OUT_DIR  = ROOT / "_4_LSTM_modules/Evaluation_after_LSTM/upper_bound"

# ── Failure thresholds (must match diagnose_zero_predictions.py) ──────────────
PRED_THR = 10.0    # W/m² — below this the prediction is considered "zero"
TRUE_THR = 200.0   # W/m² — above this the true value is considered "bright"


# ── Helper ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, R², SkillScore — all in W/m² space."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE":       float(np.sqrt(mse)),
        "MAE":        float(mean_absolute_error(y_true, y_pred)),
        "R2":         float(r2_score(y_true, y_pred)),
        "SkillScore": float(1.0 - mse / MSE_BASELINE_R),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Locate and load predictions
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 62)
print("  UPPER BOUND ANALYSIS — Zero-Failure Correction")
print("=" * 62)

# Select the best model: clim79 with standard MSE loss (no dayonly/asymloss).
# Pick the most recently modified pred_real.csv among matching folders.
candidates = sorted(
    RUNS_DIR.glob("4launch_Multfeat_sym18_clim79_BiLSTM_attn_*/pred_real.csv"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not candidates:
    sys.exit(
        "[ERROR] No pred_real.csv found matching:\n"
        "  4launch_Multfeat_sym18_clim79_BiLSTM_attn_*/pred_real.csv\n"
        f"  in {RUNS_DIR}"
    )

PRED_CSV = candidates[0]
print(f"\nSection 1 — Loading predictions")
print(f"  Model folder : {PRED_CSV.parent.name}")
print(f"  File         : {PRED_CSV.name}")

df_pred = (
    pd.read_csv(PRED_CSV, parse_dates=["time"])
    .set_index("time")
    .sort_index()
)
print(f"  Rows loaded  : {len(df_pred):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Load clear-sky GHI and apply daytime mask
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 2 — Loading clear-sky GHI and applying daytime mask")

ds_csg = xr.open_dataset(ROOT / CSI_GHI_FILE, engine="h5netcdf")
da_csg = ds_csg[CSI_VAR_NAME].squeeze()

# Reduce any remaining spatial dimensions to a scalar per timestep
for dim in list(da_csg.dims):
    if dim not in ("observation_time", "time"):
        da_csg = da_csg.isel({dim: 0})

csg_series = da_csg.to_series().rename("clear_sky_ghi")
ds_csg.close()

# Inner join: keep only timestamps present in both pred and clear-sky
df = df_pred.join(csg_series, how="inner")

# Daytime mask: clear_sky_ghi > 0
daytime_mask = df["clear_sky_ghi"] > 0
df_day = df[daytime_mask].copy()

print(f"  Clear-sky series : {len(csg_series):,} hourly values")
print(f"  Test rows (total): {len(df):,}")
print(f"  Test rows (day)  : {len(df_day):,}   (clear_sky_ghi > 0)")
print(f"  Night rows       : {(~daytime_mask).sum():,}  (excluded)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Identify zero-prediction failures and build simulated predictions
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 3 — Identifying failures and building simulated predictions")

failure_mask = (
    (df_day["y_pred"] < PRED_THR) &
    (df_day["y_true"] > TRUE_THR)
)
n_fail  = int(failure_mask.sum())
n_day   = len(df_day)
n_total = len(df)

# Simulated prediction: perfect only for the failure cases
df_day["y_pred_sim"] = df_day["y_pred"].copy()
df_day.loc[failure_mask, "y_pred_sim"] = df_day.loc[failure_mask, "y_true"]

print(f"  Failure criteria : y_pred < {PRED_THR} W/m²  AND  y_true > {TRUE_THR} W/m²")
print(f"  Failures found   : {n_fail}  "
      f"({100 * n_fail / n_total:.1f}% of all test rows, "
      f"{100 * n_fail / n_day:.1f}% of daytime rows)")
print(f"  Mean y_true (failures) : {df_day.loc[failure_mask, 'y_true'].mean():.1f} W/m²")

# Hour distribution of failures (informational)
fail_hours = df_day.loc[failure_mask].index.hour.value_counts().sort_index()
print(f"  Hour distribution of failures:")
for h, c in fail_hours.items():
    print(f"    {h:02d}h : {c}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Compute metrics for both versions
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 4 — Computing metrics (daytime only)")

y_true    = df_day["y_true"].values
y_pred    = df_day["y_pred"].values
y_pred_sim = df_day["y_pred_sim"].values

m_orig = compute_metrics(y_true, y_pred)
m_sim  = compute_metrics(y_true, y_pred_sim)

print(f"  MSE_BASELINE_R (from config) : {MSE_BASELINE_R:,.3f} W²/m⁴")
print(f"  Original  SkillScore : {m_orig['SkillScore']:+.4f}")
print(f"  Simulated SkillScore : {m_sim['SkillScore']:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Plots
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nSection 5 — Generating plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

vmax = max(float(y_true.max()), float(y_pred.max()), float(y_pred_sim.max())) * 1.05

# ── Plot 1: Side-by-side scatter ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

# Non-failure and failure point indices (for left panel colouring)
fail_idx     = np.where(failure_mask.values)[0]
non_fail_idx = np.where(~failure_mask.values)[0]

# Left — original predictions
ax = axes[0]
ax.scatter(
    y_true[non_fail_idx], y_pred[non_fail_idx],
    s=2, alpha=0.25, color="steelblue", rasterized=True, label="Normal predictions"
)
ax.scatter(
    y_true[fail_idx], y_pred[fail_idx],
    s=12, alpha=0.6, color="crimson", rasterized=True,
    label=f"Zero-pred failures (n={n_fail})"
)
ax.plot([0, vmax], [0, vmax], "r--", lw=1.2, label="1:1 line")
ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
ax.set_xlabel("SIATA observed GHI [W/m²]")
ax.set_ylabel("Predicted GHI [W/m²]")
ax.set_title(
    f"Original predictions\n"
    f"RMSE = {m_orig['RMSE']:.1f} W/m²   SkillScore = {m_orig['SkillScore']:+.3f}"
)
ax.legend(loc="upper left", fontsize=8, markerscale=2)
ax.grid(alpha=0.3)

# Right — simulated (failures corrected)
ax = axes[1]
ax.scatter(
    y_true[non_fail_idx], y_pred_sim[non_fail_idx],
    s=2, alpha=0.25, color="steelblue", rasterized=True, label="Normal predictions"
)
ax.scatter(
    y_true[fail_idx], y_pred_sim[fail_idx],
    s=12, alpha=0.6, color="forestgreen", rasterized=True,
    label=f"Corrected failures (n={n_fail})\n→ lie on 1:1 line"
)
ax.plot([0, vmax], [0, vmax], "r--", lw=1.2, label="1:1 line")
ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
ax.set_xlabel("SIATA observed GHI [W/m²]")
ax.set_title(
    f"Simulated (failures perfectly corrected)\n"
    f"RMSE = {m_sim['RMSE']:.1f} W/m²   SkillScore = {m_sim['SkillScore']:+.3f}"
)
ax.legend(loc="upper left", fontsize=8, markerscale=2)
ax.grid(alpha=0.3)

fig.suptitle(
    f"Upper-Bound Analysis — {PRED_CSV.parent.name}\n"
    f"Daytime test set  |  {n_fail} failures corrected out of {n_day:,} daytime rows",
    fontsize=11
)
fig.tight_layout()
p1 = OUT_DIR / "plot1_scatter_comparison.png"
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"  Saved: {p1}")

# ── Plot 2: Metric bar chart (2×2 subplots) ───────────────────────────────────
metric_keys   = ["RMSE", "MAE", "R2", "SkillScore"]
metric_labels = ["RMSE [W/m²]", "MAE [W/m²]", "R²", "SkillScore"]
colors        = ["steelblue", "forestgreen"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

x = np.array([0, 1])
bar_width = 0.4

for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
    ax = axes[i]
    orig_val = m_orig[key]
    sim_val  = m_sim[key]

    bars = ax.bar(x, [orig_val, sim_val], width=bar_width, color=colors, alpha=0.85,
                  edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Original", "Simulated"], fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # Annotate bar tops
    for bar, val in zip(bars, [orig_val, sim_val]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + abs(bar.get_height()) * 0.01,
            f"{val:+.3f}" if key in ("R2", "SkillScore") else f"{val:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Draw a horizontal line at the original value for easy comparison
    ax.axhline(orig_val, color="steelblue", lw=0.8, ls="--", alpha=0.5)

fig.suptitle(
    f"Metric Comparison: Original vs Simulated (failures corrected)\n"
    f"{n_fail} zero-prediction failures replaced with ground truth",
    fontsize=11
)
fig.tight_layout()
p2 = OUT_DIR / "plot2_metrics_bar_chart.png"
fig.savefig(p2, dpi=150)
plt.close(fig)
print(f"  Saved: {p2}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Summary printout
# ══════════════════════════════════════════════════════════════════════════════

def pct_change(orig, new):
    """Percentage change from orig to new."""
    if orig == 0:
        return float("nan")
    return (new - orig) / abs(orig) * 100.0

ss_positive = m_sim["SkillScore"] > 0

print()
print("=" * 62)
print("  UPPER BOUND ANALYSIS — Zero-Failure Correction")
print("=" * 62)
print(f"  Model    : {PRED_CSV.parent.name}")
print(f"  Failures : {n_fail}  "
      f"({100 * n_fail / n_total:.1f}% of all test rows, "
      f"{100 * n_fail / n_day:.1f}% of daytime rows)")
print()
print(f"  {'Metric':<14} {'Original':>12}  {'Simulated':>12}  {'Change':>12}")
print(f"  {'-'*14} {'-'*12}  {'-'*12}  {'-'*12}")
print(f"  {'RMSE_day':<14} {m_orig['RMSE']:>12.1f}  {m_sim['RMSE']:>12.1f}"
      f"  {pct_change(m_orig['RMSE'], m_sim['RMSE']):>+11.1f}%")
print(f"  {'MAE_day':<14} {m_orig['MAE']:>12.1f}  {m_sim['MAE']:>12.1f}"
      f"  {pct_change(m_orig['MAE'], m_sim['MAE']):>+11.1f}%")
print(f"  {'R2_day':<14} {m_orig['R2']:>12.3f}  {m_sim['R2']:>12.3f}"
      f"  {m_sim['R2'] - m_orig['R2']:>+12.3f}")
print(f"  {'SkillScore':<14} {m_orig['SkillScore']:>12.3f}  {m_sim['SkillScore']:>12.3f}"
      f"  {m_sim['SkillScore'] - m_orig['SkillScore']:>+12.3f}")
print("=" * 62)
print()
print(f"  Interpretation: if these {n_fail} hours were perfectly corrected,")
print(f"  SkillScore would be {m_sim['SkillScore']:+.3f} "
      f"[{'POSITIVE' if ss_positive else 'still negative'}].")
print("=" * 62)
