#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_before_after_clim.py
============================
Comparative analysis: BiLSTM model before vs after adding climatological features.

Model A (baseline, 69 features): 4launch_Multfeat_sym18_BiLSTM_attn_20260428_121958
Model B (climatology, 79 features): most recent 4launch_Multfeat_sym18_clim79_BiLSTM_attn_*

Generates 4 plots saved to:
    _4_LSTM_modules/Evaluation_after_LSTM/before_after_clim/
"""

import sys
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import CSI_GHI_FILE, VAR_REF_CLEARSKY_GHI

# ── Constants ─────────────────────────────────────────────────────────────────
MSE_BASELINE_R = 40761.472609   # GFS raw MSE (daytime-only), used for SkillScore
RUNS_DIR       = ROOT / "_4_LSTM_modules/_runs/4launch_multfeat_sym"
OUT_DIR        = ROOT / "_4_LSTM_modules/Evaluation_after_LSTM/before_after_clim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model paths ───────────────────────────────────────────────────────────────
PATH_A = RUNS_DIR / "4launch_Multfeat_sym18_BiLSTM_attn_20260428_121958" / "pred_real.csv"

clim_folders = sorted(
    glob.glob(str(RUNS_DIR / "4launch_Multfeat_sym18_clim79_BiLSTM_attn_*"))
)
if not clim_folders:
    raise FileNotFoundError("No folder matching 4launch_Multfeat_sym18_clim79_BiLSTM_attn_* found.")
PATH_B = Path(clim_folders[-1]) / "pred_real.csv"

print(f"Model A (baseline):    {PATH_A.parent.name}")
print(f"Model B (climatology): {PATH_B.parent.name}")

# ── Load predictions ──────────────────────────────────────────────────────────
df_a = pd.read_csv(PATH_A, parse_dates=["time"]).set_index("time")
df_b = pd.read_csv(PATH_B, parse_dates=["time"]).set_index("time")

# Merge on common timestamps
df = df_a.join(df_b, how="inner", lsuffix="_a", rsuffix="_b")
print(f"Common test samples:   {len(df)}")

# ── Load Clear-Sky GHI for daytime mask ───────────────────────────────────────
csi_path = ROOT / CSI_GHI_FILE
ds_csi   = xr.open_dataset(str(csi_path), engine="h5netcdf")
da_csi   = ds_csi[VAR_REF_CLEARSKY_GHI]

# Identify the time dimension (may be "time" or "observation_time")
time_dim = next((d for d in da_csi.dims
                 if "time" in d.lower() or "observ" in d.lower()), da_csi.dims[0])

# Average over all non-time spatial dims
spatial_dims = [d for d in da_csi.dims if d != time_dim]
if spatial_dims:
    da_csi = da_csi.mean(dim=spatial_dims)

# Select CSI values at test timestamps (nearest, 30-min tolerance)
timestamps = df.index
csi_vals = da_csi.sel({time_dim: timestamps}, method="nearest",
                      tolerance=pd.Timedelta("30min"))
df["clear_sky_ghi"] = csi_vals.values

# Daytime mask
day_mask = df["clear_sky_ghi"] > 0
df_day   = df[day_mask].copy()
print(f"Daytime samples:       {len(df_day)}")

y_true_a = df_day["y_true_a"].values
y_pred_a = df_day["y_pred_a"].values
y_true_b = df_day["y_true_b"].values
y_pred_b = df_day["y_pred_b"].values

# ── Metrics helper ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    ss   = 1.0 - mse / MSE_BASELINE_R
    return dict(RMSE=rmse, MAE=mae, R2=r2, SS=ss, MSE=mse)

m_a = compute_metrics(y_true_a, y_pred_a)
m_b = compute_metrics(y_true_b, y_pred_b)

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 58)
print(f"{'Metric':<18} {'Model A (baseline)':>18} {'Model B (clim79)':>18}")
print("=" * 58)
for key in ["RMSE", "MAE", "R2", "SS"]:
    va, vb = m_a[key], m_b[key]
    pct = (vb - va) / abs(va) * 100 if va != 0 else float("nan")
    sign = "+" if pct >= 0 else ""
    print(f"{key:<18} {va:>18.4f} {vb:>18.4f}   ({sign}{pct:.1f}%)")
print("=" * 58)

# ── Helper: percent change label ─────────────────────────────────────────────
def pct_label(va, vb):
    pct = (vb - va) / abs(va) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Scatter plots side by side (daytime only)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
fig.suptitle(
    "Predicted vs Observed GHI — Before and After Climatological Features",
    fontsize=13, fontweight="bold", y=1.01
)

for ax, y_true, y_pred, m, subtitle in [
    (axes[0], y_true_a, y_pred_a, m_a, "Without climatology (69 features)"),
    (axes[1], y_true_b, y_pred_b, m_b, "With climatology (79 features)"),
]:
    hb = ax.hexbin(y_true, y_pred, gridsize=60, cmap="YlOrRd",
                   extent=[0, 1000, 0, 1000], mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")
    ax.plot([0, 1000], [0, 1000], "r-", lw=1.5, label="Perfect prediction")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_xlabel("Observed GHI (W/m²)")
    ax.set_ylabel("Predicted GHI (W/m²)")
    ax.set_title(subtitle, fontsize=11)
    ax.text(
        0.04, 0.96,
        f"RMSE = {m['RMSE']:.1f} W/m²\nR² = {m['R2']:.3f}\nSkillScore = {m['SS']:.3f}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )
    # Annotate the horizontal cluster of near-zero predictions if present
    zero_preds = np.sum(y_pred < 10)
    if zero_preds > 50:
        ax.annotate(
            f"Zero-prediction cluster\n({zero_preds} pts)",
            xy=(500, 5), xytext=(500, 120),
            arrowprops=dict(arrowstyle="->", color="navy"),
            fontsize=8, color="navy", ha="center"
        )
    ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
fig.savefig(OUT_DIR / "plot1_scatter_before_after.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot1_scatter_before_after.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Error histograms overlaid (daytime, non-zero true)
# ══════════════════════════════════════════════════════════════════════════════
nz = y_true_a > 0
res_a = y_pred_a[nz] - y_true_a[nz]
res_b = y_pred_b[nz] - y_true_b[nz]

bins = np.arange(-800, 841, 40)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(res_a, bins=bins, alpha=0.6, color="steelblue",
        label=f"Model A — MAE={m_a['MAE']:.1f}  RMSE={m_a['RMSE']:.1f} W/m²")
ax.hist(res_b, bins=bins, alpha=0.6, color="darkorange",
        label=f"Model B — MAE={m_b['MAE']:.1f}  RMSE={m_b['RMSE']:.1f} W/m²")
ax.axvline(0, color="black", linestyle="--", lw=1.2, label="Zero error")
ax.set_xlabel("Residual  pred − true  (W/m²)")
ax.set_ylabel("Count")
ax.set_title("Error Distribution — Before vs After Climatological Features", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "plot2_error_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot2_error_histogram.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — MAE by hour of day (6–19)
# ══════════════════════════════════════════════════════════════════════════════
df_day["hour"] = df_day.index.hour

mae_a_hr, mae_b_hr = [], []
hours = list(range(6, 20))
for h in hours:
    mask_h = df_day["hour"] == h
    mae_a_hr.append(mean_absolute_error(df_day.loc[mask_h, "y_true_a"], df_day.loc[mask_h, "y_pred_a"])
                    if mask_h.sum() > 0 else np.nan)
    mae_b_hr.append(mean_absolute_error(df_day.loc[mask_h, "y_true_b"], df_day.loc[mask_h, "y_pred_b"])
                    if mask_h.sum() > 0 else np.nan)

mae_a_hr = np.array(mae_a_hr)
mae_b_hr = np.array(mae_b_hr)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hours, mae_a_hr, color="steelblue",   marker="o", lw=2, label="Model A (baseline)")
ax.plot(hours, mae_b_hr, color="darkorange",  marker="o", lw=2, label="Model B (clim79)")

# Shaded region: green where B < A, red where B > A
for i in range(len(hours) - 1):
    x_seg  = [hours[i], hours[i + 1]]
    a_seg  = [mae_a_hr[i], mae_a_hr[i + 1]]
    b_seg  = [mae_b_hr[i], mae_b_hr[i + 1]]
    color  = "green" if (mae_b_hr[i] + mae_b_hr[i + 1]) < (mae_a_hr[i] + mae_a_hr[i + 1]) else "red"
    ax.fill_between(x_seg, a_seg, b_seg, alpha=0.25, color=color)

# Problematic hours
for h in [8, 9, 10]:
    ax.axvline(h, color="gray", linestyle="--", lw=1.0, alpha=0.8)
    ax.text(h + 0.05, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 0 else 50,
            f"h={h}", fontsize=7, color="gray", va="top")

ax.set_xlabel("Hour of day (local time)")
ax.set_ylabel("MAE (W/m²)")
ax.set_title("Mean Absolute Error by Hour — Before vs After Climatological Features",
             fontweight="bold")
ax.set_xticks(hours)
ax.legend(fontsize=9)

# Custom legend patches for shading
green_patch = mpatches.Patch(color="green", alpha=0.4, label="Improvement (B < A)")
red_patch   = mpatches.Patch(color="red",   alpha=0.4, label="Degradation (B > A)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + [green_patch, red_patch], fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "plot3_mae_by_hour.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot3_mae_by_hour.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Metrics comparison bar chart (grouped)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(11, 6))
ax2 = ax1.twinx()

metric_groups = [
    ("RMSE\n(W/m²)",   "RMSE",  ax1, False),
    ("MAE\n(W/m²)",    "MAE",   ax1, False),
    ("R²",             "R2",    ax1, False),
    ("SkillScore",     "SS",    ax2, True),
]

x_positions = np.array([0, 2, 4, 7])   # wider gap before SkillScore (different axis)
bar_w = 0.6
colors_a = "steelblue"
colors_b = "darkorange"

for i, (label, key, axis, is_ss) in enumerate(metric_groups):
    xa = x_positions[i] - bar_w / 2
    xb = x_positions[i] + bar_w / 2
    va, vb = m_a[key], m_b[key]

    ba = axis.bar(xa, va, width=bar_w, color=colors_a, alpha=0.85,
                  label="Model A" if i == 0 else "_nolegend_")
    bb = axis.bar(xb, vb, width=bar_w, color=colors_b, alpha=0.85,
                  label="Model B" if i == 0 else "_nolegend_")

    # % change annotation above pair
    pct  = (vb - va) / abs(va) * 100 if va != 0 else 0
    sign = "+" if pct >= 0 else ""
    ymax = max(abs(va), abs(vb))
    # place label above the taller bar on the right axis if SS
    ax_ann = axis
    top = max(va, vb)
    bot = min(va, vb)
    offset = abs(ymax) * 0.07 if abs(ymax) > 0 else 0.02
    ax_ann.text(x_positions[i], top + offset,
                f"{sign}{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# SkillScore zero line on ax2
ax2.axhline(0, color="gray", linestyle="--", lw=1.0, alpha=0.7)

ax1.set_xticks(x_positions)
ax1.set_xticklabels([g[0] for g in metric_groups], fontsize=10)
ax1.set_ylabel("RMSE / MAE (W/m²)  |  R²", fontsize=10)
ax2.set_ylabel("SkillScore", fontsize=10, color="dimgray")
ax2.tick_params(axis="y", labelcolor="dimgray")

# Unified legend
patch_a = mpatches.Patch(color=colors_a, alpha=0.85, label="Model A — Without climatology (69 feat.)")
patch_b = mpatches.Patch(color=colors_b, alpha=0.85, label="Model B — With climatology (79 feat.)")
ax1.legend(handles=[patch_a, patch_b], fontsize=9, loc="upper left")

ax1.set_title("Performance Metrics — Before vs After Climatological Features",
              fontweight="bold", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "plot4_metrics_bar_chart.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot4_metrics_bar_chart.png")

print(f"\nAll plots saved to: {OUT_DIR}")
