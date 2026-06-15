#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
climatological_analysis.py
==========================
Generates 7 diagnostic plots characterising the solar radiation climatology
of Medellín, Colombia (2021-2024), based on SIATA observations and GFS
model output.

Data sources
------------
- SIATA CSI  : clearsky_index_Siata.nc
- GFS CSI    : clearsky_index_GFS_{0100,0700,1300,1900}.nc
               Best-lead-time strategy: for each hour the GFS run with the
               shortest positive lead time is selected (same logic as
               verify_skillscore.py).
- Clear-sky GHI : from config.CSI_GHI_FILE  (Ineichen, bias-corrected)

All analysis is restricted to daytime hours (clear_sky_ghi > 0).

Output
------
7 PNG files saved to  _3_Data_preparation_for_LSTM/Climatological_analysis/
"""

import sys
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 encoding errors for ±, −, ≤, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# ── Project root (two levels up from this file) ─────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    LAUNCH_TIMES,           # ["0100", "0700", "1300", "1900"]
    VAR_GFS_CSI_TEMPLATE,   # "clearsky_index_GFS_{LT}"
    VAR_REF_CLEARSKY_GHI,   # "clear_sky_ghi"
    VAR_SIATA_CSI,          # "clearsky_index_Siata"
    CSI_INDEX_DIR,          # relative path → .../clear_sky_indices/
    CSI_GHI_FILE,           # relative path → Ineichen NetCDF
    SIATA_CSI_FILE,         # relative path → SIATA CSI NetCDF
)

# ── Paths ────────────────────────────────────────────────────────────────────
CSI_DIR       = ROOT / CSI_INDEX_DIR
INEICHEN_FILE = ROOT / CSI_GHI_FILE
SIATA_FILE    = ROOT / SIATA_CSI_FILE
OUT_DIR       = ROOT / "_3_Data_preparation_for_LSTM/Climatological_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (consistent across all plots) ─────────────────────────────
C_GFS     = "orange"
C_SIATA   = "black"
C_CLEAR   = "#FFD700"   # gold / yellow
C_PARTIAL = "#ADD8E6"   # light blue
C_CLOUDY  = "#808080"   # medium gray

DPI = 150


# ===========================================================================
# SECTION 1 — Helper functions
# ===========================================================================

def _reduce_to_1d(da, time_dim="observation_time"):
    """Area-mean over lat/lon if present, then squeeze other singleton dims."""
    if {"lat", "lon"}.issubset(da.dims):
        da = da.mean(dim=("lat", "lon"))
    for dim in list(da.dims):
        if dim != time_dim:
            da = da.isel({dim: 0})
    return da


def sky_category(csi_series):
    """
    Return a categorical Series with labels 'Clear', 'Partial', 'Cloudy'
    based on CSI thresholds.
    """
    cats = pd.Series("Partial", index=csi_series.index, dtype="object")
    cats[csi_series > 0.8] = "Clear"
    cats[csi_series < 0.4] = "Cloudy"
    return cats


# ===========================================================================
# SECTION 2 — Load SIATA CSI
# ===========================================================================
print("=" * 65)
print("SECTION 2 — Loading SIATA CSI")
print("=" * 65)

ds_siata   = xr.open_dataset(SIATA_FILE, engine="h5netcdf")
da_siata   = _reduce_to_1d(ds_siata[VAR_SIATA_CSI])
siata_csi  = da_siata.to_series().rename("siata_csi").sort_index()
ds_siata.close()

print(f"  SIATA CSI: {len(siata_csi):,} rows  "
      f"({siata_csi.index.min()} -> {siata_csi.index.max()})")


# ===========================================================================
# SECTION 3 — Load Ineichen clear-sky GHI
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 3 — Loading Ineichen clear-sky GHI")
print("=" * 65)

ds_ine    = xr.open_dataset(INEICHEN_FILE, engine="h5netcdf")
da_csg    = _reduce_to_1d(ds_ine[VAR_REF_CLEARSKY_GHI])
csg_series = da_csg.to_series().rename("clear_sky_ghi").sort_index()
ds_ine.close()

print(f"  Clear-sky GHI: {len(csg_series):,} rows  |  max={csg_series.max():.1f} W/m²")


# ===========================================================================
# SECTION 4 — Load GFS CSI (best-lead strategy)
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 4 — Building GFS CSI best-lead series")
print("=" * 65)
print("  Strategy: shortest positive lead time per observation_time.")

val_frames  = []
lead_frames = []

for lt in LAUNCH_TIMES:
    nc_path  = CSI_DIR / f"clearsky_index_GFS_{lt}.nc"
    var_name = VAR_GFS_CSI_TEMPLATE.format(LT=lt)

    ds_gfs  = xr.open_dataset(nc_path, engine="h5netcdf")
    da      = _reduce_to_1d(ds_gfs[var_name])
    ser_val = da.to_series()

    launch_ser        = pd.to_datetime(ds_gfs["launch_time"].to_series())
    launch_ser.index  = ser_val.index
    lead              = ser_val.index.to_series() - launch_ser
    lead[lead <= pd.Timedelta(0)] = pd.NaT   # invalid: obs before model run

    val_frames.append(ser_val.rename(lt))
    lead_frames.append(lead.rename(lt))
    ds_gfs.close()
    print(f"  LT {lt}: {len(ser_val):,} rows  |  valid leads: {(~lead.isna()).sum():,}")

val_df   = pd.concat(val_frames,  axis=1)
lead_df  = pd.concat(lead_frames, axis=1)

idx_min   = lead_df.idxmin(axis=1)
col_pos   = val_df.columns.get_indexer(idx_min)
row_pos   = np.arange(len(val_df))
best_vals = val_df.to_numpy()[row_pos, col_pos]

gfs_csi_best = (
    pd.Series(best_vals, index=val_df.index, name="gfs_csi")
    .dropna()
    .sort_index()
)
print(f"\n  Best-lead GFS CSI: {len(gfs_csi_best):,} valid rows")


# ===========================================================================
# SECTION 5 — Align and apply daytime filter
# ===========================================================================
print("\n" + "=" * 65)
print("SECTION 5 — Aligning series and applying daytime filter")
print("=" * 65)

df = pd.DataFrame({
    "siata_csi"     : siata_csi,
    "gfs_csi"       : gfs_csi_best,
    "clear_sky_ghi" : csg_series,
}).dropna()

# Daytime: clear-sky GHI > 0
df = df[df["clear_sky_ghi"] > 0].copy()

# Derived columns
df["siata_ghi"] = df["siata_csi"] * df["clear_sky_ghi"]
df["gfs_ghi"]   = df["gfs_csi"]  * df["clear_sky_ghi"]
df["error"]     = df["gfs_csi"]  - df["siata_csi"]
df["hour"]      = df.index.hour
df["month"]     = df.index.month
df["category"]  = sky_category(df["siata_csi"])

print(f"  Daytime rows after alignment : {len(df):,}")
print(f"  Date range                   : {df.index.min()} -> {df.index.max()}")
print(f"  Hours present                : {sorted(df['hour'].unique())}")


# ===========================================================================
# PLOT 1 — Daily cycle: mean CSI by hour
# ===========================================================================
print("\n[Plot 1] Daily cycle — mean CSI by hour …")

hourly = df.groupby("hour")
siata_mean = hourly["siata_csi"].mean()
siata_std  = hourly["siata_csi"].std()
gfs_mean   = hourly["gfs_csi"].mean()

hours = siata_mean.index

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hours, siata_mean, color=C_SIATA, lw=2, label="SIATA CSI (mean)")
ax.fill_between(
    hours,
    siata_mean - siata_std,
    siata_mean + siata_std,
    color=C_SIATA, alpha=0.15, label="SIATA ±1 std",
)
ax.plot(hours, gfs_mean, color=C_GFS, lw=2, linestyle="--", label="GFS CSI (mean)")
ax.set_xlabel("Hour of Day (local time)")
ax.set_ylabel("Clear-Sky Index (CSI)")
ax.set_title("Mean CSI by Hour of Day — Medellín 2021-2024")
ax.set_xticks(hours)
ax.set_xlim(hours.min(), hours.max())
ax.set_ylim(0, 1.05)
ax.legend(framealpha=0.9)
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot1_daily_cycle.png", dpi=DPI)
plt.close(fig)
print("  Saved: plot1_daily_cycle.png")


# ===========================================================================
# PLOT 2 — Seasonal cycle: mean GHI (W/m²) by month
# ===========================================================================
print("[Plot 2] Seasonal cycle — mean GHI by month …")

monthly_siata = df.groupby("month")["siata_ghi"].mean()
monthly_gfs   = df.groupby("month")["gfs_ghi"].mean()
months        = np.arange(1, 13)
month_labels  = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(10, 5))

# Rainy season shading (Apr-May = 4-5, Oct-Nov = 10-11)
for start, end in [(3.5, 5.5), (9.5, 11.5)]:
    ax.axvspan(start, end, color="steelblue", alpha=0.10,
               label="Rainy season" if start == 3.5 else "")

ax.plot(months, monthly_siata.reindex(months), color=C_SIATA, lw=2,
        marker="o", markersize=5, label="SIATA GHI")
ax.plot(months, monthly_gfs.reindex(months), color=C_GFS, lw=2,
        linestyle="--", marker="s", markersize=5, label="GFS GHI")

ax.set_xticks(months)
ax.set_xticklabels(month_labels)
ax.set_xlabel("Month")
ax.set_ylabel("Mean Solar Radiation (W/m²)")
ax.set_title("Mean Solar Radiation by Month — Medellín 2021-2024")
ax.legend(framealpha=0.9)
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot2_seasonal_cycle.png", dpi=DPI)
plt.close(fig)
print("  Saved: plot2_seasonal_cycle.png")


# ===========================================================================
# PLOT 3 — Error heatmap: hour × month
# ===========================================================================
print("[Plot 3] Error heatmap hour × month …")

pivot = (
    df.groupby(["month", "hour"])["error"]
    .mean()
    .unstack("hour")           # columns = hours
)
# Ensure all 14 hours (6-19) and 12 months are present
all_hours  = list(range(6, 20))
all_months = list(range(1, 13))
pivot = pivot.reindex(index=all_months, columns=all_hours)

abs_max = np.nanmax(np.abs(pivot.values))
norm    = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(
    pivot.values,
    aspect="auto",
    cmap="RdBu_r",
    norm=norm,
    origin="upper",
)
cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="Mean Error (GFS − SIATA CSI)")

ax.set_xticks(range(len(all_hours)))
ax.set_xticklabels(all_hours)
ax.set_yticks(range(len(all_months)))
ax.set_yticklabels(month_labels)
ax.set_xlabel("Hour of Day (local time)")
ax.set_ylabel("Month")
ax.set_title("GFS Mean Error by Hour and Month")

# Annotate cells with values
for i in range(len(all_months)):
    for j in range(len(all_hours)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="black")

fig.tight_layout()
fig.savefig(OUT_DIR / "plot3_error_heatmap.png", dpi=DPI)
plt.close(fig)
print("  Saved: plot3_error_heatmap.png")


# ===========================================================================
# PLOT 4 — CSI distribution (SIATA daytime)
# ===========================================================================
print("[Plot 4] CSI distribution — SIATA …")

siata_vals = df["siata_csi"].values
bins = np.linspace(0, 1.4, 71)

n_clear   = (siata_vals > 0.8).sum()
n_partial = ((siata_vals >= 0.4) & (siata_vals <= 0.8)).sum()
n_cloudy  = (siata_vals < 0.4).sum()
n_total   = len(siata_vals)

pct_clear   = n_clear   / n_total * 100
pct_partial = n_partial / n_total * 100
pct_cloudy  = n_cloudy  / n_total * 100

fig, ax = plt.subplots(figsize=(10, 5))
counts, edges, patches = ax.hist(siata_vals, bins=bins, color="#CCCCCC",
                                  edgecolor="white", linewidth=0.4)

# Colour patches by category
for patch, left in zip(patches, edges[:-1]):
    if left < 0.4:
        patch.set_facecolor(C_CLOUDY)
    elif left < 0.8:
        patch.set_facecolor(C_PARTIAL)
    else:
        patch.set_facecolor(C_CLEAR)

# Vertical threshold lines
ax.axvline(0.4, color="dimgray", lw=1.2, linestyle="--")
ax.axvline(0.8, color="dimgray", lw=1.2, linestyle="--")

# Percentage labels
y_max = counts.max() * 1.05
ax.text(0.20, y_max * 0.92, f"Cloudy\n{pct_cloudy:.1f}%",
        ha="center", va="top", color=C_SIATA, fontsize=10)
ax.text(0.60, y_max * 0.92, f"Partial\n{pct_partial:.1f}%",
        ha="center", va="top", color=C_SIATA, fontsize=10)
ax.text(0.95, y_max * 0.92, f"Clear\n{pct_clear:.1f}%",
        ha="center", va="top", color=C_SIATA, fontsize=10)

legend_patches = [
    mpatches.Patch(facecolor=C_CLOUDY,  label="Cloudy  (CSI < 0.4)"),
    mpatches.Patch(facecolor=C_PARTIAL, label="Partial (0.4 ≤ CSI ≤ 0.8)"),
    mpatches.Patch(facecolor=C_CLEAR,   label="Clear   (CSI > 0.8)"),
]
ax.legend(handles=legend_patches, framealpha=0.9)
ax.set_xlabel("Clear-Sky Index (CSI)")
ax.set_ylabel("Number of Hours")
ax.set_title("Distribution of Sky Conditions — SIATA 2021-2024")
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot4_csi_distribution.png", dpi=DPI)
plt.close(fig)
print("  Saved: plot4_csi_distribution.png")


# ===========================================================================
# PLOT 5 — GFS error by sky condition (boxplot)
# ===========================================================================
print("[Plot 5] Error by sky condition — boxplot …")

cat_order  = ["Cloudy", "Partial", "Clear"]
cat_colors = [C_CLOUDY, C_PARTIAL, C_CLEAR]
data_by_cat = [df.loc[df["category"] == c, "error"].values for c in cat_order]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(
    data_by_cat,
    labels=cat_order,
    patch_artist=True,
    notch=False,
    showfliers=False,
    medianprops=dict(color="black", lw=2),
    whiskerprops=dict(lw=1.2),
    capprops=dict(lw=1.2),
)
for patch, color in zip(bp["boxes"], cat_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.axhline(0, color="black", lw=1.0, linestyle="--", alpha=0.7)
ax.set_xlabel("Sky Condition Category (SIATA)")
ax.set_ylabel("GFS CSI Error (GFS − SIATA)")
ax.set_title("GFS Forecast Error by Sky Condition Category")
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot5_error_by_condition.png", dpi=DPI)
plt.close(fig)
print("  Saved: plot5_error_by_condition.png")


# ===========================================================================
# PLOT 6 — Convective events by hour
# ===========================================================================
print("[Plot 6] Convective failure events by hour …")

conv_mask   = (df["gfs_csi"] > 0.6) & (df["siata_csi"] < 0.3)
conv_pct    = df.groupby("hour").apply(lambda g: conv_mask.loc[g.index].mean() * 100)
plot_hours  = list(range(6, 20))
conv_vals   = conv_pct.reindex(plot_hours, fill_value=0.0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(plot_hours, conv_vals, color=C_GFS, edgecolor="white", linewidth=0.5, alpha=0.85)
ax.set_xlabel("Hour of Day (local time)")
ax.set_ylabel("% of Hours with Convective Failure")
ax.set_title("GFS Overestimation Events by Hour (Convective Failures)")
ax.set_xticks(plot_hours)
ax.set_xlim(5.4, 19.6)
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot6_convective_events.png", dpi=DPI)
plt.close(fig)
print("  Saved: plot6_convective_events.png")


# ===========================================================================
# PLOT 7 — Sky condition frequency + GFS response by hour
# ===========================================================================
print("[Plot 7] Sky condition frequency + GFS response by hour …")

plot_hours = list(range(6, 20))

# Frequency of each category per hour (fraction of hours)
freq = {}
for cat in ["Clear", "Partial", "Cloudy"]:
    freq[cat] = (
        df.groupby("hour")
        .apply(lambda g: (g["category"] == cat).mean() * 100)
        .reindex(plot_hours, fill_value=0.0)
    )

# Mean GFS CSI per hour, conditioned on SIATA category
gfs_by_cat = {}
for cat in ["Clear", "Partial", "Cloudy"]:
    gfs_by_cat[cat] = (
        df[df["category"] == cat]
        .groupby("hour")["gfs_csi"]
        .mean()
        .reindex(plot_hours)
    )

x = np.arange(len(plot_hours))
width = 0.6

fig, ax1 = plt.subplots(figsize=(12, 6))

bottom = np.zeros(len(plot_hours))
bar_colors  = {"Clear": C_CLEAR, "Partial": C_PARTIAL, "Cloudy": C_CLOUDY}
for cat in ["Cloudy", "Partial", "Clear"]:   # stack bottom-up
    ax1.bar(x, freq[cat].values, bottom=bottom, width=width,
            color=bar_colors[cat], label=f"{cat} (SIATA %)", alpha=0.80)
    bottom += freq[cat].values

ax1.set_ylim(0, 105)
ax1.set_ylabel("Frequency of Sky Condition (%)")
ax1.set_xlabel("Hour of Day (local time)")
ax1.set_xticks(x)
ax1.set_xticklabels(plot_hours)

# Second axis for GFS CSI lines
ax2 = ax1.twinx()
line_styles = {"Clear": "-", "Partial": "--", "Cloudy": ":"}
line_colors = {"Clear": "#B8860B", "Partial": "steelblue", "Cloudy": "dimgray"}
for cat in ["Clear", "Partial", "Cloudy"]:
    ax2.plot(
        x, gfs_by_cat[cat].values,
        color=line_colors[cat], lw=2,
        linestyle=line_styles[cat],
        marker="o", markersize=4,
        label=f"GFS CSI | SIATA={cat}",
    )
ax2.set_ylim(0, 1.1)
ax2.set_ylabel("Mean GFS CSI")

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2,
           loc="upper left", fontsize=8, framealpha=0.9)

ax1.set_title("Sky Condition Frequency and GFS Response by Hour")
fig.tight_layout()
fig.savefig(OUT_DIR / "plot7_sky_freq_gfs_response.png", dpi=DPI)
plt.close(fig)
print("  Saved: plot7_sky_freq_gfs_response.png")


# ===========================================================================
# PLOT 8 — Standard deviation of SIATA CSI by hour
# ===========================================================================
print("[Plot 8] SIATA CSI std by hour …")

# siata_std already computed in Plot 1 section; reindex to the full hour range
std_by_hour = siata_std.reindex(plot_hours, fill_value=np.nan)
mean_std    = std_by_hour.mean()

# Colour bars by magnitude: normalise to [0,1] then map through Blues colormap
import matplotlib.cm as cm
norm_vals  = (std_by_hour - std_by_hour.min()) / (std_by_hour.max() - std_by_hour.min())
bar_colors = [cm.Blues(0.35 + 0.60 * v) for v in norm_vals]   # range 0.35–0.95

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(plot_hours, std_by_hour, color=bar_colors,
              edgecolor="white", linewidth=0.6, width=0.7)

# Horizontal mean line
ax.axhline(mean_std, color="crimson", linewidth=1.5, linestyle="--",
           label=f"Mean std = {mean_std:.4f}")

# Value labels on top of each bar
for h, v in zip(plot_hours, std_by_hour):
    if not np.isnan(v):
        ax.text(h, v + 0.003, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color="black")

ax.set_xlabel("Hour of Day (local time)", fontsize=12)
ax.set_ylabel("Std of SIATA CSI", fontsize=12)
ax.set_title("SIATA CSI Variability by Hour — Medellín 2021-2024", fontsize=13)
ax.set_xticks(plot_hours)
ax.set_xlim(5.4, 19.6)
ax.set_ylim(0, std_by_hour.max() * 1.18)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "plot08_std_by_hour.png", dpi=DPI)
plt.close(fig)
print("  Saved: plot08_std_by_hour.png")

# Print std table
print("\n  --- SIATA CSI std by hour ---")
print(f"  {'Hour':>5}  {'Std':>8}  {'vs mean':>9}")
print(f"  {'-'*5}  {'-'*8}  {'-'*9}")
for h, v in zip(plot_hours, std_by_hour):
    delta = v - mean_std
    marker = " <-- max" if v == std_by_hour.max() else (" <-- min" if v == std_by_hour.min() else "")
    print(f"  {h:>5}h  {v:>8.4f}  {delta:>+9.4f}{marker}")
print(f"  {'MEAN':>5}   {mean_std:>8.4f}")


# ===========================================================================
# PLOT 9 — CSI distribution at morning transition hours vs midday (bimodal check)
# ===========================================================================
print("[Plot 9] CSI distribution morning hours vs midday …")

from scipy.stats import gaussian_kde

MORNING_HOURS = [8, 9, 10]
REF_HOUR      = 12
bins_csi      = np.linspace(0, 1.4, 50)
bin_centres   = 0.5 * (bins_csi[:-1] + bins_csi[1:])

# Reference (hour 12) KDE — computed once, overlaid on every subplot
ref_vals = df.loc[df["hour"] == REF_HOUR, "siata_csi"].values
ref_kde  = gaussian_kde(ref_vals, bw_method=0.12)
ref_density = ref_kde(bin_centres)
# Scale to count space using the ref histogram area
ref_counts, _ = np.histogram(ref_vals, bins=bins_csi)
ref_scale = ref_counts.sum() * (bins_csi[1] - bins_csi[0])

morning_colors = {8: "#2166ac", 9: "#d6604d", 10: "#4dac26"}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.suptitle(
    "CSI Distribution at Morning Transition Hours vs Midday\n"
    "(grey dashed = hour 12 reference, the most stable hour)",
    fontsize=12,
)

for ax, h in zip(axes, MORNING_HOURS):
    vals = df.loc[df["hour"] == h, "siata_csi"].values
    n    = len(vals)

    # Histogram bars for this hour
    counts, _ = np.histogram(vals, bins=bins_csi)
    ax.bar(bin_centres, counts, width=bins_csi[1] - bins_csi[0],
           color=morning_colors[h], alpha=0.65, edgecolor="white",
           linewidth=0.4, label=f"Hour {h:02d}h  (n={n})")

    # KDE line for this hour
    if n > 5:
        hour_kde     = gaussian_kde(vals, bw_method=0.12)
        hour_density = hour_kde(bin_centres)
        hour_scale   = counts.sum() * (bins_csi[1] - bins_csi[0])
        ax.plot(bin_centres, hour_density * hour_scale,
                color=morning_colors[h], lw=2.0, label="KDE")

    # Hour-12 reference KDE overlay (scaled to this hour's count space)
    ref_scaled = ref_density * ref_scale
    ax.plot(bin_centres, ref_scaled, color="dimgray", lw=1.4,
            linestyle="--", alpha=0.7, label=f"Hour {REF_HOUR:02d}h ref KDE")

    # Threshold lines
    ax.axvline(0.4, color="dimgray", lw=0.9, linestyle=":", alpha=0.6)
    ax.axvline(0.8, color="dimgray", lw=0.9, linestyle=":", alpha=0.6)

    ax.set_xlabel("SIATA CSI", fontsize=11)
    ax.set_ylabel("Count" if h == 8 else "", fontsize=11)
    ax.set_xlim(0, 1.35)
    ax.set_title(f"Hour {h:02d}h", fontsize=12, color=morning_colors[h])
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(axis="y", alpha=0.35)

    # Annotate category fractions inside the subplot
    n_cloudy  = (vals < 0.4).sum()
    n_partial = ((vals >= 0.4) & (vals <= 0.8)).sum()
    n_clear   = (vals > 0.8).sum()
    ymax = ax.get_ylim()[1]
    ax.text(0.20, ymax * 0.93, f"{n_cloudy/n*100:.0f}%",
            ha="center", fontsize=8, color=C_CLOUDY,  style="italic")
    ax.text(0.60, ymax * 0.93, f"{n_partial/n*100:.0f}%",
            ha="center", fontsize=8, color="steelblue", style="italic")
    ax.text(1.00, ymax * 0.93, f"{n_clear/n*100:.0f}%",
            ha="center", fontsize=8, color="#B8860B",  style="italic")

fig.tight_layout()
fig.savefig(OUT_DIR / "plot09_bimodal_check.png", dpi=DPI)
plt.close(fig)
print("  Saved: plot09_bimodal_check.png")

# Print bimodal summary table
print("\n  --- CSI sky-condition breakdown by hour ---")
print(f"  {'Hour':>5}  {'n':>5}  {'Cloudy%':>8}  {'Partial%':>9}  {'Clear%':>7}")
print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*7}")
for h in MORNING_HOURS + [REF_HOUR]:
    v   = df.loc[df["hour"] == h, "siata_csi"].values
    n   = len(v)
    cl  = (v < 0.4).sum() / n * 100
    pa  = ((v >= 0.4) & (v <= 0.8)).sum() / n * 100
    cr  = (v > 0.8).sum() / n * 100
    tag = "  <- reference" if h == REF_HOUR else ""
    print(f"  {h:>5}h  {n:>5}  {cl:>7.1f}%  {pa:>8.1f}%  {cr:>6.1f}%{tag}")


# ===========================================================================
# SECTION 6 — Summary statistics table
# ===========================================================================
print("\n" + "=" * 65)
print("SUMMARY STATISTICS TABLE — Medellín Solar Climatology 2021-2024")
print("=" * 65)

mae  = df["error"].abs().mean()
rmse = np.sqrt((df["error"] ** 2).mean())
bias = df["error"].mean()

n_total   = len(df)
n_clear   = (df["category"] == "Clear").sum()
n_partial = (df["category"] == "Partial").sum()
n_cloudy  = (df["category"] == "Cloudy").sum()

n_conv = ((df["gfs_csi"] > 0.6) & (df["siata_csi"] < 0.3)).sum()

print(f"  Date range          : {df.index.min().date()} -> {df.index.max().date()}")
print(f"  Total daytime hours : {n_total:,}")
print()
print(f"  SIATA CSI  — mean : {df['siata_csi'].mean():.4f}  "
      f"std : {df['siata_csi'].std():.4f}")
print(f"  GFS CSI    — mean : {df['gfs_csi'].mean():.4f}  "
      f"std : {df['gfs_csi'].std():.4f}")
print()
print(f"  Bias (GFS − SIATA) : {bias:+.4f}")
print(f"  MAE                : {mae:.4f}")
print(f"  RMSE               : {rmse:.4f}")
print()
print(f"  Sky condition breakdown:")
print(f"    Clear   (CSI > 0.8)      : {n_clear:,}  ({n_clear/n_total*100:.1f}%)")
print(f"    Partial (0.4 ≤ CSI ≤ 0.8): {n_partial:,}  ({n_partial/n_total*100:.1f}%)")
print(f"    Cloudy  (CSI < 0.4)      : {n_cloudy:,}  ({n_cloudy/n_total*100:.1f}%)")
print()
print(f"  Convective failures (GFS>0.6 & SIATA<0.3):")
print(f"    Count : {n_conv:,}  ({n_conv/n_total*100:.2f}% of daytime hours)")
print()
print(f"  All plots saved to: {OUT_DIR}")
print("=" * 65)
