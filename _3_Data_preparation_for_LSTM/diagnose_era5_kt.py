#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_era5_kt.py
===================
Diagnostic script to inspect ERA5 ssrd and kt computation.
Run this before recomputing climatology.

Usage
-----
    set PYTHONPATH=C:\\Users\\isabe\\Projects\\codigors\\carpetasdetrabajo
    python _3_Data_preparation_for_LSTM/diagnose_era5_kt.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import PREP_DATA_DIR, EXT_GHI_FILE, EXT_VAR_NAME, UTC_OFFSET

MERGED_FILE = str(Path(PREP_DATA_DIR) / "ERA5_climatology" / "era5_medellin_1990_2020.nc")

print("=" * 66)
print("  ERA5 kt DIAGNOSTIC")
print("=" * 66)

# ── 1. Inspect merged dataset structure ───────────────────────────────────────
print("\n[1] Merged dataset structure")
ds = xr.open_dataset(MERGED_FILE, engine="h5netcdf")
print(f"  dims   : {dict(ds.dims)}")
print(f"  coords : {list(ds.coords)}")
print(f"  vars   : {list(ds.data_vars)}")
print(f"  ssrd   shape : {ds['ssrd'].shape}")
print(f"  ssrd   dims  : {ds['ssrd'].dims}")
print(f"  ssrd   dtype : {ds['ssrd'].dtype}")

# ── 2. Detect time dim and load raw ssrd for July 1990 noon UTC ──────────────
time_dim = ds["ssrd"].dims[0]
print(f"\n[2] Raw ssrd values — July 1990, 11-13h UTC (before deaccumulation)")

da_ssrd = ds["ssrd"]
times   = pd.DatetimeIndex(da_ssrd.coords[time_dim].values)

# July 1990, hours 11-13 UTC  (noon is roughly 06-08 local, but let's see full UTC day)
jul_noon_mask = (times.year == 1990) & (times.month == 7) & (times.day == 15) & \
                (times.hour >= 10) & (times.hour <= 16)
jul_noon_idx  = np.where(jul_noon_mask)[0]

if len(jul_noon_idx) == 0:
    print("  WARNING: no July 1990 timestamps found — printing first 5 timestamps")
    print(f"  First 5 times: {times[:5].tolist()}")
    jul_noon_idx = np.arange(min(5, len(times)))

for i in jul_noon_idx:
    t   = times[i]
    val = float(da_ssrd.isel({time_dim: i}).values.flat[0])
    print(f"  {t}  ssrd = {val:>14.1f} J/m^2 (accumulated)")

# ── 3. Deaccumulate ssrd ──────────────────────────────────────────────────────
print(f"\n[3] Deaccumulated ssrd (diff / 3600) — same window")
ghi_da   = da_ssrd.diff(dim=time_dim).clip(min=0) / 3600.0
times2   = pd.DatetimeIndex(ghi_da.coords[time_dim].values)
jul2_mask = (times2.year == 1990) & (times2.month == 7) & (times2.day == 15) & \
            (times2.hour >= 10) & (times2.hour <= 16)
jul2_idx  = np.where(jul2_mask)[0]

if len(jul2_idx) == 0:
    print("  WARNING: no timestamps after diff — printing first 5")
    jul2_idx = np.arange(min(5, len(times2)))

for i in jul2_idx:
    t   = times2[i]
    val = float(ghi_da.isel({time_dim: i}).values.flat[0])
    print(f"  {t}  ghi = {val:>8.2f} W/m^2")

# ── 4. Shape of ghi values array ──────────────────────────────────────────────
print(f"\n[4] ghi_da shape after diff: {ghi_da.shape}  (expected 1-D or squeezable)")
ghi_flat = ghi_da.values
print(f"  ghi_da.values shape : {ghi_flat.shape}")
if ghi_flat.ndim > 1:
    print("  WARNING: multi-dimensional — need to squeeze spatial dims!")
    ghi_flat = ghi_flat.reshape(ghi_flat.shape[0], -1).mean(axis=1)
    print(f"  After squeeze: {ghi_flat.shape}")
else:
    print("  OK: already 1-D")

# ── 5. ext_ghi for same timestamps (local time = UTC - 5) ────────────────────
print(f"\n[5] Extraterrestrial GHI lookup — July 15 1990, noon local")
ds_ext     = xr.open_dataset(EXT_GHI_FILE, engine="h5netcdf")
da_ext     = ds_ext[EXT_VAR_NAME].squeeze()
ext_series = da_ext.to_series()
ext_idx    = pd.DatetimeIndex(ext_series.index)
lookup     = ext_series.groupby([ext_idx.month, ext_idx.day, ext_idx.hour]).mean().to_dict()
ds_ext.close()

# Jul 15 local noon = UTC 17:00 (local = UTC - 5  →  UTC = local + 5)
time_local2 = times2 + pd.Timedelta(hours=UTC_OFFSET)   # UTC -> local
tl_mask     = (time_local2.year.isin([1990,1991])) & (time_local2.month == 7) & \
              (time_local2.day == 15) & (time_local2.hour >= 6) & (time_local2.hour <= 18)
tl_idx      = np.where(tl_mask)[0]

print(f"  UTC_OFFSET = {UTC_OFFSET}  (Colombia = UTC {UTC_OFFSET})")
if len(tl_idx) == 0:
    print("  WARNING: no matching local-time July 15 timestamps found")
    tl_idx = np.arange(min(8, len(times2)))

for i in tl_idx:
    t_utc   = times2[i]
    t_loc   = time_local2[i]
    ghi_val = float(ghi_da.isel({time_dim: i}).values.flat[0])
    ext_val = lookup.get((t_loc.month, t_loc.day, t_loc.hour), np.nan)
    kt_val  = ghi_val / ext_val if ext_val > 0 else np.nan
    print(f"  UTC {t_utc}  local {t_loc}  "
          f"GHI={ghi_val:>7.1f}  ext={ext_val:>7.1f}  kt={kt_val:.3f}")

# ── 6. Global stats on ghi_flat for daytime only ──────────────────────────────
print(f"\n[6] Global GHI distribution (all years)")
ghi_all = ghi_da.values.ravel()
print(f"  Total values : {len(ghi_all):,}")
print(f"  Min          : {float(np.nanmin(ghi_all)):.2f} W/m^2")
print(f"  Max          : {float(np.nanmax(ghi_all)):.2f} W/m^2")
print(f"  Mean (all)   : {float(np.nanmean(ghi_all)):.2f} W/m^2")
print(f"  > 0          : {int((ghi_all > 0).sum()):,}  ({100*(ghi_all>0).mean():.1f}%)")
print(f"  > 100        : {int((ghi_all > 100).sum()):,}")
print(f"  > 500        : {int((ghi_all > 500).sum()):,}")

# ── 7. July daytime kt ────────────────────────────────────────────────────────
print(f"\n[7] Mean kt for July daytime (local 6-18h) — all years")
time_local_all = times2 + pd.Timedelta(hours=UTC_OFFSET)
is_july    = time_local_all.month == 7
is_day6_18 = (time_local_all.hour >= 6) & (time_local_all.hour <= 18)

ext_all    = np.array([
    lookup.get((t.month, t.day, t.hour), np.nan)
    for t in time_local_all
], dtype=np.float64)

ghi_1d = ghi_da.values.ravel()  # may need shape check
if ghi_da.values.ndim > 1:
    ghi_1d = ghi_da.values.reshape(len(times2), -1).mean(axis=1)

with np.errstate(divide="ignore", invalid="ignore"):
    kt_all = np.where(ext_all > 0, ghi_1d / ext_all, np.nan)
    kt_all = np.where(np.isfinite(kt_all), np.clip(kt_all, 0, 1.2), np.nan)

mask_jul_day = is_july & is_day6_18 & (ext_all > 0)
kt_jul_day   = kt_all[mask_jul_day]
print(f"  Samples      : {len(kt_jul_day):,}")
if len(kt_jul_day) > 0:
    print(f"  Mean kt      : {np.nanmean(kt_jul_day):.3f}")
    print(f"  Max kt       : {np.nanmax(kt_jul_day):.3f}")
    print(f"  prob_clear   : {(kt_jul_day > 0.7).mean():.3f}")
    print(f"  prob_cloudy  : {(kt_jul_day < 0.3).mean():.3f}")
else:
    print("  WARNING: no July daytime samples found!")

ds.close()
print("\nDone.")
