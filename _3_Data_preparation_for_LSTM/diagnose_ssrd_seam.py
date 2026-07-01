#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_ssrd_seam.py
=====================
Print raw ssrd values around the suspected ERA5 forecast-run boundary
at 19:00 UTC (= 14:00 local) to confirm or deny the reset hypothesis.

Usage
-----
    set PYTHONPATH=C:\\Users\\isabe\\Projects\\codigors\\carpetasdetrabajo
    python _3_Data_preparation_for_LSTM/diagnose_ssrd_seam.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import PREP_DATA_DIR, UTC_OFFSET

MERGED_FILE = str(Path(PREP_DATA_DIR) / "ERA5_climatology" / "era5_medellin_1990_2020.nc")

ds       = xr.open_dataset(MERGED_FILE, engine="h5netcdf")
time_dim = ds["ssrd"].dims[0]
times    = pd.DatetimeIndex(ds["ssrd"].coords[time_dim].values)
ssrd_raw = ds["ssrd"].values.ravel()

print("=" * 72)
print("  RAW ssrd VALUES around the 19:00 UTC seam  (July 15 + 16, 1990)")
print("=" * 72)
print(f"  {'UTC time':<22}  {'Local time':<22}  {'ssrd raw [J/m2]':>16}  diff")
print("  " + "-" * 70)

# Print July 15 and 16, 1990, hours 16:00 - 22:00 UTC
for day in [15, 16]:
    mask = (
        (times.year == 1990) & (times.month == 7) & (times.day == day) &
        (times.hour >= 15) & (times.hour <= 22)
    )
    idxs = np.where(mask)[0]
    for i in idxs:
        t_utc = times[i]
        t_loc = t_utc + pd.Timedelta(hours=UTC_OFFSET)
        val   = ssrd_raw[i]
        diff  = ssrd_raw[i] - ssrd_raw[i-1] if i > 0 else float("nan")
        flag  = "  <-- seam?" if t_utc.hour == 19 else ""
        print(f"  {str(t_utc):<22}  {str(t_loc):<22}  {val:>16,.0f}  {diff:>12,.0f}{flag}")
    print()

print("\n")
print("=" * 72)
print("  DISTINCT ssrd VALUES at boundary hours across ALL years")
print("  (if run resets at 19:00 UTC, ssrd[19:00] << ssrd[18:00])")
print("=" * 72)

boundary_hours = [17, 18, 19, 20]
for h in boundary_hours:
    mask_h  = times.hour == h
    vals_h  = ssrd_raw[mask_h]
    print(f"  hour {h:02d}h UTC (local {(h + UTC_OFFSET) % 24:02d}h):  "
          f"mean={vals_h.mean():>12,.0f}  min={vals_h.min():>12,.0f}  "
          f"max={vals_h.max():>12,.0f}  n={mask_h.sum()}")

print()
print("  If ssrd mean at 19h UTC << ssrd mean at 18h UTC: run reset confirmed.")
print("  If ssrd mean at 19h UTC ~= ssrd mean at 18h UTC: just a cloudy afternoon.")

# Also check 06:00/07:00 UTC seam
print()
print("=" * 72)
print("  ALSO checking 07:00 UTC seam")
print("=" * 72)
for h in [5, 6, 7, 8]:
    mask_h  = times.hour == h
    vals_h  = ssrd_raw[mask_h]
    print(f"  hour {h:02d}h UTC (local {(h + UTC_OFFSET) % 24:02d}h):  "
          f"mean={vals_h.mean():>12,.0f}  min={vals_h.min():>12,.0f}  "
          f"max={vals_h.max():>12,.0f}  n={mask_h.sum()}")

ds.close()
print("\nDone.")
