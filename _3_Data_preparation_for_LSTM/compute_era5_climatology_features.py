#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_era5_climatology_features.py
=====================================
Read era5_climatology_1990_2020.nc and produce era5_clim_features.nc,
a clean 11-variable (12 x 24) lookup table ready for injection into
the LSTM sequence pipeline.

Features produced
-----------------
  0  era5_mean_kt        mean clearness index          daytime only (NaN at night)
  1  era5_std_kt         std  clearness index          daytime only
  2  era5_prob_cloudy    fraction kt < 0.3             daytime only
  3  era5_prob_clear     fraction kt > 0.7             daytime only
  4  era5_prob_partial   fraction 0.3 <= kt <= 0.7     daytime only
  5  era5_bimodality     prob_cloudy * prob_clear       daytime only
  6  era5_mean_tcc       mean total cloud cover         all 24 hours
  7  era5_t2m_night      mean 2 m temperature           nocturnal hours 20-06 only (NaN daytime)
  8  era5_rh_night       mean relative humidity         nocturnal hours 20-06 only (NaN daytime)
  9  era5_month_sin      sin(2 pi * month / 12)         all 24 hours
 10  era5_month_cos      cos(2 pi * month / 12)         all 24 hours

NaN values are stored as-is; the sequence builder fills them with 0.0
(same convention as SIATA climatological features).

Output
------
  _3_Data_preparation_for_LSTM/Preparation_data/era5_clim_features.nc

Usage
-----
    set PYTHONPATH=C:\\Users\\isabe\\Projects\\codigors\\carpetasdetrabajo
    python _3_Data_preparation_for_LSTM/compute_era5_climatology_features.py
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import PREP_DATA_DIR, ERA5_CLIM_FEATURES_FILE

# ── Paths ──────────────────────────────────────────────────────────────────────
CLIM_SRC  = Path(PREP_DATA_DIR) / "era5_climatology_1990_2020.nc"
OUT_PATH  = ROOT / ERA5_CLIM_FEATURES_FILE

# Night hours (local Colombia time) used for t2m_night and rh_night
NIGHT_HOURS = set(list(range(20, 24)) + list(range(0, 7)))


def main() -> None:
    print("=" * 62)
    print("  compute_era5_climatology_features.py")
    print("=" * 62)

    # ── 1. Load source climatology ────────────────────────────────────────────
    print(f"\nSection 1 — Loading {CLIM_SRC.name}")
    ds = xr.open_dataset(CLIM_SRC, engine="h5netcdf")
    print(f"  Variables : {list(ds.data_vars)}")
    print(f"  Dims      : month={ds.dims['month']}, hour={ds.dims['hour']}")

    months = ds["month"].values    # 1-12
    hours  = ds["hour"].values     # 0-23

    # ── 2. Extract the 9 source arrays (each (12, 24) float32) ───────────────
    print("\nSection 2 — Extracting source arrays")

    mean_kt      = ds["era5_mean_kt"].values.astype(np.float32)      # NaN at night
    std_kt       = ds["era5_std_kt"].values.astype(np.float32)
    prob_cloudy  = ds["era5_prob_cloudy"].values.astype(np.float32)
    prob_clear   = ds["era5_prob_clear"].values.astype(np.float32)
    prob_partial = ds["era5_prob_partial"].values.astype(np.float32)
    bimodality   = ds["era5_bimodality"].values.astype(np.float32)
    mean_tcc     = ds["era5_mean_tcc"].values.astype(np.float32)     # valid all hours
    t2m_night    = ds["era5_t2m_night"].values.astype(np.float32)    # NaN at daytime
    rh_night     = ds["era5_rh_night"].values.astype(np.float32)     # NaN at daytime
    ds.close()

    # ── 3. Build cyclic month encodings (broadcast across all 24 hours) ───────
    print("\nSection 3 — Building cyclic month encodings")
    # Shape (12, 24): same value for all hours of a given month
    month_sin = np.tile(
        np.sin(2 * np.pi * months / 12).astype(np.float32)[:, np.newaxis],
        (1, 24),
    )
    month_cos = np.tile(
        np.cos(2 * np.pi * months / 12).astype(np.float32)[:, np.newaxis],
        (1, 24),
    )

    # ── 4. Assemble output dataset ─────────────────────────────────────────────
    print("\nSection 4 — Assembling output dataset")
    feat_dict = {
        "era5_mean_kt":      (["month", "hour"], mean_kt),
        "era5_std_kt":       (["month", "hour"], std_kt),
        "era5_prob_cloudy":  (["month", "hour"], prob_cloudy),
        "era5_prob_clear":   (["month", "hour"], prob_clear),
        "era5_prob_partial": (["month", "hour"], prob_partial),
        "era5_bimodality":   (["month", "hour"], bimodality),
        "era5_mean_tcc":     (["month", "hour"], mean_tcc),
        "era5_t2m_night":    (["month", "hour"], t2m_night),
        "era5_rh_night":     (["month", "hour"], rh_night),
        "era5_month_sin":    (["month", "hour"], month_sin),
        "era5_month_cos":    (["month", "hour"], month_cos),
    }

    ds_out = xr.Dataset(
        data_vars=feat_dict,
        coords={
            "month": months.astype(np.int32),
            "hour":  hours.astype(np.int32),
        },
        attrs={
            "description":    "ERA5 climatological features for LSTM injection (1990-2020)",
            "source":         str(CLIM_SRC),
            "n_features":     "11",
            "night_hours":    "20,21,22,23,0,1,2,3,4,5,6",
            "nan_convention": "NaN stored for inapplicable hours; fill 0 at sequence-build time",
            "created_by":     "compute_era5_climatology_features.py",
        },
    )

    # ── 5. Save ───────────────────────────────────────────────────────────────
    print("\nSection 5 — Saving")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(OUT_PATH, engine="h5netcdf")
    print(f"  Saved: {OUT_PATH}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n-- Feature summary (12x24 = 288 cells per variable) --------------")
    print(f"  {'Variable':<22}  {'Valid':>6}  {'NaN':>5}  {'Min':>8}  {'Mean':>8}  {'Max':>8}")
    print("  " + "-" * 65)
    for name, (_, arr) in feat_dict.items():
        valid = arr[~np.isnan(arr)]
        n_nan = int(np.isnan(arr).sum())
        if len(valid) == 0:
            print(f"  {name:<22}  {'all NaN':>6}")
        else:
            print(f"  {name:<22}  {len(valid):>6}  {n_nan:>5}  "
                  f"{valid.min():>8.3f}  {valid.mean():>8.3f}  {valid.max():>8.3f}")

    # Sanity checks
    print("\n-- Sanity checks --------------------------------------------------")
    print(f"  kt stats have NaN at night hours (expected) : "
          f"{np.isnan(mean_kt[:, 12]).all()}  (hour=12 local midnight)")
    print(f"  t2m_night non-NaN at hour 22 (night)        : "
          f"{np.isfinite(t2m_night[:, 22]).all()}")
    print(f"  t2m_night NaN at hour 12 (noon)             : "
          f"{np.isnan(t2m_night[:, 12]).all()}")
    print(f"  mean_tcc has no NaN                         : "
          f"{not np.isnan(mean_tcc).any()}")
    print(f"  month_sin range                             : "
          f"{month_sin.min():.3f} to {month_sin.max():.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
