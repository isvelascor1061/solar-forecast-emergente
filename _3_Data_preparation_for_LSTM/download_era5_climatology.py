#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_era5_climatology.py
============================
Download ERA5 hourly single-level data for Medellín (1990-2020),
compute monthly × hourly climatological statistics, and save to NetCDF.

Pipeline
--------
  1. Download    : 4 CDS requests per year (one per variable) →
                     ERA5_climatology/raw/era5_medellin_{var}_{year}.nc
                   Merge the 4 variable files → era5_medellin_{year}.nc
                   Delete per-variable files after merge
  2. Merge       : all yearly files → ERA5_climatology/era5_medellin_1990_2020.nc
  3. Deaccumulate: ssrd (J/m²) → hourly GHI (W/m²), in UTC before time-shift
  4. Local time  : shift UTC → Colombia local (UTC-5)
  5. RH          : compute from t2m and d2m via Magnus approximation
  6. kt          : clearness index = GHI / ext_ghi  (daytime only, ext_ghi from EXT_GHI_FILE)
  7. Climatology : aggregate per (local_month, local_hour) → 9 variables on a (12, 24) grid
  8. Save        : era5_climatology_1990_2020.nc

Output variables (all on a month × hour grid, float32)
-------------------------------------------------------
  era5_mean_kt      : mean clearness index            (daytime only)
  era5_std_kt       : std  clearness index            (daytime only)
  era5_prob_cloudy  : fraction where kt < 0.3         (daytime only)
  era5_prob_clear   : fraction where kt > 0.7         (daytime only)
  era5_prob_partial : fraction where 0.3 ≤ kt ≤ 0.7  (daytime only)
  era5_bimodality   : prob_cloudy × prob_clear         (daytime only)
  era5_mean_tcc     : mean total cloud cover           (all hours)
  era5_t2m_night    : mean 2m temperature             (night hours 20-06 local, NaN otherwise)
  era5_rh_night     : mean relative humidity           (night hours 20-06 local, NaN otherwise)

Note on the 10th climatological feature
-----------------------------------------
  month_sin and month_cos (cyclic month encoding) are computed at sequence-
  building time from the timestamp, as in the existing SIATA pipeline.
  They are NOT stored in this file.

Usage
-----
    set PYTHONPATH=C:\\Users\\isabe\\Projects\\codigors\\carpetasdetrabajo
    python _3_Data_preparation_for_LSTM/download_era5_climatology.py
"""

import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi

from config import (
    PREP_DATA_DIR,
    EXT_GHI_FILE, EXT_VAR_NAME,
    UTC_OFFSET,
)

# ── Output paths ──────────────────────────────────────────────────────────────
ERA5_DIR     = os.path.join(PREP_DATA_DIR, "ERA5_climatology")
RAW_DIR      = os.path.join(ERA5_DIR, "raw")
MERGED_FILE  = os.path.join(ERA5_DIR, "era5_medellin_1990_2020.nc")
CLIM_FILE    = os.path.join(PREP_DATA_DIR, "era5_climatology_1990_2020.nc")

# ── Download parameters ───────────────────────────────────────────────────────
YEARS        = list(range(1990, 2021))                       # 1990 … 2020  (31 years)
MONTHS       = [f"{m:02d}" for m in range(1, 13)]
DAYS         = [f"{d:02d}" for d in range(1, 32)]           # CDS ignores invalid day/month combos
HOURS        = [f"{h:02d}:00" for h in range(24)]
AREA         = [6.25, -75.5, 6.25, -75.5]                   # N, W, S, E  (single point)

# Mapping: CDS long variable name → short name used in filenames and Dataset
VAR_SHORT = {
    "surface_solar_radiation_downwards": "ssrd",             # J/m², accumulated
    "total_cloud_cover":                 "tcc",              # 0-1 fraction
    "2m_temperature":                    "t2m",              # K
    "2m_dewpoint_temperature":           "d2m",              # K
}

# Local night hours used for t2m_night and rh_night features
NIGHT_HOURS = set(list(range(20, 24)) + list(range(0, 7)))  # 20,21,22,23,0,1,2,3,4,5,6

# Physical ceiling for clearness index (small margin for diffuse enhancement)
KT_MAX = 1.2


# ── Section 1 — Download ──────────────────────────────────────────────────────

def download_years() -> None:
    """
    Download ERA5 data one variable at a time per year (4 requests/year).
    Each request is ~4× smaller than a 4-variable request, avoiding the
    CDS 403 "cost limits exceeded" error.

    Resume logic (three levels):
      - If era5_medellin_{year}.nc exists   → skip the entire year
      - If era5_medellin_{var}_{year}.nc exists → skip that variable
      - Otherwise download and save the variable file

    After all 4 variables are present for a year, they are merged into a
    single era5_medellin_{year}.nc and the per-variable files are deleted.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    client = cdsapi.Client()

    for year in YEARS:
        year_file = os.path.join(RAW_DIR, f"era5_medellin_{year}.nc")

        if os.path.exists(year_file):
            size_mb = os.path.getsize(year_file) / 1e6
            print(f"Year {year}: already merged ({size_mb:.1f} MB), skipping.")
            continue

        # Download each variable in a separate CDS request
        var_paths = {}
        for cds_name, short in VAR_SHORT.items():
            var_path = os.path.join(RAW_DIR, f"era5_medellin_{short}_{year}.nc")
            var_paths[short] = var_path

            if os.path.exists(var_path):
                size_mb = os.path.getsize(var_path) / 1e6
                print(f"  Year {year} / {short}: already downloaded "
                      f"({size_mb:.1f} MB), skipping.")
                continue

            print(f"  Year {year} / {short}: downloading ...", flush=True)
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable":     [cds_name],
                    "year":         str(year),
                    "month":        MONTHS,
                    "day":          DAYS,
                    "time":         HOURS,
                    "area":         AREA,
                    "format":       "netcdf",
                },
                var_path,
            )
            size_mb = os.path.getsize(var_path) / 1e6
            print(f"  Year {year} / {short}: done ({size_mb:.1f} MB)")

        # Merge the 4 single-variable files into one year file
        print(f"Year {year}: merging variables ...", flush=True)
        open_datasets = {short: xr.open_dataset(p, engine="h5netcdf")
                         for short, p in var_paths.items()}
        ds_year = xr.merge(open_datasets.values())

        # Drop the size-1 lat/lon dimensions produced by a single-point request
        spatial_dims = [d for d in ds_year.dims if d in ("latitude", "longitude")]
        if spatial_dims:
            ds_year = ds_year.squeeze(dim=spatial_dims, drop=True)

        ds_year.to_netcdf(year_file, engine="h5netcdf")

        # Close merged dataset first, then every source dataset individually
        # before attempting deletion — Windows holds a lock on each open handle
        ds_year.close()
        for ds in open_datasets.values():
            ds.close()

        # Small delay to ensure the OS has fully released all file locks
        time.sleep(2)

        size_mb = os.path.getsize(year_file) / 1e6
        print(f"Year {year}: merged and saved ({size_mb:.1f} MB)")

        # Remove per-variable files to keep disk usage low
        for p in var_paths.values():
            if os.path.exists(p):
                try:
                    os.remove(p)
                except PermissionError:
                    print(f"  Warning: could not delete {p} (still locked) — "
                          f"file is expendable, skipping.")


# ── Section 2 — Merge ─────────────────────────────────────────────────────────

def merge_years() -> xr.Dataset:
    """
    Concatenate all yearly NetCDF files along the time axis into one Dataset.
    Spatial dimensions were already squeezed/dropped in download_years(), so
    no spatial averaging is needed here.
    Saves MERGED_FILE and returns the Dataset. Skip if MERGED_FILE exists.
    """
    if os.path.exists(MERGED_FILE):
        print(f"Merged file already exists: {MERGED_FILE}")
        return xr.open_dataset(MERGED_FILE, engine="h5netcdf")

    print("Merging yearly files ...", flush=True)
    paths = [os.path.join(RAW_DIR, f"era5_medellin_{y}.nc") for y in YEARS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} yearly file(s) not found. Run download_years() first.\n"
            f"First missing: {missing[0]}"
        )

    ds = xr.open_mfdataset(
        sorted(paths),
        combine="by_coords",
        engine="h5netcdf",
        chunks={"time": 8760},    # ~1 year per chunk for manageable memory
    )

    os.makedirs(ERA5_DIR, exist_ok=True)
    print(f"Writing merged file: {MERGED_FILE} ...", flush=True)
    ds.to_netcdf(MERGED_FILE, engine="h5netcdf")
    print(f"Merged file saved  : {MERGED_FILE}")

    # Re-open without chunks for in-memory processing
    return xr.open_dataset(MERGED_FILE, engine="h5netcdf")


# ── Section 3 — Deaccumulate ssrd ─────────────────────────────────────────────

def deaccumulate_ssrd(da_ssrd: xr.DataArray) -> xr.DataArray:
    """
    ERA5 ssrd is cumulative from 00:00 UTC each calendar day.

    Deaccumulation:
        hourly_J = ssrd(t) - ssrd(t-1)     [backward diff; result labelled at t]
        clip(min=0)                          [midnight reset gives large negative → 0]
        divide by 3600                       [J/m² → W/m²]

    The resulting time coordinate is shifted by one step (starts at index 1),
    so the first timestamp of the output corresponds to the second input timestamp.
    """
    # xr.diff with label='upper' (default): result[t] = ssrd[t] - ssrd[t-1]
    # Output length = n_time - 1; coordinate = time[1:]
    hourly_j = da_ssrd.diff(dim="time")
    hourly_j = hourly_j.clip(min=0)
    ghi_wm2  = hourly_j / 3600.0
    ghi_wm2.attrs["units"]       = "W m**-2"
    ghi_wm2.attrs["long_name"]   = "Surface solar radiation downwards (hourly, deaccumulated)"
    return ghi_wm2


# ── Section 5 — Relative humidity ─────────────────────────────────────────────

def compute_rh(t2m_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
    """
    Relative humidity (%) from temperature and dewpoint (both in Kelvin).

    Magnus approximation (Alduchov & Eskridge 1996):
        RH = 100 × exp(17.625 × Td_c / (243.04 + Td_c))
                 / exp(17.625 × T_c  / (243.04 + T_c ))
    where T_c = T_k - 273.15 (Celsius).
    """
    t_c  = t2m_k - 273.15
    td_c = d2m_k - 273.15
    alpha = 17.625 * td_c / (243.04 + td_c)
    beta  = 17.625 * t_c  / (243.04 + t_c)
    rh    = 100.0 * np.exp(alpha) / np.exp(beta)
    return np.clip(rh, 0.0, 100.0)


# ── Section 6 — Extraterrestrial GHI lookup ──────────────────────────────────

def build_ext_lookup(local_time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Load EXT_GHI_FILE (which uses local Colombia time) and build a
    (month, day, hour) → ext_ghi lookup averaged across all years in the file.

    Extraterrestrial GHI varies by less than 0.1 W/m² between years for any
    given (month, day, hour), so averaging across years is appropriate.

    Returns an array of ext_ghi values aligned with local_time_index.
    """
    print(f"Loading extraterrestrial GHI: {EXT_GHI_FILE}", flush=True)
    ds_ext  = xr.open_dataset(EXT_GHI_FILE, engine="h5netcdf")
    da_ext  = ds_ext[EXT_VAR_NAME].squeeze()

    ext_series = da_ext.to_series()
    idx        = pd.DatetimeIndex(ext_series.index)

    # Build lookup: mean ext_ghi per (month, day, hour) across all years in the file
    lookup = (
        ext_series
        .groupby([idx.month, idx.day, idx.hour])
        .mean()
        .to_dict()
    )  # keys: (month, day, hour)

    # Map every ERA5 local timestamp to its ext_ghi value
    ext_values = np.array([
        lookup.get((t.month, t.day, t.hour), np.nan)
        for t in local_time_index
    ], dtype=np.float64)

    n_nan = int(np.isnan(ext_values).sum())
    if n_nan > 0:
        print(f"  Warning: {n_nan} ERA5 timestamps have no ext_ghi match "
              f"(e.g. Feb 29 not in EXT_GHI_FILE). These will be treated as NaN.")

    return ext_values


# ── Section 7 — Aggregate per (local_month, local_hour) ──────────────────────

def to_2d_array(stat: pd.Series) -> np.ndarray:
    """
    Convert a (month, hour)-indexed Series into a (12, 24) float32 array.
    Month-hour combinations absent from `stat` remain NaN (e.g. night hours
    in a daytime-only statistic, or daytime hours in a night-only statistic).
    """
    arr = np.full((12, 24), np.nan, dtype=np.float32)
    for (m, h), val in stat.items():
        arr[int(m) - 1, int(h)] = float(val)
    return arr


def compute_climatology(ds: xr.Dataset) -> dict:
    """
    Process the merged ERA5 Dataset and return a dict of (12, 24) float32
    arrays, one per output climatological variable.
    """

    # ── 3. Deaccumulate ssrd → W/m²  (must be done in UTC before time-shift) ─
    print("Deaccumulating ssrd ...", flush=True)
    ghi_wm2  = deaccumulate_ssrd(ds["ssrd"])
    time_utc = pd.DatetimeIndex(ghi_wm2["time"].values)

    # Align tcc, t2m, d2m to the shorter time axis (same as ghi_wm2 after diff)
    tcc = ds["tcc"].sel(time=ghi_wm2["time"]).values.astype(np.float64)
    t2m = ds["t2m"].sel(time=ghi_wm2["time"]).values.astype(np.float64)
    d2m = ds["d2m"].sel(time=ghi_wm2["time"]).values.astype(np.float64)
    ghi = ghi_wm2.values.astype(np.float64)

    # ── 4. UTC → local time (Colombia UTC-5) ─────────────────────────────────
    # UTC_OFFSET is -5; shift converts UTC timestamps to local observation time
    time_local  = time_utc + pd.Timedelta(hours=UTC_OFFSET)
    local_month = time_local.month   # 1–12
    local_hour  = time_local.hour    # 0–23

    # ── 5. Relative humidity ──────────────────────────────────────────────────
    print("Computing relative humidity ...", flush=True)
    rh = compute_rh(t2m, d2m)

    # ── 6. Clearness index kt = GHI / ext_ghi  (daytime only) ────────────────
    print("Computing clearness index kt ...", flush=True)
    ext_values = build_ext_lookup(time_local)

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(ext_values > 0, ghi / ext_values, np.nan)
    kt = np.where(np.isfinite(kt), np.clip(kt, 0.0, KT_MAX), np.nan)

    is_daytime = ext_values > 0
    is_night   = np.isin(local_hour, sorted(NIGHT_HOURS))

    # ── 7. Build DataFrame for groupby ────────────────────────────────────────
    print("Building DataFrame for aggregation ...", flush=True)
    df = pd.DataFrame({
        "kt":    kt,
        "tcc":   tcc,
        "t2m":   t2m,
        "rh":    rh,
        "month": local_month,
        "hour":  local_hour,
    })

    # kt statistics — daytime only (ext_ghi > 0)
    df_day  = df[is_daytime].copy()
    grp_day = df_day.groupby(["month", "hour"])["kt"]

    mean_kt      = grp_day.mean()
    std_kt       = grp_day.std()
    prob_cloudy  = grp_day.apply(lambda x: (x < 0.3).mean())
    prob_clear   = grp_day.apply(lambda x: (x > 0.7).mean())
    prob_partial = grp_day.apply(lambda x: ((x >= 0.3) & (x <= 0.7)).mean())
    bimodality   = prob_cloudy * prob_clear     # high when clear AND cloudy are both frequent

    # tcc — all hours
    mean_tcc = df.groupby(["month", "hour"])["tcc"].mean()

    # t2m and rh — night hours only; groupby will only produce (month, night_hour) keys
    # so daytime hours will be NaN in the output array naturally
    df_night  = df[is_night].copy()
    grp_night = df_night.groupby(["month", "hour"])
    t2m_night = grp_night["t2m"].mean()
    rh_night  = grp_night["rh"].mean()

    # ── Pack into (12, 24) float32 arrays ────────────────────────────────────
    print("Packing into (12, 24) arrays ...", flush=True)
    return {
        "era5_mean_kt":      to_2d_array(mean_kt),
        "era5_std_kt":       to_2d_array(std_kt),
        "era5_prob_cloudy":  to_2d_array(prob_cloudy),
        "era5_prob_clear":   to_2d_array(prob_clear),
        "era5_prob_partial": to_2d_array(prob_partial),
        "era5_bimodality":   to_2d_array(bimodality),
        "era5_mean_tcc":     to_2d_array(mean_tcc),
        "era5_t2m_night":    to_2d_array(t2m_night),
        "era5_rh_night":     to_2d_array(rh_night),
    }


# ── Section 8 — Save ──────────────────────────────────────────────────────────

def save_climatology(clim: dict) -> None:
    """Save climatological statistics to CLIM_FILE as a NetCDF Dataset."""
    ds_out = xr.Dataset(
        data_vars={
            name: (["month", "hour"], arr)
            for name, arr in clim.items()
        },
        coords={
            "month": np.arange(1, 13, dtype=np.int32),
            "hour":  np.arange(0, 24, dtype=np.int32),
        },
        attrs={
            "description":         "ERA5 climatological statistics for Medellín (1990-2020)",
            "source_dataset":      "reanalysis-era5-single-levels",
            "location":            "Medellín, Colombia",
            "coordinates":         "6.25°N, 75.5°W",
            "area_point":          "6.25°N, 75.5°W (single ERA5 grid point)",
            "period":              "1990-2020 (31 years)",
            "time_zone":           "UTC-5 (Colombia, no daylight saving)",
            "ext_ghi_source":      EXT_GHI_FILE,
            "ssrd_processing":     "diff(ssrd)/3600, clipped to 0 → hourly W/m²",
            "kt_thresholds":       "cloudy: kt<0.3  partial: 0.3≤kt≤0.7  clear: kt>0.7",
            "kt_ceiling":          str(KT_MAX),
            "night_hours_local":   "20,21,22,23,0,1,2,3,4,5,6",
            "rh_formula":          "Magnus approx. (Alduchov & Eskridge 1996)",
            "created_by":          "download_era5_climatology.py",
        },
    )
    ds_out.to_netcdf(CLIM_FILE, engine="h5netcdf")
    print(f"\nClimatology saved: {CLIM_FILE}")


# ── Summary printout ──────────────────────────────────────────────────────────

def print_summary(clim: dict) -> None:
    """Print min / mean / max for each output variable."""
    print("\n── ERA5 climatology summary ──────────────────────────────────────")
    print(f"  {'Variable':<22}  {'Valid cells':>11}  {'Min':>8}  {'Mean':>8}  {'Max':>8}")
    print("  " + "-" * 65)
    for name, arr in clim.items():
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            print(f"  {name:<22}  {'all NaN':>11}")
        else:
            print(
                f"  {name:<22}  {len(valid):>11d}"
                f"  {valid.min():>8.3f}  {valid.mean():>8.3f}  {valid.max():>8.3f}"
            )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 66)
    print("  ERA5 climatology downloader — Medellín 1990-2020")
    print("=" * 66)

    # Step 1 — download one NetCDF per year (resume-capable)
    download_years()

    # Step 2 — merge into a single spatially-averaged Dataset (resume-capable)
    ds_merged = merge_years()

    # Steps 3-7 — deaccumulate, shift to local time, compute RH, kt, aggregate
    clim = compute_climatology(ds_merged)

    # Step 8 — save climatology NetCDF
    save_climatology(clim)

    # Summary
    print_summary(clim)


if __name__ == "__main__":
    main()
