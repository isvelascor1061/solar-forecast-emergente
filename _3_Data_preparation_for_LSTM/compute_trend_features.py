#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_trend_features.py
=========================
Compute temporal trend features from GFS best-lead CSI and TCDC series.

These 6 features give the BiLSTM explicit information about the rate and
direction of change in sky conditions — critical for distinguishing
dissipating morning fog (CSI increasing) from persistent fog (CSI flat).

Features computed (all indexed by observation_time)
----------------------------------------------------
  csi_trend_1h       : CSI(t) − CSI(t−1)           — immediate change
  csi_trend_3h       : CSI(t) − CSI(t−3)           — short-range trend
  csi_trend_6h       : CSI(t) − CSI(t−6)           — full morning evolution
  csi_volatility_3h  : rolling std over 3 h         — sky stability
  csi_is_increasing  : 1 if csi_trend_1h > 0.05    — binary clearing flag
  nocturnal_tcdc_mean: mean TCDC for hours 20-06
                       of the preceding night        — fog persistence proxy

Best-lead strategy (same as verify_skillscore.py)
--------------------------------------------------
  For each observation_time, select the GFS value from the launch time with
  the shortest positive lead time.  This produces one coherent hourly series.

Usage
-----
    set PYTHONPATH=C:\\Users\\isabe\\Projects\\codigors\\carpetasdetrabajo
    python _3_Data_preparation_for_LSTM/compute_trend_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    LAUNCH_TIMES,
    VAR_GFS_CSI_TEMPLATE,    # "clearsky_index_GFS_{LT}"
    FEAT_KC_TEMPLATE,        # path template for CSI files (clear-sky index)
    FEAT_TCDC_TEMPLATE,      # path template for TCDC files
    SIATA_CSI_FILE, SIATA_CSI_VAR,
    TREND_FEATURES_FILE,
)


# ── Section 1 — Best-lead series builder ──────────────────────────────────────

def load_best_lead_series(
    lt_list: list,
    file_template: str,
    var_template: str,
) -> pd.Series:
    """
    Load one NetCDF per launch time, compute lead times, and return a Series
    with the value from the shortest positive lead at each observation_time.

    Parameters
    ----------
    lt_list       : e.g. ["0100", "0700", "1300", "1900"]
    file_template : path string with {LT} placeholder
    var_template  : variable name string with {LT} placeholder

    Returns
    -------
    pd.Series indexed by observation_time, one value per hour.
    """
    val_frames  = []
    lead_frames = []

    for lt in lt_list:
        nc_path  = ROOT / file_template.format(LT=lt)
        var_name = var_template.format(LT=lt)

        ds  = xr.open_dataset(nc_path, engine="h5netcdf")
        da  = ds[var_name]

        # Squeeze all singleton dimensions except the time axis
        for dim in list(da.dims):
            if dim != "observation_time":
                da = da.isel({dim: 0})

        ser_val = da.to_series()   # index = observation_time

        # Lead time = observation_time − launch_time
        launch_ser        = pd.to_datetime(ds["launch_time"].to_series())
        launch_ser.index  = ser_val.index
        lead              = ser_val.index.to_series() - launch_ser
        lead[lead <= pd.Timedelta(0)] = pd.NaT   # mark non-positive leads as invalid

        val_frames.append(ser_val.rename(lt))
        lead_frames.append(lead.rename(lt))
        ds.close()

        n_valid = int((~lead.isna()).sum())
        print(f"  LT {lt}: {len(ser_val):,} rows | valid positive leads: {n_valid:,}")

    val_df  = pd.concat(val_frames,  axis=1)
    lead_df = pd.concat(lead_frames, axis=1)

    # For each row, pick the column (launch time) with the minimum positive lead
    idx_min = lead_df.idxmin(axis=1)
    col_pos = val_df.columns.get_indexer(idx_min)
    row_pos = np.arange(len(val_df))
    best    = val_df.to_numpy()[row_pos, col_pos]

    series = pd.Series(best, index=val_df.index).dropna().sort_index()
    print(f"  → Best-lead series: {len(series):,} valid hours")
    return series


# ── Section 2 — CSI trend and volatility features ────────────────────────────

def safe_diff(series: pd.Series, n_hours: int) -> pd.Series:
    """
    Compute series(t) − series(t−n) only where the actual time gap between
    consecutive index entries is exactly n hours.  Gaps caused by day
    boundaries or missing data are filled with 0.0 instead of NaN.
    """
    shifted  = series.shift(n_hours)
    time_gap = series.index.to_series().diff(n_hours)
    exact    = time_gap == pd.Timedelta(hours=n_hours)
    diff     = (series - shifted).where(exact, other=0.0)
    return diff.fillna(0.0)


def compute_csi_features(csi: pd.Series) -> pd.DataFrame:
    """
    Compute the 5 CSI-based trend and volatility features.
    All edge values and time-gap positions are filled with 0.0.
    """
    trend_1h = safe_diff(csi, 1)
    trend_3h = safe_diff(csi, 3)
    trend_6h = safe_diff(csi, 6)

    # Rolling std over 3 h; require all 3 values present (min_periods=3)
    volatility_3h = (
        csi.rolling(window=3, min_periods=3).std()
        .fillna(0.0)
    )

    # Binary flag: radiation is clearly increasing (threshold = 0.05 CSI units)
    is_increasing = (trend_1h > 0.05).astype(np.float32)

    return pd.DataFrame({
        "csi_trend_1h":      trend_1h.astype(np.float32),
        "csi_trend_3h":      trend_3h.astype(np.float32),
        "csi_trend_6h":      trend_6h.astype(np.float32),
        "csi_volatility_3h": volatility_3h.astype(np.float32),
        "csi_is_increasing": is_increasing,
    }, index=csi.index)


# ── Section 3 — Nocturnal TCDC mean ───────────────────────────────────────────

def compute_nocturnal_tcdc(
    target_index: pd.DatetimeIndex,
    tcdc: pd.Series,
) -> pd.Series:
    """
    For each calendar date D, compute the mean TCDC of the preceding night:
        hours 20-23 of day D-1  (evening before)
      + hours  0-6 of day D     (early morning)

    Then broadcast that nightly mean to ALL hours of day D (including daytime).
    A high value means last night was overcast → morning fog likely → model
    should predict slower dissipation at hours 8-10.

    The first calendar date (no previous night in the data) is filled with
    the dataset-wide mean TCDC.

    Parameters
    ----------
    target_index : DatetimeIndex of the CSI best-lead series (output index)
    tcdc         : best-lead TCDC series (observation_time index, local time)

    Returns
    -------
    pd.Series indexed by target_index, dtype float32.
    """
    hour_arr = tcdc.index.hour

    # Each night-hour timestamp is assigned to the morning date it leads into:
    #   hours  0-6  of day D   → night_date = D        (already the morning)
    #   hours 20-23 of day D-1 → night_date = D        (shift evening by +1 day)
    is_evening = np.isin(hour_arr, list(range(20, 24)))
    is_morning = np.isin(hour_arr, list(range(0, 7)))
    is_night   = is_evening | is_morning

    date_arr   = tcdc.index.normalize()    # midnight Timestamp for each entry
    night_date = date_arr + pd.to_timedelta(is_evening.astype(int), unit="D")

    # Mean TCDC per night_date, using only the night-hour rows
    night_mean = (
        pd.Series(tcdc.values[is_night].astype(np.float32),
                  index=night_date[is_night])
        .groupby(level=0)
        .mean()
    )   # index: midnight Timestamps, one per calendar date

    # Broadcast: map every target timestamp to its date's nocturnal mean
    overall_mean   = float(night_mean.mean())
    lookup_dates   = target_index.normalize()
    nocturnal_vals = lookup_dates.map(night_mean).fillna(overall_mean)

    return pd.Series(
        nocturnal_vals.values.astype(np.float32),
        index=target_index,
        name="nocturnal_tcdc_mean",
    )


# ── Section 4 — Optional correlation summary ──────────────────────────────────

def print_correlation_with_target(df_feat: pd.DataFrame) -> None:
    """
    If SIATA_CSI_FILE is accessible, print the Pearson correlation of each
    trend feature with the observed clear-sky index at matching timestamps.
    """
    try:
        ds_siata = xr.open_dataset(ROOT / SIATA_CSI_FILE, engine="h5netcdf")
        da_siata = ds_siata[SIATA_CSI_VAR].squeeze()
        for dim in list(da_siata.dims):
            if dim != "observation_time":
                da_siata = da_siata.isel({dim: 0})
        siata = da_siata.to_series().rename("siata_csi").dropna()
        ds_siata.close()

        common   = df_feat.index.intersection(siata.index)
        df_align = df_feat.loc[common]
        y_align  = siata.loc[common]
        daytime  = y_align > 0

        print("\n  Correlation with SIATA CSI:")
        print(f"  {'Feature':<25}  {'r (all)':>9}  {'r (daytime)':>12}")
        print(f"  {'-'*25}  {'-'*9}  {'-'*12}")
        for col in df_feat.columns:
            r_all = df_align[col].corr(y_align)
            r_day = df_align.loc[daytime, col].corr(y_align[daytime])
            print(f"  {col:<25}  {r_all:>9.4f}  {r_day:>12.4f}")
    except Exception as exc:
        print(f"\n  [Correlation skipped: {exc}]")


# ── Section 5 — Save ──────────────────────────────────────────────────────────

def save_features(df_feat: pd.DataFrame) -> None:
    """Save all 6 trend features to a NetCDF file."""
    out_path = ROOT / TREND_FEATURES_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_out = xr.Dataset(
        {col: (["time"], df_feat[col].values) for col in df_feat.columns},
        coords={"time": df_feat.index.values},
        attrs={
            "description":       "GFS temporal trend and nocturnal TCDC features for Medellín",
            "csi_source":        f"clearsky_index_GFS_LT.nc (best-lead across {LAUNCH_TIMES})",
            "tcdc_source":       f"TCDC_ent_LT.nc (best-lead across {LAUNCH_TIMES})",
            "lead_strategy":     "shortest positive lead time per observation_time",
            "night_hours_local": "20,21,22,23,0,1,2,3,4,5,6",
            "is_increasing_thr": "csi_trend_1h > 0.05",
            "edge_fill":         "0.0 for trend/volatility (incomplete windows or gaps)",
            "created_by":        "compute_trend_features.py",
        },
    )
    ds_out.to_netcdf(out_path, engine="h5netcdf")
    print(f"\n  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("  compute_trend_features.py")
    print("=" * 62)

    # ── 1. Best-lead GFS CSI series ──────────────────────────────────────────
    print("\nSection 1 — Building best-lead GFS CSI series")
    csi = load_best_lead_series(
        lt_list       = LAUNCH_TIMES,
        file_template = FEAT_KC_TEMPLATE,      # clearsky_index_GFS_{LT}.nc
        var_template  = VAR_GFS_CSI_TEMPLATE,  # "clearsky_index_GFS_{LT}"
    )

    # ── 2. Best-lead GFS TCDC series ─────────────────────────────────────────
    print("\nSection 2 — Building best-lead GFS TCDC series")
    tcdc = load_best_lead_series(
        lt_list       = LAUNCH_TIMES,
        file_template = FEAT_TCDC_TEMPLATE,    # TCDC_ent_{LT}.nc
        var_template  = "TCDC_ent_{LT}",
    )

    # ── 3. Compute the 5 CSI-based features ──────────────────────────────────
    print("\nSection 3 — Computing CSI trend and volatility features")
    df_csi_feat = compute_csi_features(csi)
    print(f"  csi_trend_1h non-zero: "
          f"{(df_csi_feat['csi_trend_1h'] != 0).sum():,} / {len(df_csi_feat):,}")

    # ── 4. Compute nocturnal TCDC mean ───────────────────────────────────────
    print("\nSection 4 — Computing nocturnal TCDC mean")
    nocturnal_tcdc = compute_nocturnal_tcdc(csi.index, tcdc)
    print(f"  Unique daily means: {nocturnal_tcdc.nunique():,}")
    print(f"  TCDC range: {nocturnal_tcdc.min():.1f} – {nocturnal_tcdc.max():.1f}")

    # ── 5. Assemble final DataFrame ───────────────────────────────────────────
    df_feat = df_csi_feat.copy()
    df_feat["nocturnal_tcdc_mean"] = nocturnal_tcdc.reindex(csi.index).fillna(
        float(nocturnal_tcdc.mean())
    ).values

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print("\n── Feature summary ──────────────────────────────────────────────")
    print(f"  Total hours: {len(df_feat):,}")
    print(f"  Range: {df_feat.index.min()} → {df_feat.index.max()}")
    print()
    print(f"  {'Feature':<25}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'NaN':>5}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}")
    for col in df_feat.columns:
        s     = df_feat[col]
        n_nan = int(s.isna().sum())
        print(f"  {col:<25}  {s.mean():>8.4f}  {s.std():>8.4f}"
              f"  {s.min():>8.4f}  {s.max():>8.4f}  {n_nan:>5}")

    # Sanity: nocturnal_tcdc_mean should span 0-100 range with visible variance
    tcdc_std = float(df_feat["nocturnal_tcdc_mean"].std())
    if tcdc_std < 1.0:
        print("\n  WARNING: nocturnal_tcdc_mean has very low std — check TCDC units.")
    else:
        print(f"\n  nocturnal_tcdc_mean std = {tcdc_std:.2f}  (informative range confirmed)")

    # Optional: correlation with SIATA CSI
    print_correlation_with_target(df_feat)

    # ── 7. Save ──────────────────────────────────────────────────────────────
    print("\nSection 5 — Saving to NetCDF")
    save_features(df_feat)
    print("\nDone.")


if __name__ == "__main__":
    main()
