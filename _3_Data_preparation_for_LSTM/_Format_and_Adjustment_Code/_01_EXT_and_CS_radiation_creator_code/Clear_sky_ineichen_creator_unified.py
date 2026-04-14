#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clear_sky_ineichen_creator_unified.py
======================================
Merges Clear_sky_ineichen_radiation_creator.py and
Clear_sky_readiation_creat_with_bias_correct.py into a single script.

Computes the Clear-Sky Global Horizontal Irradiance (GHI_cs) for the
GFS grid cell centred on Medellín using the Ineichen model from pvlib.

Features:
  - Real horizon profile loaded from a TIF raster file (rasterio).
  - Spatial sampling of 25 points within the 0.25° × 0.25° grid cell.
  - Configurable zenith bias correction (zenith_corr) imported from
    config.py: the Ineichen model tends to underestimate GHI at high
    zenith angles (> 60°); the factor f = 1 + α·max(0, θ − θ₀) compensates
    for this effect.
  - Output as a NetCDF with dimensions (observation_time, surface, lat, lon).

Main steps:
  1. Generate a minute-resolution time index for the period of interest.
  2. Load the horizon profile from the TIF file.
  3. For each of the 25 sampling points in the grid cell:
     a. Compute solar position and air mass.
     b. Estimate GHI_cs with the Ineichen model (Linke turbidity from table).
     c. Apply horizon mask (sun blocked by terrain relief = 0 W/m²).
     d. Apply zenith bias correction if enabled.
  4. Average the 25 points and aggregate to hourly resolution.
  5. Save the result as a NetCDF file.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import pvlib
from pvlib.clearsky import lookup_linke_turbidity
from pvlib.atmosphere import get_relative_airmass, get_absolute_airmass
from scipy.interpolate import interp1d
import rasterio

from config import (
    LAT, LON, ELEVATION,
    HORIZON_FILE, CSI_GHI_OUT_DIR,
    ZENITH_CORR, ZENITH_ALPHA, ZENITH_THRESHOLD,
)


class ClearSkyGridAverager:
    """
    Computes the Ineichen clear-sky GHI averaged over a 0.25° × 0.25° GFS
    grid cell, with an optional horizon profile and zenith bias correction.
    """

    def __init__(
        self,
        lat_center: float,
        lon_center: float,
        start_date: str,
        end_date: str,
        tz: str = "UTC",
        altitude: float = 0,
        model: str = "ineichen",
        linke_turbidity=None,
        seed: int | None = None,
        horizon_file: str | None = None,
        zenith_corr: bool = False,
        zenith_alpha: float = 0.0025,
        zenith_threshold: float = 65,
    ):
        """
        Parameters
        ----------
        lat_center       : Latitude of the GFS grid cell centre (degrees N).
        lon_center       : Longitude of the GFS grid cell centre (degrees E).
        start_date       : Start of the period (e.g. '2021-04-01 00:00').
        end_date         : End of the period (e.g. '2025-06-01 00:00').
        tz               : Time zone (e.g. 'America/Bogota').
        altitude         : Point altitude in metres.
        model            : pvlib clear-sky model (default 'ineichen').
        linke_turbidity  : Fixed Linke turbidity; None = automatic table lookup.
        seed             : Random seed for the sampling points.
        horizon_file     : Path to the TIF file with horizon elevation data.
        zenith_corr      : Enable zenith bias correction (imported from config).
        zenith_alpha     : Correction factor per degree of zenith (imported from config).
        zenith_threshold : Zenith angle above which the correction is applied (imported from config).
        """
        self.lat_center      = lat_center
        self.lon_center      = lon_center
        self.start_date      = start_date
        self.end_date        = end_date
        self.tz              = tz
        self.altitude        = altitude
        self.model           = model
        self.linke_turbidity = linke_turbidity
        self.horizon_file    = horizon_file
        self.zenith_corr     = zenith_corr
        self.zenith_alpha    = zenith_alpha
        self.zenith_threshold = zenith_threshold

        # Seed for reproducibility of the random sampling points
        if seed is not None:
            np.random.seed(seed)

        # Bounds of the 0.25° × 0.25° cell centred on lat/lon
        half = 0.125
        self.lat_min = lat_center - half
        self.lat_max = lat_center + half
        self.lon_min = lon_center - half
        self.lon_max = lon_center + half

    # ------------------------------------------------------------------
    def load_hgt(self, hgt_file: str) -> np.ndarray:
        """
        Loads the TIF elevation raster and extracts the data for
        the area of interest defined by the grid cell bounds.
        """
        with rasterio.open(hgt_file) as src:
            # Verify that the raster covers the grid cell area
            if (src.bounds.left > self.lon_max or
                    src.bounds.right < self.lon_min or
                    src.bounds.bottom > self.lat_max or
                    src.bounds.top < self.lat_min):
                raise ValueError("The TIF file does not cover the GFS grid cell area.")

            window = src.window(self.lon_min, self.lat_min,
                                self.lon_max, self.lat_max)
            hgt_data = src.read(1, window=window, boundless=True)
        return hgt_data

    # ------------------------------------------------------------------
    def calculate_horizon_profile(self, hgt_data: np.ndarray,
                                  resolution: int = 5):
        """
        Computes the horizon profile (elevation angle per azimuth) from
        the elevation data of the TIF raster.

        Parameters
        ----------
        hgt_data   : 2D array of altitudes in metres.
        resolution : Azimuthal resolution in degrees (default 5°).

        Returns
        -------
        azimuths   : Array of azimuths (0–355 °).
        elevations : Array of horizon elevation angles (°).
        """
        azimuths   = np.arange(0, 360, resolution)
        elevations = np.zeros_like(azimuths, dtype=float)

        # Centre point of the cell as the altitude reference
        center_row = hgt_data.shape[0] // 2
        center_col = hgt_data.shape[1] // 2
        center_alt = hgt_data[center_row, center_col]

        for i, az in enumerate(azimuths):
            max_distance = hgt_data.shape[1] // 2
            distances    = np.arange(1, max_distance)
            # Altitude difference relative to the centre point
            delta_elev   = hgt_data[center_row, 1:max_distance] - center_alt
            # Approximate elevation angle (distance in pixels ≈ ° × 90)
            tan_elev     = delta_elev / (distances * 90)
            elevations[i] = np.degrees(np.arctan(np.max(tan_elev)))

        return azimuths, elevations

    # ------------------------------------------------------------------
    def generate(self, output_dir: str, output_filename: str) -> None:
        """
        Runs the full computation and saves the result as a NetCDF file.

        Parameters
        ----------
        output_dir      : Destination folder for the NetCDF file.
        output_filename : Name of the output file.
        """
        os.makedirs(output_dir, exist_ok=True)

        # --- Minute-resolution time index for hourly aggregation -------
        # Computed at 1-minute resolution and then averaged per hour
        time_index   = pd.date_range(
            start=self.start_date, end=self.end_date,
            freq="min", tz=self.tz, inclusive="both"
        )
        group_labels = time_index.floor("h") + pd.Timedelta(hours=1)
        unique_hours = pd.DatetimeIndex(group_labels.unique())

        # --- Load horizon profile -------------------------------------
        # If a TIF file is provided, compute the horizon elevation profile
        # to mask hours when the sun is blocked by the terrain.
        horizon_azimuths   = None
        horizon_elevations = None
        if self.horizon_file:
            try:
                hgt_data = self.load_hgt(self.horizon_file)
                horizon_azimuths, horizon_elevations = \
                    self.calculate_horizon_profile(hgt_data)
            except Exception as e:
                print(f"Warning: could not load horizon profile: {e}")
                print("Continuing without horizon correction.")

        # --- Define the 25 sampling points within the grid cell --------
        # 9 fixed points (corners, edges, centre) + 16 random (4 per quadrant)
        lat_c, lon_c = self.lat_center, self.lon_center
        lmin,  lmax  = self.lat_min, self.lat_max
        omin,  omax  = self.lon_min, self.lon_max

        fijos = [
            (lmin, omin), (lmin, omax), (lmax, omin), (lmax, omax),
            (lat_c, omin), (lat_c, omax), (lmin, lon_c), (lmax, lon_c),
            (lat_c, lon_c)
        ]
        cuadrantes = [
            (lat_c, lmax, omin, lon_c),
            (lat_c, lmax, lon_c, omax),
            (lmin, lat_c, omin, lon_c),
            (lmin, lat_c, lon_c, omax),
        ]
        aleatorios = []
        for lat_lo, lat_hi, lon_lo, lon_hi in cuadrantes:
            pts = np.random.rand(4, 2)
            pts[:, 0] = pts[:, 0] * (lat_hi - lat_lo) + lat_lo
            pts[:, 1] = pts[:, 1] * (lon_hi - lon_lo) + lon_lo
            aleatorios.extend([tuple(pt) for pt in pts])

        muestras = fijos + aleatorios   # 25 points in total

        # --- Compute GHI_cs for each sampling point --------------------
        cs_matrix = np.zeros((len(muestras), len(time_index)))

        for idx, (lat, lon) in enumerate(muestras):
            loc = pvlib.location.Location(
                latitude=lat, longitude=lon,
                tz=self.tz, altitude=self.altitude
            )

            # Solar position for all minute-resolution timestamps
            solpos = loc.get_solarposition(time_index)

            # Relative and absolute air mass (required by Ineichen)
            airmass_rel = get_relative_airmass(solpos["apparent_zenith"])
            airmass_abs = get_absolute_airmass(airmass_rel)

            # Clear-sky GHI with the selected model
            if self.model == "ineichen":
                lt = (self.linke_turbidity if self.linke_turbidity is not None
                      else lookup_linke_turbidity(time_index, lat, lon))
                cs = loc.get_clearsky(
                    time_index, model="ineichen",
                    linke_turbidity=lt,
                    airmass_absolute=airmass_abs,
                    perez_enhancement=True,
                )
            else:
                cs = loc.get_clearsky(time_index, model=self.model)

            # Apply horizon mask: GHI = 0 when the sun is below the
            # elevation profile of the surrounding terrain
            if horizon_azimuths is not None:
                elev_interp  = interp1d(horizon_azimuths, horizon_elevations,
                                        bounds_error=False, fill_value=0)
                horizonte_el = elev_interp(solpos["azimuth"])
                sol_el       = solpos["apparent_elevation"]
                mask         = sol_el > horizonte_el
                cs["ghi"]    = cs["ghi"].where(mask, 0)

            # Zenith bias correction: the Ineichen model underestimates
            # GHI when the zenith angle exceeds zenith_threshold.
            # A factor f = 1 + α · max(0, θ − θ₀) is applied.
            if self.zenith_corr:
                zenith    = solpos["apparent_zenith"]
                factor    = 1 + self.zenith_alpha * (
                    zenith - self.zenith_threshold
                ).clip(lower=0)
                cs["ghi"] = cs["ghi"] * factor

            cs_matrix[idx, :] = cs["ghi"].values

        # --- Average the 25 points and aggregate to hourly resolution --
        df     = pd.DataFrame(cs_matrix.T, index=time_index)
        hourly = df.groupby(group_labels).mean()
        avg    = hourly.mean(axis=1)   # spatial mean of the 25 points

        # --- Build and save the NetCDF dataset -------------------------
        obs  = unique_hours.tz_localize(None).to_numpy()
        data = avg.values[:, None, None, None]
        ds = xr.Dataset(
            {"clear_sky_ghi": (("observation_time", "surface", "lat", "lon"), data)},
            coords={
                "observation_time": obs,
                "surface": [0],
                "lat": [self.lat_center],
                "lon": [self.lon_center],
            },
        )
        ds.to_netcdf(os.path.join(output_dir, output_filename))
        print(f"Clear-sky GHI saved to: {output_dir}/{output_filename}")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    gen = ClearSkyGridAverager(
        lat_center=LAT,
        lon_center=LON,
        start_date="2021-04-01 00:00",
        end_date="2025-06-01 00:00",
        tz="America/Bogota",
        altitude=ELEVATION,
        model="ineichen",
        linke_turbidity=None,   # None = uses pvlib automatic table lookup
        seed=42,
        horizon_file=HORIZON_FILE,
        zenith_corr=ZENITH_CORR,           # imported from config.py
        zenith_alpha=ZENITH_ALPHA,         # imported from config.py
        zenith_threshold=ZENITH_THRESHOLD, # imported from config.py
    )
    gen.generate(
        output_dir=CSI_GHI_OUT_DIR,
        output_filename="CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct2.nc",
    )
    """
    Execution instructions:
    1. Download the elevation TIF file from:
       https://portal.opentopography.org/apidocs/#/Public/getGlobalDem
       for the cell centred on LAT=6.25, LON=-75.5 and save it to
       HORIZON_FILE (see config.py).
    2. Adjust start_date and end_date to the available data range.
    3. Configure ZENITH_CORR, ZENITH_ALPHA and ZENITH_THRESHOLD in config.py
       to enable or disable the bias correction as desired.
    4. Run the script (may take several minutes due to the minute-level
       resolution and 25 sampling points per grid cell).
    """
