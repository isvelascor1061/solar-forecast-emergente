#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:48:49 2025

@author: leonardmerl
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import pvlib

class ExtraterrestrialGHI:
    """
    Calculates minute-resolution extraterrestrial horizontal irradiance (G0h)
    from the solar constant (pvlib.get_extra_radiation) and the solar zenith angle.
    Then, a moving hourly average is computed (11:01–12:00 → 12:00).
    Result: NetCDF with (observation_time, surface, lat, lon), surface = 0.
    """

    def __init__(
        self,
        lat_range,
        lon_range,
        start_date,
        end_date,
        lat_step=0.25,
        lon_step=0.25,
        tz="UTC",
        altitude=0
    ):
        """
        Initializes the parameters needed to calculate extraterrestrial GHI.

        :param lat_range: Latitude range for the grid.
        :param lon_range: Longitude range for the grid.
        :param start_date: Start date for the time period.
        :param end_date: End date for the time period.
        :param lat_step: Step size for latitude in degrees (default: 0.25).
        :param lon_step: Step size for longitude in degrees (default: 0.25).
        :param tz: Time zone (default: "UTC").
        :param altitude: Altitude of the location (default: 0).
        """
        # Convert the latitude and longitude range into actual grid points
        self.lat_vals = self._parse_range(lat_range, lat_step)
        self.lon_vals = self._parse_range(lon_range, lon_step)
        
        self.start_date = start_date
        self.end_date   = end_date
        self.tz = tz
        self.altitude = altitude

    @staticmethod
    def _parse_range(value_range, step):
        """
        Parses the range input to generate the grid points.

        :param value_range: Range of values (either a single value or a pair).
        :param step: The step size for generating the grid.
        :return: A numpy array of grid points.
        """
        # If a single value is provided, return that value as a 1D array
        if len(value_range) == 1:
            return np.array([value_range[0]], dtype=float)
        # If a range is provided (min, max), create a grid with the specified step size
        elif len(value_range) == 2:
            start, end = sorted(value_range)
            return np.arange(start, end + step, step, dtype=float)
        else:
            raise ValueError("lat_range and lon_range must have length 1 or 2.")

    # ------------------------------------------------------------------ #
    def generate_extraterrestrial_netcdf(
        self,
        output_dir,
        output_filename="G0h_all.nc",
        extra_radiation_method="spencer"  # pvlib method (spencer, asce, ...)
    ):
        """
        Generates extraterrestrial GHI data and saves it as a NetCDF file.

        :param output_dir: Directory to save the output NetCDF file.
        :param output_filename: Name of the output NetCDF file.
        :param extra_radiation_method: The method to calculate extraterrestrial radiation (default is "spencer").
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_filename)

        # (1) Generate a minute-level time index for the entire period
        time_min = pd.date_range(
            start=self.start_date, end=self.end_date,
            freq="T", tz=self.tz, inclusive="both"
        )

        # Label: 11:01–12:00 → 12:00 (similar logic to previous code)
        group_labels = (time_min - pd.Timedelta(minutes=1)).floor("H") + pd.Timedelta(hours=1)
        unique_hours = pd.DatetimeIndex(group_labels.unique())

        # Initialize an array to store the GHI values
        n_hours, n_lat, n_lon = len(unique_hours), len(self.lat_vals), len(self.lon_vals)
        g0h_hourly = np.zeros((n_hours, n_lat, n_lon), dtype=np.float32)

        # (2) Loop over all grid points (latitude and longitude)
        for i, lat in enumerate(self.lat_vals):
            for j, lon in enumerate(self.lon_vals):
                # Create a pvlib Location object for the grid point
                loc = pvlib.location.Location(lat, lon, altitude=self.altitude)

                # Get the solar position (zenith angle) for all minutes in the time index
                solpos = loc.get_solarposition(time_min)
                zenith = solpos["apparent_zenith"].values

                # Calculate extraterrestrial radiation (I0n) for the grid point
                i0n = pvlib.irradiance.get_extra_radiation(
                    time_min, method=extra_radiation_method
                ).values  # Units: W m⁻²

                # Calculate G0h = I0n * cos(zenith); Set values to 0 where zenith > 90° (night time)
                g0h_min = i0n * np.clip(np.cos(np.radians(zenith)), 0, None)

                # Calculate hourly averages using a moving window
                g0h_series = pd.Series(g0h_min, index=time_min)
                g0h_hour   = g0h_series.groupby(group_labels).mean()

                # Store the hourly GHI data in the array
                for k, hour in enumerate(unique_hours):
                    g0h_hourly[k, i, j] = g0h_hour.get(hour, np.nan)

        # (3) Convert observation time to a timezone-naive array
        obs_time = unique_hours.tz_localize(None).to_numpy()

        # (4) Create the xarray Dataset and save it as a NetCDF file
        ds = xr.Dataset(
            {
                "extraterrestrial_ghi": (
                    ("observation_time", "surface", "lat", "lon"),
                    g0h_hourly[:, np.newaxis, :, :]
                )
            },
            coords={
                "observation_time": obs_time,
                "surface": [0],
                "lat": self.lat_vals,
                "lon": self.lon_vals
            }
        )

        # Save the dataset to a NetCDF file
        ds.to_netcdf(out_path)
        print(f"Extraterrestrial GHI saved to: {out_path}")

# ------------------------------- CLI Example ----------------------------- #
if __name__ == "__main__":
    # Define input parameters
    lat_range  = [6.25]  # Latitude range for the grid (single latitude value)
    lon_range  = [-75.5]  # Longitude range for the grid (single longitude value)
    start_date = "2021-04-01 00:00"  # Start date for the time period
    end_date   = "2025-06-01 01:00"  # End date for the time period
    out_dir    = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Extraterrestrial_GHI"

    # Create an instance of the ExtraterrestrialGHI class
    g0 = ExtraterrestrialGHI(
        lat_range, lon_range, start_date, end_date,
        lat_step=0.25, lon_step=0.25, tz="America/Bogota", altitude=1500
    )

    # Generate the extraterrestrial GHI data and save it as a NetCDF file
    g0.generate_extraterrestrial_netcdf(
        output_dir=out_dir,
        output_filename="EXT_GHI_all.nc"
    )
    """
    Works similar to the clear sky Radiation calculator 
    1. select lat and lon range for the grid 
    2. select start and end date (corresponding to available GFS and insitu data-> Siata)
    3. lat step as 0.25 -> grid resolution on the GFS SErver
    4. tz also America/Bogota for colombia
    5. altitude -> 1500 for Medellin 
    i wasnt as accurate with this script, because there werent as many issues of excedance as with the clear sky radiation
    6. Execute and save 
    This radiation is only used to calculate the clearness indice of the GFS DSWRF1 radiation 
    so only as on of 17 features 
    """












