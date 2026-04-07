#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15 Apr 2025
Author: leonardmerl

Description
-----------
• Reads SIATA CSV files.
• Converts negative ‘radiacion’ values to NaN.
• Aggregates to hourly resolution with special rules:
    – Hours 20–05 → radiation is forced to 0 W m⁻².
    – Hour 06 and 19 → missing minutes are set to 0, then the mean is taken.
    – All other hours → if more than *threshold* % of minutes are NaN the hour is
      set to NaN, otherwise the mean of valid minutes is used.
• Saves or updates a NetCDF file, adding latitude/longitude as coordinates.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr


class CSVToNetCDF:
    """
    Convert measurement CSV files into a consolidated NetCDF dataset that
    follows the daylight-specific quality rules defined in the module doc-string.
    """

    def __init__(self, input_dir: str, output_file: str, lat: float, lon: float):
        self.input_dir = input_dir
        self.output_file = output_file
        self.lat = lat
        self.lon = lon
        self.data: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # 1) Read and unify CSV files                                        #
    # ------------------------------------------------------------------ #
    def read_csv_files(self) -> None:
        csv_files = [f for f in os.listdir(self.input_dir)
                     if f.lower().endswith(".csv")]
        frames = []

        for fname in csv_files:
            fpath = os.path.join(self.input_dir, fname)
            df = pd.read_csv(
                fpath,
                header=0,
                index_col=0,
                parse_dates=[0],
                infer_datetime_format=True,
            )

            # Fix column names if they are blank or inconsistent
            if df.columns[0] == "":
                df.columns = ["idestacion", "radiacion", "calidad"]
            elif df.columns[0] in {"radiacion", "idestacion"} and len(df.columns) == 2:
                df.columns = ["idestacion", "radiacion", "calidad"]
            elif df.columns[0] in {"timestamp", "time"}:
                df.columns = ["timestamp", "radiacion", "calidad"]

            frames.append(df)

        if frames:
            self.data = pd.concat(frames).sort_index()
        else:
            self.data = pd.DataFrame()
            print(f"[WARN] No CSV files found in: {self.input_dir}")

    # ------------------------------------------------------------------ #
    # 2) Hourly aggregation with day-/night rules                        #
    # ------------------------------------------------------------------ #
    def compute_hourly_aggregation(self, threshold: float = 0.15) -> None:
        """
        Parameters
        ----------
        threshold : float, optional
            Max. fraction of NaN minutes allowed in an hour before the entire
            hour is marked NaN (applied to hours other than 06 and 19). Default 0.15.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data present – run read_csv_files() first.")

        # Convert negatives to NaN and ensure float dtype
        self.data["radiacion"] = (
            pd.to_numeric(self.data["radiacion"], errors="coerce")
              .mask(lambda s: s < 0)
        )

        def hour_rule(series: pd.Series, gap_tol=0.75):
            h = series.name.hour
            n_nan = series.isna().sum()
        
            # 1) For hours 20:00 to 05:00 (night hours)
            if h in {20, 21, 22, 23, 0, 1, 2, 3, 4, 5}:  
                return 0.0  # All values are set to 0 during night hours
        
            # 2) For hours 06:00 and 19:00 (sunrise and sunset hours)
            if h in {6, 19}:  # Specifically for 06:00 and 19:00
                if n_nan / len(series) > 0.75:  # More than 45 minutes NaN (75% of the hour)
                    return np.nan  # Set the entire hour to NaN
                else:
                    # Replace NaNs with 0
                    series_filled = series.fillna(0)
        
                    # Calculate sum of the non-NaN values 
                    return series_filled.mean()  # Now we just calculate the mean
        
            # 3) For the remaining hours (07:00-18:00)
            if n_nan / len(series) > 0.75:  # More than 75% NaN (i.e., more than 45 minutes NaN)
                return np.nan  # Set the hour to NaN if more than 75% are NaN
            else:
                # Calculate the mean of the valid (non-NaN) values
                return series.mean()  # Calculate the mean of valid values
                     


        hourly = (
            self.data["radiacion"]
            .resample("H", closed="right", label="right")
            .apply(hour_rule)
            .to_frame(name="radiacion")
        )

        self.data = hourly

    # ------------------------------------------------------------------ #
    # 3) Save or update NetCDF                                           #
    # ------------------------------------------------------------------ #
    def save_to_netcdf(self) -> None:
        if self.data is None or self.data.empty:
            raise ValueError("No aggregated data – run compute_hourly_aggregation().")

        ds_new = xr.Dataset(
            {"GHI": ("observation_time", self.data["radiacion"].values)},
            coords={"observation_time": self.data.index.values.astype("datetime64[ns]")},
        ).assign_coords(lat=self.lat, lon=self.lon)

        if os.path.exists(self.output_file):
            ds_old = xr.open_dataset(self.output_file)
            merged = xr.concat([ds_old, ds_new], dim="observation_time")
            merged = merged.sortby("observation_time")
            merged = merged.groupby("observation_time").first()  # remove dups
            merged.to_netcdf(self.output_file, format="NETCDF4")
            print(f"[INFO] NetCDF updated: {self.output_file}")
            ds_old.close()
        else:
            ds_new.to_netcdf(self.output_file, format="NETCDF4")
            print(f"[INFO] NetCDF created: {self.output_file}")


# ---------------------------------------------------------------------- #
# Main entry point                                                       #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    INPUT_DIR = (
        "_3_Data_preparation_for_LSTM/Preparation_data/"
        "_03_Siata_GHI/CSV_Siata_GHI"
    )
    OUTPUT_DIR = (
        "_3_Data_preparation_for_LSTM/Preparation_data/"
        "_03_Siata_GHI/Netcdf_Siata_GHI"
    )
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "SIATA_GHI_all_test.nc")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    LAT, LON = 6.25, -75.5  # station coordinates for Medellín

    converter = CSVToNetCDF(INPUT_DIR, OUTPUT_FILE, LAT, LON)
    converter.read_csv_files()
    converter.compute_hourly_aggregation()   # adjust if needed
    converter.save_to_netcdf()

    """
     This script is used to merge and clean the CSV files from the Siata Server
     -> Siata CSV files contain a lot of -9999 -> at night instead of 0 and during the day when 
     there are no correct measurements 
     The script does the following: 
         It converts all negative values to NaN
         All Nans that occur during the hours 20-5 are set to 0
         
         during 6am and 19pm (Sunrise and Sunset ) 
         if more than 75% of values are NaN the whole intervall is set to NaN
         else -> NaN set to 0 and average is calculated mean with the 0 values
         -> to not overestimate radiation during these hours
         
         rest of the day 7am to 18pm 
         if more than 75% of values are NaN the whole intervall is set to NaN
         else: NaNs stay NaN but the mean for the hours is calculated as the average of 
         the valid values -> to not underestimate the relative consistent radiation during these hours
         
    """
     
     
     
     
     
     
     
     