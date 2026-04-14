#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFS‑GHI Cleaning & Clipping  (v2) – keeps *all* original coordinates
=====================================================================

This script processes GFS Global Horizontal Irradiance (GHI) data and applies a cleaning procedure by
clipping the values that exceed the corresponding values from a Clear-Sky Index (CSI) dataset.
It ensures that the cleaned data retains all original coordinates and dimensions.

Changes from the previous version
------------------------------------------
* **No reconstruction** of the `DataArray` → all dimensions and
  coordinates (e.g. ``launch_time``, ``step``, ``surface``) remain intact.
* Values replacement happens directly in a *deep copy* of the cleaned array, ensuring original data integrity.

The script follows these steps:
1. Loads the GFS and CSI data.
2. Aligns them by their time dimension to ensure both datasets are synchronized.
3. Clips the GFS data to remove values that exceed the clear-sky index.
4. Replaces missing values (NaN) in the GFS data with the mean of corresponding values from other years, considering the same month, day, and hour.
5. Saves the cleaned data as a new NetCDF file.

Usage of the pipeline and file handling remains unchanged.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

class NetCDFDataCleaner:
    """Class that loads, processes, cleans, and saves GFS and CSI datasets."""

    def __init__(
        self,
        gfs_path: str,
        csi_path: str,
        gfs_var: str = "GHI",
        csi_var: str = "clear_sky_ghi",
        min_den: float = 5.0,
        seed: int | None = None,
    ):
        """
        Initializes the cleaner object for processing GFS and CSI data.

        Args:
            gfs_path (str): Path to the GFS NetCDF file.
            csi_path (str): Path to the CSI (clear-sky index) NetCDF file.
            gfs_var (str): The name of the GFS variable to be processed (default "GHI").
            csi_var (str): The name of the CSI variable (default "clear_sky_ghi").
            min_den (float): Minimum threshold for values (values below this are set to 0.0).
            seed (int | None): Seed for the random number generator (optional).
        """
        self.gfs_path = gfs_path  # Path to the GFS NetCDF file
        self.csi_path = csi_path  # Path to the CSI NetCDF file
        self.gfs_var = gfs_var    # Name of the GFS variable (e.g., "GHI")
        self.csi_var = csi_var    # Name of the CSI variable (e.g., "clear_sky_ghi")
        self.min_den = min_den    # Minimum value threshold
        self.rng = np.random.default_rng(seed)  # Initialize random number generator with the given seed

        # Data arrays will be stored here after loading
        self.gfs_da: xr.DataArray | None = None
        self.csi_da: xr.DataArray | None = None
        self.cleaned: xr.DataArray | None = None

    # ------------------------------------------------------------------
    def load(self):
        """
        Loads the GFS and CSI datasets from the specified paths into memory as xarray DataArrays.
        """
        self.gfs_da = xr.open_dataset(self.gfs_path)[self.gfs_var]
        self.csi_da = xr.open_dataset(self.csi_path)[self.csi_var].squeeze()

    def align(self, join: str = "inner"):
        """
        Aligns the GFS and CSI datasets along the time dimension.
        The `join` parameter determines how the datasets are aligned.
        By default, an inner join is used to only keep the matching time points.

        Args:
            join (str): Type of join operation. 'inner' by default.
        """
        self.gfs_da, self.csi_da = xr.align(self.gfs_da, self.csi_da, join=join)

    def clip_exceedances(self):
        """
        Clips the GFS data by setting values that exceed the corresponding CSI values to NaN.
        This operation ensures that the GFS data will only contain values less than or equal to the clear-sky index.
        """
        self.cleaned = self.gfs_da.where(self.gfs_da <= self.csi_da)

    # ------------------------------------------------------------------
    def _draw_jitter_factor(self, size: int = 1) -> np.ndarray:
        """
        Draws random jitter factors to apply to the replacement values for NaN entries.
        These factors are used to introduce some variability when replacing NaNs, preventing exact replication.

        Args:
            size (int): The number of jitter factors to generate.
        
        Returns:
            np.ndarray: Array of jitter factors.
        """
        jitter_percent = self.rng.choice(np.arange(1.0, 10.1, 0.1), size=size) / 100.0
        return 1.0 - jitter_percent

    def replace_nans(self):
        """
        Replaces NaN values in the cleaned GFS data with the mean value from corresponding time points
        from other years, using data from the same month, day, and hour.
        The replacement is constrained by the CSI values, ensuring the replacement is realistic.
        """
        ser = self.cleaned.to_series()  # Convert the DataArray to a pandas Series for easier manipulation
        csi_ser = self.csi_da.to_series()  # Convert the CSI DataArray to a pandas Series
        times = ser.index  # Time index
        new_vals = ser.copy()  # Copy of the original values

        # Create a DataFrame with time-related columns (month, day, hour, year)
        df = ser.to_frame("ghi")
        df["month"] = times.month
        df["day"] = times.day
        df["hour"] = times.hour
        df["year"] = times.year
        grouped = df.groupby(["month", "day", "hour"])

        for i, (ts, val) in enumerate(ser.items()):
            if pd.isna(val):  # If the value is NaN
                key = (ts.month, ts.day, ts.hour)  # Group by month, day, hour
                others = grouped.get_group(key)  # Get all other entries with the same month, day, hour
                other_years = others[others["year"] != ts.year]["ghi"].dropna()  # Exclude the current year

                if other_years.empty:
                    continue
                repl = other_years.mean()  # Replace with the mean value from other years
                csi_val = csi_ser.iat[i]  # Get the corresponding CSI value for the current time step
                if repl > csi_val:  # Limit the replacement to be at most the CSI value
                    repl = csi_val * self._draw_jitter_factor()[0]
                new_vals.iat[i] = repl  # Replace the NaN value

        # Replace values below the minimum threshold with 0
        new_vals[new_vals < self.min_den] = 0.0

        # --- IMPORTANT: Preserve the original structure ----------------
        clean = self.cleaned.copy(deep=True)  # Deep copy to preserve original coordinates and dimensions
        clean.values = new_vals.values.reshape(clean.shape)  # Assign the new values
        self.cleaned = clean  # Update the cleaned DataArray

    # ------------------------------------------------------------------
    def save(self, out_dir: str, filename: str):
        """
        Saves the cleaned DataArray as a NetCDF file in the specified output directory.

        Args:
            out_dir (str): Directory to save the output file.
            filename (str): Name of the output file.
        """
        os.makedirs(out_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        path = os.path.join(out_dir, filename)  # Full path for the output file
        self.cleaned.to_dataset().to_netcdf(  # Convert to dataset and save as NetCDF
            path,
            encoding={self.gfs_var: {"zlib": True, "complevel": 4}},  # Apply compression to the variable
        )
        print(f"Cleaned NetCDF saved at: {path}")

    def process(self, out_dir: str, filename: str):
        """
        Executes the full cleaning process: loading, aligning, clipping, replacing NaNs, and saving the result.

        Args:
            out_dir (str): Directory to save the cleaned file.
            filename (str): Output filename.
        """
        self.load()  # Load the datasets
        self.align()  # Align the datasets
        self.clip_exceedances()  # Clip the GFS values exceeding the CSI values
        self.replace_nans()  # Replace NaN values with mean values from other years
        self.save(out_dir, filename)  # Save the cleaned data as a NetCDF file

# ----------------------------------------------------------------------
if __name__ == "__main__":
    from config import (
        DSWRF1_UNCLIPPED_DIR, EXT_GHI_FILE, DSWRF1_EXT_DIR,
        EXT_VAR_NAME, CLIP_MIN_DEN, CLIP_SEED,
        VAR_GFS_DSWRF_TEMPLATE,
    )

    launch_time = "1900"  # Launch time a procesar (0100, 0700, 1300 o 1900)
    GFS_PATH = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_{launch_time}.nc"   # Radiación GFS sin cortar
    CSI_PATH = EXT_GHI_FILE                                         # Radiación extraterrestre
    OUT_DIR  = DSWRF1_EXT_DIR                                       # Carpeta de salida
    OUT_FILE = f"dswrf1_EXT_{launch_time}.nc"                       # Nombre del archivo de salida

    # Initialize the NetCDFDataCleaner object with necessary parameters
    cleaner = NetCDFDataCleaner(
        gfs_path=GFS_PATH,
        csi_path=CSI_PATH,
        gfs_var=VAR_GFS_DSWRF_TEMPLATE.format(LT=launch_time),   # gfs_dswrf → "dswrf1_{LT}"
        csi_var=EXT_VAR_NAME,                                      # ref_ext_ghi → "extraterrestrial_ghi"
        min_den=CLIP_MIN_DEN,   # Umbral mínimo de radiación
        seed=CLIP_SEED,         # Semilla para el jitter
    )
    
    # Process the data and save the cleaned NetCDF file
    cleaner.process(OUT_DIR, OUT_FILE)
    """
    This script is used to clip GFS Radiation to ensure that there are no indices bigger than 1
    1. Input path to GFS file (here also just 1 launch time at a time)
    2. Input path to clear sky radiation or EXT radiation 
    3. Ouput dir -> where to save the new radiatioin 
    4. Name of output file 
    """
    
    
    
    
    
    
    
    
    
