#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean hourly SIATA GHI by clipping every clear-sky exceedance.
GHI_clipped = (1 − ε) · GHI_cs,  ε ∈ {0.005, 0.010, …, 0.030}.
Genuine gaps ≤ 5 h are linearly interpolated.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr


class NetCDFDataCleaner:
    # ----------------------------- constants ---------------------------- #
    JITTER_MAX  = 0.03     # 3 % maximum jitter
    JITTER_STEP = 0.005    # 0.5 % step size for jitter
    MAX_GAP     = 5        # Maximum allowed consecutive NaNs for interpolation
    RNG_SEED    = 42       # Seed for random number generation

    # ------------------------------------------------------------------- #
    def __init__(
        self,
        siata_path: str,
        clearsky_path: str,
        measured_var: str = "GHI",
        clearsky_var: str = "clear_sky_ghi",
    ):
        """
        Initialize the NetCDFDataCleaner with paths to the SIATA and clear-sky data.

        Args:
            siata_path: Path to the SIATA NetCDF file.
            clearsky_path: Path to the clear-sky NetCDF file.
            measured_var: Name of the variable in the SIATA file (default "GHI").
            clearsky_var: Name of the variable in the clear-sky file (default "clear_sky_ghi").
        """
        self.siata_path   = siata_path
        self.clearsky_path = clearsky_path
        self.measured_var = measured_var
        self.clearsky_var = clearsky_var

        self.measured  = None
        self.clearsky  = None
        self.ghi_clean = None
        self.csi       = None
        self.qc_clip   = None

    # ------------------------------ steps ------------------------------ #
    def load_datasets(self):
        """
        Loads the SIATA and clear-sky datasets into memory.

        This function opens the specified NetCDF files, extracts the measured GHI and
        clear-sky GHI variables, and stores them as xarray DataArrays.
        """
        self.measured = xr.open_dataset(self.siata_path, engine ="netcdf4")[self.measured_var]
        self.clearsky = xr.open_dataset(self.clearsky_path, engine ="netcdf4")[self.clearsky_var].squeeze()

    def align_times(self, join: str = "inner"):
        """
        Align the measured and clear-sky datasets along the time dimension.

        Args:
            join: Type of join to perform when aligning the datasets. Default is "inner", which
                  keeps only the times that are present in both datasets.
        """
        self.measured, self.clearsky = xr.align(self.measured, self.clearsky, join=join)

    def clip_exceedances(self):
        """
        Clips the exceedances of the measured GHI over the clear-sky GHI.
        
        Exceedances are calculated as values where measured GHI is greater than clear-sky GHI.
        A random jitter between 0.005 and 0.03 (based on the constants defined) is applied to
        the exceedance values.
        """
        exceed = self.measured > self.clearsky
    
        np.random.seed(self.RNG_SEED)
        choices = np.arange(self.JITTER_STEP,
                            self.JITTER_MAX + self.JITTER_STEP,
                            self.JITTER_STEP)  # Random choices between 0.005 and 0.030
    
        # Calculate the number of exceedances and apply jitter
        n_exc = int(exceed.sum().item())  # Convert xarray scalar to int
    
        jitter = 1 - np.random.choice(choices, size=n_exc)
    
        # Create a copy of the measured GHI and apply the jitter to exceedances
        ghi_fix = self.measured.copy()
        ghi_fix.values[exceed.values] = (
            self.clearsky.values[exceed.values] * jitter
        )
    
        self.ghi_clean = ghi_fix
        self.qc_clip   = exceed.astype("int8")  # Quality control: 1 for exceedance, 0 for valid
        self.csi       = (ghi_fix / self.clearsky).clip(0.0, 1.0)  # Clear-sky index (CSI)

    def interpolate_gaps(self):
        """
        Interpolates gaps in the GHI data where there are NaN values.
        
        Gaps of up to 5 hours (MAX_GAP) are linearly interpolated. 
        After interpolation, exceedances are clipped again to ensure consistency.
        """
        series = self.ghi_clean.to_series()

        i = 0
        n = len(series)
        while i < n:
            if pd.isna(series.iloc[i]):  # If a value is NaN
                start = i
                while i < n and pd.isna(series.iloc[i]):
                    i += 1
                gap = i - start
                if 0 < gap <= self.MAX_GAP and start > 0 and i < n:
                    series.iloc[start:i] = np.linspace(
                        series.iloc[start - 1], series.iloc[i], gap + 2
                    )[1:-1]  # Linear interpolation
            else:
                i += 1

        # Update the cleaned GHI after interpolation
        self.ghi_clean = xr.DataArray(
            series.values,
            dims=["observation_time"],
            coords={"observation_time": series.index},
            name=self.measured_var,
        )

        # Clip any exceedances that may occur after interpolation
        mask_exc = self.ghi_clean > self.clearsky
        if mask_exc.any():
            self.ghi_clean = self.ghi_clean.where(~mask_exc, self.clearsky)
            self.qc_clip   = xr.where(mask_exc, 1, self.qc_clip)

        # Update the clear-sky index (CSI)
        self.csi = (self.ghi_clean / self.clearsky).clip(0.0, 1.0)

    def save_cleaned(self, out_dir: str, out_file: str):
        """
        Saves the cleaned GHI data to a NetCDF file.
        
        If the output directory does not exist, it will be created.
        The cleaned GHI, CSI, and quality control clip are saved as variables in the NetCDF file.
        """
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, out_file)

        # Create a NetCDF dataset with the cleaned GHI, CSI, and QC clip
        xr.Dataset(
            {"GHI_clean": self.ghi_clean, "CSI": self.csi, "QC_clip": self.qc_clip},
            attrs={
                "clip_max_offset": self.JITTER_MAX,
                "clip_step": self.JITTER_STEP,
                "max_gap_interp": self.MAX_GAP,
            },
        ).to_netcdf(path)
        print(f"[INFO] Cleaned NetCDF saved to: {path}")

    def process(self, out_dir: str, out_file: str):
        """
        Full process to clean the SIATA GHI data:
        1. Load datasets.
        2. Align the times.
        3. Clip exceedances.
        4. Interpolate gaps.
        5. Save cleaned data to NetCDF.
        """
        self.load_datasets()
        self.align_times()
        self.clip_exceedances()
        self.interpolate_gaps()
        self.save_cleaned(out_dir, out_file)


# ------------------------ user configuration ------------------------- #
SIATA_PATH = (
    "_3_Data_preparation_for_LSTM/Preparation_data/_03_Siata_GHI/Netcdf_Siata_GHI/SIATA_GHI_all_test.nc"
)

CLEARSKY_PATH = (
    "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Extraterrestrial_GHI/EXT_GHI_all.nc"
)

OUTPUT_DIR  = (
    "_3_Data_preparation_for_LSTM/Preparation_data/"
    "_03_Siata_GHI/Netcdf_Siata_GHI"
)
OUTPUT_FILE = "GHI_EXT_clipped.nc"

if __name__ == "__main__":
    from config import EXT_VAR_NAME   # ref_ext_ghi → "extraterrestrial_ghi"

    cleaner = NetCDFDataCleaner(
        siata_path=SIATA_PATH,
        clearsky_path=CLEARSKY_PATH,
        clearsky_var=EXT_VAR_NAME,
    )
    cleaner.process(OUTPUT_DIR, OUTPUT_FILE)
    
    """
    This script is used to clean and clip SIATA GHI data by following the following steps:

    1. **Clipping Exceedances**: 
       Any GHI values exceeding the clear-sky GHI (extraterrestrial radiation) are clipped 
       based on a jitter factor between 0.005 and 0.03.

    2. **Interpolating Gaps**: 
       Any gaps of up to 5 hours (MAX_GAP) in the GHI data are linearly interpolated.

    3. **Saving Cleaned Data**: 
       The cleaned GHI data, along with the Clear Sky Index (CSI) and quality control flags (QC_clip), 
       are saved to a NetCDF file.

    **Configuration**:
    - **SIATA_PATH**: Path to the SIATA GHI dataset (NetCDF format).
    - **CLEARSKY_PATH**: Path to the clear-sky GHI dataset (NetCDF format).
    - **OUTPUT_DIR**: Directory to save the cleaned NetCDF file.
    - **OUTPUT_FILE**: Name of the output NetCDF file.
    """
