#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clip GFS radiation at a physical threshold:
exceedances are replaced via time interpolation,
ensuring the result never exceeds the threshold
(clear-sky or extraterrestrial GHI).

Created on 2025-04-21
@author: leonardmerl   (refactored by ChatGPT, 2025-05-12)
"""
import os
import xarray as xr
import numpy as np
import pandas as pd


class GFSThresholdClipper:
    """
    Replaces GFS values that exceed a reference radiation series with
    linearly interpolated values (along the time axis). The interpolated
    value is always ≤ the reference.
    """

    def __init__(
        self,
        gfs_path: str,
        ref_path: str,
        gfs_var: str = "dswrf1",
        ref_var: str = "clear_sky_ghi",
    ):
        """
        Parameters
        ----------
        gfs_path : str
            Path to the NetCDF file containing GFS radiation (e.g. dswrf1).
        ref_path : str
            Path to the NetCDF file containing reference radiation
            (clear-sky or extraterrestrial GHI).
        gfs_var : str
            Variable name for the GFS radiation.
        ref_var : str
            Variable name for the reference radiation.
        """
        self.gfs_path = gfs_path
        self.ref_path = ref_path
        self.gfs_var = gfs_var
        self.ref_var = ref_var

        self.gfs_da = None
        self.ref_da = None
        self.clipped_da = None

    # ------------------------------------------------------------------ #
    def load(self):
        """Load the NetCDF files."""
        self.gfs_da = xr.open_dataset(self.gfs_path)[self.gfs_var]
        self.ref_da = xr.open_dataset(self.ref_path)[self.ref_var]

    def align(self, join: str = "inner"):
        """Align time axes (observation_time)."""
        self.gfs_da, self.ref_da = xr.align(self.gfs_da, self.ref_da, join=join)

    def clip(self, dim: str = "observation_time", jitter_range: float = 0.03):
        """
        1. Build exceedance mask
        2. Set exceedances to NaN
        3. Replace with weekly mean + jitter
        4. Fill remaining edges with the threshold value
        5. Ensure: interpolated value ≤ threshold
        """
        # (1) Exceedance mask
        exceed_mask = self.gfs_da > self.ref_da

        # (2) Set exceedances to NaN
        gfs_tmp = self.gfs_da.where(~exceed_mask)

        # (3) Replace exceedances with weekly mean + jitter
        weekly_mean = self.gfs_da.groupby(pd.Grouper(freq="W")).mean("observation_time")

        # Add jitter (+/- 3%)
        jitter = np.random.uniform(-jitter_range, jitter_range, size=weekly_mean.shape)
        gfs_interp = weekly_mean + weekly_mean * jitter

        # Keep interpolated values only where they are below the threshold
        gfs_interp = gfs_interp.where(gfs_interp <= self.ref_da)

        # (4) Fill edges (in case the series starts/ends above the threshold)
        gfs_interp = gfs_interp.fillna(self.ref_da*0.95)

        # (5) Final series: exceedances → min(interpolated, threshold)
        self.clipped_da = xr.where(
            exceed_mask,
            xr.ufuncs.minimum(gfs_interp, self.ref_da),
            self.gfs_da
        )

    def save(self, output_dir: str, output_filename: str):
        """Save the clipped series as a NetCDF file."""
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_filename)
        self.clipped_da.to_dataset(name=self.gfs_var).to_netcdf(out_path)
        print(f"Clipped GFS saved to: {out_path}")

    # Convenience
    def process(self, output_dir: str, output_filename: str):
        self.load()
        self.align()
        self.clip()
        self.save(output_dir, output_filename)


# ------------------------------ Example ----------------------------------- #
if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #       1) Clip against Clear-Sky                                     #
    # ------------------------------------------------------------------ #
    GFS_PATH   = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Raw_merged/dswrf1_0100.nc"
    CLEAR_PATH = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/CSI_GHI_grid25_avg_with_horizon_and_enhancement.nc"

    OUT_DIR_CSI  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped"
    OUT_FILE_CSI = "dswrf1_CSI_0100.nc"

    clipper_csi = GFSThresholdClipper(
        gfs_path=GFS_PATH,
        ref_path=CLEAR_PATH,
        gfs_var="dswrf1",
        ref_var="clear_sky_ghi"
    )
    #clipper_csi.process(OUT_DIR_CSI, OUT_FILE_CSI)

    # ------------------------------------------------------------------ #
    #       2) Clip against Extraterrestrial Radiation                    #
    # ------------------------------------------------------------------ #
    EXTRA_PATH = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Extraterrestrial_GHI/EXT_GHI_all.nc"

    OUT_DIR_EXT  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_EXT_clipped"
    OUT_FILE_EXT = "dswrf1_EXT_0100.nc"

    clipper_ext = GFSThresholdClipper(
        gfs_path=GFS_PATH,
        ref_path=EXTRA_PATH,
        gfs_var="dswrf1",
        ref_var="extraterrestrial_ghi"
    )
    clipper_ext.process(OUT_DIR_EXT, OUT_FILE_EXT)
