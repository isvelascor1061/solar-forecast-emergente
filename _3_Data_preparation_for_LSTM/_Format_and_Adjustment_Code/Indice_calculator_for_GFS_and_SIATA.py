#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indice Calculator (v2) – preserves ALL original coordinates
===========================================================

This script calculates the **Clearness Index** (CI) or **Clear-Sky Index** (CSI) 
from GFS forecast or SIATA measurement radiation data. The resulting NetCDF files 
preserve **all dimensions and coordinates** of the original datasets, such as 
`launch_time` and `step` in GFS forecasts.

It works for both GFS and SIATA datasets; missing coordinates are simply omitted.
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from typing import Literal

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _safe_div(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    """Performs a safe division between two xarray.DataArray objects.
    
    Handles edge cases such as division by zero, ensuring:
    * 0 / 0 → 0 (instead of NaN)
    * x / 0 → NaN (for x ≠ 0)
    * Inf/−Inf values are set to NaN
    
    Args:
        numerator: The numerator DataArray.
        denominator: The denominator DataArray.

    Returns:
        A DataArray containing the result of the division.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator

    # Explicitly set 0/0 to 0
    zero_zero = (denominator == 0) & (numerator == 0)
    result = result.where(~zero_zero, 0.0)

    # Clean up any remaining Inf / NaN values
    result = result.where(np.isfinite(result))
    return result

# -----------------------------------------------------------------------------
# Main Class
# -----------------------------------------------------------------------------

class IndexCalculator:
    def __init__(
        self,
        ghi_path: str,
        ref_path: str,
        out_dir: str,
        out_file: str,
        ghi_var: str = "GHI",
        ref_var: str = "clear_sky_ghi",
        index_name: Literal["clearness_index", "clear_sky_index"] = "clearness_index",
    ) -> None:
        """
        Initialize the IndexCalculator with the paths to the GHI and reference datasets.

        Args:
            ghi_path: Path to the GHI dataset (measured radiation).
            ref_path: Path to the reference dataset (clear sky radiation).
            ghi_var: Name of the variable representing GHI in the GHI dataset (default "GHI").
            ref_var: Name of the variable representing clear sky GHI in the reference dataset (default "clear_sky_ghi").
            index_name: Name for the calculated index. Can be "clearness_index" or "clear_sky_index" (default is "clearness_index").
            out_dir: Directory where the resulting NetCDF file will be saved.
            out_file: Name of the output NetCDF file.
        """
        self.ghi_path = ghi_path
        self.ref_path = ref_path
        self.ghi_var = ghi_var
        self.ref_var = ref_var
        self.index_name = index_name
        self.out_dir = out_dir
        self.out_file = out_file

        self.ghi_da: xr.DataArray | None = None
        self.ref_da: xr.DataArray | None = None
        self.index_da: xr.DataArray | None = None

    # ------------------------------------------------------------------
    def load(self):
        """
        Loads the GHI (measured) and reference (clear-sky) datasets into memory.
        """
        self.ghi_da = xr.open_dataset(self.ghi_path)[self.ghi_var]
        self.ref_da = xr.open_dataset(self.ref_path)[self.ref_var]

    def align(self, join: str = "inner"):
        """
        Aligns the GHI and reference datasets by their common coordinates.
        
        This ensures that both datasets have matching time coordinates for division.

        Args:
            join: The type of join to use. 'inner' will only keep the common coordinates between both datasets.
        """
        self.ghi_da, self.ref_da = xr.align(self.ghi_da, self.ref_da, join=join)

    def calc_index(self):
        """
        Calculates the desired index (Clearness Index or Clear Sky Index) 
        by dividing GHI by the reference clear-sky GHI dataset.
        
        The resulting DataArray will preserve all coordinates and dimensions of the input datasets.
        """
        # Perform safe division with automatic broadcasting for correct shape
        idx = _safe_div(self.ghi_da, self.ref_da)

        # Make a deep copy of the result (not the original GHI array!)
        idx_clean = idx.copy(deep=True)
        idx_clean.name = self.index_name
        self.index_da = idx_clean

    def save(self):
        """
        Saves the calculated index as a NetCDF file in the specified directory and file name.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, self.out_file)
        self.index_da.to_dataset().to_netcdf(
            path,
            encoding={self.index_name: {"zlib": True, "complevel": 4}},
        )
        print(f"Index file written: {path}")

    # ------------------------------------------------------------------
    def process(self):
        """
        Runs the entire processing pipeline: loads datasets, aligns them, calculates the index, and saves the result.
        """
        self.load()
        self.align()
        self.calc_index()
        self.save()


# -----------------------------------------------------------------------------
# Minimal Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from config import (
        SIATA_GHI_FILE, SIATA_GHI_VAR,
        CSI_GHI_FILE, CSI_VAR_NAME,
        CSI_INDEX_DIR, SIATA_CSI_FILE, SIATA_CSI_VAR,
    )

    calculator = IndexCalculator(
        ghi_path=SIATA_GHI_FILE,
        ref_path=CSI_GHI_FILE,
        out_dir=CSI_INDEX_DIR,
        out_file=os.path.basename(SIATA_CSI_FILE),
        ghi_var=SIATA_GHI_VAR,        # siata_ghi → "GHI_clean"
        ref_var=CSI_VAR_NAME,          # ref_clearsky_ghi → "clear_sky_ghi"
        index_name=SIATA_CSI_VAR,      # siata_csi → "clearsky_index_Siata"
    )
    calculator.process()
    
    """
    This script is used to calculate the Clear sky and the Clearness index as a fraction of the 
    GFS /Siata RAdiation and the clear sky GHI or ext GHI
    1. get GHI Path (denominator) -> GFS or SIata GHI
    2. get Reference path (numerator) -> Clear sky ghi or EXT radiation
    3. out file name of the index file
    4. ghi var -> name of variable in the GFS or Siara file
    5. ref var -> name of the variable in the Clear sky GHI or EXT GHI file 
    6. index nams -> name of the index variable in the new indice file
    
    
    
    
    
    """






