#!/usr/bin/env python3
# merge_hourly_nc.py  ·  without CLI-Parsing
# -----------------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr


class HourlyObservationMerger:
    """Merges NetCDF hourly datasets along the *observation_time* dimension."""

    def __init__(
        self,
        input_dir: str | Path,
        output_path: str | Path,
        pattern: str = "*.nc",
        concat_dim: str = "observation_time",
    ) -> None:
        """
        Initializes the HourlyObservationMerger class to merge NetCDF files.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing the raw NetCDF files to be merged.
        output_path : str or Path
            The location where the merged NetCDF file will be saved.
        pattern : str, optional
            The file pattern to match for input files (default is '*.nc').
        concat_dim : str, optional
            The dimension along which the datasets will be concatenated (default is 'observation_time').
        """
        self.input_dir = Path(input_dir)  # Directory where raw NetCDF files are stored
        self.output_path = Path(output_path)  # Path to save the merged file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

        self.pattern = pattern  # File pattern to match input files
        self.concat_dim = concat_dim  # Dimension to concatenate along (e.g., 'observation_time')

    # ---------------------------------------------------------------- helpers
    def _promote_scalar_coord(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Promotes a scalar coordinate to a true dimension.
        If the coordinate 'observation_time' is scalar, it will be promoted to a dimension of length 1.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to modify.

        Returns
        -------
        xr.Dataset
            The modified dataset with 'observation_time' promoted to a dimension if necessary.
        """
        dim = self.concat_dim
        if dim in ds.dims:  # If 'observation_time' is already a dimension, return the dataset
            return ds

        if dim not in ds.coords:  # If the coordinate is not found in the dataset, raise an error
            raise KeyError(f"'{dim}' not found in the dataset.")

        coord = ds[dim]

        if not coord.dims:  # If the coordinate is scalar, promote it to a dimension of length 1
            value = coord.values
            ds = ds.drop_vars(dim)  # Drop the scalar coordinate
            ds = ds.expand_dims({dim: 1})  # Expand it to a dimension of length 1
            ds = ds.assign_coords({dim: (dim, [value])})  # Assign the scalar value as the dimension's value
            return ds

        # If the coordinate is 1-dimensional but part of another dimension (e.g., 'time')
        base = coord.dims[0]
        ds = ds.swap_dims({base: dim})  # Swap the dimensions if needed
        if base in ds.coords and base not in ds.dims:  # If the base dimension is not a data dimension, drop it
            ds = ds.drop_vars(base)
        return ds

    # ----------------------------------------------------------------- merge
    def merge(self) -> None:
        """
        Merges the NetCDF files from the input directory into a single NetCDF file.
        It groups the files by 'observation_time' and concatenates them along this dimension.
        The merged file is saved at the specified output path.
        """
        # List all NetCDF files in the input directory matching the specified pattern
        files = sorted(self.input_dir.glob(self.pattern))
        if not files:
            print("⚠️  No source files found.")
            return

        # Check if the output file already exists (i.e., if it's a continuation of a previous merge)
        if self.output_path.exists():
            master = xr.open_dataset(self.output_path, engine="netcdf4")
            known = set(master[self.concat_dim].values)  # The existing observation times in the master file
            print(f"→ Master loaded: {self.output_path.name}")
        else:
            master = None  # If the output file does not exist, start with an empty master
            known = set()

        new_parts: list[xr.Dataset] = []  # List to store datasets that need to be added
        for f in files:
            try:
                ds = xr.open_dataset(f, engine="netcdf4")  # Open the current NetCDF file
                ds = self._promote_scalar_coord(ds)  # Ensure 'observation_time' is a true dimension

                ts = ds[self.concat_dim].values[0]  # Get the first value of 'observation_time'
                if ts in known:  # Skip this file if it has already been processed (based on observation_time)
                    ds.close()  # Close the dataset to free up resources
                    continue

                new_parts.append(ds.load())  # Fully load the dataset into memory
                ds.close()  # Close the dataset
            except Exception as err:
                print(f"⚠️  {f.name}: {err} (skipped)")

        if not new_parts:
            print("✅ No new hours – everything is up to date.")
            if master is not None:
                master.close()  # Close the master file if it was opened
            return

        # Merge the new datasets along the 'observation_time' dimension
        combined = xr.concat(new_parts, dim=self.concat_dim)
        merged = xr.concat([master, combined], dim=self.concat_dim) if master else combined  # Concatenate with the master file
        merged = merged.sortby(self.concat_dim)  # Sort by observation time to maintain order

        # Ensure the output file is overwritten if it already exists
        if self.output_path.exists():
            self.output_path.unlink()  # Delete the existing output file
        
        # Save the merged dataset to a new NetCDF file
        merged.to_netcdf(self.output_path, mode='w')

        # Print a summary of how many new hours were added
        added = len(combined[self.concat_dim])
        print(f"✅ {added} new hour{'s' if added != 1 else ''} → {self.output_path}")

        merged.close()  # Close the merged dataset
        if master is not None:
            master.close()  # Close the master file if it was opened
        for ds in new_parts:
            ds.close()  # Close all newly loaded datasets

# ---------------------------------------------------------------- USER-INPUTS
if __name__ == "__main__":
    """
    Main entry point for the script.
    - Defines the input and output directories.
    - Specifies the variable pattern to match the NetCDF files.
    - Calls the merging process.
    """
    from config import RAW_MULTGFS_0100_DIR, MERGED_MULTGFS_0100_FILE
    INPUT_DIR = RAW_MULTGFS_0100_DIR
    OUTPUT_FILE = MERGED_MULTGFS_0100_FILE
    FILE_PATTERN = "*.nc"  # The file pattern for the NetCDF files to be merged

    # Initialize the HourlyObservationMerger with the directories and variable pattern
    HourlyObservationMerger(
        input_dir=INPUT_DIR,
        output_path=OUTPUT_FILE,
        pattern=FILE_PATTERN,
    ).merge()  # Start the merging process

    """
    This is the first post-processing script to handle interval parameters.
    The script performs the following:
    1. It reads the 1-hour files (this code needs to be executed once for each launch time).
    2. Merges the files for the whole Timespan (merged along observation_time with the exact same launch time).
    3. Saves them in a new directory (_2_data_postprocessing/_03_Merged_MultGFS_data + each launch time).
    4. Renames 'time' to 'launch_time' and 'valid_time' to 'observation_time' for consistency.
    -> Output is 1 Netcdf file for each launch time (0100, 0700, 1300, 1900)
    aligned on the observation_time dimension 
    containing all downloaded parameters 
    Tldr: Merges 1 hour locally saved files into 1 file for the whole timeseries 
    """
