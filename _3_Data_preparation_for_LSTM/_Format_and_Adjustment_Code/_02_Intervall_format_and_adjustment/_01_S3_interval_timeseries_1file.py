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
from pathlib import Path

class DSWRF1Merger:
    """
    Merges NetCDF files with the specified variable (e.g., dswrf1_0100)
    and saves them as a single NetCDF file.

    Dynamically handles different launch times and variable names.
    """

    def __init__(
        self,
        input_folder: str | Path,
        output_folder: str | Path,
        launch_time: str,
        variable_name: str = None,
        file_pattern: str = None
    ):
        """
        Args:
            input_folder: Path to the input NetCDF files
            output_folder: Path to the folder where merged NetCDF will be saved
            launch_time: Launch time (e.g., "0100") used for file search and variable names
            variable_name: Name of the variable in the dataset (e.g., "dswrf1_0100"), 
                           if None, the variable is automatically set to "dswrf1_{launch_time}"
            file_pattern: Glob pattern for input files. 
                          If None, defaults to "dswrf1_*{launch_time}.nc"
        """
        # Assigning input, output folder, and launch time values to class attributes
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.launch_time = launch_time
        
        # Set the variable name, defaulting to "dswrf1_{launch_time}" if not provided
        self.variable_name = variable_name or f"dswrf1_{launch_time}"
        
        # Set file pattern, defaulting to match files with the given launch time
        self.file_pattern = file_pattern or f"dswrf1_*{launch_time}.nc"

        # Ensure the output folder exists, creating it if necessary
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _rename_variable(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Renames the variable in the dataset to the desired name.
        Assumption: The dataset contains exactly one variable other than coordinates.
        """
        # Extract the list of data variables in the dataset
        vars_in_ds = list(ds.data_vars)
        
        # Raise error if no data variables are found
        if len(vars_in_ds) == 0:
            raise ValueError("No data variable found in the dataset.")
        
        # If the variable name is already correct, return the dataset as is
        if self.variable_name in vars_in_ds:
            return ds
        else:
            # Rename the first variable in the dataset to the desired variable name
            original_var = vars_in_ds[0]
            renamed_ds = ds.rename({original_var: self.variable_name})
            return renamed_ds

    def merge_and_save(self, output_filename: str) -> None:
        """
        Merges the NetCDF files and saves them as a single output NetCDF file.
        It also handles the case where the output file already exists.
        
        Args:
            output_filename: The name of the output file where the merged data will be stored.
        """
        # Define the output file path
        out_file = self.output_folder / output_filename
        
        # Search for all matching files in the input folder
        files = sorted(self.input_folder.glob(self.file_pattern))
        
        # If no files are found, raise an error
        if not files:
            raise FileNotFoundError(
                f"No files found matching pattern '{self.file_pattern}' in folder '{self.input_folder}'."
            )

        # Check if the output file already exists
        if out_file.exists():
            print(f"Output file {out_file} already exists. Searching for new observation times...")
            
            # Load existing data from the output file
            existing_ds = xr.load_dataset(out_file)
            existing_times = existing_ds["observation_time"].values

            new_ds_list = []  # List to store datasets with new observation times
            
            # Loop through the input files to find new observation times
            for f in files:
                ds_tmp = xr.open_dataset(f)
                ds_tmp = self._rename_variable(ds_tmp)
                obs_times = ds_tmp["observation_time"].values
                
                # Create a mask to filter out existing observation times
                new_mask = ~np.isin(obs_times, existing_times)
                if new_mask.any():
                    ds_new = ds_tmp.sel(observation_time=obs_times[new_mask])
                    new_ds_list.append(ds_new)
                ds_tmp.close()

            # If new data is found, merge it with the existing data
            if new_ds_list:
                new_combined = xr.concat(new_ds_list, dim="observation_time")
                ds_merged = xr.concat([existing_ds, new_combined], dim="observation_time")
                ds_merged = ds_merged.sortby("observation_time")
                ds_merged.to_netcdf(out_file)
                ds_merged.close()
                print(f"File {out_file} has been updated with new observation times.")
            else:
                print(f"No new observation times found. File {out_file} remains unchanged.")
            existing_ds.close()
        else:
            # If the output file does not exist, merge all files and save them
            ds_list = []
            for f in files:
                ds_tmp = xr.open_dataset(f)
                ds_tmp = self._rename_variable(ds_tmp)
                ds_list.append(ds_tmp)

            # Concatenate all datasets along the observation_time dimension
            ds_merged = xr.concat(ds_list, dim="observation_time")
            ds_merged = ds_merged.sortby("observation_time")
            ds_merged.to_netcdf(out_file)
            ds_merged.close()
            print(f"NetCDF saved: {out_file}")


if __name__ == "__main__":
    # Set the paths for the input and output folders
    input_folder = "_2_data_postprocessing/_02_merged_Rad1data/dswrf/merged_raw_dswrf1_1900"
    output_folder = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Unclipped_merged_dswrf1"

    # Define the launch time and the variable name
    launch_time = "1900"
    variable_name = f"dswrf1_{launch_time}"

    # Define the file pattern for the input files
    file_pattern = f"dswrf1_*{launch_time}.nc"
    
    # Set the output filename
    output_filename = f"dswrf1_{launch_time}.nc"

    # Create an instance of the DSWRF1Merger class
    merger = DSWRF1Merger(
        input_folder=input_folder,
        output_folder=output_folder,
        launch_time=launch_time,
        variable_name=variable_name,
        file_pattern=file_pattern
    )
    
    # Call the merge_and_save method to merge the files and save the output
    merger.merge_and_save(output_filename)
    """
    this code merges the daily dswrf1 files from the data processing part into 1 timeseries 
    with the same launch time
    Execution:
        1. Input folder -> Folder of the by day merged dswrf1 values -> for one launch time e.g. 
        _2_data_postprocessing/_02_merged_Rad1data/dswrf/merged_raw_dswrf1_0100
        2. Output folder is the unclipped GFS Folder with the same launch time 
        e.g.: _3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Unclipped_merged_dswrf1
        3. launch_time -> launch time for the corresponding folders and files
        eg: 0100
        4. file pattern in the input folder -> dswrf1_{launch_time}
        5. output filename -> dswrf1_{launch_time}
        
        All of the daily files are merged into 1 netdcf4 file with the dimension observvation time 
        -> timeseries for all available data with 1 launch time
        
    """












