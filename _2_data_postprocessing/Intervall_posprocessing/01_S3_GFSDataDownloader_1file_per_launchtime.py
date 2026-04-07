from pathlib import Path
import xarray as xr
import pandas as pd

class GFSDataMerger:
    """
    This class is responsible for merging GFS NetCDF files by their launch time.
    The files are grouped by launch time and then merged along the observation time dimension.
    The merged datasets are saved as new NetCDF files in a specified output directory.
    """

    def __init__(self, input_dir, output_dir, variable_name=None):
        """
        Initializes the GFSDataMerger class with the given input and output directories.
        If a variable name is provided, only that variable will be merged.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing the raw NetCDF files.
        output_dir : str or Path
            Directory to save the merged files.
        variable_name : str or None
            The name of the data variable to merge.
            If None, all variables will be merged.
        """
        # Convert input and output directories to Path objects for ease of use
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)  # Ensure output directory exists
        self.variable_name = variable_name

    def merge_files_by_launch_time(self):
        """
        This method processes the NetCDF files in the input directory, groups them by launch time,
        merges the files for each launch time, and saves the merged datasets in the output directory.
        """
        # List all NetCDF files in the input directory
        nc_files = list(self.input_dir.glob("*.nc"))
        launch_groups = {}

        # Loop through each NetCDF file to group them by launch time
        for nc_file in nc_files:
            try:
                ds = xr.open_dataset(nc_file,engine="netcdf4")  # Open the NetCDF file using xarray

                # If a variable name is provided, limit the dataset to that variable
                if self.variable_name:
                    if self.variable_name in ds.data_vars:
                        ds = ds[[self.variable_name]]  # Select only the specified variable
                    else:
                        # If the variable is not found, print a warning and skip the file
                        print(f"Warning: Variable '{self.variable_name}' not found in {nc_file.name}, skipping.")
                        ds.close()  # Close the dataset to free resources
                        continue

                # Extract the launch time from the dataset (assumes 'time' dimension exists)
                launch_time = pd.to_datetime(ds.time.values.item())

                # Group datasets by their launch time
                if launch_time not in launch_groups:
                    launch_groups[launch_time] = []

                launch_groups[launch_time].append(ds)  # Append the dataset to the corresponding launch time group

            except Exception as e:
                print(f"Error processing {nc_file.name}: {e}")

        # Merge files for each launch time group and save them
        for launch_time, datasets in launch_groups.items():
            try:
                # Concatenate all datasets along the 'valid_time' dimension
                merged = xr.concat(datasets, dim="valid_time")
                merged = merged.rename({"valid_time": "observation_time"})  # Rename time dimensions for clarity
                merged = merged.rename({"time": "launch_time"})
                merged = merged.sortby("observation_time")  # Sort by observation time for consistency

                # Construct a filename for the merged dataset based on launch time and variable
                filename = f"{self.variable_name or 'merged'}_{launch_time.strftime('%Y%m%d_%H%M')}.nc"
                merged.to_netcdf(self.output_dir / filename)  # Save the merged dataset as a NetCDF file

                print(f"Saved: {filename} for launch_time {launch_time}")

                # Close all datasets after merging
                for ds in datasets:
                    ds.close()

            except Exception as e:
                print(f"Error merging for launch_time {launch_time}: {e}")

def main():
    """
    Main function to run the GFSDataMerger with specified directories and variable name.
    It reads the raw NetCDF files, merges them by launch time, and saves the merged files.
    """
    # Set the input and output directories, and specify the variable to merge
    input_dir = "_1_data_acquisition/_01_raw_rad_data/dswrf/raw_dswrf_0100"
    output_dir = "_2_data_postprocessing/_01_merged_Radrawdata/dswrf/merged_raw_dswrf_0100"
    variable_name = "sdswrf"  # Example: Specify the variable to merge

    # Initialize the GFSDataMerger object
    merger = GFSDataMerger(input_dir, output_dir, variable_name=variable_name)

    # Merge the files by launch time
    merger.merge_files_by_launch_time()

if __name__ == "__main__":
    main()

    """
    This is the first post-processing script to handle interval parameters.
    It performs the following:
    1. It reads the 1-hour files (this code needs to be executed once for each launch time).
    2. Merges the files for a corresponding day (merged on observation time with the exact same launch time).
    3. Saves them in a new directory (_2_data_postprocessing/_01_merged_Radrawdata + each launch time).
    4. Renames 'time' to 'launch_time' and 'valid_time' to 'observation_time'.
    
    Important -> At this point this pipeline only works with 1 tile in the grid 
    -> to work with multiple it would need to subset with each corresponding tile 
    -> Code to do that would need to be added 
    
    """
