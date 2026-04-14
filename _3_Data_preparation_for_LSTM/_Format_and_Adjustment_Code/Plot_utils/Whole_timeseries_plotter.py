#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:17:45 2025

@author: leonardmerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the merged DSWRF1 file and save it to a specified directory.
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path

def plot_merged_dswrf1(
    nc_file: str | Path,
    output_dir: str | Path,
    title: str = "DSWRF1 Time Series",
    out_filename: str = None
):
    """
    Loads the specified NetCDF file (containing 'dswrf1' and 'observation_time'),
    creates a time-series plot, and saves it to 'output_dir'.

    Args:
        nc_file (str | Path): Path to the NetCDF file to be plotted.
        output_dir (str | Path): Directory where the plot will be saved.
        title (str): Title of the plot.
        out_filename (str): Name of the PNG file to save.
                            If None, it is derived from nc_file.
    """
    nc_file = Path(nc_file)
    if not nc_file.is_file():
        raise FileNotFoundError(f"File not found: {nc_file}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If no out_filename is set, derive it from the NC filename
    # e.g. "GFS_merged_dswrf1_all.png" if nc_file="GFS_merged_dswrf1_all.nc"
    if out_filename is None:
        out_filename = nc_file.with_suffix('.png').name

    out_path = output_dir / out_filename

    # Load NetCDF
    ds = xr.open_dataset(nc_file, engine="netcdf4")

    # Convert to DataFrame
    df = ds[["dswrf1"]].to_dataframe().reset_index()
    df["observation_time"] = pd.to_datetime(df["observation_time"])

    # Create plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["observation_time"], df["dswrf1"], marker=".", linestyle="-")

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("DSWRF1 (W/m²)")
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")

    # Display plot
    plt.show()


if __name__ == "__main__":
    # Example:
    merged_file = "_3_Data_preparation_for_LSTM/Preparation_data/merged_timerseries/GFS_dswrf1/GFS_merged_dswrf1_all.nc"
    save_dir = "_3_Data_preparation_for_LSTM/Preparation_data/Merged_Plots/Merged_timeseries_plots/GFS_dswrf1_all"

    plot_merged_dswrf1(
        nc_file=merged_file,
        output_dir=save_dir,
        title="GFS DSWRF1 timeseries launch_time: 0100 (merged)",
        out_filename="GFS_dswrf1_merged_timeseries"  # Optional
    )
