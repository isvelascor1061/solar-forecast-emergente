#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOMADS Data Downloader

Downloads and processes shortwave radiation flux data from NOAA NOMADS.

@author: leonardmerl
"""

import os
import xarray as xr
import logging
import numpy as np
from datetime import datetime, timedelta
from _1_data_acquisition.Extra_code.nomads_dswrf1.NOMADS_coordinate_query_xarray import CoordIndexConverter, CoordinateInfo

class Forecast1():
    def __init__(self, folder_path, date, hour, timestep):
        """
        Initializes the Forecast_1 object with the specified forecast properties.
        """
        self.folder_path = folder_path
        self.date = datetime(*date)
        self.hour = hour
        self.step = int(timestep)

        # Create URL for NOAA NOMADS
        self.url = self.build_url()
        print(f"🔗 Dataset URL : {self.url}")

        # Activate logging
        logging.basicConfig(filename="forecast.log", level=logging.INFO)

        # Download dataset
        self.dataset = self.download_dataset()

    def build_url(self):
        """
        Builds the URL for downloading the dataset from NOAA NOMADS.
        """
        date_str = self.date.strftime("%Y%m%d")  
        return f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date_str}/gfs_0p25_{self.hour}z"

    def download_dataset(self):
        """
        Downloads the dataset from the generated URL on NOAA NOMADS.
        """
        try:
            dataset = xr.open_dataset(self.url, engine="netcdf4")
            return dataset
        except Exception as e:
            logging.error(f"❌ Error while opening dataset from {self.url}: {e}")
            return None

    def get_variable_name(self):
        """
        Determines the correct variable name for NOAA NOMADS data.
        Uses dswrfsfc for shortwave radiation.
        """
        available_vars = list(self.dataset.variables)

        if "dswrfsfc" in available_vars:  
            return "dswrfsfc"
        if "uswrfsfc" in available_vars:  
            return "uswrfsfc"
        if "ulwrfsfc" in available_vars:  
            return "ulwrfsfc"

        raise KeyError("No suitable radiation flux variable found in dataset.")

    def get_1(self, lat_range, lon_range):
        """
        Retrieves data from the dataset for given coordinates and variables, and saves it.
        """
        try:
            logging.info("✅ Starting data extraction...")
    
            # Check if dataset was loaded
            if self.dataset is None:
                raise ValueError("❌ Dataset could not be loaded.")
    
            # 1️⃣ Create `coord_query` dynamically
            coord_info = CoordinateInfo(self.date, self.hour)
            coords_available = coord_info.get_coordinates()
            coord_query = CoordIndexConverter(coords_available)
    
            # 2️⃣ Determine the correct variable
            variable_name = self.get_variable_name()
    
            # Check if the variable exists
            if variable_name not in self.dataset:
                raise KeyError(f"❌ Variable '{variable_name}' not found in dataset. Available variables: {list(self.dataset.variables)}")
    
            var = self.dataset[variable_name]
            logging.info(f"✅ Using variable: {variable_name}")
    
            # 3️⃣ Check `lat` and `lon`
            if "lat" not in self.dataset.coords or "lon" not in self.dataset.coords:
                raise KeyError(f"❌ Latitude or longitude not found in dataset. Available coordinates: {list(self.dataset.coords)}")
    
            # 4️⃣ Retrieve `lat` and `lon`
            lat_slice = coord_query.get_coord_slice(lat_range, "lat")
            lon_slice = coord_query.get_coord_slice(lon_range, "lon")
    
            # 5️⃣ If `var` is empty, raise an error
            if var is None:
                raise ValueError("❌ The selected variable is empty.")
    
            # 6️⃣ Extract values
            index = (0, lat_slice, lon_slice)
            data = var[index]
    
            # 7️⃣ Generate filename with `DSWRF1`
            abbreviation = "DSWRF1" if variable_name == "dswrfsfc" else "DSWRF3"
            reftime_str = self.date.strftime("%Y-%m-%d")
            forecast_time = self.date + timedelta(hours=self.step)
            forecast_time_str = forecast_time.strftime("%Y-%m-%d_%H%M")
    
            filename = f"{self.hour}_{abbreviation}_{reftime_str}_{self.hour}00_{forecast_time_str}.nc"
    
            full_path = os.path.join(self.folder_path, filename)
    
            # 8️⃣ Save data as NetCDF
            data.to_netcdf(full_path)
            logging.info(f"✅ Data successfully saved to {full_path}")
            return data
    
        except Exception as e:
            logging.error(f"❌ Error in get_1(): {e}")
            return None



# Main script execution for NOMADS
if __name__ == "__main__":
    # Set output directory
    output_folder = "raw_data"

    # Define the date range for forecasts
    start_date = "2025-03-11"
    end_date = "2025-03-12"
    date_range = np.arange(start_date, end_date, dtype="datetime64[D]")
    date_range = [tuple(map(int, str(d).split("-"))) for d in date_range]

    # Define forecast hours (GFS model run times)
    hours = ["00", "06", "12", "18"]

    # Define timesteps (how far ahead the forecast should be) → Hourly for the first 24 hours
    start = 1
    end = 24
    step = 1
    timesteps = [str(i).zfill(3) for i in range(start, end + 1, step)]

    # Define the spatial range for Medellín
    lat_range = "[6.0:6.5]"
    lon_range = "[-75.75:-75.25]"

    # Loop through each date, hour, and timestep to retrieve forecasts
    for single_date in date_range:
        for hour in hours[:1]:
            for timestep in timesteps:
                try:
                    print(f"Processing: {single_date}, Hour: {hour}, Timestep: {timestep}")

                    # Create Forecast instance for NOMADS
                    forecast = Forecast1(folder_path=output_folder, date=single_date, hour=hour, timestep=timestep)
                    result = forecast.get_1(lat_range, lon_range)

                    if result is not None:
                        print("✅ Data successfully downloaded and saved.")
                    else:
                        print("❌ Error downloading data.")

                except Exception as e:
                    print(f"❌ An error occurred: {str(e)}")
                    
"""
Filename:
12_DSWRF1_2025-03-14_1200_2025-03-14_1300.nc
DSWRF1: Downward shortwave radiation flux, 1-hour average
2025-03-14: Date
1200 GFS launch time
1300: Forecast for 1 hour later
"""
