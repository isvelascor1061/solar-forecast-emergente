#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast_1 Module

This module defines the Forecast_1 class for retrieving and processing GFS forecast data.
The Forecast_1 class is designed to extract short-wave radiation flux data from the GFS dataset 
and save it to NetCDF files.

Usage:
    - Define the date range, hours, and timesteps for the forecast.
    - Specify the folder where the forecast data will be saved.
    - Iterate over each date, hour, and timestep, and use the Forecast_1 class to retrieve and save the data.
"""

import os
import xarray as xr
import logging
import numpy as np
from datetime import datetime, timedelta
from _1_data_acquisition.Extra_code.thredds_dswrf36.coordinate_query_xarray2 import CoordIndexConverter, CoordinateInfo

class Forecast1():
    def __init__(self, folder_path, date, hour, timestep):
        """
        Initializes the Forecast_1 object with the specified forecast properties.

        Parameters
        ----------
        folder_path : str
            The directory where forecast data will be saved.
        date : tuple 
            The date of the forecast issuance in UTC Time (Year, Month, Day).
        hour : str
            The hour of the forecast issuance (e.g., "00", "06").
        timestep : str
            The timestep of the forecast (e.g., "003").
        """
        self.folder_path = folder_path
        self.date = datetime(*date)  # Ensure date is a datetime object
        self.hour = hour
        self.step = int(timestep)

        # Create URL
        self.url = self.build_url()
        print(f"Dataset URL : {self.url}")

        # Activate logging
        logging.basicConfig(filename="forecast.log", level=logging.INFO)

        # Download dataset
        self.dataset = self.download_dataset()

    def build_url(self):
        """
        Builds the URL for downloading the dataset.

        Returns
        -------
        str
            The formatted URL for the dataset.
        """
        url_template = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/ds084.1/{year}/{date}/gfs.0p25.{date}{hour}.f{step}.grib2"
        
        return url_template.format(
            hour=self.hour,
            step=str(self.step).zfill(3),
            date=self.date.strftime("%Y%m%d"),
            year=self.date.strftime("%Y")
        )
    
    def download_dataset(self):
        """
        Downloads the dataset from the generated URL.

        Returns
        -------
        xarray.Dataset or None
            The downloaded dataset, or None if an error occurs.
        """  
        try: 
            dataset = xr.open_dataset(self.url, engine="netcdf4")
            return dataset
        except Exception as e:
            logging.error(f"Error while opening dataset from {self.url}: {e}")
            return None

    def get_variable_name(self):
        """
        Determines the correct variable name based on the timestep.

        Returns
        -------
        str
            The variable name to extract from the dataset.
        """
        return "Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average" if self.step % 6 == 0 else "Downward_Short-Wave_Radiation_Flux_surface_3_Hour_Average"

    def get_1(self, lat_range, lon_range):
        """
        Retrieves data from the dataset for given coordinates and variables, and saves it.

        Parameters
        ----------
        lat_range : str
            The latitude range in the format "[min:max]".
        lon_range : str
            The longitude range in the format "[min:max]".

        Returns
        -------
        data : xarray.Dataset
            The extracted Dataset containing the gridded radiation data.
        """
        try:
            logging.info("Starting data extraction...")

            # 1️⃣ Create `coord_query` dynamically
            coord_info = CoordinateInfo(self.date, self.hour, str(self.step).zfill(3))
            coords_available = coord_info.get_coordinates()
            coord_query = CoordIndexConverter(coords_available)

            # 2️⃣ Determine the correct variable based on the forecast step
            variable_name = self.get_variable_name()
            if variable_name not in self.dataset:
                raise KeyError(f"Variable '{variable_name}' not found in dataset.")

            var = self.dataset[variable_name]
            logging.info(f"Using variable: {variable_name}")

            # 3️⃣ Determine spatial indices based on latitude and longitude ranges
            lat_slice = coord_query.get_coord_slice(lat_range, "lat")
            lon_slice = coord_query.get_coord_slice(lon_range, "lon")

            index = (0, lat_slice, lon_slice)

            # 4️⃣ Extract the required data
            data = var[index]

            # 5️⃣ Generate filename
            abbreviation = "DSWRF6" if variable_name == "Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average" else "DSWRF3"
            
            # Convert reference time to string format
            reftime_str = self.date.strftime("%Y-%m-%d")
            
            # Calculate forecast time
            forecast_time = self.date + timedelta(hours=self.step)
            forecast_time_str = forecast_time.strftime("%Y-%m-%d_%H%M")

            # Define filename
            filename = f"{self.hour}_{abbreviation}_{reftime_str}_{self.hour}00_{forecast_time_str}.nc"

            full_path = os.path.join(self.folder_path, filename)

            # 6️⃣ Save data as NetCDF file
            data.to_netcdf(full_path)
            logging.info(f"Data successfully saved to {full_path}")
            return data

        except Exception as e:
            logging.error(f"Error in get_1(): {e}")
            return None

# Main script execution
if __name__ == "__main__":
    # Set output directory
    output_folder = "raw_data"

    # Define the date range for forecasts
    start_date = "2016-01-01"
    end_date= "2016-01-02"
    date_range =np.arange(start_date,end_date,dtype="datetime64[D]")
    date_range = [tuple(map(int, str(d).split("-"))) for d in date_range]

    # Define forecast hours (GFS model run times)
    hours = ["00", "06", "12", "18"]

    # Define timesteps (how far ahead the forecast should be)
    start = 3
    end = 24
    step = 3
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

                    forecast = Forecast1(folder_path=output_folder, date=single_date, hour=hour, timestep=timestep)
                    result = forecast.get_1(lat_range, lon_range)

                    if result is not None:
                        print("✅ Data successfully downloaded and saved.")
                    else:
                        print("❌ Error downloading data.")

                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    
"""
Filename:
12_DSWRF6_2019-01-01_1200_2019-01-01_1500.nc
DSWRF6: Downwards shortwave radiation flux, six hour average
2019-01-01: Date
1200 GFS launch time
1500: of hours ahead the reference time 12 + 15, interval for 2019-01-02 from 00 to 6  
12: Also GFS Launchtime, but again at the front for a better overview and organization  
"""
