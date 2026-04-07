#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 13:14:43 2025

@author: leonardmerl
"""
import requests
import xarray as xr 
import os
from datetime import datetime, timedelta
import time 
import pandas as pd
import numpy as np


class GFSDataDownloader:
    """
    A class to automate the retrieval, processing, and storage of GFS forecast data.
    The data is downloaded for a specified time period, geographical region, and selected forecast hours.

    Attributes:
    -----------
    save_dir : str
        The directory where NetCDF files will be saved.
    """
    def __init__(self,save_dir="./_1_data_acquisition/raw_data"):
        """
        Initializes the GFSDataDownloader class.

        Parameters:
        -----------
        save_dir : str, optional
            Directory where downloaded NetCDF files will be stored (default is "/_1_data_acquisition/raw_data").

        Actions:
        --------
        - Creates the specified directory if it does not exist.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok= True)
        
    def construct_url(self, date, forecast_hour, forecast_step):
        """
        Constructs the URL for the NOAA GFS dataset. Start 1.4.2021
    
        Parameters:
        -----------
        date : str
            The forecast initialization date in YYYYMMDD format.
        forecast_hour : int or float
            The GFS cycle time (00, 06, 12, 18 UTC).
        forecast_step : int or float
            The forecast hour step (e.g., 001, 006, 012).
    
        Returns:
        --------
        tuple(str, str)
            A tuple containing:
            - grib_url: The direct GRIB2 file URL for the given parameters.
            - idx_url: The corresponding .idx file URL.
        """
        base_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

    
        # 🔹 Ensure forecast_hour and forecast_step are integers before formatting
        forecast_hour = int(forecast_hour)
        forecast_step = int(forecast_step)
        
        grib_url = f"{base_url}/gfs.{date}/{forecast_hour:02d}/atmos/gfs.t{forecast_hour:02d}z.pgrb2.0p25.f{forecast_step:03d}"
        idx_url = f"{grib_url}.idx"
    
        return grib_url, idx_url
        
    def get_dswrf_variable_name(self,timestep):
        """
        Determines the appropriate DSWRF variable name based on the forecast timestep. 
        This step is crucial in finding the appropriate byte offset in the idx files

        Parameters:
        -----------
        timestep : int
            The forecast hour for which the DSWRF variable is needed.

        Returns:
        --------
        str
            The DSWRF variable name corresponding to the given timestep.

        Raises:
        -------
        ValueError
            If the provided timestep is not in the valid range (1-23).
        """
        match timestep:
            case 1 | 2 | 3 | 4 | 5:  
                return f"DSWRF:surface:0-{timestep} hour ave fcst"
            case 6:
                return "DSWRF:surface:0-6 hour ave fcst"
            case 7 | 8 | 9 | 10 | 11:
                return f"DSWRF:surface:6-{timestep} hour ave fcst"
            case 12:
                return "DSWRF:surface:6-12 hour ave fcst"
            case 13 | 14 | 15 | 16 | 17:
                return f"DSWRF:surface:12-{timestep} hour ave fcst"
            case 18:
                return "DSWRF:surface:12-18 hour ave fcst"
            case 19 | 20 | 21 | 22 | 23|24:
                return f"DSWRF:surface:18-{timestep} hour ave fcst"
            case _:
                raise ValueError(f"❌ Invalid timestep: {timestep}")
                
    def get_dswrf_interval(self, hour):
        """
        Determines the DSWRF averaging period based on the forecast hour. This is only important for the correct notation of the output files.
        For each timestep the DSWRF value is different, because its calculated over a different intervall 
        -> this function for a correct notation of the DSWRF intervall

        Parameters:
        -----------
        hour : int
            The forecast hour.

        Returns:
        --------
        int
            The DSWRF averaging period (e.g., 6 for a 6-hour average).

        Raises:
        -------
        ValueError
            If the forecast hour is outside the valid range (1-24).
        """
        if hour in [6, 12, 18, 24]:  
            return 6  
        elif hour in [1, 2, 3, 4, 5]:  
            return hour  
        elif hour in [7, 8, 9, 10, 11]:  
            return hour - 6  
        elif hour in [13, 14, 15, 16, 17]:  
            return hour - 12  
        elif hour in [19, 20, 21, 22, 23, 24]:  
            return hour - 18  
        else:
            raise ValueError(f"❌ Invalid forecast hour: {hour}")
            
            
    def find_dswrf_in_idx(self, idx_url, timestep):
        """
        Extracts the byte offsets for DSWRF data from the .idx file.
        For each grib2 file in the SW3 bucket there is a idx file saved, containing all available indices of the corresponding file.

        Parameters:
        -----------
        idx_url : str
            URL of the .idx file containing index information for the GRIB2 file.
        timestep : int
            The forecast hour for which DSWRF data is needed.

        Returns:
        --------
        tuple
            A tuple containing the start and end byte positions for the DSWRF data.

        Raises:
        -------
        Exception
            If the DSWRF variable is not found in the .idx file.
        """
        dswrf_name = self.get_dswrf_variable_name(timestep)
        response = requests.get(idx_url)
        if response.status_code != 200:
            raise Exception(f"❌ Failed to load .idx file: {response.status_code}")

        lines = response.text.splitlines()
        dswrf_offsets = None

        for i, line in enumerate(lines):
            if dswrf_name in line:
                parts = line.split(":")
                try:
                    start_byte = int(parts[1])
                    end_byte = int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None
                    dswrf_offsets = (start_byte, end_byte)
                    break
                except (IndexError, ValueError):
                    continue

        if not dswrf_offsets:
            raise Exception(f"❌ DSWRF variable '{dswrf_name}' not found in .idx file.")

        return dswrf_offsets
    
    def find_dswrf_variable(self, ds):
        """
        Identifies the correct DSWRF variable from the dataset.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            The dataset loaded from the GRIB2 file.
    
        Returns:
        --------
        str or None
            The correct DSWRF variable name if found, otherwise None.
        """
        possible_names = [name for name in ds.data_vars.keys() if "DSWRF" in name or "sdswrf" in name ]
        return possible_names[0] if possible_names else None
    
    def download_and_extract(self, grib_url, idx_url, timestep, lat_range, lon_range, UTC_OFFSET):
        """
        Downloads and extracts DSWRF data for a specified region or point.
    
        Parameters:
        -----------
        grib_url : str
            URL of the GRIB2 file.
        idx_url : str
            URL of the .idx file for the GRIB2 data.
        timestep : int
            The forecast hour.
        lat_range : list
            The latitude range [min, max] or single-element list.
        lon_range : list
            The longitude range [min, max] or single-element list.
        UTC_OFFSET: int
            hours of time difference between UTC and local region
    
        Returns:
        --------
        xarray.DataArray
            The extracted DSWRF data for the specified region or grid point.
        """
        dswrf_offsets = self.find_dswrf_in_idx(idx_url, timestep)
        headers = {"Range": f"bytes={dswrf_offsets[0]}-{dswrf_offsets[1] if dswrf_offsets[1] else ''}"}
        response = requests.get(grib_url, headers=headers)
    
        if response.status_code not in [206, 200]:
            raise Exception(f"❌ HTTP Range Request failed: {response.status_code}")
    
        temp_file = "temp.grib2"
        with open(temp_file, "wb") as f:
            f.write(response.content)
    
        try:
            ds = xr.open_dataset(temp_file, engine="cfgrib")
            dswrf_var = self.find_dswrf_variable(ds)
    
            if not dswrf_var:
                return None
    
            # Longitude anpassen (z. B. -75 → 285)
            adj_lon_range = (
                [lon + 360 if lon < 0 else lon for lon in lon_range]
                if isinstance(lon_range, list)
                else [lon_range + 360 if lon_range < 0 else lon_range]
            )
    
            # Einzelpunkt oder Region?
            if len(lat_range) == 1 and len(lon_range) == 1:
                ds_region = ds[dswrf_var].sel(
                    latitude=lat_range[0],
                    longitude=adj_lon_range[0],
                    method="nearest"
                )
            else:
                ds_region = ds[dswrf_var].sel(
                    latitude=slice(lat_range[1], lat_range[0]),
                    longitude=slice(adj_lon_range[0], adj_lon_range[1])
                )
    
            # Zeitkorrektur für lokale Zeitzone
            if "time" in ds_region.coords:
                ds_region["time"] = ds_region["time"] + np.timedelta64(UTC_OFFSET, 'h')
            if "valid_time" in ds_region.coords:
                ds_region["valid_time"] = ds_region["valid_time"] + np.timedelta64(UTC_OFFSET, 'h')
    
            return ds_region
    
        except Exception as e:
            print(f"❌ Error loading GRIB file: {e}")
            return None


    def download_for_period(self, start_date, end_date, lat_range, lon_range,UTC_OFFSET,launch_times_utc, start_forecasthour, end_forecasthour):
        """
        Downloads DSWRF data for a specified period and saves it as NetCDF files.
        The forecast hours are automatically converted to Colombian local time (UTC-5).
        The filename includes:
            - The forecast issue time in Colombian time.
            - The time difference in hours between the forecast issue and the prediction.
            - The local start time instead of the UTC start time.
    
        Parameters:
        -----------
        start_date : str
            Start date in YYYYMMDD format.
        end_date : str
            End date in YYYYMMDD format.
        lat_range : list
            Latitude range [min, max].
        lon_range : list
            Longitude range [min, max].
        """
        date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")
        forecast_hours = list(range(start_forecasthour, end_forecasthour+1))
        
        
        while date <= end_date:
            start_time_utc = date.strftime("%Y%m%d")
    
            for forecast_hour_utc in launch_times_utc:
                # 🔁 Sonderbehandlung für 00 UTC (bzw. 24 als Eingabe)
                if forecast_hour_utc == 24 or forecast_hour_utc == 0:
                    utc_launch_date = date + timedelta(days=1)
                    actual_forecast_hour = 0
                else:
                    utc_launch_date = date
                    actual_forecast_hour = forecast_hour_utc
            
                # GFS-Zeitstempel in UTC
                utc_launch_datetime = datetime.combine(utc_launch_date, datetime.min.time()) + timedelta(hours=actual_forecast_hour)
                utc_launch_str = utc_launch_datetime.strftime("%Y-%m-%d_%H00")
                # Lokale Forecast-Startzeit (z. B. Kolumbien = UTC-5)
                local_forecast_hour = (actual_forecast_hour + UTC_OFFSET) % 24
                local_forecast_date = utc_launch_date if actual_forecast_hour >= abs(UTC_OFFSET) else utc_launch_date - timedelta(days=1)
            
                local_start_datetime = utc_launch_datetime + timedelta(hours=UTC_OFFSET)
                local_start_time_str = local_start_datetime.strftime("%Y-%m-%d_%H00")

                            
    
                for hour in forecast_hours:  # Modify range for more forecast hours
                    adjusted_hour = hour  # No need to shift, we are using local start time directly
                    forecast_hour_difference = adjusted_hour  # Forecast step in hours
    
                    # Compute valid forecast time in Colombian time correctly
                    valid_datetime_local = local_forecast_date + timedelta(hours=local_forecast_hour + forecast_hour_difference)
                    valid_date_str_local = valid_datetime_local.strftime("%Y-%m-%d_%H00")

    
                    grib_url, idx_url = self.construct_url(
                        utc_launch_datetime.strftime("%Y%m%d"),
                        utc_launch_datetime.hour,
                        adjusted_hour
                    )
                    print(f"📥 Downloading DSWRF Data: {grib_url}")
    
                    ds = self.download_and_extract(grib_url, idx_url, adjusted_hour, lat_range, lon_range,UTC_OFFSET)
                    #print(f"coords are {ds.coords}") #check the coordinates of the data array
                    #print(ds.values) check values for sdswrf
                    #print(ds.time.values)
                    #print(ds.valid_time.values)
                    #print(f"dimensions: {ds.dims}")
                    #print(ds.valid_time.values)
                    #print(ds.values)
                    #print(ds.sdswrf.values)
                    #print(ds)
                    
                    
                    
                    
                    
                    if ds is not None:
                        dswrf_interval = self.get_dswrf_interval(hour)
                        forecast_date_str = local_forecast_date.strftime("%Y-%m-%d")
    
                        print(f"✅ Adjusted Forecast Time (Local Time): {valid_date_str_local}")
    
                        # 🎯 Updated filename with correct local start time and valid forecast time
                        nc_filename = f"{self.save_dir}/{forecast_date_str}_{local_forecast_hour:02d}_{forecast_hour_difference:02d}_DSWRF{dswrf_interval}_{local_start_time_str}_{valid_date_str_local}_UTCSTART:{utc_launch_str}.nc"
                        ds.to_netcdf(nc_filename)
                        print(f"💾 Saved as: {nc_filename}")
    
            date += timedelta(days=1)
    
if __name__ == "__main__":
    start_time = time.perf_counter()
    downloader = GFSDataDownloader()
    downloader.download_for_period(
        start_date="20210401",
        end_date="20210430",
        lat_range=[6.25],
        lon_range=[-75.5],
        UTC_OFFSET=-5,
        launch_times_utc= [6,12,18,00], 
        start_forecasthour = 1,
        end_forecasthour=24
        
        
        
    )
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    '''
    _1_data_acquisition/raw_data/2021-04-01_01_01_DSWRF1_2021-04-01_0100_2021-04-01_0200_UTCSTART:2021-04-01_0600.nc 
    --------------------------
    DSWRF -> downward short wave radiation as a mean of a given intervall (intevall is determined in the function)
    Mean of the intervall of hours but resetting every 6 hours 
    prediction for 1 hour after forecast time is dswrf1 because the intervall ist 0-1
    ''for 4th hour its dswrf4 -> intervall from 0-4 for 
    ''9th hour its dswrf3 -> intervall from6-9
    ''for 23 hour its dswrf5 -> intervall from 18-23
    -------------------------------------------------
    2024-03-18 -> Forecast launch date -> at the start for sorting and better outline
    -------------------------------------------------
    _07_ -> local forecast launch time -> at the start for better sorting and outline
    -------------------------------------------------
    _09_ -> hour of the forecast -> prediction for the 9th hour after the launch time -> start at 7-> forecast for 16
    -------------------------------------------------
    _20240318_0700_ -> lauch time 0700 corresponding to 7am also possible -> 01 am (0100) 13pm (1300) and 19 pm (1900)
    -> very important!!! -> this is local columbian time -> these launch times are equivalent to a utc launch time of 6,12,18,24 
    -> but its easier to just ge thr forecast data for these times and then change the variables in the file description
    -------------------------------------------------
    20240318_1400 -> local time of the prediction == 0600+0800 = 1400 -> Gets the exact time for which the prediction is made 
        
    -------------------------------------------------
    UTCSTART:2021-04-01_0600.nc 
     -> UTC time of launch -> can be changed later on -> as a reminder, that there is a time difference
    -> time difference is hardcoded to be 6 hours 
    -> will need to be changed later
    
    
    '''


















