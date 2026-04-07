#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:14:29 2025

@author: leonardmerl
"""

import requests
import xarray as xr 
import os
from datetime import datetime, timedelta
import time 
import numpy as np
import matplotlib.pyplot as plt

class GFSDataDownloader:
    """
    A class to automate the retrieval, processing, and storage of GFS forecast data.
    The data is downloaded for a specified time period, geographical region, and selected forecast hours.

    Attributes:
    -----------
    save_dir : str
        The directory where NetCDF files will be saved.
    """
    def __init__(self,save_dir="raw_data"):
        """
        Initializes the GFSDataDownloader class.

        Parameters:
        -----------
        save_dir : str, optional
            Directory where downloaded NetCDF files will be stored (default is "raw_data").

        Actions:
        --------
        - Creates the specified directory if it does not exist.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok= True)
        
    def construct_url(self, date, forecast_hour, forecast_step):
        """
        Constructs the URL for the NOAA GFS dataset.
    
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
            - `grib_url`: The direct GRIB2 file URL for the given parameters.
            - `idx_url`: The corresponding .idx file URL.
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
    
    def get_optimal_forecast_hours(self, utc_offset):
        """
        Dynamically selects the best GFS forecast start times for a given UTC offset.
        Ensures that the first forecast hour is >= 12:00 local time.
    
        Parameters:
        -----------
        utc_offset : int
            The local time zone offset from UTC (e.g., -5 for Colombia, +6 for Bangladesh).
    
        Returns:
        --------
        list
            A list of four optimized UTC forecast start times.
        """
        # Standard GFS Forecast Times in UTC
        available_forecast_hours = [0, 6, 12, 18]
    
        # Berechne die lokalen Zeiten für jede GFS-Vorhersagezeit
        local_times = [(hour + utc_offset) % 24 for hour in available_forecast_hours]
    
        # Finde die kleinste UTC-Zeit, die in der lokalen Zeitzone >= 12:00 Uhr ist
        best_start_index = next(i for i in range(len(local_times)) if local_times[i] >= 12)
    
        # Wähle 4 aufeinanderfolgende Forecast-Zeiten ausgehend von diesem Startpunkt
        optimal_forecast_hours = [available_forecast_hours[(best_start_index + i) % 4] for i in range(4)]
    
        return optimal_forecast_hours
    
    def download_for_period(self, start_date, end_date, lat_range, lon_range, UTC_OFFSET):
        """
        Downloads DSWRF data for a specified period and saves it as NetCDF files.
        The forecast hours are dynamically adjusted for the given UTC offset to always start at >= 12:00 local time.
        
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
        UTC_OFFSET : int
            Time zone difference from UTC (e.g., -5 for Colombia).
        """
        date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")
    
        # Berechne die besten Forecast-Stunden für die angegebene Zeitzone
        optimal_forecast_hours = self.get_optimal_forecast_hours(UTC_OFFSET)
        print(f"🌍 Optimized forecast hours (UTC) for UTC{UTC_OFFSET}: {optimal_forecast_hours}")
    
        while date <= end_date:
            start_time_utc = date.strftime("%Y%m%d")
    
            for forecast_hour_utc in optimal_forecast_hours:
                # Berechne die lokale Forecast-Zeit
                local_forecast_hour = (forecast_hour_utc + UTC_OFFSET) % 24
                local_forecast_date = date if local_forecast_hour >= 12 else date - timedelta(days=1)
    
                # Falls forecast_hour_utc == 24, setze es auf 0 und gehe zum nächsten Tag
                if forecast_hour_utc == 24:
                    forecast_hour_utc = 0
                    date += timedelta(days=1)
    
                local_start_datetime = datetime(date.year, date.month, date.day, forecast_hour_utc) - timedelta(hours=UTC_OFFSET)
                local_start_time_str = local_start_datetime.strftime("%Y%m%d_%H00")
    
                for hour in range(1, 25):  # Modify range for more forecast hours
                    forecast_hour_difference = hour  # Forecast step in hours
    
                    # Berechnung der gültigen Vorhersagezeit in lokaler Zeit
                    valid_datetime_local = local_forecast_date + timedelta(hours=local_forecast_hour + forecast_hour_difference)
                    valid_date_str_local = valid_datetime_local.strftime("%Y-%m-%d_%H00")
    
                    grib_url, idx_url = self.construct_url(start_time_utc, forecast_hour_utc, hour)
                    print(f"📥 Downloading DSWRF Data: {grib_url}")
    
                    ds = self.download_and_extract(grib_url, idx_url, hour, lat_range, lon_range)
    
                    if ds is not None:
                        dswrf_interval = self.get_dswrf_interval(hour)
                        forecast_date_str = local_forecast_date.strftime("%Y-%m-%d")
    
                        print(f"✅ Adjusted Forecast Time (Local Time UTC{UTC_OFFSET}): {valid_date_str_local}")
    
                        # 🎯 Aktualisierter Dateiname mit korrekter lokaler Startzeit
                        nc_filename = f"{self.save_dir}/{forecast_date_str}_{local_forecast_hour:02d}_{forecast_hour_difference:02d}_DSWRF{dswrf_interval}_{local_start_time_str}_{forecast_hour_utc:02d}00_{valid_date_str_local}.nc"
                        ds.to_netcdf(nc_filename)
                        print(f"💾 Saved as: {nc_filename}")
    
            date += timedelta(days=1)



   









































