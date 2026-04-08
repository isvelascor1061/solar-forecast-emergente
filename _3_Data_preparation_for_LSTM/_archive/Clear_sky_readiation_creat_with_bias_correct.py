#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 10:23:33 2025

@author: leonardmerl

Created on Thu Jun 19 10:23:33 2025

@author: leonardmerl

This script calculates the clear-sky Global Horizontal Irradiance (GHI) over a specified region and time period.
The clear-sky GHI is an estimate of the solar radiation that would be received under clear-sky conditions.
The script uses a model (default is 'ineichen') to estimate solar radiation under ideal conditions.
It also includes options to apply corrections based on the solar zenith angle (for low sun elevation) and 
to mask out areas where the sun is obstructed by the horizon (using elevation data).
The output is a NetCDF file containing the clear-sky GHI values for each time step.

The main components of this script are:
1. **Clear-Sky GHI Calculation** using a chosen model (e.g., 'ineichen').
2. **Zenith Angle Correction** for low sun elevation.
3. **Horizon Masking** using elevation data to exclude areas where the sun is blocked by the terrain.
4. **NetCDF Output** that stores the calculated GHI values for future analysis.

"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import pvlib
from pvlib.clearsky import lookup_linke_turbidity
from pvlib.atmosphere import get_relative_airmass, get_absolute_airmass
import rasterio
from pvlib.tools import cosd

class ClearSkyGridAverager:
    def __init__(
        self,
        lat_center: float,
        lon_center: float,
        start_date: str,
        end_date: str,
        tz: str = "UTC",
        altitude: float = 0,
        model: str = "ineichen",
        linke_turbidity=None,
        seed: int | None = None,
        horizon_file: str | None = None,
        zenith_corr: bool = False,     # Zenith angle correction activated?
        zenith_alpha: float = 0.0025,  # Slope [%/°], 0.25% per degree.
        zenith_threshold: float = 65,  # Threshold for applying correction
   ):
        """
    Class to calculate clear-sky Global Horizontal Irradiance (GHI) for a given location and time range.
    Uses a model (ineichen or others) to compute clear-sky solar radiation under ideal clear-sky conditions,
    with optional zenith angle correction and horizon masking (elevation data).
    
    Parameters:
        lat_center (float): Latitude of the center point of the region to calculate for.
        lon_center (float): Longitude of the center point of the region to calculate for.
        start_date (str): Start date for the calculation period (e.g., '2021-04-01 00:00').
        end_date (str): End date for the calculation period (e.g., '2025-06-01 00:00').
        tz (str): Timezone for the location (default is 'UTC').
        altitude (float): Altitude of the location in meters (default is 0).
        model (str): The model used for the clear-sky calculations (default is 'ineichen').
        linke_turbidity (float or None): Linke turbidity value for the 'ineichen' model (optional).
        seed (int or None): Optional random seed for reproducibility.
        horizon_file (str or None): Path to a file containing elevation (horizon) data (optional).
        zenith_corr (bool): Whether to apply a correction for the solar zenith angle (default is False).
        zenith_alpha (float): Correction factor for zenith angle in percent per degree (default is 0.0025).
        zenith_threshold (float): Zenith angle threshold above which the correction applies (default is 65°).
    """
        self.lat_center = lat_center
        self.lon_center = lon_center
        self.start_date = start_date
        self.end_date = end_date
        self.tz = tz
        self.altitude = altitude
        self.model = model
        self.linke_turbidity = linke_turbidity
        self.horizon_file = horizon_file
        if seed is not None:
            np.random.seed(seed)
        half = 0.125
        self.lat_min = lat_center - half
        self.lat_max = lat_center + half
        self.lon_min = lon_center - half
        self.lon_max = lon_center + half
        self.zenith_corr = zenith_corr
        self.zenith_alpha = zenith_alpha
        self.zenith_threshold = zenith_threshold

    def load_hgt(self, hgt_file):
        """
       Load the HGT (elevation) file and extract elevation data for the target area.

       :param hgt_file: Path to the HGT file (elevation data).
       :return: The extracted elevation data.
       """
        with rasterio.open(hgt_file) as src:
            # Check if HGT bounds cover our area of interest
            if (src.bounds.left > self.lon_max or 
                src.bounds.right < self.lon_min or
                src.bounds.bottom > self.lat_max or 
                src.bounds.top < self.lat_min):
                raise ValueError("HGT file does not cover the target area")
            
            window = src.window(self.lon_min, self.lat_min, 
                               self.lon_max, self.lat_max)
            hgt_data = src.read(1, window=window, boundless=True)
        return hgt_data

    def calculate_horizon_profile(self, hgt_data, resolution=5):
        """
        Calculate the horizon profile (elevation) based on the HGT (elevation) data.
        
        This function generates the azimuthal and corresponding elevation angles for a given area.

        :param hgt_data: Elevation data from the HGT file.
        :param resolution: Resolution for azimuths in degrees (default is 5°).
        :return: Arrays of azimuths and corresponding elevations.
        """
        # Generate azimuth bins (0-360 degrees)
        azimuths = np.arange(0, 360, resolution)
        elevations = np.zeros_like(azimuths, dtype=float)
        
        # Assume HGT data is a square around the center point
        center_row = hgt_data.shape[0] // 2
        center_col = hgt_data.shape[1] // 2
        center_alt = hgt_data[center_row, center_col]
        
        # Calculate elevation angles for each azimuth
        for i, az in enumerate(azimuths):
            # Get elevation profile along this azimuth
            max_distance = hgt_data.shape[1] // 2
            distances = np.arange(1, max_distance)
            # Slice elevation data to match distances length
            delta_elev = hgt_data[center_row, 1:max_distance] - center_alt
            tan_elev = delta_elev / (distances * 90)  # Approximate angular distance
            elevations[i] = np.degrees(np.arctan(np.max(tan_elev)))
    
        return azimuths, elevations

    def generate(self, output_dir: str, output_filename: str):
        
        """
       Generate the clear-sky GHI data for the given time period, location, and model.
       The results are saved as a NetCDF file.

       :param output_dir: Directory where the output file will be saved.
       :param output_filename: The name of the output file.
       """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate time index (fixed to use 'min' instead of deprecated 'T')
        time_index = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq="min",  # Updated from 'T' to 'min'
            tz=self.tz,
            inclusive="both"
        )
        group_labels = time_index.floor('h') + pd.Timedelta(hours=1)
        unique_hours = pd.DatetimeIndex(group_labels.unique())
        # Load and calculate horizon profile if available
        horizon_azimuths = None
        horizon_elevations = None
        if self.horizon_file:
            try:
                hgt_data = self.load_hgt(self.horizon_file)
                horizon_azimuths, horizon_elevations = self.calculate_horizon_profile(hgt_data)
            except Exception as e:
                print(f"Error loading horizon file: {e}")
                print("Proceeding without horizon correction")
        
        lat_c, lon_c = self.lat_center, self.lon_center
        lmin, lmax = self.lat_min, self.lat_max
        omin, omax = self.lon_min, self.lon_max
        fixed = [
            (lmin, omin), (lmin, omax), (lmax, omin), (lmax, omax),
            (lat_c, omin), (lat_c, omax), (lmin, lon_c), (lmax, lon_c),
            (lat_c, lon_c)
        ]
        quads = [
            (lat_c, lmax, omin, lon_c),
            (lat_c, lmax, lon_c, omax),
            (lmin, lat_c, omin, lon_c),
            (lmin, lat_c, lon_c, omax)
        ]
        random_pts = []
        for lat_lo, lat_hi, lon_lo, lon_hi in quads:
            pts = np.random.rand(4,2)
            pts[:,0] = pts[:,0] * (lat_hi - lat_lo) + lat_lo
            pts[:,1] = pts[:,1] * (lon_hi - lon_lo) + lon_lo
            random_pts.extend([tuple(pt) for pt in pts])
        samples = fixed + random_pts  # 25 points
        
        horizon_profile = self.load_hgt(self.horizon_file)
        horizon_profile = self.calculate_horizon_profile(horizon_profile)

        # Initialize matrix for clear sky values
        cs_matrix = np.zeros((len(samples), len(time_index)))
        
        for idx, (lat, lon) in enumerate(samples):
            loc = pvlib.location.Location(
                latitude=lat, longitude=lon,
                tz=self.tz, altitude=self.altitude
            )
            
            # Get solar position for all times
            solpos = loc.get_solarposition(time_index)
            
            # Calculate airmass using solar zenith angle
            airmass_relative = get_relative_airmass(solpos['apparent_zenith'])
            airmass_absolute = get_absolute_airmass(airmass_relative)
            
            # Get clearsky data
            if self.model == 'ineichen':
                lt = (self.linke_turbidity if self.linke_turbidity is not None 
                      else lookup_linke_turbidity(time_index, lat, lon))
                cs = loc.get_clearsky(time_index, 
                                      model='ineichen', 
                                    linke_turbidity=lt,
                                    airmass_absolute=airmass_absolute, 
                                    perez_enhancement=True)
            else:
                cs = loc.get_clearsky(time_index, model=self.model)
            
            # Apply horizon mask if available
            if horizon_azimuths is not None:
                # Interpolate horizon elevation for each timestep's azimuth
                from scipy.interpolate import interp1d
                elev_interp = interp1d(horizon_azimuths, horizon_elevations,
                                      bounds_error=False, fill_value=0)
                horizon_elev = elev_interp(solpos['azimuth'])
                
                # Mask when sun elevation < horizon elevation
                sun_elevation = solpos['apparent_elevation']
                mask = sun_elevation > horizon_elev
                cs['ghi'] = cs['ghi'].where(mask, 0)
            if self.zenith_corr:
                zenith = solpos['apparent_zenith']
                # Faktor   f = 1 + α · max(0, zenith - θ0)
                factor = 1 + self.zenith_alpha * (zenith - self.zenith_threshold).clip(lower=0)
                cs['ghi'] = cs['ghi'] * factor
            
            cs_matrix[idx, :] = cs['ghi'].values

        df = pd.DataFrame(cs_matrix.T, index=time_index)

        hourly = df.groupby(group_labels).mean()
        avg = hourly.mean(axis=1)

        obs = unique_hours.tz_localize(None).to_numpy()
        data = avg.values[:, None, None, None]
        ds = xr.Dataset(
            {'clear_sky_ghi': (('observation_time','surface','lat','lon'), data)},
            coords={
                'observation_time': obs,
                'surface': [0],
                'lat': [self.lat_center],
                'lon': [self.lon_center]
            }
        )

        ds.to_netcdf(os.path.join(output_dir, output_filename))
        print(f"Clear-sky grid average saved to {output_dir}/{output_filename}")

if __name__ == '__main__':
    gen = ClearSkyGridAverager(
        lat_center=6.25, lon_center=-75.5,
        start_date='2021-04-01 00:00', end_date='2025-06-01 00:00',
        tz='America/Bogota', altitude=1582,
        model='ineichen',
        linke_turbidity=None,
        seed=42,
        horizon_file='_3_Data_preparation_for_LSTM/Preparation_data/Elevation_data/Medellin.tif',  # Set path to your HGT file here
        zenith_corr=True,      # acitvates bias correction
        zenith_alpha=0.003,   # degree value of increse 0.3° 
        zenith_threshold=60    #
    )
    gen.generate(
        output_dir='_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI',
        output_filename='CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct2.nc'
    )
    
    
    """
    Important -> How to execute 
    1. https://portal.opentopography.org/apidocs/#/Public/getGlobalDem go to this website and download a hight profile for the desired area
    2. Input start_date and end_date -> should be the Same range as 
    3. select time zone (america/Bogota for colombia)
    4. Select altitude (i used the exact one of the Siata Station)
    5. if linke turbidity = None -> gets linke turbidity from a lookup table which is more accurate
    6. Seed is fot the random points in the selected tile
    7.horizon file is the directory to the file you downloaded
    8. zenith_corr = True enables a bias correction 
    9. zenith alpha -> correctioin factor -> the higher this factor, the higher the increase at high zenith angles
    10.Input lat and lon center of the desired tile -> corresponding to the one with the GFS data 
    11. zenith_threshol -> the angle at whick we start a correction (model usually underestimates at high angles > 65)
    -> Execute (keep in mind that this can take some time because the code is calculating a timeseries for 
                multiple points inside the tile and outputs the mean)
    """
    
    
    
    
    
    
    
