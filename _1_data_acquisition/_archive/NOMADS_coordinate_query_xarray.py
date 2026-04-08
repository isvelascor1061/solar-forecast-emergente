#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:20:12 2025

@author: leonardmerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate Query Creator for NOMADS

This script extracts latitude and longitude coordinates from NOAA NOMADS datasets.
It retrieves grid information and enables coordinate-to-index conversions.
"""

import re
import numpy as np
import xarray as xr
import logging
from datetime import datetime

from utils.math_utils import scientific_to_decimal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CoordinateInfo:
    """
    Extracts and processes latitude and longitude information from NOAA NOMADS datasets.
    
    Attributes
    ----------
    date : datetime
        Forecast date.
    hour : str
        Forecast initialization hour (e.g., '06').
    dataset : xarray.Dataset or None
        Opened NOAA NOMADS dataset, or None if loading fails.
    """
    def __init__(self, date: datetime, hour: str):
        self.date = date
        self.hour = hour
        self.url = f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date.strftime('%Y%m%d')}/gfs_0p25_{hour}z"
        
        try:
            self.dataset = xr.open_dataset(self.url, engine='netcdf4')
            logging.info(f"✅ Successfully opened NOMADS dataset from {self.url}")
        except Exception as e:
            logging.error(f"❌ Failed to open NOMADS dataset from {self.url}: {e}")
            self.dataset = None
            
    
    def _extract_coord_data(self, variable_name: str):
        """
        Extracts metadata about latitude or longitude.

        Parameters
        ----------
        variable_name : str
            'lat' or 'lon' to specify the coordinate type.

        Returns
        -------
        dict
            Contains grads_size, min, max, and resolution of the coordinate grid.
        """
        if self.dataset is None or variable_name not in self.dataset.variables:
            logging.warning(f"⚠️ Variable {variable_name} not found in dataset.")
            return None
        
        values = scientific_to_decimal(self.dataset[variable_name].values).astype(float)
        grads_size = self.dataset.dims.get(variable_name, len(values))

        return {
            'grads_size': float(grads_size),
            'minimum': np.min(values),
            'maximum': np.max(values),
            'resolution': np.abs(np.mean(np.diff(values)))
        }
    
    
    def get_coordinates(self):
        """
        Extracts latitude and longitude details from the NOMADS dataset.

        Returns
        -------
        dict or None
            Dictionary containing grid size, min/max, and resolution for lat/lon, or None if dataset is unavailable.
        """
        if self.dataset is None:
            logging.error("❌ Dataset is not available. Cannot extract coordinates.")
            return None
        
        return {
            "lon": self._extract_coord_data("lon"),
            "lat": self._extract_coord_data("lat")
        }

class CoordIndexConverter:
    """
    Converts latitude and longitude values into grid indices.
    
    Attributes
    ----------
    coords_available : dict
        Dictionary with min/max/resolution for latitude and longitude.
    
    Methods
    -------
    value_input_to_index(inpt, coord_name)
        Converts a value or range of values to a coordinate index.
    value_to_index(coord_name, value)
        Converts a value to a coordinate index.
    """
  
    def __init__(self, coords_available):
        self.coords_available = coords_available

    def value_to_index(self, coord_name, value):
        """
        Converts a coordinate value into its corresponding grid index.

        Parameters
        ----------
        coord_name : str
            The name of the coordinate ('lat' or 'lon').
        value : float
            The value of the coordinate to convert.

        Returns
        -------
        int
            The index of the coordinate value.
        """
        resolution = float(self.coords_available[coord_name]["resolution"])
        min_val = float(self.coords_available[coord_name]["minimum"])
        grads_size = int(self.coords_available[coord_name]["grads_size"])
    
        possibles = [resolution * n + min_val for n in range(grads_size)] #creates a list where the index of a coordinate value correspond to its index in the GFS raster
        
        if coord_name == "lat":
            possibles = possibles[::-1] 
    
        closest = min(possibles, key=lambda x: abs(x - value))  #finds closest value of the coordinate
        return possibles.index(closest)   #returns index of the given coordinate

    def get_coord_slice(self, inpt, coord_name):
        """
        Converts a value or a range of values to a coordinate index and returns a slice object.

        Parameters
        ----------
        inpt : str or float
            The value or range of values to convert. If it is a string, it should be in the format "[min:max]"
            for a range, or just a single value if it is not a range.
        coord_name : str
            The name of the coordinate ('lat' or 'lon').

        Returns
        -------
        slice or int
            A slice object for range inputs or an integer index for single values.
        """
        if isinstance(inpt, str) and inpt.startswith("[") and inpt.endswith("]") and ":" in inpt:
            val_1, val_2 = map(float, re.findall(r"[-+]?[0-9]*\.?[0-9]+", inpt))
            if coord_name == "lon":
                val_1 = val_1 % 360
                val_2 = val_2 % 360
            val_min = self.value_to_index(coord_name, min(val_1, val_2))
            val_max = self.value_to_index(coord_name, max(val_1, val_2))
            return slice(min(val_min, val_max), max(val_min, val_max) + 1)
        else:
            return self.value_to_index(coord_name, float(inpt))


if __name__ == "__main__":
    date = datetime.strptime('2025-03-10', '%Y-%m-%d')
    hour = "06"
    
    coord_info = CoordinateInfo(date, hour)
    coords_available = coord_info.get_coordinates()
    
    if coords_available:
        converter = CoordIndexConverter(coords_available)
        lon_index = converter.get_coord_slice("[-75.25:-75.75]", "lon")
        lat_index = converter.get_coord_slice("[6:6.5]", "lat")
        
        print(f"Longitude Index: {lon_index}")
        print(f"Latitude Index: {lat_index}")
