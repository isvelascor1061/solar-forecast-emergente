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
        horizon_file: str | None = None
    ):
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

    def load_hgt(self, hgt_file):
        """Load HGT file and extract elevation data for the target area."""
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
        Calculate horizon elevation profile from HGT data.
        Returns: azimuths (degrees), elevations (degrees)
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
        horizon_file='_3_Data_preparation_for_LSTM/Preparation_data/Elevation_data/Medellin.tif'  # Set path to your HGT file here
    )
    gen.generate(
        output_dir='_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI',
        output_filename='CSI_GHI_grid25_avg_with_horizon_and_enhancement2.nc'
    )
