import os
import uuid
import requests
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import threading

netcdf_lock = threading.Lock()

class GFSDataDownloader:
    """
    A parallel downloader and processor for GFS (Global Forecast System) radiation data.
    Downloads DSWRF (Downward Shortwave Radiation Flux) data from NOAA's GFS forecasts,
    processes it, and saves as NetCDF files.
    
    Features:
    - Parallel downloads using ThreadPoolExecutor
    - Automatic variable name detection
    - Timezone adjustment
    - Spatial subsetting
    - Progress tracking
    """
    
    def __init__(self, save_dir="_1_data_acquisition/raw_data", max_workers=4):
        """
        Initialize the downloader.
        
        Args:
            save_dir (str): Directory to save downloaded files
            max_workers (int): Maximum number of parallel threads
        """
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.max_workers = max_workers
        self.session = requests.Session()  # Reuse session for better performance

    def _get_dswrf_variable_name(self, timestep):
        """
        Get the exact variable name for DSWRF based on forecast timestep.
        
        GFS uses different variable names for different forecast intervals.
        This maps timesteps to the correct variable name in the GRIB files.
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
            case 19 | 20 | 21 | 22 | 23 | 24:
                return f"DSWRF:surface:18-{timestep} hour ave fcst"
            case _:
                raise ValueError(f"Invalid timestep: {timestep}")

    def _get_dswrf_interval(self, hour):
        """Calculate the averaging interval in hours for a given forecast hour"""
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
            raise ValueError(f"Invalid forecast hour: {hour}")

    def _construct_url(self, date_str, forecast_hour, forecast_step):
        """Build the URL for GFS GRIB data and index files"""
        base_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
        grib_url = (
            f"{base_url}/gfs.{date_str}/{forecast_hour:02d}/atmos/"
            f"gfs.t{forecast_hour:02d}z.pgrb2.0p25.f{forecast_step:03d}"
        )
        return grib_url, grib_url + ".idx"

    def _find_dswrf_in_idx(self, idx_url, timestep):
        """
        Locate DSWRF data in the index file.
        Returns byte range (start, end) if found, None otherwise.
        """
        dswrf_name = self._get_dswrf_variable_name(timestep)
        resp = self.session.get(idx_url)
        if resp.status_code != 200:
            return None

        for i, line in enumerate(resp.text.splitlines()):
            if dswrf_name in line:
                try:
                    parts = line.split(":")
                    start_byte = int(parts[1])
                    end_byte = int(resp.text.splitlines()[i+1].split(":")[1]) if (i+1) < len(resp.text.splitlines()) else None
                    return (start_byte, end_byte)
                except (IndexError, ValueError):
                    continue
        return None

    def _find_dswrf_variable(self, ds):
        """Identify the DSWRF variable in an xarray Dataset"""
        possible_names = [n for n in ds.data_vars if "DSWRF" in n or "sdswrf" in n]
        return possible_names[0] if possible_names else None

    def _download_and_process(self, job):
        """
        Combined download and processing task for a single forecast timestep.
        Designed to run in parallel threads.
        """
        # Download phase
        grib_url, idx_url = job["grib_url"], job["idx_url"]
        timestep = job["timestep"]
        
        # Locate data in index file
        offsets = self._find_dswrf_in_idx(idx_url, timestep)
        if not offsets:
            return f"❌ DSWRF not found in {idx_url}"

        # Download only the relevant portion of the GRIB file
        headers = {"Range": f"bytes={offsets[0]}-{offsets[1] if offsets[1] else ''}"}
        resp = self.session.get(grib_url, headers=headers)
        if resp.status_code not in [200, 206]:
            return f"❌ Download failed for {grib_url}"

        # Save temporary file
        temp_path = os.path.join(self.save_dir, f"temp_{uuid.uuid4().hex}.grib2")
        with open(temp_path, "wb") as f:
            f.write(resp.content)

        # Processing phase
        try:
            ds = xr.open_dataset(temp_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
            varname = self._find_dswrf_variable(ds)
            if not varname:
                raise ValueError("No DSWRF variable found")
            
            # Spatial subsetting
            lat_range, lon_range = job["lat_range"], job["lon_range"]
            adj_lon_range = [lon + 360 if lon < 0 else lon for lon in lon_range]
            
            if len(lat_range) == 1 and len(lon_range) == 1:
                ds_region = ds[varname].sel(
                    latitude=lat_range[0], 
                    longitude=adj_lon_range[0], 
                    method="nearest"
                )
            else:
                ds_region = ds[varname].sel(
                    latitude=slice(lat_range[1], lat_range[0]),
                    longitude=slice(adj_lon_range[0], adj_lon_range[1]))
            
            # Timezone adjustment
            UTC_OFFSET = job["UTC_OFFSET"]
            for time_coord in ["time", "valid_time"]:
                if time_coord in ds_region.coords:
                    ds_region[time_coord] = ds_region[time_coord] + np.timedelta64(UTC_OFFSET, 'h')

            # Generate output filename
            dswrf_interval = self._get_dswrf_interval(timestep)
            local_date, local_hour = job["local_forecast_date"], job["local_forecast_hour"]
            valid_time = local_date + timedelta(hours=local_hour + job["forecast_hour_diff"])
            
            outname = (
                f"{self.save_dir}/{local_date.strftime('%Y-%m-%d')}_{local_hour:02d}_"
                f"{job['forecast_hour_diff']:02d}_DSWRF{dswrf_interval}_"
                f"{valid_time.strftime('%Y-%m-%d_%H00')}_UTCSTART:{job['utc_launch_str']}.nc"
            )

            # Save as NetCDF
            with netcdf_lock:
                ds_region.to_dataset(name=varname).to_netcdf(outname)
            
        except Exception as e:
            return f"❌ Error processing {temp_path}: {str(e)}"
        finally:
            ds.close() if 'ds' in locals() else None
            os.remove(temp_path) if os.path.exists(temp_path) else None

        return f"✅ Saved: {os.path.basename(outname)}"

    def download_for_period(self, start_date, end_date, lat_range, lon_range,
                          UTC_OFFSET, launch_times_utc, start_forecasthour, end_forecasthour):
        """
        Main method to download and process data for a date range.
        
        Args:
            start_date/end_date (str): Dates in YYYYMMDD format
            lat_range/lon_range (list): [min, max] or [single_point]
            UTC_OFFSET (int): Hours to adjust timestamps
            launch_times_utc (list): Model initialization times (UTC)
            start/end_forecasthour (int): Forecast hour range
        """
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        tasks = []
        gc.collect()

        # Generate all download tasks
        while start_dt <= end_dt:
            for utc_hour in launch_times_utc:
                # Handle UTC hour rollover
                if utc_hour in [24, 0]:
                    launch_date = start_dt + timedelta(days=1)
                    actual_hour = 0
                else:
                    launch_date = start_dt
                    actual_hour = utc_hour

                launch_dt = datetime.combine(launch_date, datetime.min.time()) + timedelta(hours=actual_hour)
                utc_str = launch_dt.strftime("%Y-%m-%d_%H00")

                # Convert to local time
                local_hour = (actual_hour + UTC_OFFSET) % 24
                local_date = launch_date if actual_hour >= abs(UTC_OFFSET) else launch_date - timedelta(days=1)

                # Create jobs for each forecast hour
                for hour in range(start_forecasthour, end_forecasthour + 1):
                    grib_url, idx_url = self._construct_url(launch_dt.strftime("%Y%m%d"), launch_dt.hour, hour)
                    tasks.append({
                        "grib_url": grib_url,
                        "idx_url": idx_url,
                        "timestep": hour,
                        "lat_range": lat_range,
                        "lon_range": lon_range,
                        "UTC_OFFSET": UTC_OFFSET,
                        "local_forecast_date": local_date,
                        "local_forecast_hour": local_hour,
                        "forecast_hour_diff": hour,
                        "utc_launch_str": utc_str
                    })
            start_dt += timedelta(days=1)

        # Execute tasks in parallel
        print(f"Starting {len(tasks)} parallel tasks with {self.max_workers} workers...")
        t0 = datetime.now()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._download_and_process, job) for job in tasks]
            for i, future in enumerate(as_completed(futures)):
                print(f"[{i+1}/{len(tasks)}] {future.result()}")

        print(f"Completed! Total duration: {datetime.now() - t0}")


if __name__ == "__main__":
    # Example usage
    downloader = GFSDataDownloader(
        save_dir="_1_data_acquisition/raw_rad_data/raw_dswrf_1900",
        max_workers=6 # Adjust based on your system capabilities
    )

    downloader.download_for_period(
        start_date="20250501",
        end_date="20250531",
        lat_range=[6.25],  # Single point
        lon_range=[-75.5],
        UTC_OFFSET=-5,  # UTC-5 timezone
        launch_times_utc=[00],  # 06Z model run
        start_forecasthour=1,
        end_forecasthour=24
    )