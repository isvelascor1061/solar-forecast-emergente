#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel GFS radiation downloader (DSWRF, DLWRF …), class-based.

• Each UTC launch (0100, 0700, 1200 …) is saved in its own sub-folder
  inside `save_dir`.
• spawn ProcessPool + max_tasks_per_child=8  → stable RAM, no seg-fault pool kills
• Worker retries network fetches up to 3 times before emitting a ✖-string.

Author: Emergente · Jun 2025
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import uuid, requests, xarray as xr, numpy as np, time, random, gc
from concurrent.futures import ThreadPoolExecutor as Pool, as_completed
    # Every worker reuses the same semaphore, so we manage concurrent access to shared resources

# NETCDF_LOCK is used to ensure that only one worker writes to a NetCDF file at a time
NETCDF_LOCK = None  

def _init_worker(shared_lock):  
    """This function initializes each worker to have access to the shared NetCDF lock."""
    global NETCDF_LOCK
    NETCDF_LOCK = shared_lock    

def _count_tasks(start_date, end_date, launch_hours_utc, fcst_start, fcst_end):
    """
    This helper function calculates the total number of tasks (files) that need to be processed.
    It takes into account the date range, UTC launch times, and forecast hours.
    """
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt   = datetime.strptime(end_date, "%Y%m%d")
    n_days   = (end_dt - start_dt).days + 1          # Number of days (including the end day)
    n_launch = len(launch_hours_utc)                  # Number of UTC launch times (e.g., 0100, 0700)
    n_fcst   = fcst_end - fcst_start + 1              # Number of forecast hours (e.g., from 12 to 12)
    return n_days * n_launch * n_fcst  # Total tasks = days * launches * forecast hours

# ------------------------------------------------------------------ #
class GFSDataDownloader:
    """
    A class to download GFS (Global Forecast System) radiation data (e.g., DSWRF, DLWRF) in parallel.
    This class uses multiprocessing for parallel downloads to speed up the process and reduce memory usage.
    """
    def __init__(self, save_dir: str, variable: str, max_workers: int = 4):
        """
        Initializes the downloader with parameters:
        - save_dir: Directory where the downloaded data will be stored.
        - variable: The variable (e.g., DSWRF) to download.
        - max_workers: Number of parallel workers to use (default is 4).
        """
        self.save_dir    = Path(save_dir).expanduser().resolve()  # The base directory where data will be saved
        self.var         = variable.upper()                       # The GFS variable (e.g., DSWRF, DLWRF)
        self.max_workers = max_workers                            # Maximum number of parallel workers

    # -------------- Helper Methods --------------------------------------- #
    @staticmethod
    def _gfs_urls(yyyymmdd: str, cycle: int, step: int):
        """
        This method generates the URLs for downloading the GRIB and IDX files based on the date, cycle, and forecast step.
        The URLs are used to fetch the data from the NOAA server.
        """
        root = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
        key  = f"gfs.{yyyymmdd}/{cycle:02d}/atmos/gfs.t{cycle:02d}z.pgrb2.0p25.f{step:03d}"
        return f"{root}/{key}", f"{root}/{key}.idx"  # GRIB file URL and IDX file URL

    def _grib_key(self, step: int):
        """
        Generates a specific GRIB variable key for each forecast step.
        The key determines which variable to download based on the forecast hour.
        """
        v = self.var
        match step:
            case 1 | 2 | 3 | 4 | 5:           return f"{v}:surface:0-{step} hour ave fcst"  # For steps 1-5
            case 6:                           return f"{v}:surface:0-6 hour ave fcst"      # For step 6
            case 7 | 8 | 9 | 10 | 11:         return f"{v}:surface:6-{step} hour ave fcst"  # For steps 7-11
            case 12:                          return f"{v}:surface:6-12 hour ave fcst"     # For step 12
            case 13 | 14 | 15 | 16 | 17:      return f"{v}:surface:12-{step} hour ave fcst" # For steps 13-17
            case 18:                          return f"{v}:surface:12-18 hour ave fcst"     # For step 18
            case 19 | 20 | 21 | 22 | 23 | 24: return f"{v}:surface:18-{step} hour ave fcst" # For steps 19-24
            case _:   raise ValueError(step)

    @staticmethod
    def _interval(hour: int):
        """
        This method converts the forecast hour into an interval value for processing the forecast data.
        It adjusts the hour according to the forecast step intervals.
        """
        if hour in (6, 12, 18, 24):                  return 6
        if hour in (1, 2, 3, 4, 5):                  return hour
        if hour in (7, 8, 9, 10, 11):                return hour - 6
        if hour in (13, 14, 15, 16, 17):             return hour - 12
        if hour in (19, 20, 21, 22, 23, 24):         return hour - 18
        raise ValueError(hour)

    # -------------- Worker Process Method (Static) -------------------------------- #
    @staticmethod
    def _worker(job: dict) -> str:
         """
         The worker method processes the download for a specific forecast step:
         - Fetches IDX and GRIB files
         - Extracts the variable data
         - Subsets it based on latitude and longitude ranges
         - Saves the data as a NetCDF file
         """
         from pathlib import Path
         import requests, xarray as xr, numpy as np
         var      = job["var"]               # Variable to download (e.g., DSWRF)

         save_dir = Path(job["save_dir"])    # Directory to save data
         session  = requests.Session()

        # ---- Fetch IDX file with retry logic ----------------------------------
         idx_lines = None
         for att in range(5):  # Retry up to 5 times
            try:
                r = session.get(job["idx_url"], timeout=30)
                r.raise_for_status()
                idx_lines = r.text.splitlines()  # Parse IDX file content
                break
            except requests.exceptions.RequestException as e:
                if att == 2:
                    return f"✖ IDX fail {job['utc_launch']} {job['fcst_hr']}h: {e}"
                time.sleep(1.5 * (att + 1) + random.random())  # Backoff

        # Find byte-range for the selected variable from the IDX file
         span = None
         for i, ln in enumerate(idx_lines):
            if job["grib_key"] in ln:  # Find the corresponding GRIB variable key in IDX
                start = int(ln.split(":")[1])
                end   = int(idx_lines[i+1].split(":")[1]) if i+1 < len(idx_lines) else None
                span  = (start, end); break
         if not span:
            return f"✖ key not in IDX {job['utc_launch']} {job['fcst_hr']}h"

        # ---- Fetch GRIB file with retry logic ---------------------------------
         hdr = {"Range": f"bytes={span[0]}-{span[1] if span[1] else ''}"}
         blob = None
         for att in range(3):  # Retry up to 3 times for the GRIB file
            try:
                r = session.get(job["grib_url"], headers=hdr, timeout=60)
                r.raise_for_status()
                blob = r.content  # GRIB file content
                break
            except requests.exceptions.RequestException as e:
                if att == 2:
                    return f"✖ GRIB fail {job['utc_launch']} {job['fcst_hr']}h: {e}"
                time.sleep(1.5 * (att + 1) + random.random())  # Backoff

         tmp = save_dir / f"tmp_{uuid.uuid4().hex}.grib2"  # Temporary file for the GRIB data
         tmp.write_bytes(blob)

         try:
            ds = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""})  # Read GRIB data into xarray
            varname = next(n for n in ds.data_vars if var in n.upper())  # Find the variable in the GRIB data
            da = ds[varname]  # Extract the variable data

            # Subset the data based on the latitude and longitude ranges
            lat_r, lon_r = job["lat_range"], job["lon_range"]
            lon_r = [lon + 360 if lon < 0 else lon for lon in lon_r]  # Adjust longitudes
            if len(lat_r) == len(lon_r) == 1:
                da = da.sel(latitude=lat_r[0], longitude=lon_r[0], method="nearest")
            else:
                da = da.sel(latitude=slice(lat_r[1], lat_r[0]),
                            longitude=slice(lon_r[0], lon_r[1]))

            # Time zone shift for the dataset
            off = np.timedelta64(job["UTC_OFFSET"], "h")
            for tc in ("time", "valid_time"):
                if tc in da.coords:
                    da[tc] += off

            # Construct the filename and save the data as NetCDF
            loc_start = job["loc_start_dt"]
            fcst_hr = job["fcst_hr"]
            valid_dt = loc_start + timedelta(hours=fcst_hr)
            fname = (f"{loc_start:%Y-%m-%d}_{loc_start.hour:02d}_{fcst_hr:02d}_"
                     f"{var}{job['interval']}_{valid_dt:%Y-%m-%d_%H00}_"
                     f"UTCSTART:{job['utc_launch']}.nc")

            out = save_dir / fname

            # Save the data as NetCDF
            with NETCDF_LOCK:
                da.to_dataset(name=varname).to_netcdf(out, encoding={varname: {"dtype": "float32", "zlib": True, "complevel": 1, "shuffle": True}})

         except Exception as e:
            return f"✖ Proc error {job['utc_launch']} {job['fcst_hr']}h: {e}"
         finally:
            ds.close() if 'ds' in locals() else None  # Ensure dataset is closed
            tmp.unlink(missing_ok=True)  # Remove temporary GRIB file

         return f"✅ {out.name}"  # Return the file name upon success
    
    def _task_iter(self, start_date, end_date, lat_range, lon_range, utc_offset, launch_hours_utc, fcst_start, fcst_end):
        """
        Generates job dictionaries on demand, minimizing RAM usage.
        This method yields job configurations for downloading data based on the forecast start and end dates.
        """
        cur = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        while cur <= end:
            for utc_hr in launch_hours_utc:
                utc_launch = datetime.combine(cur, datetime.min.time()) + timedelta(hours=utc_hr)
                utc_lbl = utc_launch.strftime("%Y-%m-%d_%H00")  # For logging

                # Local start time adjusted by UTC offset
                loc_start = utc_launch + timedelta(hours=utc_offset)
                local_lbl = loc_start.strftime("%H%M")

                # Destination directory based on local time
                launch_dir = f"{self.save_dir}{local_lbl}"

                for fhr in range(fcst_start, fcst_end + 1):
                    g_url, i_url = self._gfs_urls(utc_launch.strftime("%Y%m%d"), utc_launch.hour, fhr)

                    yield {
                        "grib_url": g_url,
                        "idx_url": i_url,
                        "fcst_hr": fhr,
                        "lat_range": lat_range,
                        "lon_range": lon_range,
                        "UTC_OFFSET": utc_offset,
                        "loc_start_dt": loc_start,  # Local start time as datetime
                        "utc_launch": utc_lbl,  # For filenames/logging
                        "save_dir": launch_dir,  # Directory for saving files
                        "grib_key": self._grib_key(fhr),  # Key for variable in the GRIB data
                        "interval": self._interval(fhr),  # Forecast time interval
                        "var": self.var,  # Variable to download (e.g., DSWRF)
                    }
            cur += timedelta(days=1)

    # -------------- public ----------------------------------------- #
    def download_for_period(self, start_date: str, end_date: str, lat_range: list[float], lon_range: list[float], UTC_OFFSET: int, launch_times_utc: list[int], start_fcst: int, end_fcst: int):
        """
        Public method to download data for the specified period, with parallel processing.
        This method orchestrates the entire downloading process, utilizing multiprocessing.
        """
        t0 = datetime.now()   # Start time for duration tracking
        print("Tasks are streamed out of the generator")

        total = _count_tasks(start_date, end_date, launch_times_utc, start_fcst, end_fcst)
        print(f"Starting the Download of {total} files")

        ctx = mp.get_context("spawn")
        lock = ctx.Lock()

        generator = self._task_iter(start_date, end_date, lat_range, lon_range, UTC_OFFSET, launch_times_utc, start_fcst, end_fcst)

        with ctx.Pool(processes=self.max_workers, maxtasksperchild=4, initializer=_init_worker, initargs=(lock,)) as pool:
            for i, res in enumerate(pool.imap_unordered(self._worker, generator, chunksize=1), 1):
                print(f"[{i}/{total}] {res}")

        print("Done – Duration:", datetime.now() - t0)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    dl = GFSDataDownloader(
        save_dir="_1_data_acquisition/_01_raw_rad_data/dswrf/raw_dswrf_",   # base dir for saving data
        variable="DSWRF",  # Variable to download (e.g., DSWRF)
        max_workers=8,  # Number of workers for parallel processing
    )

    dl.download_for_period(
        start_date="20210402",  # Start date for the data download
        end_date="20210402",    # End date for the data download or just one tile in the raster
        lat_range=[6.25],  # Latitude range for the data or just one tile in the raster
        lon_range=[-75.5],  # Longitude range for the data
        UTC_OFFSET=-5,  # UTC offset to adjust the local time
        launch_times_utc=[6],  # The UTC launch times to consider (e.g., 0600)
        start_fcst=1,  # Start forecast hour
        end_fcst=1,    # End forecast hour
    )
    
    
    """
    Important: 
        this was the first version of the download code for firstly just radiation 
        -> This scirpt only includes the time adjustment but doesnt rename them to launch_time and observation time
        because this was done in later scripts
        
        This script doesnt download direct forecast, but the Intervall ones -> different postprocessing 
        To asure a stable and consistent download do the following -> 
        1. As save dir -> absolute path including the folders with the different launch times 
        but without the final number so e.g.: _1_data_acquisition/_01_raw_rad_data/dswrf/raw_dswrf_
        2. select lat an lon ranges -> script then saves the data in a matrix corresponding to how many tiles are selected
        3. select the launch times utc -> 6-> 0100, 12-> 0700 -> 18 -> 1300 24-> 1900(code automatically chooses 00 of the following day )
        4. select start and end of fcst hour (1,24) code can only handle 24 forecast hours 
        5. Seclect configuraition, save, close spyder and start trough terminal
        
        Theres 1 saved file for each launch hour and fcst hours so 96 files for every day 
        saved in _1_data_acquisition/_01_raw_rad_data
        dswrf for short wave radiation 
        dlwrf for long wave radiatin 
        but script can be used to download any kind of intervall parameter in the idx files
    """








