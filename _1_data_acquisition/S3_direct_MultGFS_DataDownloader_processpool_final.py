
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFS-Downloader (0.25°) — Downloads multiple variables in parallel
and saves a NetCDF file for each forecast hour.

Filename format:
YYYY-MM-DD_locHH_FORECASThr_<Variables>_YYYY-MM-DD_HH00_UTCSTART:YYYY-MM-DD_HH00.nc

This script downloads multiple meteorological variables from NOAA's GFS model. 
It downloads data for a specified date range, for given forecast hours, 
and for multiple launch times (e.g., 00z, 06z, 12z, 18z). 
The data is then processed and stored as NetCDF files for further analysis.


Filename: YYYY-MM-DD_locHH_FORECASThr_<Variables>_YYYY-MM-DD_HH00_UTCSTART:YYYY-MM-DD_HH00.nc

The filename structure follows a specific pattern that ensures each file can be uniquely identified and contains all the relevant information.
Here is a detailed explanation of each component of the filename:

1. **YYYY-MM-DD**:  
   - **Meaning**: The start date of the forecast period.
   - **Example**: `2021-04-01` for April 1st, 2021.
   - **Usage**: This part of the filename helps to quickly identify the date of the forecast. It indicates the day when the forecast for the specified period began.

2. **locHH**:  
   - **Meaning**: The hour and minute of the forecast in local time, adjusted by the UTC offset.
   - **Example**: `0100` for 01:00 local time (if UTC offset = -5 hours).
   - **Usage**: This component provides the start time of the forecast in the local time of the specific location. It allows for tracking the exact start time of the forecast and ensures there are no overlaps in forecast times.

3. **FORECASThr**:  
   - **Meaning**: The forecast hour for the data stored in the file.
   - **Example**: `12` for the 12th forecast hour.
   - **Usage**: This part shows the specific forecast hour for which the data is saved. For example, a file with `12` indicates that the stored data corresponds to the forecast for the 12th hour after the forecast start point.


4. **YYYY-MM-DD_HH00**:  
   - **Meaning**: The valid date and time of the forecast (in UTC format).
   - **Example**: `2021-04-01_12` for April 1st, 2021 at 12:00 UTC.
   - **Usage**: This component indicates the time point when the forecast is valid. It shows the time for which the meteorological forecast is relevant, for example, for the 12th forecast hour.

5. **UTCSTART:YYYY-MM-DD_HH00**:  
   - **Meaning**: The start time of the forecast in UTC format.
   - **Example**: `UTCSTART:2021-04-01_00`.
   - **Usage**: This part of the filename indicates the exact start time of the forecast in UTC format. The start time is crucial for interpreting the data in the correct context.

6. **.nc**:  
   - **Meaning**: The file extension indicates the NetCDF (Network Common Data Form) format.
   - **Usage**: NetCDF is a widely used format for storing scientific data, particularly meteorological and climatological data. This extension shows that the file is a NetCDF file, which can be processed by tools like `xarray` and `netCDF4`.

**Example of a full filename**:
2021-04-01_01_01_MultGFS_2021-04-01_0200_UTCSTART:2021-04-01_0600.nc
"""

import os
os.environ["GRIB_DISABLE_CACHE"] = "1"  # Disable GRIB file caching to avoid re-downloading
from pathlib import Path
from datetime import datetime, timedelta
import uuid, requests, xarray as xr, numpy as np
import concurrent.futures as cf
import multiprocessing as mp
import gc, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, time, random, xarray as xr, numpy as np
from requests.exceptions import RequestException
from requests.exceptions import RequestException, ConnectionError, Timeout
import time, random
import gc, tracemalloc


# --------------------------------------------------------------------------- #
# Global variable to handle NetCDF locking across processes
NETCDF_LOCK = None      # This will be set in the worker process to ensure synchronized access to the NetCDF files.

# Initializes the worker with the lock object for NetCDF file access.
def _init_worker(shared_lock):
    """
   This function initializes the worker process by providing access to a shared lock object. 
   This lock is used to ensure that only one process writes to a NetCDF file at a time.
   
   :param shared_lock: Lock object that will be shared among all worker processes.
   """
    global NETCDF_LOCK
    NETCDF_LOCK = shared_lock
    
# Start memory tracking using tracemalloc to monitor memory usage during the script's execution
tracemalloc.start()


def _slice_geo(arr, lat_rng, lon_rng):
    """Subset a DataArray auf Punkt oder Kachel – liefert neues, kleines Array."""
    lat_name = next((c for c in ("latitude", "lat", "y")  if c in arr.coords), None)
    lon_name = next((c for c in ("longitude", "lon", "x") if c in arr.coords), None)
    if lat_name is None or lon_name is None:
        return arr            # kein slicing möglich

    lon_adj = [l + 360 if l < 0 else l for l in lon_rng]

    # Punkt- oder Kachelmodus
    if len(lat_rng) == 1 and len(lon_rng) == 1:            # Punkt
        return arr.sel({lat_name: lat_rng[0],
                        lon_name: lon_adj[0]}, method="nearest").squeeze(drop=True)

    # Kachel  -------------------------------------------------------------
    def _smart_slice(coord, rng):
        lo, hi = sorted(rng)
        ascend = coord[0] < coord[-1]
        return slice(lo, hi) if ascend else slice(hi, lo)

    return arr.sel({lat_name: _smart_slice(arr[lat_name], lat_rng),
                    lon_name: _smart_slice(arr[lon_name], lon_adj)})





# --------------------------------------------------------------------------- #
# Counts the number of tasks that need to be processed based on the given parameters.
def _count_tasks(start_date, end_date,
                 launch_hours_utc, fcst_start, fcst_end):
    
    """
   This function counts the total number of tasks (forecast data downloads) that need to be processed based on 
   the specified start and end date, forecast hours, and launch hours. Each forecast hour for each launch hour 
   will result in a separate task.

   :param start_date: The start date of the forecast period in 'YYYYMMDD' format.
   :param end_date: The end date of the forecast period in 'YYYYMMDD' format.
   :param launch_hours_utc: List of UTC hours for forecast launches (e.g., [0, 6, 12, 18]).
   :param fcst_start: The starting forecast hour (e.g., 0).
   :param fcst_end: The ending forecast hour (e.g., 12).

   :return: The total number of tasks to be processed.
   """
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt   = datetime.strptime(end_date,   "%Y%m%d")
    n_days   = (end_dt - start_dt).days + 1          # incl. Enddate
    n_launch = len(launch_hours_utc)
    n_fcst   = fcst_end - fcst_start + 1
    return n_days * n_launch * n_fcst

# --------------------------------------------------------------------------- #
# Class to handle the downloading and processing of GFS forecast data
class GFSDownloader:
    def __init__(self, save_dir: str, var_specs: list[tuple[str, str]], max_workers: int = 4):
        """
       Initialize the downloader with the directory to save files, variable specifications, and maximum workers.
       
       :param save_dir: Directory where the downloaded files will be saved (full path).
       :param var_specs: List of variable specifications (each as a tuple of GRIB variable name and output name).
                         Each tuple specifies the GRIB variable and the desired output name for the variable.
       :param max_workers: Maximum number of parallel processes to be used (default is 4).
       """
        self.save_dir = Path(save_dir).expanduser().resolve()  # Resolves user directory to absolute path
        self.max_workers = max_workers  # Maximum parallel download processes
        self.var_specs = var_specs  # Variable specifications for data to be downloaded
        

     # ----------------------------- Helper Functions ------------------------- #    
    @staticmethod
    def _construct_urls(date_str, cycle_hour, fcst_hour):
        """
        Constructs the URL for downloading the GRIB and IDX files from NOAA's GFS server.

        :param date_str: Date string in "YYYYMMDD" format.
        :param cycle_hour: Hour of the forecast cycle (e.g., 00, 06, 12, 18).
        :param fcst_hour: Forecast hour (e.g., 3, 5, 11, etc.).

        :return: A tuple containing the GRIB URL and the IDX URL for the given forecast hour.
        """
        
        base = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
        
        g = f"{base}/gfs.{date_str}/{cycle_hour:02d}/atmos/" \
            f"gfs.t{cycle_hour:02d}z.pgrb2.0p25.f{fcst_hour:03d}"
        return g, g + ".idx"

    def _find_key_in_idx(self, idx_url: str,
                         var_specs: list[tuple[str, str]],
                         fcst_hour: int) -> dict[str, tuple[int, int | None]]:
        
        """
       Finds the byte ranges for each variable in the IDX file.

       :param idx_url: URL of the IDX file.
       :param var_specs: List of variable specifications, which includes the GRIB variable name and output name.
       :param fcst_hour: Forecast hour (e.g., 0, 6, 12, etc.) for which we are retrieving data.

       :return: A dictionary where each key is an output variable name and the value is a tuple of start and end byte ranges.
       """
       
        r = requests.get(idx_url, timeout=20)  # Send HTTP GET request to download the IDX file
        r.raise_for_status()  # Check for errors in the response
        lines = r.text.splitlines()  # Split the response into lines
        wanted, spec_map = {}, {}
        # Map output names to GRIB variable and level
        for idx_name, out_name in var_specs:
            var, level = idx_name.split(":", 1)
            spec_map[out_name] = (var, level)
            
            
        # Process the IDX file to find the required variables
        for i, ln in enumerate(lines):
            parts = ln.split(":")
            if len(parts) < 6 or not parts[5].startswith(f"{fcst_hour} hour fcst"):
                continue

            var, level = parts[3], parts[4]  # Get variable and level from the IDX line
            for out_name, (v_want, l_want) in spec_map.items():
                if var == v_want and level == l_want: # If variable and level match, get byte ranges
                    start = int(parts[1])
                    end   = (int(lines[i + 1].split(":")[1]) - 1
                             if i + 1 < len(lines) else None)
                    wanted[out_name] = (start, end) # Store byte range
                    break
            if len(wanted) == len(var_specs):
                break # Stop if all desired variables are found
        return wanted

    # ------------------------------------------------------------------ #
    def _download_and_process(self, job: dict) -> str:
        """
        Download one forecast step, spatially subset (point or tile),
        shift times to local, and save as NetCDF.
    
        Parameters
        ----------
        job : dict
            All metadata for this forecast hour (urls, fcst_hour, lat/lon range …)
    
        Returns
        -------
        str
            Result message for logging.
        """
        # ---- build launch folder -------------------------------------------------
        label      = job["loc_start_dt"].strftime("%H%M")          # e.g. 0100
        launch_dir = Path(str(self.save_dir) + label)
        launch_dir.mkdir(parents=True, exist_ok=True)
    
        session   = requests.Session()
        grib_url  = job["grib_url"]
        idx_url   = job["idx_url"]
        fcst_hr   = job["fcst_hour"]
    
        # ---- find byte-ranges in IDX --------------------------------------------
        for attempt in range(3):
            try:
                ranges = self._find_key_in_idx(idx_url, self.var_specs, fcst_hr)
                break
            except RequestException as e:
                if attempt == 2:
                    return f"✖ IDX network error ({fcst_hr} h): {e}"
                time.sleep(2.0 * (attempt + 1) + random.random())
    
        if len(ranges) < len(self.var_specs):
            missing = [n for _, n in self.var_specs if n not in ranges]
            return f"✖ missing vars: {', '.join(missing)} ({fcst_hr} h)"
    
        # ---- helpers -------------------------------------------------------------
        def _coord_name(arr, cand):
            """return first coordinate name present in arr or None"""
            return next((c for c in cand if c in arr.coords), None)
    
        def _smart_slice(coord, rng):
            lo, hi = min(rng), max(rng)
            ascend = coord[0] < coord[-1]
            return slice(lo, hi) if ascend else slice(hi, lo)
    
        # ---- loop over variables -------------------------------------------------
        open_ds, tmp_files, pieces = [], [], {}
        launch_label = f"{job['loc_start_dt'].hour:02d}{job['loc_start_dt'].minute:02d}"
    
        for idx_name, out_name in self.var_specs:
            start, end = ranges[out_name]
            headers = {"Range": f"bytes={start}-{end if end else ''}"}
    
            # retry on network errors
            for attempt in range(3):
                try:
                    resp = session.get(grib_url, headers=headers, timeout=30)
                    resp.raise_for_status()
                    break
                except RequestException as e:
                    if attempt == 2:
                        return f"✖ GRIB network error {out_name} ({fcst_hr} h): {e}"
                    time.sleep(2.0 * (attempt + 1) + random.random())
    
            tmp_path = launch_dir / f"tmp_{uuid.uuid4().hex}.grib2"
            tmp_files.append(tmp_path)
            tmp_path.write_bytes(resp.content)
    
            ds = xr.open_dataset(tmp_path, engine="cfgrib",
                                 backend_kwargs={"indexpath": ""})
            if "heightAboveGround" in ds.coords:
                ds = ds.drop_vars("heightAboveGround")
            open_ds.append(ds)
    
            arr = ds[next(iter(ds.data_vars))]
    
            arr = ds[next(iter(ds.data_vars))]
            
            # -------   HIER  ALTEN Slicing-Block entfernen  -------------------------
            arr = _slice_geo(arr, job["lat_range"], job["lon_range"])

    
            # ---- store in dict ---------------------------------------------------
            arr = arr.astype("float32")           # save memory
            arr.name = f"{out_name}_{launch_label}"
            pieces[arr.name] = arr
    
        # ---- derived wind speed --------------------------------------------------
        u_key = f"UGRD_10m_{launch_label}"
        v_key = f"VGRD_10m_{launch_label}"
        if u_key in pieces and v_key in pieces:
            u = pieces.pop(u_key)
            v = pieces.pop(v_key)
            wind = (u ** 2 + v ** 2) ** 0.5
            wind.name = f"Wind_10m_{launch_label}"
            pieces[wind.name] = wind
            
        out_ds = xr.Dataset(pieces).squeeze(drop=True)
        
     

    
        offset = np.timedelta64(job["UTC_OFFSET"], "h")
        if "time" in out_ds.coords:
            out_ds = out_ds.rename({"time": "launch_time"})
            out_ds["launch_time"] += offset
        if "valid_time" in out_ds.coords:
            out_ds = out_ds.rename({"valid_time": "observation_time"})
            out_ds["observation_time"] += offset
    
        # ---- build filename ------------------------------------------------------
        loc_start = job["loc_start_dt"]
        valid_dt  = loc_start + timedelta(hours=fcst_hr)
        fname = (
            f"{loc_start:%Y-%m-%d}_{loc_start.hour:02d}_{fcst_hr:02d}_"
            f"MultGFS_{valid_dt:%Y-%m-%d_%H00}_UTCSTART:{job['utc_launch']}.nc"
        )
        out_path = launch_dir / fname
    
        # ---- save & clean up -----------------------------------------------------
        with NETCDF_LOCK:
            enc = {v: {"dtype": "float32"} for v in out_ds}
            out_ds.to_netcdf(out_path, encoding=enc)
    
        out_ds.close()
        pieces.clear()
        for ds in open_ds:
            ds.close()
        for p in tmp_files:
            p.unlink(missing_ok=True)
        gc.collect()
    
        print("Peak-RAM (MiB):", tracemalloc.get_traced_memory()[1] / 1_048_576,
              flush=True)
        return f"✅ Saved: {fname} | fcst_hr={fcst_hr}"


    
    # --------------------------------------------------------------------------- #
    # Task generator that dynamically yields job dictionaries for each forecast step 
    def _task_iter(self, start_date, end_date,
               lat_range, lon_range, utc_offset,
               launch_hours_utc, fcst_start, fcst_end):
        """
        Generates job dictionaries on-demand for each forecast step within the date range.
    
        :param start_date: The start date of the forecast period
        :param end_date: The end date of the forecast period
        :param lat_range: Latitude range for the data to be extracted
        :param lon_range: Longitude range for the data to be extracted
        :param utc_offset: The UTC offset (e.g., -5 for Colombia)
        :param launch_hours_utc: List of UTC hours for forecast launches (e.g., [0, 6, 12, 18])
        :param fcst_start: Start forecast hour (e.g., 0)
        :param fcst_end: End forecast hour (e.g., 12)
        
        :return: A generator yielding job dictionaries for each forecast step
        """
        cur = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date,   "%Y%m%d")
    
        while cur <= end:
            for hh in launch_hours_utc:
                utc_launch = datetime.combine(cur, datetime.min.time()) + timedelta(hours=hh)
                utc_str    = utc_launch.strftime("%Y-%m-%d_%H00")
                loc_start  = utc_launch + timedelta(hours=utc_offset)
                
                # Create launch folder
                launch_folder = str(self.save_dir) + utc_str
    
                for fhr in range(fcst_start, fcst_end + 1):
                    g_url, i_url = self._construct_urls(utc_launch.strftime("%Y%m%d"),
                                                        utc_launch.hour, fhr)
                    yield {
                        "grib_url": g_url,
                        "idx_url": i_url,
                        "fcst_hour": fhr,
                        "lat_range": lat_range,
                        "lon_range": lon_range,
                        "UTC_OFFSET": utc_offset,
                        "loc_start_dt": loc_start,
                        "utc_launch": utc_str,
                        "save_dir":    str(launch_folder),
                    }
            cur += timedelta(days=1)


  
    # --------------------------- Manager & Pool --------------------------- #
    def download_for_period(self, start_date, end_date,
                            lat_range, lon_range, utc_offset,
                            launch_hours_utc, fcst_start, fcst_end):
        """
       Downloads GFS forecast data for a specified date range, latitude and longitude range, and forecast hours.
       The function utilizes multiple worker processes to parallelize the task.

       :param start_date: The start date of the forecast period in 'YYYYMMDD' format.
       :param end_date: The end date of the forecast period in 'YYYYMMDD' format.
       :param lat_range: The latitude range (list of two values [min_lat, max_lat]) to download data for.
       :param lon_range: The longitude range (list of two values [min_lon, max_lon]) to download data for.
       :param utc_offset: The UTC offset (e.g., -5 for Colombia) to adjust the times for local time zones.
       :param launch_hours_utc: List of UTC hours for forecast launches (e.g., [0, 6, 12, 18]).
       :param fcst_start: The start forecast hour (e.g., 0).
       :param fcst_end: The end forecast hour (e.g., 12).

       :return: None. The method downloads and processes the data, saving it as NetCDF files.
       """

        print("Tasks are being streamed from the generator – memory usage stays low!")
        total_tasks = _count_tasks(start_date, end_date,
                               launch_hours_utc, fcst_start, fcst_end)
        print(f"Starting {total_tasks} tasks")
        t0   = datetime.now()
        ctx  = mp.get_context("spawn")
        lock = ctx.Lock()

        generator = self._task_iter(start_date, end_date,
                                    lat_range, lon_range, utc_offset,
                                    launch_hours_utc, fcst_start, fcst_end)

        with ctx.Pool(processes=self.max_workers, maxtasksperchild=4,
                      initializer=_init_worker, initargs=(lock,)) as pool:

            for i, res in enumerate(pool.imap_unordered(self._download_and_process, generator, chunksize=1), 1):
                print(f"[{i}/{total_tasks}] {res}")                      # reduzierte Ausgabe reicht

        print("Done-Duration:", datetime.now() - t0)



# --------------------------------------------------------------------------- #
"""
The `var_specs` list contains tuples that define which variables to download from the GFS model's GRIB files, 
and how they should be named when saved in the final NetCDF file. Each tuple contains two elements:

1. **IDX Variable Name** (e.g., "TMP:surface"):
   - The **IDX variable name** comes from the **IDX file**, not directly from the GRIB file.
   - The IDX file serves as an index for the GRIB file, containing metadata for each variable (e.g., temperature, wind speed) available in the forecast data.
   - The general format is `<variable_name>:<level>`, where:
     - `<variable_name>` refers to the meteorological variable (e.g., temperature, wind speed).
     - `<level>` refers to the atmospheric level at which the variable is measured (e.g., "surface", "2 m above ground").
   - Example:
     - `"TMP:surface"` refers to **temperature at the surface**.
     - `"UGRD:10 m above ground"` refers to the **U-component of wind at 10 meters above the ground**.

2. **Output Variable Name** (e.g., "TMP_surface"):
   - The **output variable name** is the name you want to use when storing the variable in the final NetCDF file.
   - This is the name you will reference in the final processed data file. It can be any descriptive name you choose.
   - Example:
     - `"TMP_surface"` will be the name used for **surface temperature** data in the final NetCDF file.

### How to Extract the Necessary Information from IDX Files:

1. **IDX File**:
   - The **IDX file** is an index file containing a list of all available variables and their byte positions within the GRIB file.
   - Each line in the IDX file corresponds to one variable and provides information such as:
     - **Start and end byte positions** of the variable's data in the GRIB file.
     - **Variable Name** (e.g., "TMP:surface") and **Level** (e.g., "surface").
     - **Forecast Hour** (e.g., "6h" for a 6-hour forecast).
   
the idx files are on the S3 Server , can be downloded an inspectet as a txt file
"""



if __name__ == "__main__":
    VAR_SPECS = [
        ("TMP:surface",                                          "TMP_surface"),
        ("RH:2 m above ground",                                  "RH_2m"),
        ("PWAT:entire atmosphere (considered as a single layer)", "PWAT_ent"),
        ("TCDC:entire atmosphere",                               "TCDC_ent"),
        ("HCDC:high cloud layer",                                "HCDC_high"),
        ("MCDC:middle cloud layer",                              "MCDC_mid"),
        ("LCDC:low cloud layer",                                 "LCDC_low"),
        ("HGT:cloud ceiling",                                    "HGT_cloud_ceiling"),
        ("UGRD:10 m above ground",                               "UGRD_10m"),
        ("VGRD:10 m above ground",                               "VGRD_10m"),
        ("CAPE:surface",                                         "CAPE_surface"),
        ("HPBL:surface",                                         "HPBL_surface"),
        ("SUNSD:surface",                                        "SUNSD_surface"),
    ]

    dl = GFSDownloader(
        save_dir    = "/Users/leonardmerl/Internship_emergente/K-SolarForecast/K-SolarForecast_rework_sp/_1_data_acquisition/_02_raw_MultGFS_data/raw_MultGFS_",
        var_specs   = VAR_SPECS,
        max_workers = 8,
    )

    dl.download_for_period(
        start_date       = "20210401",
        end_date         = "20210401",
        lat_range        = [6.25,6.5],
        lon_range        = [-75.5,-75.25],
        utc_offset       = -5,
        launch_hours_utc = [12],
        fcst_start       = 2,
        fcst_end         =2,
    )
    
    """
    Important:
        For a stable and save run the following procedure ist recommended
        1. select the lat and lon range (raster you want to download data for)
        2. Choose the utc offest based on time region -> -5 for all of colombia 
        ->this automatically adjusts the launch and observation time inside the files
        3. choose launch time -> all can be downloaded at the same time (6,12, 18, 24)
        4. choose save dir (important to use absolute path)-> folder blueprint were the files are stored without launch time 
        -> code automatically sorts them into the correct launch time folder
        5. selecet fcst start and end -> from 1-24 to download the next 24 steps for each launch time 
        6. save the code, close spyder and start it trough the console 
        
        every files contains all variables 
        theres 1 file for every hour and launch time -> 96 files every day
        saved in the  _1_data_acquisition/_02_raw_MultGFS_data directory -> depending on each launch time
    """
        
        
        
        
        
