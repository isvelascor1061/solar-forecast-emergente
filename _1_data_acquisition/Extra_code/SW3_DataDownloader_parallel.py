import os
import uuid
import requests
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_dswrf_variable_name(timestep):
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
            raise ValueError(f"❌ Invalid timestep: {timestep}")

def get_dswrf_interval(hour):
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

def construct_url(date_str, forecast_hour, forecast_step):
    base_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    grib_url = (
        f"{base_url}/gfs.{date_str}/{forecast_hour:02d}/atmos/"
        f"gfs.t{forecast_hour:02d}z.pgrb2.0p25.f{forecast_step:03d}"
    )
    idx_url = grib_url + ".idx"
    return grib_url, idx_url

def find_dswrf_in_idx(idx_url, timestep, session):
    dswrf_name = get_dswrf_variable_name(timestep)
    resp = session.get(idx_url)
    if resp.status_code != 200:
        return None

    lines = resp.text.splitlines()
    for i, line in enumerate(lines):
        if dswrf_name in line:
            parts = line.split(":")
            try:
                start_byte = int(parts[1])
                end_byte = (
                    int(lines[i + 1].split(":")[1]) if (i + 1) < len(lines) else None
                )
                return (start_byte, end_byte)
            except (IndexError, ValueError):
                pass
    return None

class GFSDataDownloaderParallel:
    def __init__(self,
        save_dir="./_1_data_acquisition/raw_data",
        max_workers=12
    ):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.max_workers = max_workers
        self.session = requests.Session()

    def download_dswrf_snippet(self, job):
        """ Thread-Funktion: Lädt per Byte-Range DSWRF herunter und speichert lokal als GRIB2. """
        grib_url = job["grib_url"]
        idx_url = job["idx_url"]
        timestep = job["timestep"]

        offsets = find_dswrf_in_idx(idx_url, timestep, self.session)
        if offsets is None:
            return None  # DSWRF existiert nicht

        start_byte, end_byte = offsets
        headers = {"Range": f"bytes={start_byte}-{end_byte if end_byte else ''}"}
        resp = self.session.get(grib_url, headers=headers)
        if resp.status_code not in [200, 206]:
            return None

        # Temporären Dateinamen erzeugen
        temp_filename = f"temp_{uuid.uuid4().hex}.grib2"
        temp_path = os.path.join(self.save_dir, temp_filename)
        with open(temp_path, "wb") as f:
            f.write(resp.content)

        # Merke im job-Dict, wo die Datei liegt
        job["grib_path"] = temp_path
        return job  # Erfolg

    @staticmethod
    def find_dswrf_variable(ds):
        possible_names = [n for n in ds.data_vars if "DSWRF" in n or "sdswrf" in n]
        return possible_names[0] if possible_names else None

    def open_and_save_netcdf(self, job):
        """ Sequentielle Funktion: Öffnet GRIB2 mit cfgrib, beschneidet, speichert als NetCDF. """
      

        grib_path = job["grib_path"]
        if not os.path.isfile(grib_path):
            return f"❌ GRIB-Datei nicht gefunden: {grib_path}"

        lat_range = job["lat_range"]
        lon_range = job["lon_range"]
        UTC_OFFSET = job["UTC_OFFSET"]
        local_forecast_date = job["local_forecast_date"]
        local_forecast_hour = job["local_forecast_hour"]
        forecast_hour_diff = job["forecast_hour_diff"]
        utc_launch_str = job["utc_launch_str"]
        timestep = job["timestep"]

        try:
            ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        except Exception as e:
            return f"❌ Fehler beim Öffnen von {grib_path}: {e}"

        varname = self.find_dswrf_variable(ds)
        if not varname:
            ds.close()
            return f"❌ Keine DSWRF-Variable in {grib_path} gefunden!"

        # Lat/Lon anpassen
        if any(l < 0 for l in lon_range):
            adj_lon_range = [l + 360 if l < 0 else l for l in lon_range]
        else:
            adj_lon_range = lon_range

        if len(lat_range) == 1 and len(lon_range) == 1:
            ds_region = ds[varname].sel(
                latitude=lat_range[0], longitude=adj_lon_range[0], method="nearest"
            )
        else:
            ds_region = ds[varname].sel(
                latitude=slice(lat_range[1], lat_range[0]),
                longitude=slice(adj_lon_range[0], adj_lon_range[1])
            )

        # Zeitkorrektur
        if "time" in ds_region.coords:
            ds_region["time"] = ds_region["time"] + np.timedelta64(UTC_OFFSET, 'h')
        if "valid_time" in ds_region.coords:
            ds_region["valid_time"] = ds_region["valid_time"] + np.timedelta64(UTC_OFFSET, 'h')

        ds.close()

        # Output-Filename
        dswrf_interval = get_dswrf_interval(timestep)
        forecast_date_str = local_forecast_date.strftime("%Y-%m-%d")
        valid_datetime_local = local_forecast_date + timedelta(hours=local_forecast_hour + forecast_hour_diff)
        valid_date_str_local = valid_datetime_local.strftime("%Y-%m-%d_%H00")
        local_start_datetime = (
            datetime.combine(local_forecast_date, datetime.min.time()) +
            timedelta(hours=local_forecast_hour)
        )
        local_start_time_str = local_start_datetime.strftime("%Y-%m-%d_%H00")

        outname = (
            f"{self.save_dir}/{forecast_date_str}_{local_forecast_hour:02d}_"
            f"{forecast_hour_diff:02d}_DSWRF{dswrf_interval}_{local_start_time_str}_"
            f"{valid_date_str_local}_UTCSTART:{utc_launch_str}.nc"
        )

        # Als Dataset abspeichern
        try:
            # ds_region ist ein DataArray; in ein Dataset umwandeln für .to_netcdf():
            ds_region.to_dataset(name=varname).to_netcdf(outname)
        except Exception as e:
            return f"❌ Fehler beim Speichern in {outname}: {e}"

        # Temp.-GRIB löschen
        try:
            os.remove(grib_path)
        except OSError:
            pass

        return f"✅ Gespeichert: {outname}"

    def download_for_period(
        self, start_date, end_date, lat_range, lon_range,
        UTC_OFFSET, launch_times_utc, start_forecasthour, end_forecasthour
    ):
        """ 
        1) Erst alle GRIB-Dateien parallel herunterladen.
        2) Danach jede Datei sequentiell öffnen und in NetCDF umwandeln.
        """
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        tasks = []
        # --- Aufgabenliste erstellen
        while start_dt <= end_dt:
            for forecast_hour_utc in launch_times_utc:
                if forecast_hour_utc in [24, 0]:
                    utc_launch_date = start_dt + timedelta(days=1)
                    actual_forecast_hour = 0
                else:
                    utc_launch_date = start_dt
                    actual_forecast_hour = forecast_hour_utc

                utc_launch_datetime = (
                    datetime.combine(utc_launch_date, datetime.min.time()) +
                    timedelta(hours=actual_forecast_hour)
                )
                utc_launch_str = utc_launch_datetime.strftime("%Y-%m-%d_%H00")

                # Lokale Zeit bestimmen
                local_forecast_hour = (actual_forecast_hour + UTC_OFFSET) % 24
                if actual_forecast_hour >= abs(UTC_OFFSET):
                    local_forecast_date = utc_launch_date
                else:
                    local_forecast_date = utc_launch_date - timedelta(days=1)

                for hour in range(start_forecasthour, end_forecasthour + 1):
                    date_str = utc_launch_datetime.strftime("%Y%m%d")
                    grib_url, idx_url = construct_url(date_str, utc_launch_datetime.hour, hour)

                    job = {
                        "grib_url": grib_url,
                        "idx_url": idx_url,
                        "timestep": hour,
                        "lat_range": lat_range,
                        "lon_range": lon_range,
                        "UTC_OFFSET": UTC_OFFSET,
                        "local_forecast_date": local_forecast_date,
                        "local_forecast_hour": local_forecast_hour,
                        "forecast_hour_diff": hour,
                        "utc_launch_str": utc_launch_str
                    }
                    tasks.append(job)
            start_dt += timedelta(days=1)

        # --- Phase 1: Paralleler Download
        print(f"Starte {len(tasks)} parallele Downloads mit {self.max_workers} Threads ...")
        t0 = datetime.now()

        completed_jobs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.download_dswrf_snippet, job) for job in tasks]
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:  
                    # Download ok => speicher Job
                    completed_jobs.append(result)

        # --- Phase 2: Sequentielles Öffnen & Speichern
        print(f"Öffne und speichere {len(completed_jobs)} GRIB-Dateien einzeln ...")
        for job in completed_jobs:
            msg = self.open_and_save_netcdf(job)
            print(msg)

        print(f"Fertig! Gesamtdauer: {datetime.now() - t0}")


# -----------------------------------------------------------------------------
# Beispielaufruf
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    start_time = time.perf_counter()

    downloader = GFSDataDownloaderParallel(
        save_dir="./_1_data_acquisition/raw_data",
        max_workers=6  # paralleler Download
    )

    downloader.download_for_period(
        start_date="20210904",
        end_date="20210904",
        lat_range=[6.25],
        lon_range=[-75.5],
        UTC_OFFSET=-5,
        launch_times_utc=[6],
        start_forecasthour=1,
        end_forecasthour=24
    )

    end_time = time.perf_counter()
    print(f"Total runtime: {end_time - start_time:.2f} s")
