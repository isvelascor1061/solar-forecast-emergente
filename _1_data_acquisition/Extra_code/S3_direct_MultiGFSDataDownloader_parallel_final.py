#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFS-Downloader (0.25°) für
  • TMP   (surface)
  • RH    (2 m above ground)
  • TCDC  (entire atmosphere)

Legt pro Vorhersagestunde eine NetCDF-Datei an:

YYYY-MM-DD_locHH_FORECASThr_TMP_RH_TCDC_YYYY-MM-DD_HH00_UTCSTART:YYYY-MM-DD_HH00.nc
"""

from pathlib import Path
from datetime import datetime, timedelta
import uuid, requests, xarray as xr, numpy as np
import concurrent.futures as cf, threading

# --------------------------------------------------------------------------- #
NETCDF_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
class GFSDownloader:
    def __init__(self, save_dir: str, var_specs: list[tuple[str,str]], max_workers: int = 4):
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.session     = requests.Session()      # Session Hauptprozess
        self.var_specs  = var_specs

    # ----------------------------- Hilfsfunktionen -------------------------
    @staticmethod
    def _construct_urls(date_str, cycle_hour, fcst_hour):
        base = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
        g = f"{base}/gfs.{date_str}/{cycle_hour:02d}/atmos/" \
            f"gfs.t{cycle_hour:02d}z.pgrb2.0p25.f{fcst_hour:03d}"
        return g, g + ".idx"
    import requests
    
    
    def _find_key_in_idx(self,idx_url: str,
                         var_specs: list[tuple[str, str]],
                         fcst_hour: int) -> dict[str, tuple[int, int | None]]:
        """
        Liefert für alle Einträge in VAR_SPECS das Byteintervall (start, end)
        innerhalb der GRIB-Datei zurück.
    
        Parameters
        ----------
        idx_url   : URL der *.idx*-Datei
        var_specs : Liste aus (idx_name, out_name) – genau wie in VAR_SPECS
                    z. B. ("TMP:surface", "TMP_surface")
        fcst_hour : Vorhersagehorizont (int), z. B. 3  →  "3 hour fcst"
    
        Returns
        -------
        dict[out_name] -> (start, end)   # end=None, wenn Zeile die letzte ist
        """
        # --------------------------------------------------------- IDX laden --
        r = requests.get(idx_url, timeout=20)
        r.raise_for_status()
        lines = r.text.splitlines()
    
        # ------------------------------------------------------ vorbereiten ---
        wanted = {}
        spec_map = {}                     # map out_name -> (var, level)
        for idx_name, out_name in var_specs:
            var, level = idx_name.split(":", 1)
            spec_map[out_name] = (var, level)
    
        # ------------------------------------------------------ Zeilen suchen --
        for i, ln in enumerate(lines):
            parts = ln.split(":")
            if len(parts) < 6:
                continue            # unvollständige Zeile überspringen
    
            var, level = parts[3], parts[4]
            if not parts[5].startswith(f"{fcst_hour} hour fcst"):
                continue
    
            # prüfen, ob diese (var, level) in unserer Wunschliste steht
            for out_name, (v_want, l_want) in spec_map.items():
                if var == v_want and level == l_want:
                    start = int(parts[1])
                    end = (int(lines[i + 1].split(":")[1]) - 1
                           if i + 1 < len(lines) else None)
                    wanted[out_name] = (start, end)
                    break                       # zur nächsten Zeile
    
            # abbrechen, wenn schon alle Variablen gefunden
            if len(wanted) == len(var_specs):
                break
    
        return wanted        # fehlende Keys erscheinen nicht im Dict


    

    # ----------------------------- Worker ----------------------------------
    
    def _download_and_process(self, job: dict) -> str:
         """
         Lädt alle in VAR_SPECS definierten Felder für einen Vorhersageschritt,
         schneidet auf Punkt/Ausschnitt zu, wandelt Zeitkoordinaten in Lokalzeit
         und speichert eine NetCDF-Datei.
         """
         # eigene Session im Sub-Prozess
         if not hasattr(self, "session") or self.session is None:
             self.session = requests.Session()
    
         grib_url, idx_url = job["grib_url"], job["idx_url"]
         fcst_hr           = job["fcst_hour"]
    
         # -------------------------------------------------- Byte-Ranges ermitteln
         ranges = self._find_key_in_idx(idx_url, self.var_specs, fcst_hr)
         if len(ranges) < len(self.var_specs):
             missing = [name for _, name in self.var_specs if name not in ranges]
             return f"❌ fehlend: {', '.join(missing)} ({fcst_hr} h)"
    
         open_datasets, tmp_files, pieces = [], [], {}
         coords = None
    
         # Label aus lokaler Launch-Zeit, z. B. "0100" oder "0600"
         launch_label = f"{job['loc_start_dt'].hour:02d}{job['loc_start_dt'].minute:02d}"
    
         # -------------------------------------------------- jede Variable holen
         for idx_name, out_name in self.var_specs:
             start, end = ranges[out_name]
             headers = {"Range": f"bytes={start}-{end if end else ''}"}
             resp = self.session.get(grib_url, headers=headers, timeout=30)
             resp.raise_for_status()
    
             tmp_path = (self.save_dir / f"tmp_{uuid.uuid4().hex}.grib2").resolve()
             tmp_files.append(tmp_path)
             tmp_path.write_bytes(resp.content)
    
             ds = xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
             
             if 'heightAboveGround' in ds.coords:
                    ds = ds.drop_vars('heightAboveGround')
             open_datasets.append(ds)
    
             arr = ds[next(iter(ds.data_vars))]
    
             # -------------- räumlicher Zuschnitt -------------------------
             lat_rng, lon_rng = job["lat_range"], job["lon_range"]
             lon_adj = [lon + 360 if lon < 0 else lon for lon in lon_rng]
    
             if len(lat_rng) == 1:        # Punkt-Modus
                 arr = arr.sel(latitude=lat_rng[0],
                               longitude=lon_adj[0],
                               method="nearest").squeeze(drop=True)
             else:                        # Ausschnitt
                 arr = arr.sel(latitude=slice(lat_rng[1], lat_rng[0]),
                               longitude=slice(lon_adj[0], lon_adj[1]))
    
             arr = arr.load()
             arr.name = f"{out_name}_{launch_label}"      # Suffix anhängen
             pieces[arr.name] = arr
             
             u_key = f"UGRD_10m_{launch_label}"
             v_key = f"VGRD_10m_{launch_label}"
             if u_key in pieces and v_key in pieces:
                u = pieces.pop(u_key)
                v = pieces.pop(v_key)
                wind10 = (u**2 + v**2)**0.5
                wind10.name = f"Wind10m_{launch_label}"
                pieces[wind10.name] = wind10
           
             
             raw_key = f"SUNSD_surface_{launch_label}"
             if raw_key in pieces:
                # Rohwert in Sekunden holen
                secs = pieces.pop(raw_key)
                # In Minuten umwandeln
                mins = secs / 60.0
                mins.name = f"SUNSD_minutes_{launch_label}"
                pieces[mins.name] = mins
    
             if coords is None:           # einmalig Koordinaten übernehmen
                 point_mode = len(lat_rng) == 1
                 coords = {c: ds.coords[c].load()
                           for c in ds.coords
                           if not (point_mode and c in ("latitude", "longitude"))}
            
    
         out_ds = xr.Dataset(pieces, coords=coords).squeeze(drop=True)
    
         # -------------------------------------------------- Zeitkoordinaten lokal
         offset = np.timedelta64(job["UTC_OFFSET"], "h")
         if "time" in out_ds.coords:
             out_ds = out_ds.rename({"time": "launch_time"})
             out_ds = out_ds.assign_coords(
                 launch_time=out_ds["launch_time"] + offset
             )
         if "valid_time" in out_ds.coords:
             out_ds = out_ds.rename({"valid_time": "observation_time"})
             out_ds = out_ds.assign_coords(
                 observation_time=out_ds["observation_time"] + offset
             )
    
         # -------------------------------------------------- Dateiname
         loc_start = job["loc_start_dt"]
         valid_dt  = loc_start + timedelta(hours=fcst_hr)
         fname = (
             f"{loc_start:%Y-%m-%d}_{loc_start.hour:02d}_{fcst_hr:02d}_"
             f"MultGFS_{valid_dt:%Y-%m-%d_%H00}_"
             f"UTCSTART:{job['utc_launch']}.nc"
         )
         out_path = self.save_dir / fname
    
         # -------------------------------------------------- Speichern & Aufräumen
         with NETCDF_LOCK:
            out_ds.to_netcdf(out_path)
        
         for ds in open_datasets:
             ds.close()
             del ds
        
         del pieces, out_ds                 # alles freigeben
         import gc; gc.collect()            # RAM wirklich räumen
         for p in tmp_files:
             try:
                 p.unlink()
             except FileNotFoundError:
                 pass
    
         return f"✅ Saved: {fname} | fcst_hr={fcst_hr}"


    # ----------------------------- Manager & Taskgen ------------------------
    def download_for_period(self, start_date, end_date,
                            lat_range, lon_range,
                            utc_offset, launch_hours_utc,
                            fcst_start, fcst_end):

        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt   = datetime.strptime(end_date,   "%Y%m%d")
        tasks = []

        while start_dt <= end_dt:
            for hh in launch_hours_utc:
                utc_launch = datetime.combine(start_dt, datetime.min.time()) + timedelta(hours=hh)
                utc_str    = utc_launch.strftime("%Y-%m-%d_%H00")
                loc_start  = utc_launch + timedelta(hours=utc_offset)

                for fhr in range(fcst_start, fcst_end + 1):
                    g_url, i_url = self._construct_urls(
                        utc_launch.strftime("%Y%m%d"), utc_launch.hour, fhr)

                    tasks.append({
                        "grib_url":   g_url,
                        "idx_url":    i_url,
                        "fcst_hour":  fhr,
                        "lat_range":  lat_range,
                        "lon_range":  lon_range,
                        "UTC_OFFSET": utc_offset,
                        "loc_start_dt": loc_start,
                        "utc_launch": utc_str
                    })
            start_dt += timedelta(days=1)

        print(f"Starte {len(tasks)} Tasks mit {self.max_workers} Prozessoren …")
        t0 = datetime.now()

        with cf.ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for i, fut in enumerate(cf.as_completed(
                    pool.submit(self._download_and_process, t) for t in tasks), 1):
                print(f"[{i}/{len(tasks)}] {fut.result()}")

        print("Fertig – Dauer:", datetime.now() - t0)


#  Beispiel
# --------------------------------------------------------------------------- #
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
    ("SUNSD:surface",                                        "SUNSD_surface")
    ]


    dl = GFSDownloader(
        save_dir    = "_1_data_acquisition/_02_raw_MultGFS_data/raw_MultGFS_0700",
        var_specs   = VAR_SPECS,
        max_workers = 4
    )

    dl.download_for_period(
        start_date       = "20250103",
        end_date         = "20250531",
        lat_range        = [6.25],      # Punkt
        lon_range        = [-75.5],
        utc_offset       = -5,          # Kolumbien
        launch_hours_utc = [12],         # 06-UTC-Lauf
        fcst_start       = 1,
        fcst_end         = 24
    )
