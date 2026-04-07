#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:54:26 2025

@author: leonardmerl

Split a multi-variable NetCDF file into one NetCDF4 file per variable.

Special handling:
    * `SUNSD_minutes_0100` is stored cumulatively (minutes of sunshine since
      the last six-hour reset).  We convert it to true hourly values before
      saving by differencing adjacent timesteps.

All coordinates, attributes and original encodings are preserved.
Created Jun 2025 · Author: Emergente (ChatGPT refactor)
"""
from __future__ import annotations
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd



class NetCDFVarSplitter:
    """
    Load one NetCDF file, (optionally) transform SUNSD, and write every
    variable into its own NetCDF4 file.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the source NetCDF file that contains *all* variables.
    mapping : dict
        Dictionary of the form {variable_name: output_path}.
        Every key **must** exist in the dataset; otherwise a KeyError is raised.
    """

    # ------------------------------------------------------------------ #
    def __init__(self, input_path: str | Path, mapping: dict[str, str | Path]):
        self.input_path = Path(input_path).expanduser().resolve()
        self.mapping = {k: Path(v).expanduser().resolve()
                        for k, v in mapping.items()}

        # Lazily open the dataset (no data are read until a variable is touched)
        self.ds = xr.open_dataset(self.input_path, chunks={})
        
        
        
    def drop_days_with_hour_gaps(da: xr.DataArray) -> xr.DataArray:
        """
        Removes complete calendar days as soon as they contain ≥1 missing hour step
        (NaN).  Expects a 1 h grid.
        """
        # Vollständigen Index anlegen …
        full_idx = pd.date_range(
            da['observation_time'][0].item(),
            da['observation_time'][-1].item(),
            freq='1H'
        )
        da_full = da.reindex(observation_time=full_idx)
    
        # Für jeden Tag prüfen, ob ≥1 NaN vorkommt
        bad_days = da_full.isnull().groupby('observation_time.date').any()
        # Mapping Tag → True/False auf DataArray-Ebene
        keep_mask = da_full['observation_time'].dt.floor('D').map(~bad_days)
    
        # Tage mit Lücken verwerfen
        da_clean = da_full.where(keep_mask, drop=True)
        return da_clean

    # ------------------------------------------------------------------ #
    @staticmethod
    def sunsd_to_hourly_gapproof(da: xr.DataArray,
                                 max_per_hour: float = 60.) -> xr.DataArray:
        da_min = (da / np.timedelta64(1, "m")).astype("float32") \
                 if np.issubdtype(da.dtype, np.timedelta64) else da.astype("float32")
    
        prev      = da_min.shift(observation_time=1).fillna(0.)
        dt_hours  = (da_min['observation_time']
                     - da_min['observation_time'].shift(observation_time=1)
                    ) / np.timedelta64(1, 'h')
        dt_hours  = dt_hours.fillna(1.)
    
        delta   = da_min - prev
        reset   = delta < 0
        growth  = delta > 0
    
        hourly = xr.zeros_like(da_min)
        hourly = xr.where(growth, delta / dt_hours, hourly)
        hourly = xr.where(reset,  da_min / dt_hours, hourly)
    
        hourly = hourly.clip(0., max_per_hour)
        hourly.attrs = da.attrs | {
            "long_name": "Hourly sunshine duration (cleaned, ≤60 min)",
            "units": "minutes",
        }
        return hourly
    
   

    # ------------------------------------------------------------------ #
    def _write_single_var(self, var: str, out_path: Path) -> None:
        if var not in self.ds:
            raise KeyError(f"Variable '{var}' not found.")
    
        da = self.ds[var]
    
        # --- SUNSD special handling ----------------------------------------
        if var.lower().startswith("sunsd"):
            da = self.sunsd_to_hourly_gapproof(da)
    
            # 1) ensure pure float32 in memory
            da = da.astype("float32")
    
            # 2) remove CF-time attributes that trigger timedelta decoding
            for key in ("units", "calendar", "standard_name", "axis"):
                da.attrs.pop(key, None)
            # keep a *non-CF* hint instead
            da.attrs["unit_label"] = "minutes"
    
        # --- build dataset & save ------------------------------------------
        new_ds = xr.Dataset({var: da})
    
        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
        # encode explicitly as float32 + moderate compression
        new_ds.to_netcdf(
            out_path,
            format="NETCDF4",
            encoding={var: {"dtype": "float32", "zlib": True, "complevel": 4}},
        )
        print(f"✅  {var:<25s} → {out_path}")


    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Loop over the mapping and write all variables."""
        for var, path in self.mapping.items():
            self._write_single_var(var, path)

        self.ds.close()
        print("🎉  All files written successfully.")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # -------------------------------------------------- #
    # 1) Set the launch-time tag only here  (e.g. "0700", "1300", …)
    LT = "1900"

    # -------------------------------------------------- #
    # 2) Source file that contains ALL variables for this launch time
    INPUT_NC = f"_2_data_postprocessing/_03_Merged_MultGFS_data/MultGFS_{LT}.nc"

    # -------------------------------------------------- #
    # 3) Define where each single-variable file should go
    #    Key  : *base* variable name (without launch tag)
    #    Value: sub-directory (relative to BASE_OUT) that should hold the NC file
    VAR_DIRS = {
        "TMP_surface"       : "_05_Temp_surface",
        "RH_2m"             : "_06_RH_2m",
        "PWAT_ent"          : "_13_PWAT_ent",
        "TCDC_ent"          : "_07_CDC_ent/_01_TCDC",
        "HCDC_high"         : "_07_CDC_ent/_02_HCDC",
        "MCDC_mid"          : "_07_CDC_ent/_03_MCDC",
        "LCDC_low"          : "_07_CDC_ent/_04_LCDC",
        "HGT_cloud_ceiling" : "_08_HGT_cloud_ceiling",
        "Wind10m"           : "_09_Wind10m",
        "CAPE_surface"      : "_10_CAPE_surface",
        "HPBL_surface"      : "_11_HPBL",
        "SUNSD_minutes"     : "_11_SUNSD",          # will become hourly_SUNSD_…
    }

    # Base folder for all prepared data
    BASE_OUT = "_3_Data_preparation_for_LSTM/Preparation_data"

    # -------------------------------------------------- #
    # 4) Build OUT_MAP automatically
    OUT_MAP = {}
    for base_name, subdir in VAR_DIRS.items():
        # Full variable name inside the dataset
        var_name = f"{base_name}_{LT}"

       
        if base_name == "SUNSD_minutes":
            file_name = f"SUNSD_minutes_{LT}.nc"
        else:
            file_name = f"{base_name}_{LT}.nc"

        OUT_MAP[var_name] = f"{BASE_OUT}/{subdir}/{file_name}"

    # -------------------------------------------------- #
    splitter = NetCDFVarSplitter(INPUT_NC, OUT_MAP)
    splitter.run()
    """
    This script splits the MultGFS File with all direct parameters and one timeseries per launch time 
    into one file for each parameter 
    1. LT -> launch time -> 0100, 0700, 1300, 1900
    2. input nc is the directory of the MULT_GFS files
    3. Var dirs is a dict with the var name and the corresponding subfolder -> in preperation_data
    4. Sunsd is also a 6 hour block value but acculmalative and not a mean 
    -> The values are convertet back to being 1 hour values
    Script needs to be executed ones for every launch time 
    """

