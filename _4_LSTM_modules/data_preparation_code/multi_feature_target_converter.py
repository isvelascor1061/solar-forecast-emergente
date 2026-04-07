#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Launch-Time Feature-Target Converter (v1.2)
=================================================

This module assembles **GFS forecast features** (optionally also SIATA
measurements) from *multiple model-launch times* into a single, *flat* feature
vector, perfectly shaped for a sequence-to-sequence or sequence-to-value
network (e.g. an LSTM).

Key design choices
------------------
* **Per-launch block**  
  For every launch time *LT* (e.g. “0100”, “0700”, “1300”, “1900”) the loader
  concatenates all requested variables plus the launch-to-observation `step`
  (in days) – if `include_step=True`.  
  Resulting block layout:

  ``[feat1_LT, feat2_LT, …, featN_LT, step_LT]``

* **Global flags**  
  After *all* launch-specific blocks an optional daylight flag (`day_flag`)
  is appended once.  Additional cyclic encodings – hour-of-day (*hod*),
  day-of-year (*doy*) – and a solar-zenith feature can be enabled as well.

* **Normalisation per variable**  
  Every feature can be left untouched, min-max scaled to 0-1, z-score
  standardised, divided by its long-term mean, **or** automatically rescaled
  to a fixed physical range (see `FIXED_RANGES`).

Output shapes
-------------
* **`to_dataframe()`** → `pd.DataFrame` with one column per feature +
  the target column (optionally NaNs retained).
* **`to_numpy()`** → `(samples, n_features)` **X** and `(samples,)` **y** –
  directly feedable into your model.

Typical workflow
----------------
```python
launch_times = ["0100", "0700", "1300", "1900"]

feat_templates = [
    # alias   path template                               var template                norm
    ("kc",   ".../clear_sky_index_GFS_{LT}.nc",  "clear_sky_index_GFS_{LT}",  "none"),
    ("ks",   ".../clearness_index_GFS_{LT}.nc",  "clearness_index_GFS_{LT}",  "none"),
    ("ds",   ".../dswrf1_{LT}.nc",              "dswrf1_{LT}",              "min_max"),
    ("tmp",  ".../temp_{LT}.nc",                "temperature_{LT}",         "z_score"),
]

loader = MultiLaunchTimeLoader(
    launch_times      = launch_times,
    feature_templates = feat_templates,
    target_path       = ".../clearness_index_Siata.nc",
    target_var        = "clearness_index_Siata",
    tz_offset         = -5,        # shift UTC→local
    dayflag_tz        = "local",   # day/night flag in local time
)
loader.load()
X, y = loader.to_numpy()
print(X.shape)   # (n_samples, n_flat_features)

"""


from __future__ import annotations

import xarray as xr
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Tuple, Dict, Sequence, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import timedelta

import pvlib

# -----------------------------------------------------------------------------
# Helperfunktions
# -----------------------------------------------------------------------------

def _parse_hour(lt: str) -> int:
    return int(lt[:2])


def _shift_hour(hour_utc: int, offset: int) -> int:
    return (hour_utc + offset) % 24




def _day_flag(idx: pd.DatetimeIndex, tz_offset: int, use_local: bool, flag_start: int, flag_end:int) -> xr.DataArray:
    if use_local:
        idx = idx + pd.Timedelta(hours=tz_offset)
    flag = ((idx.hour > flag_end) & (idx.hour < flag_start)).astype(np.int8)
    return xr.DataArray(flag, coords={"observation_time": idx}, dims="observation_time")


# -----------------------------------------------------------------
# 1) Hour-of-Day  (periodical 24 h)  →  hod_sin, hod_cos ∈ [0,1]
# -----------------------------------------------------------------
def _encode_hod(idx: pd.DatetimeIndex) -> xr.Dataset:
    h = idx.hour + idx.minute / 60
    hod_sin = (np.sin(2 * np.pi * h / 24) + 1) / 2      # 0-1
    hod_cos = (np.cos(2 * np.pi * h / 24) + 1) / 2
    return xr.Dataset(
        data_vars=dict(
            hod_sin=("observation_time", hod_sin),
            hod_cos=("observation_time", hod_cos),
        ),
        coords={"observation_time": idx},
    )

# -----------------------------------------------------------------
# 2) Day-of-Year (periodical in 365 d)  →  doy_sin, doy_cos ∈ [0,1]
# -----------------------------------------------------------------
def _encode_doy(idx: pd.DatetimeIndex) -> xr.Dataset:
    doy = idx.dayofyear - 1                       # 0 … 364
    doy_sin = (np.sin(2 * np.pi * doy / 365) + 1) / 2
    doy_cos = (np.cos(2 * np.pi * doy / 365) + 1) / 2
    return xr.Dataset(
        data_vars=dict(
            doy_sin=("observation_time", doy_sin),
            doy_cos=("observation_time", doy_cos),
        ),
        coords={"observation_time": idx},
    )

# -----------------------------------------------------------------
# 3) Solar-Zenith-Angle  →  zenith ∈ [0,1]
#    1 = Sun in zenith (θ = 0°), 0 = Nacht (θ ≥ 90°)
# -----------------------------------------------------------------
def _solar_zenith(idx, lat, lon):
    if idx.tz is None:        # tz-naiv → als UTC interpretieren
        idx_aware = idx.tz_localize("UTC").tz_convert("America/Bogota")
    else:
        idx_aware = idx.tz_convert("America/Bogota")

    sp = pvlib.solarposition.get_solarposition(
        idx_aware, lat, lon, method="nrel_numpy"
    )
    zen_norm = np.clip(1 - sp["apparent_zenith"]/90, 0, 1)

    # ⇓ TZ wieder entfernen, damit alle Arrays gleich sind
    idx_naive = idx_aware.tz_localize(None)

    return xr.DataArray(
        zen_norm,
        dims="observation_time",
        coords={"observation_time": idx_naive},
        name="zenith",
    )



# -----------------------------------------------------------------------------
# fixed pyhsical min / max values (checked in the dataframes)
# -----------------------------------------------------------------------------
FIXED_RANGES = {
    "DLWRF":        (274, 400),
    "TMP_surface":  (281.4, 302),
    "RH_2m" :        (32, 100.0),
    "HGT_cloud_ceiling":(2160,20001),
    "CAPE_surface": (0.0, 1660),
    "HPBL_surface": (10.0, 2500),
    "dlwrf1":       (280, 400)


}


# -----------------------------------------------------------------------------
# Main Class
# -----------------------------------------------------------------------------

class MultiLaunchTimeLoader:
    def __init__(
        self,
        launch_times: Sequence[str],
        feature_templates: Sequence[Tuple[str, str, str, str]] | Dict[str, Tuple[str, str, str]],
        target_path: str,
        target_var: str,
        normalize_target: str = "none",
        tz_offset: int = 0,
        dayflag_tz: str = "local",
        flag_start: int =19,
        flag_end: int= 6,   # 'local' oder 'utc'
        include_step: bool = True,
        include_dayflag:bool= True,
        add_hod: bool = False,
                 add_doy: bool = False,
                 add_zenith: bool = False,
                 lat: float | None = None,
                 lon: float | None = None,
    ) -> None:

        self.launch_times = list(launch_times)
        self.tz_offset = tz_offset
        self.dayflag_local = (dayflag_tz == "local")
        self.flag_start = flag_start
        self.flag_end = flag_end

        #feature templates to list
        if isinstance(feature_templates, dict):
            self.feat_tpls = list(feature_templates.items())
        else:  # already list/tuple
            self.feat_tpls = list(feature_templates)

        # check consistency of normalisation 
        allowed_norm = {"none", "min_max", "z_score", "average", "auto"}  # ← "auto" neu

        for tpl in self.feat_tpls:
            if len(tpl) != 4:
                raise ValueError("feature_templates‑Eintrag muss (alias, path_tpl, var_tpl, norm) sein")
            if tpl[3] not in allowed_norm:
                raise ValueError(f"Unbekannte Normierung: {tpl[3]}")
        if normalize_target not in allowed_norm:
            raise ValueError("Unbekannte Target‑Normierung")

        self.target_path = target_path
        self.target_var = target_var
        self.normalize_target = normalize_target
        self.include_step = include_step
        self.include_dayflag= include_dayflag

        self.feature_das: List[xr.DataArray] = []
        self.feature_vars: List[str] = []
        self.target_da: xr.DataArray | None = None
        
        self.add_hod    = add_hod
        self.add_doy    = add_doy
        self.add_zenith = add_zenith
        self.lat, self.lon = lat, lon

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(da: xr.DataArray, method: str = "auto") -> xr.DataArray:
        if method == "none":
            return da

        vals = da.values
        mask = np.isfinite(vals)
        if not mask.any():
            return da

        # ---------- fixed Ranges (auto/fixed) ---------------------------
        if method in {"auto", "fixed"} and da.name in FIXED_RANGES:
            x_min, x_max = FIXED_RANGES[da.name]
            span = x_max - x_min or 1.0
            da_n = da.copy()
            da_n.values[mask] = np.clip((vals[mask] - x_min) / span, 0.0, 1.0)
            return da_n

        # ---------- Data driven scalars -> z score would need other neural network ------------------------------
        if method == "min_max":
            scaler = MinMaxScaler((0, 1))
        elif method == "z_score":
            scaler = StandardScaler()
        else:           # "average"
            mean = vals[mask].mean()
            da_n = da.copy()
            da_n.values[mask] = vals[mask] / mean if mean else vals[mask]
            return da_n

        da_n = da.copy()
        da_n.values[mask] = scaler.fit_transform(vals[mask].reshape(-1,1)).flatten()
        return da_n


    # ------------------------------------------------------------------
    def load(self):
        feat_das = []
        feat_names = []
    
        for lt in self.launch_times:
            ds_first = None
            for alias, path_tpl, var_tpl, norm in self.feat_tpls:
                path = path_tpl.format(LT=lt)
                var  = var_tpl.format(LT=lt)
            
                ds = xr.open_dataset(path)
                da = ds[var].squeeze()
                if da.ndim > 1:
                    dims = [d for d in da.dims if d != "observation_time"]
                    da = da.mean(dim=dims)
            
                da = da.rename(alias)              
                da = self._normalize(da, norm)
                da = da.rename(f"{alias}_{lt}")    
            
                feat_das.append(da)
                feat_names.append(f"{alias}_{lt}")

                if ds_first is None:
                    ds_first = ds
    
            # calculate Step for each launh time 
            if ds_first is not None and "step" in ds_first.coords:
                step_raw = ds_first.coords["step"]
               
                step_hours = step_raw.values.astype("timedelta64[h]").astype(float)
                
                step_days = step_hours / 24.0
                
                step_da = xr.DataArray(
                    step_days,
                    coords={"observation_time": ds_first.coords["observation_time"]},
                    dims="observation_time",
                )
                if self.include_step:
                    feat_das.append(step_da)
                    feat_names.append(f"step_{lt}")
                else:
                    print("Step-Feature wird nicht hinzugefügt (include_step=False)")
    
                
    
           
    
        # load target and align 
        tgt_da = xr.open_dataset(self.target_path)[self.target_var].squeeze()
        if tgt_da.ndim > 1:
            dims = [d for d in tgt_da.dims if d != "observation_time"]
            tgt_da = tgt_da.mean(dim=dims)
        tgt_da = self._normalize(tgt_da, self.normalize_target)
    
        aligned = xr.align(*(feat_das + [tgt_da]), join="inner")
        self.feature_das = list(aligned[:-1])
        self.target_da = aligned[-1]
        self.feature_vars = feat_names
    
        #if necesarry add dayflag as feature -> NN does this already
        idx = self.target_da.coords["observation_time"].to_index()
        day_da = _day_flag(idx, self.tz_offset, self.dayflag_local, self.flag_start, self.flag_end)
        if self.include_dayflag:
            self.feature_das.append(day_da)
            self.feature_vars.append("day_flag")
        else: 
            print("No dayflag included")
            
        idx = self.target_da["observation_time"].to_index()

        if self.add_hod:
            hod_ds = _encode_hod(idx)
            for var in hod_ds:
                self.feature_das.append(hod_ds[var])
                self.feature_vars.append(var)
        
        if self.add_doy:
            doy_ds = _encode_doy(idx)
            for var in doy_ds:
                self.feature_das.append(doy_ds[var])
                self.feature_vars.append(var)
        
        if self.add_zenith:
            if self.lat is None or self.lon is None:
                raise ValueError("lat & lon erforderlich für Zenith-Feature")
            zen_da = _solar_zenith(idx, self.lat, self.lon)
            self.feature_das.append(zen_da)
            self.feature_vars.append("zenith")



    # ------------------------------------------------------------------
    def to_dataframe(self, dropna: bool = True) -> pd.DataFrame:
        if self.target_da is None:
            raise RuntimeError("load() zuerst aufrufen!")
        series = [da.to_series().rename(n) for da, n in zip(self.feature_das, self.feature_vars)]
        series.append(self.target_da.to_series().rename(self.target_var))
        df = pd.concat(series, axis=1)
        return df.dropna() if dropna else df

    def to_numpy(self):
        df = self.to_dataframe()
        X = df[self.feature_vars].values.astype(np.float32)
        y = df[self.target_var].values.astype(np.float32)
        return X, y


# -----------------------------------------------------------------------------
# Smoke‑Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
   
    launch_times = ["0100","0700","1300","1900"]

    feat_templates = [
        # --- Indices, already normalized 0-1 -----------------------------------------------
       ("kc",  "_3_Data_preparation_for_LSTM/Preparation_data/_04_indices/clear_sky_indices/clearsky_index_GFS_{LT}.nc",  "clearsky_index_GFS_{LT}",  "none"),
       ("ks",  "_3_Data_preparation_for_LSTM/Preparation_data/_04_indices/clearness_indices/clearness_index_GFS_{LT}.nc", "clearness_index_GFS_{LT}",  "none"),
       # --- Radiation -----------------------------------------------------------
       ("dswrf1",  "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Unclipped_merged_dswrf1/dswrf1_{LT}.nc",              "dswrf1_{LT}",              "min_max"),
       ("dlwrf1",  "_3_Data_preparation_for_LSTM/Preparation_data/_12_DLWRF/dlwrf1_{LT}.nc",                               "dlwrf1_{LT}",              "auto"),
       # --- Atmosphere ----------------------------------------------------------
       ("TMP_surface",  "_3_Data_preparation_for_LSTM/Preparation_data/_05_Temp_surface/TMP_surface_{LT}.nc",              "TMP_surface_{LT}",            "auto"),
       ("RH_2m",  "_3_Data_preparation_for_LSTM/Preparation_data/_06_RH_2m/RH_2m_{LT}.nc",                                  "RH_2m_{LT}",                  "auto"),
       ("CAPE_surface",  "_3_Data_preparation_for_LSTM/Preparation_data/_10_CAPE_surface/CAPE_surface_{LT}.nc",             "CAPE_surface_{LT}",            "auto"),
       ("HPBL_surface",  "_3_Data_preparation_for_LSTM/Preparation_data/_11_HPBL/HPBL_surface_{LT}.nc",                     "HPBL_surface_{LT}",            "auto"),
       ("PWAT_ent",      "_3_Data_preparation_for_LSTM/Preparation_data/_13_PWAT_ent/PWAT_ent_{LT}.nc",                     "PWAT_ent_{LT}",            "min_max"),
       # --- Clouds & Visibility ------------------------------------------------------
       ("TCDC_ent",  "_3_Data_preparation_for_LSTM/Preparation_data/_07_CDC_ent/_01_TCDC/TCDC_ent_{LT}.nc",              "TCDC_ent_{LT}",            "min_max"),
       ("HCDC_ent",  "_3_Data_preparation_for_LSTM/Preparation_data/_07_CDC_ent/_02_HCDC/HCDC_high_{LT}.nc",              "HCDC_high_{LT}",            "min_max"),
       ("MCDC_ent",  "_3_Data_preparation_for_LSTM/Preparation_data/_07_CDC_ent/_03_MCDC/MCDC_mid_{LT}.nc",              "MCDC_mid_{LT}",            "min_max"),
       ("LCDC_ent",  "_3_Data_preparation_for_LSTM/Preparation_data/_07_CDC_ent/_04_LCDC/LCDC_low_{LT}.nc",              "LCDC_low_{LT}",            "min_max"),
       ("HGT_cloud_ceiling",  "_3_Data_preparation_for_LSTM/Preparation_data/_08_HGT_cloud_ceiling/HGT_cloud_ceiling_{LT}.nc",     "HGT_cloud_ceiling_{LT}",  "auto"),
       # --- Windspeed -----------------------------------------------------------------
       ("Wind10m",  "_3_Data_preparation_for_LSTM/Preparation_data/_09_Wind10m/Wind10m_{LT}.nc",              "Wind10m_{LT}",            "min_max"),
       # --- Duration of sunshine ---------------------------------------------------
       ("SUNSD_minutes",  "_3_Data_preparation_for_LSTM/Preparation_data/_11_SUNSD/SUNSD_minutes_{LT}.nc",              "SUNSD_minutes_{LT}",            "min_max"),




       ]

    loader = MultiLaunchTimeLoader(
        launch_times=launch_times,
        feature_templates=feat_templates,
        target_path="_3_Data_preparation_for_LSTM/Preparation_data/_04_indices/clear_sky_indices/clearsky_index_Siata.nc",
        target_var="clearsky_index_Siata",
        tz_offset=-0,
        dayflag_tz="local",
        flag_start = 19,
        flag_end = 6,
        include_step = True,
        include_dayflag=False,
        add_hod=False,
        add_doy=False,
        add_zenith=True,
        lat=6.25,          # Medellín-Koordinate
        lon=-75.5,
    )
    loader.load()
    df = loader.to_dataframe(dropna=False)   # dropna=True → remove NaN values
    df.to_csv("_4_LSTM_modules/Prepared_data/feature_targetkc_ks.csv", index=True)
    print(df.filter(regex="_0100$").describe().T[["min","max"]].head(15))
    print(df.head(20))          
    print(df.columns.tolist())  
    print(df.shape)      
    t = pd.Timestamp("2023-10-05 19:00:00")
    print("Hour:", t.hour)
    print("Dayflag:", (t.hour >= 6 and t.hour < 19))
