#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:05:52 2025

@author: leonardmerl

NetCDFTimeSeriesLoader mit string-basierten Normalisierungs-Methoden:
'none', 'min_max', 'z_score', 'average'
Und optionaler Target-Normalisierung per normalize_target.
"""

import xarray as xr
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

class NetCDFTimeSeriesLoader:
    """
    Lädt mehrere NetCDF-Features und ein NetCDF-Target, aligned auf
    'observation_time' und bietet DataFrame-/NumPy-Schnittstellen.

    Für jedes Feature gibt man in `normalize_features` einen String an:
      - "none"    → keine Normalisierung
      - "min_max" → Min-Max-Skalierung auf [0,1]
      - "z_score" → Standard-Z-Score (nur Tagsüber: 06–18 Uhr)
      - "average" → Division durch Mittelwert

    Über `normalize_target` kann man zusätzlich das Target normalisieren.
    """

    def __init__(
        self,
        feature_paths: List[str],
        feature_vars: List[str],
        normalize_features: List[str],  # Methoden für Features
        target_path: str,
        target_var: str,
        normalize_target: str = "none"  # Methode für Target
    ):
        allowed = {"none", "min_max", "z_score", "average"}
        assert len(feature_paths) == len(feature_vars) == len(normalize_features), \
            "feature_paths, feature_vars und normalize_features müssen gleich lang sein"
        for m in list(normalize_features) + [normalize_target]:
            if m not in allowed:
                raise ValueError(f"Unbekannte Normalisierungs-Methode: {m!r}")

        self.feature_paths = feature_paths
        self.feature_vars = feature_vars
        self.normalize_features = normalize_features
        self.target_path = target_path
        self.target_var = target_var
        self.normalize_target = normalize_target

        self.feature_das: List[xr.DataArray] = []
        self.target_da: xr.DataArray

    def load(self):
        """
        Lädt alle Features, wendet ggf. Normalisierung an, und lädt das Target.
        """
        self.feature_das = []
        for path, var, method in zip(self.feature_paths, self.feature_vars, self.normalize_features):
            ds = xr.open_dataset(path, engine="netcdf4")
            da = ds[var].squeeze()
            # falls räumliche Dims vorhanden, mitteln
            if da.ndim > 1:
                dims = [d for d in da.dims if d != "observation_time"]
                da = da.mean(dim=dims)
            # Feature-Normalisierung
            if method != "none":
                da = self._normalize(da, method)
            self.feature_das.append(da)

        # Target laden
        ds_t = xr.open_dataset(self.target_path, engine="netcdf4")
        da_t = ds_t[self.target_var].squeeze()
        if da_t.ndim > 1:
            dims = [d for d in da_t.dims if d != "observation_time"]
            da_t = da_t.mean(dim=dims)
        # Target-Normalisierung
        if self.normalize_target != "none":
            da_t = self._normalize(da_t, self.normalize_target)
        self.target_da = da_t
    def _normalize(self, da: xr.DataArray, method: str) -> xr.DataArray:
        """
        Wendet die gewählte Normalisierungsmethode an (ignoriert NaN-Werte komplett).
        Für 'z_score': Nutzt nur Tagwerte (06–18 Uhr), aber überspringt NaN-Werte.
        """
        arr = da.values.astype(float)
        arr_n = arr.copy()  # Kopie für normalisierte Werte
    
        if method == "min_max":
            scaler = MinMaxScaler(feature_range=(0, 1))
            valid_mask = np.isfinite(arr)
            if valid_mask.any():
                arr_n[valid_mask] = scaler.fit_transform(arr[valid_mask].reshape(-1, 1)).flatten()
    
    
    
        elif method == "z_score":
        # Extrahiere nur gültige Werte (ignoriert NaN)
            valid_mask = np.isfinite(arr)
            valid_values = arr[valid_mask].reshape(-1, 1)  # Als 2D-Array für sklearn
            
            if len(valid_values) > 1:  # Mind. 2 Werte für std
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(valid_values)
                arr_n[valid_mask] = scaled_values.flatten()
          
            
    
        elif method == "average":
            valid_mask = np.isfinite(arr)
            if valid_mask.any():
                mean = np.mean(arr[valid_mask])
                if mean > 1e-6:  # Vermeidet Division durch ~0
                    arr_n[valid_mask] = arr[valid_mask] / mean
    
        # Erstelle neue DataArray mit normalisierten Werten
        da_norm = da.copy(deep=True)
        da_norm.values = arr_n
        return da_norm

    def align(self, join: str = "inner"):
        """
        Align all features + target on 'observation_time'.
        """
        all_das = self.feature_das + [self.target_da]
        aligned = xr.align(*all_das, join=join)
        self.feature_das = list(aligned[:-1])
        self.target_da    = aligned[-1]

    def to_dataframe(self, dropna: bool = False) -> pd.DataFrame:
        """
        Gibt ein DataFrame zurück, indexed by 'observation_time'.
        Wenn dropna=True, werden alle Zeilen mit mindestens einem NaN entfernt.
        """
        series = []
        for da, var in zip(self.feature_das, self.feature_vars):
            series.append(da.to_series().rename(var))
        series.append(self.target_da.to_series().rename(self.target_var))

        df = pd.concat(series, axis=1)
        if dropna:
            df = df.dropna()
        return df

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gibt X (n_samples × n_features) und y (n_samples,) als NumPy-Arrays zurück.
        """
        df = self.to_dataframe(dropna=True)
        X = df[self.feature_vars].values
        y = df[self.target_var].values
        return X, y



if __name__ == "__main__":
    # Beispiel-Aufruf:
    loader = NetCDFTimeSeriesLoader(
        feature_paths=[
            "_3_Data_preparation_for_LSTM/Preparation_data/_04_indices/clear_sky_indices/clear_sky_index_GFS_0100.nc",
            "_3_Data_preparation_for_LSTM/Preparation_data/_04_indices/clearness_indices/clearness_index_GFS_0100.nc"
        ],
        feature_vars=[ 'clear_sky_index_gfs_0100',"clearness_index_gfs_0100"],
        normalize_features=["none","none"],  # strings only
        target_path='_3_Data_preparation_for_LSTM/Preparation_data/_04_indices/clear_sky_indices/clear_sky_index_Siata.nc',
        target_var="clear_sky_index_Siata",
        normalize_target ="none"
    )
    loader.load()
    loader.align(join='inner')
    df = loader.to_dataframe()
    df.to_csv("_4_LSTM_modules/z_score_check.csv",index=True,na_rep="NaN")
    print(df)
    X, y = loader.to_numpy()
    print("Shape:", X.shape, y.shape)
