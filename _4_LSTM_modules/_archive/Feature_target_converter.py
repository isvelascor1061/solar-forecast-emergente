#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 09:29:33 2025

@author: leonardmerl

Module to load multiple NetCDF4 files as features and a NetCDF4 file as target,
align them on 'observation_time', and convert to pandas DataFrame or NumPy arrays.
"""
import os
from typing import List, Optional, Tuple
import xarray as xr
import pandas as pd
import numpy as np


class NetCDFTimeSeriesLoader:
    """
    Loads multiple NetCDF files (features) and a NetCDF file (target), aligns on
    'observation_time', and provides methods to get data as DataFrame or NumPy arrays.

    Usage:
        loader = NetCDFTimeSeriesLoader(
            feature_paths=['f1.nc','f2.nc'],
            feature_vars=['var1','var2'],
            target_path='tgt.nc',
            target_var='tgtvar'
        )
        loader.load()
        df = loader.to_dataframe()
        X, y = loader.to_numpy()
    """

    def __init__(
        self,
        feature_paths: List[str],
        feature_vars: List[str],
        target_path: str,
        target_var: str
    ):
        assert len(feature_paths) == len(feature_vars), \
            "feature_paths and feature_vars must have same length"
        self.feature_paths = feature_paths
        self.feature_vars = feature_vars
        self.target_path = target_path
        self.target_var = target_var

        # xarray DataArrays
        self.feature_das: List[xr.DataArray] = []
        self.target_da: Optional[xr.DataArray] = None

    def load(self):
        """
        Load feature and target DataArrays from disk, squeeze and collapse
        spatial dimensions if present.
        """
        # load features
        self.feature_das = []
        for path, var in zip(self.feature_paths, self.feature_vars):
            ds = xr.open_dataset(path, engine = "netcdf4")
            da = ds[var].squeeze()
            # average over spatial dims if any
            if da.ndim > 1:
                dims = [d for d in da.dims if d != 'observation_time']
                da = da.mean(dim=dims)
            self.feature_das.append(da)
        # load target
        ds_t = xr.open_dataset(self.target_path, engine="netcdf4")
        da_t = ds_t[self.target_var].squeeze()
        if da_t.ndim > 1:
            dims = [d for d in da_t.dims if d != 'observation_time']
            da_t = da_t.mean(dim=dims)
        self.target_da = da_t

    def align(self, join: str = 'inner'):
        """
        Align all feature arrays and the target array on their 'observation_time' dimension.

        Parameters
        ----------
        join : str
            Alignment mode: 'inner', 'outer', 'left', or 'right'.
        """
        all_das = self.feature_das + [self.target_da]
        aligned = xr.align(*all_das, join=join)
        # split back
        n = len(self.feature_das)
        self.feature_das = list(aligned[:n])
        self.target_da = aligned[n]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame with columns for each feature and the target,
        indexed by 'observation_time'. Rows with any NaN are dropped.
        """
        series = []
        for da, var in zip(self.feature_das, self.feature_vars):
            s = da.to_series().rename(var)
            series.append(s)
        tgt = self.target_da.to_series().rename(self.target_var)
        series.append(tgt)
        df = pd.concat(series, axis=1)
        df = df.dropna()
        return df

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return feature matrix X and target array y as NumPy arrays.
        X has shape (n_samples, n_features), y has shape (n_samples,).
        """
        df = self.to_dataframe()
        X = df[self.feature_vars].values
        y = df[self.target_var].values
        return X, y


if __name__ == '__main__':
    # Example usage
    loader = NetCDFTimeSeriesLoader(
        feature_paths=[
            '_3_Data_preparation_for_LSTM/Preparation_data/Clear-sky-indices/GFS_clear-sky/clear_sky_index_GFS.nc'
        ],
        feature_vars=['clear_sky_index_GFS'],
        target_path='_3_Data_preparation_for_LSTM/Preparation_data/Clear-sky-indices/Siata_clear-sky/clear_sky_index_Siata.nc',
        target_var='clear_sky_index_Siata'
    )
    loader.load()
    loader.align(join='inner')
    df = loader.to_dataframe()
    print(df)
    X, y = loader.to_numpy()
    print(X.shape, y.shape)
