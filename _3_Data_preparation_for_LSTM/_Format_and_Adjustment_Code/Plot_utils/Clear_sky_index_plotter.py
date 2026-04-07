
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 14:30:00 2025

@author: leonardmerl

Script to plot a time series of radiation indices from a NetCDF file.
"""

import os
import xarray as xr
import matplotlib.pyplot as plt

class MultiIndexPlotter:
    """
    Plots time series of radiation indices from multiple NetCDF files.
    Each file may contain spatial dimensions that are averaged if necessary.
    """
    def __init__(
        self,
        file_configs: list,
        output_fig: str,
        title: str = None,
        ylabel: str = "Index",
        figsize: tuple = (12, 6),
    ):
        """
        Parameters
        ----------
        file_configs : list of dicts
            Each dict with keys:
            - 'path': Path to NetCDF file
            - 'var': Variable name to plot
            - 'label': Legend label for the series
        output_fig : str
            Path (including filename) to save the plot (PNG).
        title : str
            Plot title.
        ylabel : str
            Y-axis label.
        figsize : tuple
            Figure size in inches.
        """
        self.file_configs = file_configs
        self.output_fig = output_fig
        self.title = title
        self.ylabel = ylabel
        self.figsize = figsize
        self.series = []  # list of (label, DataArray)

    def load_and_prepare(self):
        """
        Loads each NetCDF, squeezes length-1 dims, averages remaining spatial dims,
        and stores a 1D DataArray per config.
        """
        for cfg in self.file_configs:
            ds = xr.open_dataset(cfg['path'])
            da = ds[cfg['var']]
            # remove length-1 dimensions
            da = da.squeeze()
            # if still multidimensional, average over all dims except time
            if da.ndim > 1:
                time_dim = 'observation_time'
                other_dims = [d for d in da.dims if d != time_dim]
                da = da.mean(dim=other_dims)
            self.series.append((cfg['label'], da))

    def plot(self):
        """
        Generates the plot and saves it to the configured path.
        """
        plt.figure(figsize=self.figsize)
        for label, da in self.series:
            plt.plot(da['observation_time'], da.values, label=label)
        plt.xlabel('Time')
        plt.ylabel(self.ylabel)
        plt.title(self.title or 'Radiation Index Time Series')
        plt.grid(True)
        plt.legend()

        out_dir = os.path.dirname(self.output_fig)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(self.output_fig, dpi=300)
        print(f"Plot saved to {self.output_fig}")

if __name__ == '__main__':
    # === User configuration: define each index file and label ===
    FILE_CONFIGS = [
        {
            'path': (
                "_3_Data_preparation_for_LSTM/Preparation_data/Clear-sky-indices/GFS_clear-sky/clear_sky_index_GFS.nc"
            ),
            'var': 'clear_sky_index_GFS',
            'label': 'GFS Clear-Sky Index'
        },
        {
            'path': (
                "_3_Data_preparation_for_LSTM/Preparation_data/Clear-sky-indices/Siata_clear-sky/clear_sky_index_Siata.nc"
            ),
            'var': 'clear_sky_index_Siata',
            'label': 'Siata Clear-Sky Index'
        }
    ]
    # Path for the output plot (PNG)
    OUTPUT_FIG = (
        None)

    plotter = MultiIndexPlotter(
        file_configs=FILE_CONFIGS,
        output_fig=OUTPUT_FIG,
        title='SIATA clear sky index vs. GFS clear sky index',
        ylabel='Index',
        figsize=(12, 6)
    )
    plotter.load_and_prepare()
    plotter.plot()
