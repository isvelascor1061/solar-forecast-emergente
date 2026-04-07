import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import os


def plot_nc_heatmap(file_path, variable_name=None, title="Heatmap Plot", cmap="viridis", vmin=0, vmax=1000):
    """
    Plots a heatmap from a NetCDF (.nc) file with a consistent color scale.

    Parameters:
    -----------
    file_path : str
        Path to the NetCDF file.
    variable_name : str, optional
        The name of the variable to plot. If None, the first available variable is used.
    title : str, optional
        The title of the plot (default: "Heatmap Plot").
    cmap : str, optional
        The colormap for the heatmap (default: "viridis").
    vmin : float, optional
        Minimum value for the color scale (default: 0).
    vmax : float, optional
        Maximum value for the color scale (default: 1400).
    """
    # Öffne die NetCDF-Datei
    ds = xr.open_dataset(file_path)

    # Falls keine Variable angegeben wurde, verwende die erste verfügbare Variable
    if variable_name is None:
        variable_name = list(ds.data_vars.keys())[0]  # Erste Variable im Dataset nehmen

    if variable_name not in ds:
        raise ValueError(f"Variable '{variable_name}' not found in dataset. Available variables: {list(ds.data_vars.keys())}")

    # Wähle die Daten aus
    data = ds[variable_name]

    # Stelle sicher, dass Zeitdimension entfernt wird (falls vorhanden)
    if "time" in data.dims:
        data = data.isel(time=0)  # Erstes Zeit-Frame auswählen

    # Lese Latitude- und Longitude-Koordinaten
    lat = ds.coords["latitude"].values
    lon = ds.coords["longitude"].values
    
  

    # Erstelle die Heatmap mit fester Farbschema-Skalierung
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(lon, lat, data, shading="auto" ,cmap="plasma", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(label=data.attrs.get("units", "Unknown Unit"))
    cbar.set_label(f"Surface downward shortwave Radiation flux \n Average over 3 hour intervall in W/m^2", fontsize = 12)
    
    # Achsenbeschriftungen und Titel
    plt.xlabel("Longitude (degress east)",fontsize = 12 )
    plt.ylabel("Latitude(degress north)",fontsize = 12)
    plt.title(title)

    # Zeige den Plot
    plt.show()
    
# Beispiel: Heatmap für mehrere Dateien mit konsistentem Farbschema
file_list = ["raw_data/2024-03-18_01_15_DSWRF3_20240318_1100_0600_2024-03-18_1600.nc"]  # <-- Hier die tatsächlichen Datei-Pfade einfügen

for file in file_list:
    plot_nc_heatmap(file, title=f"Historical GFS Forecast Medellín, Date of Forecast: 2024-03-18, GFS Launch time: 0100, Time of forecast: 2024-03-18-1600 ")
    