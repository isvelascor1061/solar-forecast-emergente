#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:17:45 2025

@author: leonardmerl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot der zusammengeführten DSWRF1-Datei und Speichern in eine gewünschte Directory.
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path

def plot_merged_dswrf1(
    nc_file: str | Path,
    output_dir: str | Path,
    title: str = "DSWRF1 Zeitreihe",
    out_filename: str = None
):
    """
    Lädt die angegebene NetCDF-Datei (mit 'dswrf1' und 'observation_time'),
    erstellt einen Zeitreihenplot und speichert ihn in 'output_dir'.
    
    Args:
        nc_file (str | Path): Pfad zur NetCDF-Datei, die geplottet werden soll.
        output_dir (str | Path): Ordner, in dem der Plot abgespeichert wird.
        title (str): Titel des Plots.
        out_filename (str): Name der zu speichernden PNG-Datei. 
                           Wenn None, wird aus nc_file abgeleitet.
    """
    nc_file = Path(nc_file)
    if not nc_file.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {nc_file}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Falls kein out_filename gesetzt, ableiten aus NC-Filename
    if out_filename is None:
        # z.B. "GFS_merged_dswrf1_all.png" falls nc_file="GFS_merged_dswrf1_all.nc"
        out_filename = nc_file.with_suffix('.png').name

    out_path = output_dir / out_filename

    # NetCDF laden
    ds = xr.open_dataset(nc_file, engine="netcdf4")

    # Zu DataFrame konvertieren
    df = ds[["dswrf1"]].to_dataframe().reset_index()
    df["observation_time"] = pd.to_datetime(df["observation_time"])

    # Plot erstellen
    plt.figure(figsize=(12, 5))
    plt.plot(df["observation_time"], df["dswrf1"], marker=".", linestyle="-")

    # X-Achse formatieren
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.title(title)
    plt.xlabel("Zeit")
    plt.ylabel("DSWRF1 (W/m²)")
    plt.grid(True)
    plt.tight_layout()

    # Plot speichern
    plt.savefig(out_path, dpi=150)
    print(f"Plot gespeichert unter: {out_path}")

    # Plot anzeigen
    plt.show()


if __name__ == "__main__":
    # Beispiel: 
    merged_file = "_3_Data_preparation_for_LSTM/Preparation_data/merged_timerseries/GFS_dswrf1/GFS_merged_dswrf1_all.nc"
    save_dir = "_3_Data_preparation_for_LSTM/Preparation_data/Merged_Plots/Merged_timeseries_plots/GFS_dswrf1_all"

    plot_merged_dswrf1(
        nc_file=merged_file,
        output_dir=save_dir,
        title="GFS DSWRF1 timeseries launch_time: 0100 (merged)",
        out_filename="GFS_dswrf1_merged_timeseries"  # Optional
    )
