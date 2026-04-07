import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
import re

def plot_dswrf1_by_month(
    folder_path: str | Path, 
    output_folder: str | Path
):
    """
    Sucht in 'folder_path' nach Dateien im Format 'dswrf1_YYYYMMDD_0100.nc',
    gruppiert sie nach Monat (YYYYMM), erstellt für jede Monatsgruppe
    einen Zeitreihen-Plot und speichert diesen als PNG nach 'output_folder'.
    """
    folder_path = Path(folder_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)  # Output-Verzeichnis anlegen, falls nicht vorhanden
    
    # Alle Dateien suchen, die z.B. ein Datum + Launch 01:00 im Namen haben
    pattern_nc = re.compile(r"dswrf1_(\d{8})_0100\.nc$")
    all_files = sorted(folder_path.glob("dswrf1_*_0100.nc"))
    
    # Dictionary: { 'YYYYMM': [Datei1, Datei2, ...], ... }
    monthly_groups = {}
    
    for f in all_files:
        match = pattern_nc.match(f.name)
        if match:
            yyyymmdd = match.group(1)    # z.B. "20220105"
            year = yyyymmdd[:4]          # "2022"
            month = yyyymmdd[4:6]        # "01"
            year_month = year + month    # "202201"
            
            if year_month not in monthly_groups:
                monthly_groups[year_month] = []
            monthly_groups[year_month].append(f)
    
    # Für jede Monatsgruppe: Daten einlesen, zusammenführen & plotten
    for ym, files in monthly_groups.items():
        # NetCDFs einlesen und entlang observation_time zusammenführen
        datasets = [xr.open_dataset(f, engine="netcdf4") for f in files]
        ds_merged = xr.concat(datasets, dim="observation_time")
        ds_merged = ds_merged.sortby("observation_time")
        
        # In DataFrame umwandeln
        df = ds_merged[["dswrf1"]].to_dataframe().reset_index()
        df["observation_time"] = pd.to_datetime(df["observation_time"])
        
        # Plot erstellen
        plt.figure(figsize=(12, 5))
        plt.plot(df["observation_time"], df["dswrf1"], marker=".", linestyle="-")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        plt.title(f"DSWRF1 (Monat {ym}, Launch 01:00)")
        plt.xlabel("Zeit")
        plt.ylabel("DSWRF1 (W/m²)")
        plt.grid(True)
        plt.tight_layout()
        
        # PNG-Datei im output_folder speichern, z. B. "dswrf1_202201.png"
        out_png = output_folder / f"dswrf1_{ym}.png"
        plt.savefig(out_png, dpi=150)
        
        # Fenster schließen, um Speicher zu sparen
        plt.close()

        print(f"Plot gespeichert: {out_png}")

    print("Fertig! Alle Monatsplots erstellt.")


if __name__ == "__main__":
    # Beispiel: 
    data_dir = "_2_data_postprocessing/DSWRF1_merged_data"
    out_dir = "_3_Data_preparation_for_LSTM/Preparation_data/Merged_Plots/Merged_monthly_plots/GFS_dswrf1_monthly"
    
    plot_dswrf1_by_month(data_dir, out_dir)
