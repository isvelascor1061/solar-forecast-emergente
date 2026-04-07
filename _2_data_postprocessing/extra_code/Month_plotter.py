import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib.dates as mdates

# 📁 Ordner & Datei-Filter
data_dir = Path("_2_data_postprocessing/DSWRF1_merged_data")
files = sorted(data_dir.glob("dswrf1_202201*_0100.nc"))  # z. B. nur Launch um 01:00 Uhr

# 📦 Dateien als Liste laden
datasets = [xr.open_dataset(f, engine="netcdf4") for f in files]

# 🔗 Zusammenführen entlang 'observation_time'
ds_merged = xr.concat(datasets, dim="observation_time")

# 🧹 Nach Zeit sortieren, falls nötig
ds_merged = ds_merged.sortby("observation_time")

# 🔁 In DataFrame umwandeln für besseren Plot
df = ds_merged[["dswrf1"]].to_dataframe().reset_index()
df["observation_time"] = pd.to_datetime(df["observation_time"])

# 📈 Plot
plt.figure(figsize=(12, 5))
plt.plot(df["observation_time"], df["dswrf1"], marker=".", linestyle="-", color="tab:blue")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.title("DSWRF1 Zeitreihe (nach observation_time, January 2022, Launch 01:00)")
plt.xlabel("Zeit")
plt.ylabel("DSWRF1 (W/m²)")
plt.grid(True)
plt.tight_layout()
plt.show()
 