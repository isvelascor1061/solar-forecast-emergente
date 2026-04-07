


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import xarray as xr
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
nc_path   = Path("_1_data_acquisition/_01_raw_rad_data/dswrf/raw_dswrf_0100/"
                 "2021-04-02_01_12_DSWRF6_2021-04-02_1300_UTCSTART:2021-04-02_0600.nc")
zoom      = 10        # OSM-Zoomstufe
cell_half = 0.125     # halbe Zellbreite/-höhe (0.25°-GFS)

# --------------------------------------------------
ds   = xr.open_dataset(nc_path, engine="netcdf4")
var  = ds["sdswrf"]
launch_time =ds["time"].values
launch_str=pd.to_datetime(launch_time).strftime("%Y-%m-%d %H:%M COT")
vt = ds["valid_time"].values          # kann 0-D oder 1-D sein
if np.ndim(vt) == 0:                  # 0-D → direkt verwenden
    time_str = pd.to_datetime(vt).strftime("%Y-%m-%d %H:%M COT")
else:                                 # 1-D → erstes Element
    time_str = pd.to_datetime(vt[0]).strftime("%Y-%m-%d %H:%M COT")

# Koordinaten (lat/lon) robust auslesen
lat_name = "latitude"  if "latitude"  in var.coords else "lat"
lon_name = "longitude" if "longitude" in var.coords else "lon"
lat = float(var[lat_name].item())
lon = float(var[lon_name].item())
val = float(var.values.squeeze())

# 2×2-Kanten­gitter + 1×1-Matrix
lat_edges = np.array([lat - cell_half, lat + cell_half])
lon_edges = np.array([lon - cell_half, lon + cell_half])
lon_edges_2d, lat_edges_2d = np.meshgrid(lon_edges, lat_edges)
raster = np.array([[val]])

# --------------------------------------------------
osm_tiles = cimgt.OSM()
proj      = osm_tiles.crs
fig = plt.figure(figsize=(10, 9))
ax  = fig.add_subplot(1, 1, 1, projection=proj)
ax.add_image(osm_tiles, zoom, alpha=0.7)

# Kartenfenster etwas größer
margin = 0.35
ax.set_extent([lon - margin, lon + margin, lat - margin, lat + margin])

# Rasterplot
mesh = ax.pcolormesh(lon_edges_2d, lat_edges_2d, raster,
                     cmap="inferno", alpha=0.8,
                     transform=ccrs.PlateCarree())

# Marker in der Mitte
ax.plot(lon, lat, 'ro', transform=ccrs.PlateCarree(), markersize=6)

# ---------- Gridlines ----------
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}

# Farbskala & Titel
cbar = plt.colorbar(mesh, ax=ax, shrink=0.7, pad=0.03)
cbar.set_label("DSWRF6 (W m⁻²)")
plt.title("DSWRF6 around Medellin(7-13h)\n"
          f"– Observation time: {time_str}\n"
          f"-Launch time: {launch_str}", pad=14)

plt.tight_layout()
plt.show()
