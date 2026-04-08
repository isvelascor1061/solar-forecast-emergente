import xarray as xr
import fsspec
import os
import requests
from datetime import datetime, timedelta

# 🔹 Konstante für Medellín (Koordinaten)
MEDELLIN_LAT = slice(334, 336+1)   # Change this part to work with a given location, not fixed

MEDELLIN_LON =slice(1137, 1139+1)

# 🔹 Zielverzeichnis für die NetCDF-Dateien
SAVE_DIR = "raw_data"
os.makedirs(SAVE_DIR, exist_ok=True)  # Erstellt das Verzeichnis, falls es nicht existiert

# 🔹 Funktion zum Prüfen, ob eine Datei existiert
def file_exists(url):
    """
    Überprüft, ob eine Datei unter der angegebenen URL existiert.

    Parameter:
    ----------
    url : str
        Die URL zur GRIB2-Datei.

    Rückgabe:
    ---------
    bool
        True, wenn die Datei existiert, sonst False.
    """
    response = requests.head(url)
    return response.status_code == 200

# 🔹 Funktion zum Laden der GRIB2-Dateien direkt von NOAA-GFS über HTTPS
def load_dswrf_data_http(grib_url):
    """
    Lädt die DSWRF-Daten direkt von NOAA über HTTPS mit fsspec und gibt ein xarray.Dataset zurück.

    Parameter:
    ----------
    grib_url : str
        URL zur GRIB2-Datei im NOAA-Server.

    Rückgabe:
    ---------
    xarray.Dataset
        Das Dataset mit den geladenen DSWRF-Daten oder None, falls ein Fehler auftritt.
    """
    if not file_exists(grib_url):
        print(f"⚠️ Datei existiert nicht: {grib_url}")
        return None

    try:
        # Öffne die Datei direkt mit fsspec über HTTP
        with fsspec.open(grib_url, mode="rb") as file:
            ds = xr.open_dataset(file, engine="cfgrib")

        print("\n📊 Verfügbare Variablen im GRIB-Datensatz:")
        print(ds.data_vars)  # Gibt alle geladenen Variablen aus

        # Suche nach der DSWRF-Variable
        dswrf_var = next((v for v in ds.data_vars.keys() if "DSWRF" in v or "sdswrf" in v), None)

        if dswrf_var:
            print(f"✅ Gefundene DSWRF-Variable: {dswrf_var}")
            return ds[dswrf_var]
        else:
            print("⚠️ DSWRF-Variable nicht gefunden.")
            return None
    except Exception as e:
        print(f"❌ Fehler beim Laden der GRIB-Datei von {grib_url}: {e}")
        return None

# 🔹 Hauptskript
def main():
    """
    Lädt DSWRF-Daten für die nächsten 6 Zeitschritte aus GFS-Daten für Medellín und speichert sie als NetCDF.
    """
    start_time = "20190101"  # Beispielhaftes Datum
    forecast_hour = 0  # GFS Startzeit (00 UTC)

    # Nutze HTTPS statt S3!
    base_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{start_time}/{forecast_hour:02d}/atmos/gfs.t{forecast_hour:02d}z.pgrb2.0p25.f"

    for hour in range(1, 7):  # Nur die ersten 6 Stunden
        grib_url = f"{base_url}{hour:03d}"

        print(f"📥 Lade Daten für Stunde {hour} von {grib_url}...")

        try:
            ds = load_dswrf_data_http(grib_url)
            if ds is not None:
                # 🔹 Filtere die Daten nur für Medellín
                lat_idx = abs(ds.latitude - MEDELLIN_LAT).argmin()
                lon_idx = abs(ds.longitude - MEDELLIN_LON).argmin()
                ds_medellin = ds.isel(latitude=lat_idx, longitude=lon_idx)

                # 🔹 NetCDF-Dateinamen erstellen
                nc_filename = f"raw_data/00_DSWRF{hour}_{start_time}_0000_{start_time[:8]}{hour:02d}00_.nc"

                # 🔹 NetCDF speichern
                ds_medellin.to_netcdf(nc_filename)
                print(f"💾 Gespeichert als: {nc_filename}")

        except Exception as e:
            print(f"❌ Fehler beim Laden der Daten für Stunde {hour}: {e}")

if __name__ == "__main__":
    main()
