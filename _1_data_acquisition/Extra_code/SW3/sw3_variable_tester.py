import requests
import xarray as xr
import os

# 🔹 Funktion zur Generierung des korrekten DSWRF-Variablennamens
def get_dswrf_variable_name(timestep):
    match timestep:
        case 1 | 2 | 3 | 4 | 5:  
            return f"DSWRF:surface:0-{timestep} hour ave fcst"
        case 6:
            return "DSWRF:surface:0-6 hour ave fcst"
        case 7 | 8 | 9 | 10 | 11:
            return f"DSWRF:surface:6-{timestep} hour ave fcst"
        case 12:
            return "DSWRF:surface:6-12 hour ave fcst"
        case 13 | 14 | 15 | 16 | 17:
            return f"DSWRF:surface:12-{timestep} hour ave fcst"
        case 18:
            return "DSWRF:surface:12-18 hour ave fcst"
        case 19 | 20 | 21 | 22 | 23:
            return f"DSWRF:surface:18-{timestep} hour ave fcst"
        case _:
            raise ValueError(f"❌ Ungültiger Timestep: {timestep}")

# 🔹 Funktion zum Finden der korrekten DSWRF-Variable in der `.idx`-Datei
def find_dswrf_in_idx(idx_url, timestep):
    dswrf_name = get_dswrf_variable_name(timestep)

    response = requests.get(idx_url)
    if response.status_code != 200:
        raise Exception(f"❌ Fehler beim Laden der .idx-Datei: {response.status_code}")

    lines = response.text.splitlines()
    dswrf_offsets = None

    print("\n📄 Verfügbare DSWRF-Einträge in der .idx-Datei:")
    for line in lines:
        if "DSWRF" in line:
            print(line)  # Zeige alle DSWRF-Einträge zur Überprüfung

    for i, line in enumerate(lines):
        if dswrf_name in line:
            parts = line.split(":")
            try:
                start_byte = int(parts[1])
                end_byte = int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None
                dswrf_offsets = (start_byte, end_byte)
                break
            except (IndexError, ValueError) as e:
                print(f"⚠️ Fehler beim Parsen der Zeile: {line}. Fehler: {e}")
                continue

    if not dswrf_offsets:
        raise Exception(f"❌ DSWRF-Variable '{dswrf_name}' nicht in der .idx-Datei gefunden.")

    return dswrf_offsets

# 🔹 Funktion zum Finden der richtigen DSWRF-Variable im GRIB2-File
def find_dswrf_variable(ds):
    """
    Sucht nach der DSWRF-Variable in einem xarray.Dataset.
    Falls DSWRF nicht exakt existiert, prüft die Funktion auch alternative Namen.
    """
    possible_names = [name for name in ds.data_vars.keys() if "DSWRF" in name or "sdswrf" in name or "Short-Wave" in name]
    return possible_names[0] if possible_names else None

# 🔹 Funktion zum Laden der DSWRF-Daten aus dem GRIB2-File
def load_dswrf_data(grib_url, idx_url, timestep):
    dswrf_offsets = find_dswrf_in_idx(idx_url, timestep)
    if not dswrf_offsets:
        raise Exception("❌ Keine gültigen DSWRF-Offsets gefunden.")

    headers = {"Range": f"bytes={dswrf_offsets[0]}-{dswrf_offsets[1] if dswrf_offsets[1] else ''}"}
    response = requests.get(grib_url, headers=headers)
    if response.status_code not in [206, 200]:
        raise Exception(f"❌ Fehler beim HTTP Range Request: {response.status_code}")

    temp_file = "temp.grib2"
    with open(temp_file, "wb") as f:
        f.write(response.content)

    try:
        ds = xr.open_dataset(temp_file, engine="cfgrib")
        print("\n📊 Verfügbare Variablen im GRIB-Datensatz:")
        print(ds.data_vars)  # Gibt alle geladenen Variablen aus

        dswrf_var = find_dswrf_variable(ds)

        if dswrf_var:
            print(f"✅ Gefundene DSWRF-Variable: {dswrf_var}")
            return ds[dswrf_var]
        else:
            print("⚠️ DSWRF-Variable nicht gefunden.")
            return None
    except Exception as e:
        print(f"❌ Fehler beim Laden der GRIB-Datei: {e}")
        return None
  

# 🔹 Hauptskript
def main():
    start_time = "20250310"
    forecast_hour = 0  
    base_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{start_time}/{forecast_hour:02d}/atmos/gfs.t{forecast_hour:02d}z.pgrb2.0p25.f"

    dswrf_data = []

    for hour in range(1, 7):
        grib_url = f"{base_url}{hour:03d}"
        idx_url = f"{grib_url}.idx"

        print(f"📥 Lade Daten für Stunde {hour}...")
        try:
            ds = load_dswrf_data(grib_url, idx_url, hour)
            if ds is not None:
                dswrf_data.append(ds)
        except Exception as e:
            print(f"❌ Fehler beim Laden der Daten für Stunde {hour}: {e}")

    if dswrf_data:
        combined_ds = xr.concat(dswrf_data, dim="time")
        print("\n📊 Kombinierte DSWRF-Daten:")
        print(combined_ds)
    else:
        print("⚠️ Keine DSWRF-Daten gefunden.")

# 🔹 Skript ausführen
if __name__ == "__main__":
    try:
        main()
    finally:
        if os.path.exists("temp.grib2"):
            os.remove("temp.grib2")

import matplotlib.pyplot as plt


