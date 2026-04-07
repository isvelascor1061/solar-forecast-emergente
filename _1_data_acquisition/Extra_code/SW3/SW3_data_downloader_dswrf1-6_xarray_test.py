import requests
import xarray as xr
import os

# 🔹 Koordinaten für Medellín

# 🔹 Zielverzeichnis für NetCDF-Dateien
SAVE_DIR = "raw_data"
os.makedirs(SAVE_DIR, exist_ok=True)  # Erstellt das Verzeichnis, falls es nicht existiert

# 🔹 Funktion zum Generieren des DSWRF-Variablennamens
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

def get_dswrf_interval(hour):
    """
    Berechnet das korrekte DSWRF-Intervall für die Vorhersagezeit.

    Parameter:
    ----------
    hour : int
        Die vorhergesagte Stunde.

    Rückgabe:
    ---------
    int
        Der DSWRF-Zeitraum, z. B. 6 für einen Durchschnitt über 6 Stunden.
    """
    if hour in [6, 12, 18, 24]:  
       return 6  # DSWRF6 für diese Hauptstunden
    elif hour in [1, 2, 3, 4, 5]:  
        return hour  # 0-1, 0-2, 0-3, 0-4, 0-5
    elif hour in [7, 8, 9, 10, 11]:  
        return hour - 6  # 6-7 → DSWRF1, 6-8 → DSWRF2, ...
    elif hour in [13, 14, 15, 16, 17]:  
        return hour - 12  # 12-13 → DSWRF1, 12-14 → DSWRF2, ...
    elif hour in [19, 20, 21, 22, 23]:  
        return hour - 18  # 18-19 → DSWRF1, 18-20 → DSWRF2, ...
    else:
        raise ValueError(f"❌ Ungültige Vorhersagestunde: {hour}")


# 🔹 Funktion zum Finden der DSWRF-Variable in der `.idx`-Datei
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
            print(line)

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
    possible_names = [name for name in ds.data_vars.keys() if "DSWRF" in name or "sdswrf" in name or "Short-Wave" in name]
    return possible_names[0] if possible_names else None

# 🔹 Funktion zum Laden der DSWRF-Daten für Medellín
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
        lat_min, lat_max = 6.0,6.50     # Nur 1.5° um Medellín herum
        lon_min, lon_max = -75.75,-75.25   # Nur 1° um Medellín herum
        if ds.longitude.min() <= 0:
            if lon_min < 0:
                lon_min += 360
            if lon_max < 0:
                lon_max += 360
        print("\n📊 Verfügbare Variablen im GRIB-Datensatz:")

        dswrf_var = find_dswrf_variable(ds)

        if dswrf_var:
            print(f"✅ Gefundene DSWRF-Variable: {dswrf_var}")

   
            ds_medellin_region = ds[dswrf_var].sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
           


            return ds_medellin_region
        else:
            print("⚠️ DSWRF-Variable nicht gefunden.")
            return None
    except Exception as e:
        print(f"❌ Fehler beim Laden der GRIB-Datei: {e}")
        return None

# 🔹 Hauptskript
def main():
    lat_min, lat_max = 6.0,6.50     # Nur 1.5° um Medellín herum
    lon_min, lon_max = -75.75,-75.25   # Nur 1° um Medellín herum
    if lon_min and lon_max <=0:
        lon_min += 360
        lon_max +=360
    start_time = "20250316"
    forecast_hour = 00  # Beispiel für GFS-Startzeit: 00, 06, 12 oder 18 UTC

    # URL-Basis für NOAA GFS-Daten
    base_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{start_time}/{forecast_hour:02d}/atmos/gfs.t{forecast_hour:02d}z.pgrb2.0p25.f"

    dswrf_data = []

    for hour in range(14, 15):
        grib_url = f"{base_url}{hour:03d}"
        idx_url = f"{grib_url}.idx"

        print(f"📥 Lade DSWRF-Daten für Stunde {hour}...")

        try:
            ds = load_dswrf_data(grib_url, idx_url, hour)
            
            if ds is not None:
                dswrf_data.append(ds)
                
                dswrf_interval = get_dswrf_interval(hour)
                # 🔹 NetCDF-Dateinamen erstellen mit `forecast_hour` vorne
                nc_filename = f"{SAVE_DIR}/{forecast_hour:02d}_DSWRF{dswrf_interval}_{start_time}_{forecast_hour:02d}00_{start_time[:8]}{hour:02d}00_.nc"
                
                # 🔹 NetCDF speichern
                ds.to_netcdf(nc_filename)
                print(f"💾 Gespeichert als: {nc_filename}")

        except Exception as e:
            print(f"❌ Fehler beim Laden der Daten für Stunde {hour}: {e}")
            

if __name__ == "__main__":
    try:
        main()
        
    finally:
        if os.path.exists("temp.grib2"):
            os.remove("temp.grib2")
