"""
Download GOES-16 COD (Cloud Optical Depth) single-pixel values and compute
CSI_SIATA for the training period 2021-04-01 to 2024-12-31.

Extracts the nearest pixel to the SIATA reference station in Medellin
for every daytime 10-minute timestamp in Colombia local time.

Output: cod_pixel_2021_2024.csv
Columns:
    timestamp_local  - Colombia local time (America/Bogota)
    cod_value        - COD at nearest GOES-16 pixel (NaN if unavailable)
    zenith           - Solar apparent zenith angle in degrees
    ghi_siata        - SIATA GHI at exact timestamp in W/m2 (NaN if missing)
    clear_sky_ghi    - Ineichen clear-sky GHI in W/m2
    csi_siata        - ghi_siata / clear_sky_ghi, clipped [0, 1]

Usage:
    python download_cod_pixel.py

Resolution note:
    - Before March 22 2023: COD Full Disk at 4 km/pixel
    - From March 22 2023 onwards: COD Full Disk at 2 km/pixel
    - For single pixel extraction both periods are treated identically
      (nearest pixel via scan angle argmin)

Satellite: GOES-16 | Product: ABI-L2-CODF
Colombia local time is UTC-5, so UTC = local + 5h.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
import pvlib
import s3fs
import xarray as xr
from goes2go import GOES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAT = 6.2591538
LON = -75.5884924
ALT = 1582  # meters, Medellin elevation

SATELLITE = 16
PRODUCT = "ABI-L2-CODF"

ZENITH_LIMIT = 82  # degrees — official daytime limit

# Save buffer to CSV every N timestamps to avoid data loss on interruption
SAVE_INTERVAL = 50

# Print progress every N timestamps
PRINT_INTERVAL = 100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "cod_pixel_2021_2024.csv")

SIATA_GHI_DIR = os.path.normpath(
    os.path.join(
        SCRIPT_DIR, "../..",
        "_3_Data_preparation_for_LSTM",
        "Preparation_data", "_03_Siata_GHI", "CSV_Siata_GHI",
    )
)

# Expected output columns — used to detect schema changes and reset the file
EXPECTED_COLUMNS = {
    "timestamp_local", "cod_value", "zenith",
    "ghi_siata", "clear_sky_ghi", "csi_siata",
}

# pvlib location (created once, reused for clear-sky and solar position)
LOCATION = pvlib.location.Location(
    latitude=LAT,
    longitude=LON,
    altitude=ALT,
    tz="America/Bogota",
)

# ---------------------------------------------------------------------------
# SIATA data loader
# ---------------------------------------------------------------------------

def load_siata_data():
    """
    Load all monthly SIATA GHI CSV files from SIATA_GHI_DIR into a single
    DataFrame indexed by naive local datetime at 1-minute resolution.

    Corrections applied:
        - Replace -9999 with NaN in radiacion
        - Set radiacion = NaN where calidad == -9999
        - Clip negative radiacion values to 0
    """
    csv_files = sorted(glob.glob(os.path.join(SIATA_GHI_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No SIATA CSV files found in:\n  {SIATA_GHI_DIR}")

    print(f"Loading {len(csv_files)} SIATA CSV files ...", end=" ", flush=True)

    frames = []
    for path in csv_files:
        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
            usecols=[0, 2, 3],  # datetime index, radiacion, calidad
        )
        frames.append(df)

    siata = pd.concat(frames).sort_index()

    # Apply corrections
    siata["radiacion"] = siata["radiacion"].replace(-9999, np.nan)
    siata.loc[siata["calidad"] == -9999, "radiacion"] = np.nan
    siata["radiacion"] = siata["radiacion"].clip(lower=0)

    print(f"done. {len(siata):,} rows | {siata.index[0]} to {siata.index[-1]}")
    return siata


# ---------------------------------------------------------------------------
# Output file schema check
# ---------------------------------------------------------------------------

def reset_if_schema_changed():
    """
    If the output CSV exists but has different columns than EXPECTED_COLUMNS,
    delete it so a fresh run begins with the correct schema.
    """
    if not os.path.exists(OUTPUT_FILE):
        return
    try:
        header = pd.read_csv(OUTPUT_FILE, nrows=0)
        if set(header.columns) != EXPECTED_COLUMNS:
            os.remove(OUTPUT_FILE)
            print("Output file had old schema -- deleted, starting fresh.")
    except Exception:
        os.remove(OUTPUT_FILE)
        print("Output file unreadable -- deleted, starting fresh.")


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def find_resume_timestamp():
    """
    If the output CSV exists, return the last timestamp already processed
    as a tz-aware Colombia timestamp. Otherwise return None.
    """
    if not os.path.exists(OUTPUT_FILE):
        return None
    try:
        existing = pd.read_csv(OUTPUT_FILE, usecols=["timestamp_local"])
        if existing.empty:
            return None
        last_ts = pd.to_datetime(existing["timestamp_local"]).max()
        return last_ts.tz_localize("America/Bogota")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Geostationary scan angle conversion
# ---------------------------------------------------------------------------

def latlon_to_scanangle(lat, lon, ds):
    """
    Convert a geographic lat/lon to the nearest (row, col) pixel index
    in a GOES-16 full-disk xarray dataset using the geostationary scan
    angle approximation. No full lat/lon grid is computed.
    """
    proj  = ds["goes_imager_projection"]
    lon0  = proj.attrs["longitude_of_projection_origin"]
    H     = proj.attrs["perspective_point_height"] + proj.attrs["semi_major_axis"]
    r_eq  = proj.attrs["semi_major_axis"]
    r_pol = proj.attrs["semi_minor_axis"]

    lat_r = np.radians(lat)
    lon_r = np.radians(lon - lon0)

    # Solve for rs (distance from satellite to point on ellipsoid)
    a = np.cos(lat_r)**2 + (r_eq / r_pol)**2 * np.sin(lat_r)**2
    b = -2 * H * np.cos(lat_r) * np.cos(lon_r)
    c = H**2 - r_eq**2

    rs = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # Satellite-centered Cartesian components
    sx = rs * np.cos(lat_r) * np.cos(lon_r)
    sy = -rs * np.sin(lon_r)
    sz = rs * np.cos(lat_r) * np.sin(lon_r)

    # Convert to scan angles (radians)
    x_scan = np.arctan(-sy / (H - sx))
    y_scan = np.arctan(sz / np.sqrt((H - sx)**2 + sy**2))

    col = int(np.argmin(np.abs(ds["x"].values - x_scan)))
    row = int(np.argmin(np.abs(ds["y"].values - y_scan)))

    return row, col


# ---------------------------------------------------------------------------
# Variable name detection
# ---------------------------------------------------------------------------

def try_variable_names(ds):
    """
    Try candidate variable names for COD in order of preference.
    Returns (name, DataArray) or raises KeyError if none found.
    """
    candidates = ["COD", "Optical_Depth_at_Band_2"]
    for name in candidates:
        if name in ds:
            return name, ds[name]

    # Fallback: first variable with float data
    for name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.floating):
            return name, var

    raise KeyError("No suitable COD variable found in dataset.")


# ---------------------------------------------------------------------------
# COD pixel download
# ---------------------------------------------------------------------------

def get_cod_pixel(ts_local):
    """
    Download GOES-16 COD for the timestamp nearest to ts_local and extract
    the single pixel value at the SIATA station location.

    Returns float or NaN.
    """
    ts_utc = ts_local.tz_convert("UTC")
    start  = (ts_utc - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")
    end    = (ts_utc + pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")

    try:
        G  = GOES(satellite=SATELLITE, product=PRODUCT)
        df = G.df(start=start, end=end)

        if df is None or len(df) == 0:
            return np.nan

        fs      = s3fs.S3FileSystem(anon=True)
        s3_path = df.iloc[0]["file"]

        with fs.open(s3_path, "rb") as f:
            ds = xr.open_dataset(f, engine="h5netcdf")

            var_name, cod_da = try_variable_names(ds)
            row, col = latlon_to_scanangle(LAT, LON, ds)

            value = float(cod_da.values[row, col])

            # Reject fill value
            fill_val = cod_da.attrs.get("_FillValue", None)
            if fill_val is not None and value == float(fill_val):
                return np.nan

            # Sanity range check
            if value <= 0 or value > 200:
                return np.nan

            return value

    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check for schema change and reset output file if needed
    reset_if_schema_changed()

    # --- Load SIATA data ---
    siata_df = load_siata_data()

    # --- Generate all daytime timestamps ---
    print("Generating daytime timestamps 2021-04-01 to 2024-12-31 ...", end=" ", flush=True)

    timestamps = pd.date_range(
        start="2021-04-01 00:00",
        end="2024-12-31 23:50",
        freq="10min",
        tz="America/Bogota",
    )

    solar_pos      = LOCATION.get_solarposition(timestamps)
    daytime_mask   = solar_pos["apparent_zenith"] <= ZENITH_LIMIT
    timestamps_day = timestamps[daytime_mask]
    zenith_day     = solar_pos.loc[daytime_mask, "apparent_zenith"]

    print(f"done. {len(timestamps_day):,} daytime timestamps.")

    # --- Resume logic ---
    last_ts = find_resume_timestamp()

    if last_ts is not None:
        resume_mask     = timestamps_day > last_ts
        timestamps_todo = timestamps_day[resume_mask]
        zenith_todo     = zenith_day[resume_mask]
        print(f"Resuming from {last_ts}")
    else:
        timestamps_todo = timestamps_day
        zenith_todo     = zenith_day
        print("Starting fresh")

    total_todo = len(timestamps_todo)
    if total_todo == 0:
        print("Nothing to do -- all timestamps already processed.")
        return

    print(f"Timestamps remaining: {total_todo:,}\n")

    # --- Main loop ---
    buffer      = []
    first_write = not os.path.exists(OUTPUT_FILE)
    t_start     = time.time()

    for i, (ts, zen) in enumerate(zip(timestamps_todo, zenith_todo)):

        # 1. COD pixel from GOES-16
        cod = get_cod_pixel(ts)

        # 2. SIATA GHI at exact timestamp (naive index lookup)
        ts_naive  = ts.replace(tzinfo=None)
        ghi_siata = np.nan
        if ts_naive in siata_df.index:
            ghi_siata = siata_df.at[ts_naive, "radiacion"]

        # 3. Ineichen clear-sky GHI
        cs        = LOCATION.get_clearsky(
            pd.DatetimeIndex([ts]),
            model="ineichen",
            linke_turbidity=3.0,
        )
        clear_sky = float(cs["ghi"].iloc[0])

        # 4. CSI
        if clear_sky > 0 and not np.isnan(ghi_siata):
            csi_siata = float(np.clip(ghi_siata / clear_sky, 0.0, 1.0))
        else:
            csi_siata = np.nan

        buffer.append({
            "timestamp_local": ts.strftime("%Y-%m-%d %H:%M"),
            "cod_value":       cod,
            "zenith":          round(float(zen), 4),
            "ghi_siata":       ghi_siata,
            "clear_sky_ghi":   round(clear_sky, 4),
            "csi_siata":       csi_siata,
        })

        # Flush buffer to CSV every SAVE_INTERVAL timestamps
        if len(buffer) >= SAVE_INTERVAL:
            df_buf = pd.DataFrame(buffer)
            df_buf.to_csv(
                OUTPUT_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False
            buffer = []

        # Progress print every PRINT_INTERVAL timestamps
        if (i + 1) % PRINT_INTERVAL == 0:
            elapsed = time.time() - t_start
            rate    = elapsed / (i + 1)
            eta_min = rate * (total_todo - i - 1) / 60

            cod_str = f"{cod:.1f}" if not np.isnan(cod)      else "NaN"
            csi_str = f"{csi_siata:.2f}" if not np.isnan(csi_siata) else "NaN"

            print(
                f"[{i+1:>6}/{total_todo}] "
                f"{ts.strftime('%Y-%m-%d %H:%M')} | "
                f"COD={cod_str:>5} | "
                f"CSI={csi_str} | "
                f"zenith={zen:>5.1f} | "
                f"ETA {eta_min:.0f} min"
            )

    # Flush remaining buffer
    if buffer:
        df_buf = pd.DataFrame(buffer)
        df_buf.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

    # --- Final summary ---
    print(f"\n{'='*60}")
    print("Download complete.")

    result    = pd.read_csv(OUTPUT_FILE)
    total     = len(result)
    valid_cod = result["cod_value"].notna().sum()
    valid_csi = result["csi_siata"].notna().sum()

    print(f"Total timestamps processed : {total:,}")
    print(f"Valid COD values           : {valid_cod:,}  ({100*valid_cod/total:.1f}%)")
    print(f"Valid CSI (SIATA)          : {valid_csi:,}  ({100*valid_csi/total:.1f}%)")
    print(f"NaN COD                    : {total-valid_cod:,}  ({100*(total-valid_cod)/total:.1f}%)")
    print(f"NaN CSI                    : {total-valid_csi:,}  ({100*(total-valid_csi)/total:.1f}%)")
    print(f"Date range                 : {result['timestamp_local'].min()} to {result['timestamp_local'].max()}")
    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"Output file size           : {size_mb:.2f} MB")
    print(f"Output file                : {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
