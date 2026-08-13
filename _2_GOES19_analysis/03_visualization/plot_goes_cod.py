"""
Visualize GOES-16 Cloud Optical Depth (COD) over the Valle de Aburrá.

Generates a single 3x3 figure with 9 panels, one per 10-minute step
starting from a user-supplied local Colombia time.

Usage:
    python plot_goes_cod.py --date 2023-10-15 --start "09:00" --save
    python plot_goes_cod.py --date 2023-11-20 --start "14:30" --save

Arguments:
    --date   YYYY-MM-DD   Date to visualize (required)
    --start  HH:MM        Local Colombia start time (required)
                          9 panels are generated every 10 minutes:
                          start, start+10, start+20 ... start+80
    --save                Save the figure to outputs/ (optional)

Satellite: GOES-16 | Product: ABI-L2-CODF
Colombia local time is UTC-5, so UTC = local + 5h.
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from goes2go import GOES

# -- RESOLUTION NOTE ----------------------------------------------------------
# COD Full Disk: was 4 km/pixel before March 22, 2023
#                changed to 2 km/pixel on March 22, 2023
# ACM Full Disk: always 2 km/pixel (no change)
# BCM Full Disk: always 2 km/pixel (no change)
# For training data 2021-2024, COD needs resampling
# to a common 2km resolution for the pre-March 2023 period.
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fixed configuration (location, product, colormap)
# ---------------------------------------------------------------------------

LAT_CENTER = 6.2591538
LON_CENTER = -75.5884924

LAT_MIN = LAT_CENTER - 0.45
LAT_MAX = LAT_CENTER + 0.45
LON_MIN = LON_CENTER - 0.46
LON_MAX = LON_CENTER + 0.46

SATELLITE  = 16
PRODUCT    = "ABI-L2-CODF"
UTC_OFFSET = 5  # Colombia is UTC-5 -> UTC = local + 5

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COD_CMAP = "YlOrRd"
COD_VMIN = 0
COD_VMAX = 25

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def local_to_utc(date_str: str, hour: int, minute: int) -> datetime:
    """Convert a Colombia local time (UTC-5) to UTC datetime."""
    year, month, day = map(int, date_str.split("-"))
    local_dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return local_dt + timedelta(hours=UTC_OFFSET)


def format_local_time(hour: int, minute: int) -> str:
    """Return a human-readable local time string, e.g. '9:00 AM'."""
    dt = datetime(2000, 1, 1, hour, minute)
    return dt.strftime("%I:%M %p").lstrip("0") or "12:00 AM"


def try_variable_names(ds):
    """
    Try candidate variable names for COD in order.
    Returns (name, DataArray) or raises KeyError if none found.
    """
    candidates = ["COD", "Optical_Depth_at_Band_2"]
    for name in candidates:
        if name in ds:
            return name, ds[name]

    # Fallback: first variable that holds float data
    for name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.floating):
            return name, var

    raise KeyError("No suitable COD variable found in dataset.")


def get_latlon(ds):
    """
    Extract 2-D latitude and longitude arrays from a GOES xarray dataset.

    Tries goes2go's accessor first, then falls back to manual calculation
    from the geostationary projection parameters stored in the file.
    Returns (lats, lons) as numpy arrays, or (None, None) on failure.
    """
    # Preferred: goes2go accessor
    try:
        lats, lons = ds.GOES.latlon()
        return np.array(lats), np.array(lons)
    except Exception:
        pass

    # Fallback: manual calculation from geostationary projection
    try:
        proj_info = ds["goes_imager_projection"]
        lon_origin = proj_info.attrs["longitude_of_projection_origin"]
        H   = proj_info.attrs["perspective_point_height"] + proj_info.attrs["semi_major_axis"]
        r_eq  = proj_info.attrs["semi_major_axis"]
        r_pol = proj_info.attrs["semi_minor_axis"]

        x = ds["x"].values  # scan angle in radians
        y = ds["y"].values

        X, Y = np.meshgrid(x, y)

        a = (np.sin(X)**2 +
             np.cos(X)**2 * (np.cos(Y)**2 + (r_eq / r_pol)**2 * np.sin(Y)**2))
        b = -2 * H * np.cos(X) * np.cos(Y)
        c = H**2 - r_eq**2

        discriminant = b**2 - 4 * a * c
        rs = (-b - np.sqrt(np.where(discriminant >= 0, discriminant, np.nan))) / (2 * a)

        sx = rs * np.cos(X) * np.cos(Y)
        sy = -rs * np.sin(X)
        sz = rs * np.cos(X) * np.sin(Y)

        lats = np.degrees(np.arctan((r_eq / r_pol)**2 * sz / np.sqrt((H - sx)**2 + sy**2)))
        lons = lon_origin - np.degrees(np.arctan(sy / (H - sx)))

        return lats, lons
    except Exception as e:
        print(f"    lat/lon error: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_cod(utc_dt: datetime):
    """
    Download GOES-16 ABI-L2-CODF nearest to utc_dt and return cropped arrays.

    Crops to the study domain before returning so that cartopy only transforms
    ~50x50 pixels instead of ~2713x2713, avoiding memory allocation errors.

    Returns
    -------
    (cod_vals, lons, lats) numpy arrays cropped to the study domain,
    or None if download or processing fails.
    """
    try:
        G = GOES(satellite=SATELLITE, product=PRODUCT)

        ts    = pd.Timestamp(utc_dt)
        start = (ts - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")
        end   = (ts + pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")

        print(f"    Searching files {start} - {end} UTC ...", end=" ", flush=True)

        df = G.df(start=start, end=end)
        if df is None or len(df) == 0:
            print("no files found.")
            return None

        print(f"{len(df)} file(s) found. Opening from S3 ...", end=" ", flush=True)

        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        s3_path = df.iloc[0]["file"]

        with fs.open(s3_path, "rb") as f:
            ds = xr.open_dataset(f, engine="h5netcdf")
            print("done.")

            var_name, cod_da = try_variable_names(ds)
            print(f"    Variable used: '{var_name}'")

            lats, lons = get_latlon(ds)
            if lats is None:
                print("    Could not determine lat/lon.")
                return None

            # Find row/col indices within domain bounding box
            row_mask = np.any((lats >= LAT_MIN) & (lats <= LAT_MAX), axis=1)
            col_mask = np.any((lons >= LON_MIN) & (lons <= LON_MAX), axis=0)

            if not row_mask.any() or not col_mask.any():
                print("    WARNING: domain mask returned no pixels.")
                return None

            cod_vals = cod_da.values.astype(float)

            # Crop all arrays to domain bounding box before passing to cartopy
            lats_crop = lats[np.ix_(row_mask, col_mask)]
            lons_crop = lons[np.ix_(row_mask, col_mask)]
            cod_crop  = cod_vals[np.ix_(row_mask, col_mask)]

            # Apply precise domain mask within the cropped region
            mask_crop = (
                (lats_crop >= LAT_MIN) & (lats_crop <= LAT_MAX) &
                (lons_crop >= LON_MIN) & (lons_crop <= LON_MAX)
            )
            cod_crop[~mask_crop] = np.nan

            return cod_crop, lons_crop, lats_crop

    except Exception as exc:
        print(f"FAILED ({exc})")
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sequence(slug: str, label: str, local_times: list,
                  date_str: str, save: bool):
    """
    Build a 3x3 panel figure for the given date and list of local times.

    Parameters
    ----------
    slug        : filename slug, e.g. "09-00"
    label       : figure title label
    local_times : list of 9 (hour, minute) tuples in Colombia local time
    date_str    : 'YYYY-MM-DD'
    save        : whether to write the figure to disk
    """
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        3, 3,
        figsize=(14, 14),
        subplot_kw={"projection": proj},
        constrained_layout=False,
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.05, bottom=0.08, top=0.93)

    fig.suptitle(
        f"GOES-16 Cloud Optical Depth | {label}",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )

    print(f"\n{'='*60}")
    print(f"COD | {label}")
    print(f"{'='*60}")

    pcm_ref = None

    for idx, (hour, minute) in enumerate(local_times):
        ax = axes.flat[idx]
        utc_dt     = local_to_utc(date_str, hour, minute)
        time_label = format_local_time(hour, minute)

        print(f"\n  Panel {idx+1}/9 - {time_label} local ({utc_dt.strftime('%H:%M')} UTC):")

        result = download_cod(utc_dt)

        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
        ax.add_feature(
            cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "50m"),
            linewidth=0.5, edgecolor="steelblue", facecolor="none",
        )

        if result is not None:
            cod_vals, lons_crop, lats_crop = result

            valid     = np.isfinite(lons_crop) & np.isfinite(lats_crop) & np.isfinite(cod_vals)
            cod_plot  = np.where(valid, cod_vals, np.nan)
            lons_plot = np.where(np.isfinite(lons_crop), lons_crop, 0.0)
            lats_plot = np.where(np.isfinite(lats_crop), lats_crop, 0.0)

            if valid.any():
                pcm = ax.pcolormesh(
                    lons_plot, lats_plot, cod_plot,
                    cmap=COD_CMAP, vmin=COD_VMIN, vmax=COD_VMAX,
                    transform=proj, shading="auto",
                )
                pcm_ref = pcm
            else:
                ax.set_facecolor("#d0d0d0")
                ax.text(0.5, 0.5, "No data\navailable",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="gray")
        else:
            ax.set_facecolor("#d0d0d0")
            ax.text(0.5, 0.5, "No data available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#444444")

        # Red star at SIATA station
        ax.plot(LON_CENTER, LAT_CENTER,
                marker="*", color="red", markersize=10,
                transform=proj, zorder=5)

        ax.set_title(time_label, fontsize=10, pad=4)

        gl = ax.gridlines(draw_labels=False,
                          linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
        gl.xlocator = mticker.FixedLocator(
            np.arange(round(LON_MIN, 1) - 0.1, LON_MAX + 0.1, 0.2))
        gl.ylocator = mticker.FixedLocator(
            np.arange(round(LAT_MIN, 1) - 0.1, LAT_MAX + 0.1, 0.2))

    # Shared horizontal colorbar
    cbar_ax = fig.add_axes([0.15, 0.04, 0.70, 0.018])
    if pcm_ref is not None:
        cb = fig.colorbar(pcm_ref, cax=cbar_ax, orientation="horizontal")
    else:
        norm = mcolors.Normalize(vmin=COD_VMIN, vmax=COD_VMAX)
        sm   = plt.cm.ScalarMappable(cmap=COD_CMAP, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Cloud Optical Depth - higher = denser cloud", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    if save:
        out_path = os.path.join(OUTPUT_DIR, f"COD_GOES16_{date_str}_{slug}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"\n  Saved -> {out_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot GOES-16 COD over Valle de Aburra for a given date and start time."
    )
    parser.add_argument("--date", required=True,
                        help="Date in YYYY-MM-DD format")
    parser.add_argument("--start", required=True,
                        help="Start time in HH:MM local Colombia time")
    parser.add_argument("--save", action="store_true",
                        help="Save the output image")
    args = parser.parse_args()

    start_hour, start_min = map(int, args.start.split(":"))
    local_times = []
    for i in range(9):
        total_min = start_hour * 60 + start_min + i * 10
        local_times.append((total_min // 60, total_min % 60))

    slug  = args.start.replace(":", "-")
    label = f"{args.date} from {args.start} (local Colombia)"

    plot_sequence(slug, label, local_times, args.date, args.save)
