"""
Visualize GOES-16 ABI Clear Sky Mask (ACM) over the Valle de Aburrá.

Generates a single 3x3 figure with 9 panels, one per 10-minute step
starting from a user-supplied local Colombia time.

Usage:
    python plot_goes_acm.py --date 2023-10-15 --start "09:00" --save
    python plot_goes_acm.py --date 2023-11-20 --start "14:30" --save

Arguments:
    --date   YYYY-MM-DD   Date to visualize (required)
    --start  HH:MM        Local Colombia start time (required)
                          9 panels are generated every 10 minutes:
                          start, start+10, start+20 ... start+80
    --save                Save the figure to outputs/ (optional)

Satellite: GOES-16 | Product: ABI-L2-ACMF
Colombia local time is UTC-5, so UTC = local + 5h.

ACM categories:
  0 = Clear
  1 = Probably Clear
  2 = Probably Cloudy
  3 = Cloudy
"""

import os
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
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
PRODUCT    = "ABI-L2-ACMF"
UTC_OFFSET = 5  # Colombia is UTC-5 -> UTC = local + 5

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ACM: 4-category discrete colormap
ACM_COLORS = ["#2ECC71", "#FED7AA", "#FB923C", "#C2410C"]
ACM_LABELS = ["Clear", "Prob. Clear", "Prob. Cloudy", "Cloudy"]
ACM_TICKS  = [0, 1, 2, 3]
ACM_CMAP   = mcolors.ListedColormap(ACM_COLORS)
ACM_NORM   = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], ACM_CMAP.N)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def local_to_utc(date_str: str, hour: int, minute: int) -> datetime:
    """Convert a Colombia local time (UTC-5) to UTC datetime.

    date_str is the panel-specific date (may differ from the start date
    when the sequence crosses midnight).
    """
    local_dt = datetime.strptime(
        f"{date_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M"
    )
    return local_dt + timedelta(hours=UTC_OFFSET)


def format_local_time(hour: int, minute: int) -> str:
    """Return a human-readable local time string, e.g. '9:00 AM'."""
    dt = datetime(2000, 1, 1, hour, minute)
    return dt.strftime("%I:%M %p").lstrip("0") or "12:00 AM"


def try_variable_names(ds):
    """
    Try candidate variable names for ACM in order.
    Returns (name, DataArray) or raises KeyError if none found.
    """
    # Prefer the 4-category ACM variable over the binary BCM variable
    candidates = ["ACM", "BCM"]
    for name in candidates:
        if name in ds:
            return name, ds[name]

    # Fallback: first variable that holds integer or float data
    for name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.integer) or np.issubdtype(var.dtype, np.floating):
            return name, var

    raise KeyError("No suitable ACM variable found in dataset.")


def get_latlon_domain(ds, lat_min, lat_max, lon_min, lon_max, buffer=1.0):
    """
    Return lat/lon arrays cropped to the study domain, plus 1-D row/col indices
    so the caller can index the data variable the same way.

    Memory-safe strategy:
      1. Try goes2go accessor (may OOM on large grids -> caught).
         If it succeeds, convert to float32 immediately, subset, then free
         the full-disk arrays.
      2. Fallback: use 1-D scan-angle approximations (no 2-D allocation) to
         find which rows/cols fall inside the domain, then build a small
         meshgrid only for that subset.

    Returns
    -------
    (lats_sub, lons_sub, row_idx, col_idx) - all for the domain subset
    or (None, None, None, None) on failure.
    """
    lats_sub = lons_sub = row_idx = col_idx = None

    # --- attempt 1: goes2go accessor ---
    try:
        lats_full, lons_full = ds.GOES.latlon()
        lats_full = np.asarray(lats_full, dtype=np.float32)
        lons_full = np.asarray(lons_full, dtype=np.float32)

        row_mask = np.any(
            (lats_full >= lat_min - buffer) & (lats_full <= lat_max + buffer), axis=1
        )
        col_mask = np.any(
            (lons_full >= lon_min - buffer) & (lons_full <= lon_max + buffer), axis=0
        )
        r_idx = np.where(row_mask)[0]
        c_idx = np.where(col_mask)[0]

        if len(r_idx) > 0 and len(c_idx) > 0:
            lats_sub = lats_full[np.ix_(r_idx, c_idx)]
            lons_sub = lons_full[np.ix_(r_idx, c_idx)]
            row_idx = r_idx
            col_idx = c_idx

        del lats_full, lons_full  # free full-disk memory immediately
    except Exception:
        pass

    if lats_sub is not None:
        return lats_sub, lons_sub, row_idx, col_idx

    # --- attempt 2: memory-safe manual geostationary projection ---
    try:
        proj_info = ds["goes_imager_projection"]
        lon_origin = float(proj_info.attrs["longitude_of_projection_origin"])
        H = float(proj_info.attrs["perspective_point_height"]) + \
            float(proj_info.attrs["semi_major_axis"])
        r_eq  = float(proj_info.attrs["semi_major_axis"])
        r_pol = float(proj_info.attrs["semi_minor_axis"])

        x_full = ds["x"].values.astype(np.float64)  # 1-D, e.g. 5424 elements
        y_full = ds["y"].values.astype(np.float64)  # 1-D

        c_scalar = H**2 - r_eq**2  # constant term in quadratic

        # Approximate lon as a function of x alone (evaluated at Y=0):
        #   a=1, b=-2H*cos(x), c=H^2-r_eq^2
        b_x   = -2.0 * H * np.cos(x_full)
        disc_x = b_x**2 - 4.0 * c_scalar
        rs_x  = (-b_x - np.sqrt(np.maximum(disc_x, 0.0))) / 2.0
        sy_x  = -rs_x * np.sin(x_full)
        sx_x  =  rs_x * np.cos(x_full)
        approx_lons = lon_origin - np.degrees(np.arctan2(sy_x, H - sx_x))

        # Approximate lat as a function of y alone (evaluated at X=0):
        #   a=cos^2(y)+(r_eq/r_pol)^2*sin^2(y), b=-2H*cos(y), c=H^2-r_eq^2
        a_y   = np.cos(y_full)**2 + (r_eq / r_pol)**2 * np.sin(y_full)**2
        b_y   = -2.0 * H * np.cos(y_full)
        disc_y = b_y**2 - 4.0 * a_y * c_scalar
        rs_y  = (-b_y - np.sqrt(np.maximum(disc_y, 0.0))) / (2.0 * a_y)
        sx_y  = rs_y * np.cos(y_full)
        sz_y  = rs_y * np.sin(y_full)
        approx_lats = np.degrees(
            np.arctan2((r_eq / r_pol)**2 * sz_y, H - sx_y)
        )

        # Find index ranges with buffer (generous to account for approx error)
        col_idx = np.where(
            (approx_lons >= lon_min - buffer) & (approx_lons <= lon_max + buffer)
        )[0]
        row_idx = np.where(
            (approx_lats >= lat_min - buffer) & (approx_lats <= lat_max + buffer)
        )[0]

        if len(row_idx) == 0 or len(col_idx) == 0:
            print("    WARNING: 1-D approximation found no domain pixels.")
            return None, None, None, None

        x_sub = x_full[col_idx]
        y_sub = y_full[row_idx]

        # Build the small subset meshgrid (e.g. ~100x100 instead of 5424x5424)
        X, Y = np.meshgrid(x_sub, y_sub)

        a = (np.sin(X)**2 +
             np.cos(X)**2 * (np.cos(Y)**2 + (r_eq / r_pol)**2 * np.sin(Y)**2))
        b = -2.0 * H * np.cos(X) * np.cos(Y)
        discriminant = b**2 - 4.0 * a * c_scalar
        rs = (-b - np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * a)

        sx = rs * np.cos(X) * np.cos(Y)
        sy = -rs * np.sin(X)
        sz = rs * np.cos(X) * np.sin(Y)

        lats_sub = np.degrees(
            np.arctan2((r_eq / r_pol)**2 * sz, np.sqrt((H - sx)**2 + sy**2))
        )
        lons_sub = lon_origin - np.degrees(np.arctan2(sy, H - sx))

        return lats_sub, lons_sub, row_idx, col_idx

    except Exception as e:
        print(f"    lat/lon error: {e}")
        return None, None, None, None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_acm(utc_dt: datetime):
    """
    Download GOES-16 ABI-L2-ACMF nearest to utc_dt and return cropped arrays.

    Only the domain subset (~100x100 pixels) is ever held in memory --
    the full-disk geostationary grid (5424x5424) is never fully allocated,
    avoiding the 'Unable to allocate 224 MiB' error.

    Returns
    -------
    (acm_vals, lons, lats) cropped to the study domain, or None on failure.
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

            var_name, acm_da = try_variable_names(ds)
            print(f"    Variable used: '{var_name}'")

            lats_sub, lons_sub, row_idx, col_idx = get_latlon_domain(
                ds, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
            )
            if lats_sub is None:
                print("    Could not determine lat/lon for domain.")
                return None

            # Index the data variable using the same row/col indices
            acm_crop = acm_da.values.astype(float)[np.ix_(row_idx, col_idx)]

            # Apply precise geographic mask within the cropped region
            mask_crop = (
                (lats_sub >= LAT_MIN) & (lats_sub <= LAT_MAX) &
                (lons_sub >= LON_MIN) & (lons_sub <= LON_MAX)
            )
            acm_crop[~mask_crop] = np.nan

            return acm_crop, lons_sub, lats_sub

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
    local_times : list of 9 (date_str, hour, minute) tuples in Colombia local time;
                  date_str per panel may differ from the start date on midnight rollover
    date_str    : 'YYYY-MM-DD' of the start time (used for the figure title/filename)
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
        f"GOES-16 Clear Sky Mask (ACM) | {label}",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )

    print(f"\n{'='*60}")
    print(f"ACM | {label}")
    print(f"{'='*60}")

    pcm_ref = None

    for idx, (panel_date, hour, minute) in enumerate(local_times):
        ax = axes.flat[idx]
        utc_dt     = local_to_utc(panel_date, hour, minute)
        time_label = format_local_time(hour, minute)

        print(f"\n  Panel {idx+1}/9 - {time_label} local ({utc_dt.strftime('%H:%M')} UTC):")

        result = download_acm(utc_dt)

        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
        ax.add_feature(
            cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "50m"),
            linewidth=0.5, edgecolor="steelblue", facecolor="none",
        )

        if result is not None:
            acm_vals, lons_crop, lats_crop = result

            valid     = np.isfinite(lons_crop) & np.isfinite(lats_crop) & np.isfinite(acm_vals)
            acm_plot  = np.where(valid, acm_vals, np.nan)
            lons_plot = np.where(np.isfinite(lons_crop), lons_crop, 0.0)
            lats_plot = np.where(np.isfinite(lats_crop), lats_crop, 0.0)

            if valid.any():
                pcm = ax.pcolormesh(
                    lons_plot, lats_plot, acm_plot,
                    cmap=ACM_CMAP, norm=ACM_NORM,
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

    # Shared horizontal colorbar with discrete ACM category labels
    cbar_ax = fig.add_axes([0.15, 0.04, 0.70, 0.018])
    if pcm_ref is not None:
        cb = fig.colorbar(pcm_ref, cax=cbar_ax, orientation="horizontal")
    else:
        sm = plt.cm.ScalarMappable(cmap=ACM_CMAP, norm=ACM_NORM)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks(ACM_TICKS)
    cb.set_ticklabels(ACM_LABELS)
    cb.set_label("Clear Sky Mask category", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    if save:
        out_path = os.path.join(OUTPUT_DIR, f"ACM_GOES16_{date_str}_{slug}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"\n  Saved -> {out_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot GOES-16 ACM over Valle de Aburra for a given date and start time."
    )
    parser.add_argument("--date", required=True,
                        help="Date in YYYY-MM-DD format")
    parser.add_argument("--start", required=True,
                        help="Start time in HH:MM local Colombia time")
    parser.add_argument("--save", action="store_true",
                        help="Save the output image")
    args = parser.parse_args()

    base = datetime.strptime(f"{args.date} {args.start}", "%Y-%m-%d %H:%M")
    local_times = []
    for i in range(9):
        t = base + timedelta(minutes=i * 10)
        local_times.append((t.date().isoformat(), t.hour, t.minute))

    slug  = args.start.replace(":", "-")
    label = f"{args.date} from {args.start} (local Colombia)"

    plot_sequence(slug, label, local_times, args.date, args.save)
