"""
Visualize GOES-16 Cloud Optical Depth (COD) over the Valle de Aburrá.

Three 3×3 panel figures covering:
  - Early morning (madrugada): 02:00–03:20 local
  - Dawn transition (amanecer): 05:50–07:10 local
  - Midday (mediodia): 12:00–13:20 local

Date: 2023-10-15 | Satellite: GOES-16 | Product: ABI-L2-CODF
Colombia local time is UTC-5, so UTC = local + 5h.
"""

import os
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from goes2go import GOES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAT_CENTER = 6.2591538
LON_CENTER = -75.5884924

LAT_MIN = LAT_CENTER - 0.45
LAT_MAX = LAT_CENTER + 0.45
LON_MIN = LON_CENTER - 0.46
LON_MAX = LON_CENTER + 0.46

DATE_STR = "2023-10-15"
SATELLITE = 16
PRODUCT = "ABI-L2-CODF"
UTC_OFFSET = 5  # Colombia is UTC-5  →  UTC = local + 5

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COD_CMAP = "YlOrRd"
COD_VMIN = 0
COD_VMAX = 25

# ---------------------------------------------------------------------------
# Sequence definitions
# Each entry: (slug, display_label, list_of_local_(hour, minute) tuples)
# ---------------------------------------------------------------------------

SEQUENCES = [
    (
        "madrugada",
        "Early Morning (Madrugada)",
        [(2, 0), (2, 10), (2, 20), (2, 30), (2, 40), (2, 50), (3, 0), (3, 10), (3, 20)],
    ),
    (
        "amanecer",
        "Dawn Transition (Amanecer)",
        [(5, 50), (6, 0), (6, 10), (6, 20), (6, 30), (6, 40), (6, 50), (7, 0), (7, 10)],
    ),
    (
        "mediodia",
        "Midday (Mediodía)",
        [(12, 0), (12, 10), (12, 20), (12, 30), (12, 40), (12, 50), (13, 0), (13, 10), (13, 20)],
    ),
]


def local_to_utc(date_str: str, hour: int, minute: int) -> datetime:
    """Convert a Colombia local time (UTC-5) to UTC datetime."""
    year, month, day = map(int, date_str.split("-"))
    local_dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return local_dt + timedelta(hours=UTC_OFFSET)


def format_local_time(hour: int, minute: int) -> str:
    """Return a human-readable local time string, e.g. '2:00 AM'."""
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
        H = proj_info.attrs["perspective_point_height"] + proj_info.attrs["semi_major_axis"]
        r_eq = proj_info.attrs["semi_major_axis"]
        r_pol = proj_info.attrs["semi_minor_axis"]

        x = ds["x"].values  # scan angle in radians
        y = ds["y"].values  # scan angle in radians

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


def download_cod(utc_dt: datetime):
    """
    Download GOES-16 ABI-L2-CODF nearest to utc_dt and return cropped arrays.

    Uses a ±15 minute window (wider than scan cadence) so goes2go's timestamp
    comparison reliably finds a match, then opens the first file directly.

    Returns
    -------
    (cod_vals, lons, lats) — all numpy arrays masked to the study domain
    or None if download or processing fails.
    """
    try:
        G = GOES(satellite=SATELLITE, product=PRODUCT)

        ts = pd.Timestamp(utc_dt)
        start = (ts - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")
        end   = (ts + pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")

        print(f"    Searching files {start} – {end} UTC …", end=" ", flush=True)

        df = G.df(start=start, end=end)
        if df is None or len(df) == 0:
            print("no files found.")
            return None

        print(f"{len(df)} file(s) found. Opening from S3 …", end=" ", flush=True)

        # Open the nearest file directly from AWS S3 (no local download)
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        s3_path = df.iloc[0]["file"]

        with fs.open(s3_path, "rb") as f:
            ds = xr.open_dataset(f, engine="h5netcdf")

            print("done.")

            # Extract COD variable — try_variable_names returns (name, DataArray)
            var_name, cod_da = try_variable_names(ds)
            print(f"    Variable used: '{var_name}'")

            # Get lat/lon arrays
            lats, lons = get_latlon(ds)
            if lats is None:
                print("    Could not determine lat/lon.")
                return None

            # Find row/col indices within domain bounding box
            row_mask = np.any(
                (lats >= LAT_MIN) & (lats <= LAT_MAX), axis=1
            )
            col_mask = np.any(
                (lons >= LON_MIN) & (lons <= LON_MAX), axis=0
            )

            if not row_mask.any() or not col_mask.any():
                print("    WARNING: domain mask returned no pixels — check lat/lon range.")
                return None

            cod_vals = cod_da.values.astype(float)

            # Crop all arrays to domain bounding box (~50×50 pixels instead of ~2713×2713)
            # This avoids the "Unable to allocate 168 MiB" error when cartopy tries to
            # transform every pixel of the full-disk image before clipping.
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


def plot_sequence(slug: str, label: str, local_times: list):
    """
    Build and save a 3×3 panel figure for one sequence.

    Parameters
    ----------
    slug        : filename slug, e.g. "madrugada"
    label       : figure title label, e.g. "Early Morning (Madrugada)"
    local_times : list of 9 (hour, minute) tuples in Colombia local time
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
        f"GOES-16 Cloud Optical Depth — {label}\n{DATE_STR}",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )

    print(f"\n{'='*60}")
    print(f"Sequence: {label}")
    print(f"{'='*60}")

    pcm_ref = None  # store one pcolormesh handle for the colorbar

    for idx, (hour, minute) in enumerate(local_times):
        ax = axes.flat[idx]
        utc_dt = local_to_utc(DATE_STR, hour, minute)
        time_label = format_local_time(hour, minute)

        print(f"\n  Panel {idx+1}/9 — {time_label} local ({utc_dt.strftime('%H:%M')} UTC):")

        result = download_cod(utc_dt)

        # Map extent
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)

        # Cartopy features
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "rivers_lake_centerlines", "50m"
            ),
            linewidth=0.5,
            edgecolor="steelblue",
            facecolor="none",
        )

        if result is not None:
            cod_vals, lons_plot_raw, lats_plot_raw = result

            # Mask limb pixels where geostationary projection yields NaN lat/lon
            valid = (
                np.isfinite(lons_plot_raw) &
                np.isfinite(lats_plot_raw) &
                np.isfinite(cod_vals)
            )
            cod_plot  = np.where(valid, cod_vals, np.nan)
            lons_plot = np.where(np.isfinite(lons_plot_raw), lons_plot_raw, 0.0)
            lats_plot = np.where(np.isfinite(lats_plot_raw), lats_plot_raw, 0.0)

            if valid.any():
                pcm = ax.pcolormesh(
                    lons_plot, lats_plot, cod_plot,
                    cmap=COD_CMAP,
                    vmin=COD_VMIN,
                    vmax=COD_VMAX,
                    transform=proj,
                    shading="auto",
                )
                pcm_ref = pcm
            else:
                ax.set_facecolor("#d0d0d0")
                ax.text(
                    0.5, 0.5, "No data\navailable",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=10, color="gray",
                )
        else:
            ax.set_facecolor("#d0d0d0")
            ax.text(
                0.5, 0.5, "No data available",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=10, color="#444444",
            )

        # Red star at SIATA station
        ax.plot(
            LON_CENTER, LAT_CENTER,
            marker="*", color="red", markersize=10,
            transform=proj, zorder=5,
        )

        # Panel title
        ax.set_title(time_label, fontsize=10, pad=4)

        # Minimal gridlines for orientation
        gl = ax.gridlines(
            draw_labels=False,
            linewidth=0.3, color="gray", alpha=0.5, linestyle="--",
        )
        gl.xlocator = mticker.FixedLocator(
            np.arange(round(LON_MIN, 1) - 0.1, LON_MAX + 0.1, 0.2)
        )
        gl.ylocator = mticker.FixedLocator(
            np.arange(round(LAT_MIN, 1) - 0.1, LAT_MAX + 0.1, 0.2)
        )

    # Shared horizontal colorbar
    cbar_ax = fig.add_axes([0.15, 0.04, 0.70, 0.018])
    if pcm_ref is not None:
        cb = fig.colorbar(pcm_ref, cax=cbar_ax, orientation="horizontal")
        cb.set_label("Cloud Optical Depth — higher = denser cloud", fontsize=11)
        cb.ax.tick_params(labelsize=9)
    else:
        # All panels failed — draw a placeholder colorbar from a dummy image
        import matplotlib.cm as mplcm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=COD_VMIN, vmax=COD_VMAX)
        sm = plt.cm.ScalarMappable(cmap=COD_CMAP, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cb.set_label("Cloud Optical Depth — higher = denser cloud", fontsize=11)
        cb.ax.tick_params(labelsize=9)

    out_path = os.path.join(OUTPUT_DIR, f"COD_GOES16_{slug}_{DATE_STR}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for slug, label, local_times in SEQUENCES:
        plot_sequence(slug, label, local_times)

    print("\nAll sequences complete.")
