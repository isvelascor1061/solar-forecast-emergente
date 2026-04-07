#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline-Check für beliebige NetCDF-Paare auf IDENTISCHEM Test-Split.
Berechnet MSE, MAE, RMSE, R², Pearson-r und die Varianz der Residuen
und gibt alles im deutschen Zahlenformat aus (1.234.567,89).
"""

from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# -----------------------------------------------------------
#    Deutsches Zahlformat (Tausender-Punkt, Dezimal-Komma)
# -----------------------------------------------------------
def fmt_de(value, prec=5):
    """
    Formatiert float/np.number im deutschen Stil:
    1.234.567,89000   (prec = Nachkommastellen)
    """
    s = f"{value:,.{prec}f}"           # 1,234,567.89000
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

# -----------------------------------------------------------
# 1)  Dateipaare (beliebig erweiterbar)
# -----------------------------------------------------------
PAIRS = [
    ("_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped/dswrf1_CSI_0100.nc",
     "dswrf1_0100",
     "_3_Data_preparation_for_LSTM/Preparation_data/_03_Siata_GHI/Netcdf_Siata_GHI/GHI_CSI_clipped.nc",
     "GHI_clean",
     "GFS-DSWRF1  vs  SIATA-GHI")
]

TEST_IDX_FILE = Path("_4_LSTM_modules/test_indices/test_indices_multfeat_caus24_CSI_0100.npy")
SUBSET_COORDS = None
USE_AREA_MEAN = True

# -----------------------------------------------------------
# 2)  Test-Indizes laden
# -----------------------------------------------------------
test_idx = pd.to_datetime(np.load(TEST_IDX_FILE))
print(f"{len(test_idx):,} Zeitpunkte aus {TEST_IDX_FILE.name} werden verwendet.\n")

# -----------------------------------------------------------
# 3)  Hilfsfunktion → 1-D-Serie aus DataArray
# -----------------------------------------------------------
def to_timeseries(da, subset=None, area_mean=False):
    time_dim = next((d for d in da.dims if np.issubdtype(da[d].dtype, np.datetime64)), None)
    if time_dim is None:
        raise ValueError("Keine Zeitachse gefunden.")
    if subset:
        da = da.sel(subset, method="nearest")
    elif area_mean and {"lat", "lon"} <= set(da.dims):
        da = da.mean(dim=("lat", "lon"))
    for dim in da.dims:
        if dim != time_dim:
            da = da.isel({dim: 0})
    return da.to_series()

# -----------------------------------------------------------
# 4)  Evaluierungsfunktion
# -----------------------------------------------------------
def evaluate_pair(gfs_path, gfs_var, siata_path, siata_var):
    ds_gfs   = xr.open_dataset(gfs_path)
    ds_siata = xr.open_dataset(siata_path)

    gfs_ser   = to_timeseries(ds_gfs[gfs_var],   SUBSET_COORDS, USE_AREA_MEAN)
    siata_ser = to_timeseries(ds_siata[siata_var], SUBSET_COORDS, USE_AREA_MEAN)

    joined = (
        pd.concat([gfs_ser.rename("gfs"), siata_ser.rename("siata")], axis=1, join="inner")
          .dropna(subset=["siata"])
    )
    joined_test = joined[joined.index.isin(test_idx)].dropna()
    print(f"Anzahl der Teststunden nach Filter: {len(joined_test):,}")

    y_true, y_pred = joined_test["siata"].values, joined_test["gfs"].values
    residuals = y_true - y_pred
   

    ## ----- Baseline (1-h Persistence) ----------------------------------
    siata       = joined_test["siata"]
    baseline_df = pd.concat([siata, siata.shift(1)], axis=1).dropna()
    baseline_df.columns = ["y_true", "y_base"]
    
    y_b_true, y_b_pred = baseline_df["y_true"].values, baseline_df["y_base"].values
    mse_base = mean_squared_error(y_b_true, y_b_pred)
    
    # ----- Modell (GFS) auf dieselbe Indexmenge ------------------------
    gfs_aligned = joined_test["gfs"].loc[baseline_df.index].values
    
    
    
    
    mse_gfs  = mean_squared_error(y_b_true, gfs_aligned)
    skill_gfs = np.nan if mse_base == 0 else 1 - mse_gfs / mse_base
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    corr, pval = pearsonr(y_true, y_pred)
    

    # Scatter-Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=8, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f"Predicted vs Actual ({len(joined_test)} Test-Stunden)")
    plt.xlabel("SIATA [W/m²]"); plt.ylabel("GFS [W/m²]")
    plt.grid(True); plt.tight_layout(); plt.show()

    # Residuen-Histogramm
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, edgecolor="k", alpha=0.7)
    plt.title("Residuen-Histogramm"); plt.xlabel("Residual [W/m²]")
    plt.grid(True); plt.tight_layout(); plt.show()

    return len(joined_test), mse, rmse, mae, r2, corr, pval, residuals, mse_base, skill_gfs

# -----------------------------------------------------------
# 5)  Alle Paare durchlaufen und Ergebnisse deutsch formatiert ausgeben
# -----------------------------------------------------------
for gfs_p, gfs_v, siata_p, siata_v, label in PAIRS:
    try:
        n, mse, rmse, mae, r2, corr, pval, residuals, mse_base,skill_gfs = evaluate_pair(gfs_p, gfs_v, siata_p, siata_v)
        print("------------------------------------------------------------")
        print(label)
        print(f"Test-Stunden                : {fmt_de(n, 0)}")
        print(f"MSE   [W m⁻²]               : {fmt_de(mse)}")
        print(f"RMSE  [W m⁻²]               : {fmt_de(rmse)}")
        print(f"MAE   [W m⁻²]               : {fmt_de(mae)}")
        print(f"R²                          : {fmt_de(r2)}")
        print(f"Pearson-r                   : {fmt_de(corr)}  (p = {pval:.1e})")
        print(f"Varianz der Residuen [W m⁻²]: {fmt_de(np.var(residuals))}")
        print(f"Baseline-MSE          [W m⁻²]: {fmt_de(mse_base)}")
        print(f"Skill-Score (GFS vs Baseline) : {fmt_de(skill_gfs, 4)}")

    except Exception as e:
        print("------------------------------------------------------------")
        print(f"{label}  ➜  FEHLER: {e}")

print("------------------------------------------------------------")
