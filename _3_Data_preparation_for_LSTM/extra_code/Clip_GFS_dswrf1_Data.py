#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFS-Strahlung an physikalischer Schwelle «clippen»:
Überschreitungen werden per Zeit-Interpolation ersetzt,
dabei nie größer als die Schwelle (clear-sky oder extraterrestrial GHI).

Created on 2025-04-21
@author: leonardmerl   (Modifikation von ChatGPT, 2025-05-12)
"""
import os
import xarray as xr
import numpy as np
import pandas as pd


class GFSThresholdClipper:
    """
    Ersetzt GFS-Werte, die oberhalb einer Referenzstrahlung liegen, durch
    linear interpolierte Werte (Zeitachse).  Interpolat ≤ Referenz.
    """

    def __init__(
        self,
        gfs_path: str,
        ref_path: str,
        gfs_var: str = "dswrf1",
        ref_var: str = "clear_sky_ghi",
    ):
        """
        Parameters
        ----------
        gfs_path : str
            Pfad zur NetCDF-Datei mit GFS-Strahlung (z. B. dswrf1).
        ref_path : str
            Pfad zur NetCDF-Datei mit Referenzstrahlung (Clear-Sky oder G0h).
        gfs_var : str
            Variablenname der GFS-Strahlung.
        ref_var : str
            Variablenname der Referenzstrahlung.
        """
        self.gfs_path = gfs_path
        self.ref_path = ref_path
        self.gfs_var = gfs_var
        self.ref_var = ref_var

        self.gfs_da = None
        self.ref_da = None
        self.clipped_da = None

    # ------------------------------------------------------------------ #
    def load(self):
        """NetCDF-Dateien laden."""
        self.gfs_da = xr.open_dataset(self.gfs_path)[self.gfs_var]
        self.ref_da = xr.open_dataset(self.ref_path)[self.ref_var]

    def align(self, join: str = "inner"):
        """Zeitleisten angleichen (observation_time)."""
        self.gfs_da, self.ref_da = xr.align(self.gfs_da, self.ref_da, join=join)

    def clip(self, dim: str = "observation_time", jitter_range: float = 0.03):
        """
        1. Maske der Überschreitungen
        2. Überschreitungen → NaN
        3. Ersetzen durch Wochen-Durchschnitt + Jitter
        4. Überreste (am Rand) mit Schwelle auffüllen
        5. Sicherstellen: Interpolat ≤ Schwelle
        """
        # (1) Maske der Überschreitungen
        exceed_mask = self.gfs_da > self.ref_da
    
        # (2) Überschreitungen auf NaN setzen
        gfs_tmp = self.gfs_da.where(~exceed_mask)
    
        # (3) Ersetzen von Überschreitungen mit Wochen-Durchschnitt + Jitter
        # Um die Woche zu extrahieren, nutzen wir `observation_time`
        weekly_mean = self.gfs_da.groupby(pd.Grouper(freq="W")).mean("observation_time")
        
        # Hinzufügen von Jitter (+- 3%)
        jitter = np.random.uniform(-jitter_range, jitter_range, size=weekly_mean.shape)
        gfs_interp = weekly_mean + weekly_mean * jitter
    
        # Interpolation, wenn Werte nicht über der Schwelle sind
        gfs_interp = gfs_interp.where(gfs_interp <= self.ref_da)
    
        # (4) Ränder auffüllen (falls Serie am Anfang/Ende > Schwelle)
        gfs_interp = gfs_interp.fillna(self.ref_da*0.95)
    
        # (5) Endgültige Serie: Überschreitungen → min(Interpolat, Schwelle)
        self.clipped_da = xr.where(
            exceed_mask,
            xr.ufuncs.minimum(gfs_interp, self.ref_da),
            self.gfs_da
        )

    def save(self, output_dir: str, output_filename: str):
        """Geschnittene Serie als NetCDF sichern."""
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_filename)
        self.clipped_da.to_dataset(name=self.gfs_var).to_netcdf(out_path)
        print(f"Clipped GFS saved to: {out_path}")

    # Convenience
    def process(self, output_dir: str, output_filename: str):
        self.load()
        self.align()
        self.clip()
        self.save(output_dir, output_filename)


# ------------------------------ Beispiel ---------------------------------- #
if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #       1) Clip gegen Clear-Sky                                     #
    # ------------------------------------------------------------------ #
    GFS_PATH   = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/Raw_merged/dswrf1_0100.nc"
    CLEAR_PATH = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Ineichen_GHI/CSI_GHI_grid25_avg_with_horizon_and_enhancement.nc"

    OUT_DIR_CSI  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_CSI_clipped"
    OUT_FILE_CSI = "dswrf1_CSI_0100.nc"

    clipper_csi = GFSThresholdClipper(
        gfs_path=GFS_PATH,
        ref_path=CLEAR_PATH,
        gfs_var="dswrf1",
        ref_var="clear_sky_ghi"
    )
    #clipper_csi.process(OUT_DIR_CSI, OUT_FILE_CSI)

    # ------------------------------------------------------------------ #
    #       2) Clip gegen Extraterrestrische Strahlung                   #
    # ------------------------------------------------------------------ #
    EXTRA_PATH = "_3_Data_preparation_for_LSTM/Preparation_data/_01_CSI_EXT_radiation/Extraterrestrial_GHI/EXT_GHI_all.nc"

    OUT_DIR_EXT  = "_3_Data_preparation_for_LSTM/Preparation_data/_02_GFS_dswrf1/GFS_merged_EXT_clipped"
    OUT_FILE_EXT = "dswrf1_EXT_0100.nc"

    clipper_ext = GFSThresholdClipper(
        gfs_path=GFS_PATH,
        ref_path=EXTRA_PATH,
        gfs_var="dswrf1",
        ref_var="extraterrestrial_ghi"
    )
    clipper_ext.process(OUT_DIR_EXT, OUT_FILE_EXT)
    
    
    
