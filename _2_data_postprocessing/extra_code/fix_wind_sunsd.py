#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 19:43:15 2025
"""
import xarray as xr
import numpy as np
import os

def process_netcdf(input_file, output_folder, output_filename):
    # Eingabedatei laden
    ds = xr.open_dataset(input_file)

    # Ersetze NaN-Werte in Wind10m_0100 mit Berechnungen aus ugrd und vgrd
    wind10m_nan = ds['Wind10m_0100'].isnull()
    ds['Wind10m_0100'] = ds['Wind10m_0100'].where(~wind10m_nan, np.sqrt(ds['UGRD_10m_0100']**2 + ds['VGRD_10m_0100']**2))

    # Ersetze NaN-Werte in SUNSD_minutes_0100 mit SUNSD_surface_0100 / 60
    sunds_minutes_nan = ds['SUNSD_minutes_0100'].isnull()
    ds['SUNSD_minutes_0100'] = ds['SUNSD_minutes_0100'].where(~sunds_minutes_nan, ds['SUNSD_surface_0100'] / 60)

    # Lösche die Variablen ugrd_10m_0100, vgrd_10m_0100 und sundsd_surface_0100
    ds = ds.drop_vars(['UGRD_10m_0100', 'VGRD_10m_0100', 'SUNSD_surface_0100'])

    # Ausgabedatei speichern
    output_file = os.path.join(output_folder, output_filename)
    ds.to_netcdf(output_file)

    print(f"Die bearbeitete Datei wurde erfolgreich gespeichert als {output_file}")

# Beispielaufruf der Funktion
input_file = "_2_data_postprocessing/_03_Merged_MultGFS_data/MultGFS_0100.nc"  # Pfad zur Eingabedatei
output_folder = "_2_data_postprocessing/_03_Merged_MultGFS_data"  # Ordner, in dem die Ausgabedatei gespeichert werden soll
output_filename = "MULTGFS_0100_fixed.nc"  # Name der Ausgabedatei

process_netcdf(input_file, output_folder, output_filename)

