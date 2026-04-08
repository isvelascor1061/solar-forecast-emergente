#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 19:49:40 2025

@author: leonardmerl
"""

import xarray as xr
import numpy as np
import os

def process_netcdf(input_file, output_folder, output_filename):
    # Eingabedatei laden
    ds = xr.open_dataset(input_file)

    # Berechne Wind10m_0100 aus UGRD_10m_0700 und VGRD_10m_0700
    ds['Wind10m_1900'] = np.sqrt(ds['UGRD_10m_1900']**2 + ds['VGRD_10m_1900']**2)

    # Berechne SUNSD_minutes_0700 aus SUNSD_surface_0700 / 60
    ds['SUNSD_minutes_1900'] = ds['SUNSD_surface_1900'] / 60

    # Lösche die Variablen UGRD_10m_0700, VGRD_10m_0700 und SUNSD_surface_0700
    ds = ds.drop_vars(['UGRD_10m_1900', 'VGRD_10m_1900', 'SUNSD_surface_1900'])

    # Ausgabedatei speichern
    output_file = os.path.join(output_folder, output_filename)
    ds.to_netcdf(output_file)

    print(f"Die bearbeitete Datei wurde erfolgreich gespeichert als {output_file}")

# Beispielaufruf der Funktion
input_file = "_2_data_postprocessing/_03_Merged_MultGFS_data/MultGFS_1900.nc"  # Pfad zur Eingabedatei
output_folder = "_2_data_postprocessing/_03_Merged_MultGFS_data"  # Ordner, in dem die Ausgabedatei gespeichert werden soll
output_filename = "1900_fixed.nc"  # Name der Ausgabedatei

process_netcdf(input_file, output_folder, output_filename)
