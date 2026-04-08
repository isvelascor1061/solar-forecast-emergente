#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_wind_sunsd_unified.py
=========================
Unifica fix_wind_sunsd.py y fix_wind_sunsd_and_rename.py en un único script
configurable por launch time.

Pasos que realiza:
  1. Carga el archivo NetCDF MultiGFS del launch time indicado.
  2. Calcula la velocidad del viento como sqrt(U² + V²) a partir de las
     componentes UGRD_10m y VGRD_10m.
  3. Verifica NaNs antes de calcular: si Wind10m ya existe, solo rellena
     las posiciones con NaN; si no existe, crea la variable desde cero.
  4. Convierte SUNSD de segundos a minutos con la misma lógica de NaN.
  5. Elimina las variables UGRD, VGRD y SUNSD_surface del dataset.
  6. Guarda el resultado como un nuevo archivo NetCDF.
"""

import xarray as xr
import numpy as np
import os
from config import MERGED_MULTGFS_DIR, LAUNCH_TIME_DEFAULT


# -----------------------------------------------------------------------
# Parámetro de ejecución — ajustar según el launch time a procesar
# -----------------------------------------------------------------------
LAUNCH_TIME = LAUNCH_TIME_DEFAULT   # Ej.: "0100", "0700", "1300", "1900"


def process_netcdf(input_file: str, output_folder: str, output_filename: str,
                   launch_time: str) -> None:
    """
    Procesa un archivo NetCDF MultiGFS: calcula viento vectorial, convierte
    SUNSD a minutos y elimina las variables componente.

    Parámetros
    ----------
    input_file     : Ruta al archivo NetCDF de entrada.
    output_folder  : Carpeta donde se guardará el archivo procesado.
    output_filename: Nombre del archivo de salida.
    launch_time    : Cadena de hora de lanzamiento (ej. "0100").
    """
    # --- Nombres de variables dependientes del launch time -------------
    var_wind   = f"Wind10m_{launch_time}"
    var_ugrd   = f"UGRD_10m_{launch_time}"
    var_vgrd   = f"VGRD_10m_{launch_time}"
    var_sunsd_min = f"SUNSD_minutes_{launch_time}"
    var_sunsd_sec = f"SUNSD_surface_{launch_time}"

    # --- Cargar el dataset de entrada ----------------------------------
    ds = xr.open_dataset(input_file)

    # --- Velocidad del viento: sqrt(U² + V²) ---------------------------
    # Se calcula siempre a partir de las componentes vectoriales para
    # garantizar consistencia. Si la variable Wind10m ya existe en el
    # dataset (con posibles NaNs), solo se rellenan las posiciones vacías;
    # si no existe, se crea desde cero.
    viento_calculado = np.sqrt(ds[var_ugrd] ** 2 + ds[var_vgrd] ** 2)

    if var_wind in ds.data_vars:
        # La variable existe: verificar NaNs y rellenar solo esas posiciones
        mascara_nan = ds[var_wind].isnull()
        ds[var_wind] = ds[var_wind].where(~mascara_nan, viento_calculado)
    else:
        # La variable no existe: crearla directamente
        ds[var_wind] = viento_calculado

    # --- Duración de sol: convertir de segundos a minutos --------------
    # SUNSD_surface almacena la duración en segundos; dividiendo entre 60
    # obtenemos SUNSD_minutes. Misma lógica de NaN que para el viento.
    sunsd_minutos = ds[var_sunsd_sec] / 60

    if var_sunsd_min in ds.data_vars:
        # La variable existe: rellenar solo posiciones con NaN
        mascara_nan_sun = ds[var_sunsd_min].isnull()
        ds[var_sunsd_min] = ds[var_sunsd_min].where(~mascara_nan_sun, sunsd_minutos)
    else:
        # La variable no existe: crearla directamente
        ds[var_sunsd_min] = sunsd_minutos

    # --- Eliminar variables componente ya no necesarias ----------------
    # Las componentes individuales U y V quedan redundantes tras calcular
    # la magnitud vectorial; SUNSD_surface queda redundante tras la conversión.
    ds = ds.drop_vars([var_ugrd, var_vgrd, var_sunsd_sec])

    # --- Guardar el dataset procesado ----------------------------------
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, output_filename)
    ds.to_netcdf(output_file)
    ds.close()

    print(f"Archivo procesado guardado en: {output_file}")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Ruta de entrada: archivo MultiGFS mergeado para el launch time elegido
    input_file = os.path.join(MERGED_MULTGFS_DIR, f"MultGFS_{LAUNCH_TIME}.nc")

    # Ruta de salida: misma carpeta, nombre con sufijo "_fixed"
    output_folder   = MERGED_MULTGFS_DIR
    output_filename = f"MultGFS_{LAUNCH_TIME}_fixed.nc"

    process_netcdf(input_file, output_folder, output_filename, LAUNCH_TIME)
