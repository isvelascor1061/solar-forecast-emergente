#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clear_sky_ineichen_creator_unified.py
======================================
Unifica Clear_sky_ineichen_radiation_creator.py y
Clear_sky_readiation_creat_with_bias_correct.py en un único script.

Calcula la Irradiancia Global Horizontal de cielo despejado (GHI_cs) para
la celda GFS centrada en Medellín, usando el modelo Ineichen con pvlib.

Características:
  - Perfil de horizonte real cargado desde un archivo raster TIF (rasterio).
  - Muestreo espacial de 25 puntos dentro de la celda 0.25° × 0.25°.
  - Corrección de sesgo zenital configurable (zenith_corr) importada desde
    config.py: el modelo Ineichen tiende a subestimar la GHI a ángulos
    cenitales altos (> 60°); el factor f = 1 + α·max(0, θ − θ₀) compensa
    este efecto.
  - Salida como NetCDF con dimensiones (observation_time, surface, lat, lon).

Pasos principales:
  1. Generar índice temporal minutal para el período de interés.
  2. Cargar el perfil de horizonte desde el archivo TIF.
  3. Para cada uno de los 25 puntos de muestreo de la celda:
     a. Calcular posición solar y masa de aire.
     b. Estimar GHI_cs con el modelo Ineichen (turbidez de Linke de tabla).
     c. Aplicar máscara de horizonte (sol tapado por el relieve = 0 W/m²).
     d. Aplicar corrección de sesgo zenital si está activada.
  4. Promediar los 25 puntos y agregar a resolución horaria.
  5. Guardar el resultado como NetCDF.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import pvlib
from pvlib.clearsky import lookup_linke_turbidity
from pvlib.atmosphere import get_relative_airmass, get_absolute_airmass
from scipy.interpolate import interp1d
import rasterio

from config import (
    LAT, LON, ELEVATION,
    HORIZON_FILE, CSI_GHI_OUT_DIR,
    ZENITH_CORR, ZENITH_ALPHA, ZENITH_THRESHOLD,
)


class ClearSkyGridAverager:
    """
    Calcula la GHI de cielo despejado (modelo Ineichen) promediada sobre
    una celda GFS de 0.25° × 0.25°, con perfil de horizonte y corrección
    de sesgo zenital opcional.
    """

    def __init__(
        self,
        lat_center: float,
        lon_center: float,
        start_date: str,
        end_date: str,
        tz: str = "UTC",
        altitude: float = 0,
        model: str = "ineichen",
        linke_turbidity=None,
        seed: int | None = None,
        horizon_file: str | None = None,
        zenith_corr: bool = False,
        zenith_alpha: float = 0.0025,
        zenith_threshold: float = 65,
    ):
        """
        Parámetros
        ----------
        lat_center       : Latitud del centro de la celda GFS (grados N).
        lon_center       : Longitud del centro de la celda GFS (grados E).
        start_date       : Inicio del período (ej. '2021-04-01 00:00').
        end_date         : Fin del período (ej. '2025-06-01 00:00').
        tz               : Zona horaria (ej. 'America/Bogota').
        altitude         : Altitud del punto en metros.
        model            : Modelo pvlib de cielo despejado (default 'ineichen').
        linke_turbidity  : Turbidez de Linke fija; None = tabla automática.
        seed             : Semilla aleatoria para los puntos de muestreo.
        horizon_file     : Ruta al archivo TIF con datos de elevación del horizonte.
        zenith_corr      : Activar corrección de sesgo zenital (importado de config).
        zenith_alpha     : Factor de corrección por grado de cénit (importado de config).
        zenith_threshold : Ángulo cenital a partir del cual se corrige (importado de config).
        """
        self.lat_center      = lat_center
        self.lon_center      = lon_center
        self.start_date      = start_date
        self.end_date        = end_date
        self.tz              = tz
        self.altitude        = altitude
        self.model           = model
        self.linke_turbidity = linke_turbidity
        self.horizon_file    = horizon_file
        self.zenith_corr     = zenith_corr
        self.zenith_alpha    = zenith_alpha
        self.zenith_threshold = zenith_threshold

        # Semilla para reproducibilidad de los puntos aleatorios de muestreo
        if seed is not None:
            np.random.seed(seed)

        # Límites de la celda de 0.25° × 0.25° centrada en lat/lon
        half = 0.125
        self.lat_min = lat_center - half
        self.lat_max = lat_center + half
        self.lon_min = lon_center - half
        self.lon_max = lon_center + half

    # ------------------------------------------------------------------
    def load_hgt(self, hgt_file: str) -> np.ndarray:
        """
        Carga el archivo raster TIF de elevación y extrae los datos para
        el área de interés definida por los límites de la celda.
        """
        with rasterio.open(hgt_file) as src:
            # Verificar que el raster cubre el área de la celda
            if (src.bounds.left > self.lon_max or
                    src.bounds.right < self.lon_min or
                    src.bounds.bottom > self.lat_max or
                    src.bounds.top < self.lat_min):
                raise ValueError("El archivo TIF no cubre el área de la celda GFS.")

            window = src.window(self.lon_min, self.lat_min,
                                self.lon_max, self.lat_max)
            hgt_data = src.read(1, window=window, boundless=True)
        return hgt_data

    # ------------------------------------------------------------------
    def calculate_horizon_profile(self, hgt_data: np.ndarray,
                                  resolution: int = 5):
        """
        Calcula el perfil de horizonte (ángulo de elevación por azimut) a
        partir de los datos de elevación del raster TIF.

        Parámetros
        ----------
        hgt_data   : Matriz 2D con altitudes en metros.
        resolution : Resolución azimutal en grados (default 5°).

        Retorna
        -------
        azimuths   : Array de azimuts (0–355 °).
        elevations : Array de ángulos de elevación del horizonte (°).
        """
        azimuths   = np.arange(0, 360, resolution)
        elevations = np.zeros_like(azimuths, dtype=float)

        # Punto central de la celda como referencia de altitud
        center_row = hgt_data.shape[0] // 2
        center_col = hgt_data.shape[1] // 2
        center_alt = hgt_data[center_row, center_col]

        for i, az in enumerate(azimuths):
            max_distance = hgt_data.shape[1] // 2
            distances    = np.arange(1, max_distance)
            # Diferencia de altitud respecto al punto central
            delta_elev   = hgt_data[center_row, 1:max_distance] - center_alt
            # Ángulo de elevación aproximado (distancia en píxeles ≈ ° × 90)
            tan_elev     = delta_elev / (distances * 90)
            elevations[i] = np.degrees(np.arctan(np.max(tan_elev)))

        return azimuths, elevations

    # ------------------------------------------------------------------
    def generate(self, output_dir: str, output_filename: str) -> None:
        """
        Ejecuta el cálculo completo y guarda el resultado como NetCDF.

        Parámetros
        ----------
        output_dir      : Carpeta de destino del archivo NetCDF.
        output_filename : Nombre del archivo de salida.
        """
        os.makedirs(output_dir, exist_ok=True)

        # --- Índice temporal minutal para integrar por hora ------------
        # Se calcula a resolución de 1 minuto y luego se promedia por hora
        time_index   = pd.date_range(
            start=self.start_date, end=self.end_date,
            freq="min", tz=self.tz, inclusive="both"
        )
        group_labels = time_index.floor("h") + pd.Timedelta(hours=1)
        unique_hours = pd.DatetimeIndex(group_labels.unique())

        # --- Cargar perfil de horizonte --------------------------------
        # Si se proporciona un archivo TIF, se calcula el perfil de elevación
        # del horizonte para enmascarar horas en que el sol está tapado.
        horizon_azimuths   = None
        horizon_elevations = None
        if self.horizon_file:
            try:
                hgt_data = self.load_hgt(self.horizon_file)
                horizon_azimuths, horizon_elevations = \
                    self.calculate_horizon_profile(hgt_data)
            except Exception as e:
                print(f"Aviso: no se pudo cargar el perfil de horizonte: {e}")
                print("Continuando sin corrección de horizonte.")

        # --- Definir los 25 puntos de muestreo dentro de la celda ------
        # 9 puntos fijos (esquinas, bordes y centro) + 16 aleatorios (4 por cuadrante)
        lat_c, lon_c = self.lat_center, self.lon_center
        lmin,  lmax  = self.lat_min, self.lat_max
        omin,  omax  = self.lon_min, self.lon_max

        fijos = [
            (lmin, omin), (lmin, omax), (lmax, omin), (lmax, omax),
            (lat_c, omin), (lat_c, omax), (lmin, lon_c), (lmax, lon_c),
            (lat_c, lon_c)
        ]
        cuadrantes = [
            (lat_c, lmax, omin, lon_c),
            (lat_c, lmax, lon_c, omax),
            (lmin, lat_c, omin, lon_c),
            (lmin, lat_c, lon_c, omax),
        ]
        aleatorios = []
        for lat_lo, lat_hi, lon_lo, lon_hi in cuadrantes:
            pts = np.random.rand(4, 2)
            pts[:, 0] = pts[:, 0] * (lat_hi - lat_lo) + lat_lo
            pts[:, 1] = pts[:, 1] * (lon_hi - lon_lo) + lon_lo
            aleatorios.extend([tuple(pt) for pt in pts])

        muestras = fijos + aleatorios   # 25 puntos en total

        # --- Calcular GHI_cs para cada punto de muestreo ---------------
        cs_matrix = np.zeros((len(muestras), len(time_index)))

        for idx, (lat, lon) in enumerate(muestras):
            loc = pvlib.location.Location(
                latitude=lat, longitude=lon,
                tz=self.tz, altitude=self.altitude
            )

            # Posición solar para todos los instantes minutales
            solpos = loc.get_solarposition(time_index)

            # Masa de aire relativa y absoluta (necesaria para Ineichen)
            airmass_rel = get_relative_airmass(solpos["apparent_zenith"])
            airmass_abs = get_absolute_airmass(airmass_rel)

            # GHI de cielo despejado con el modelo seleccionado
            if self.model == "ineichen":
                lt = (self.linke_turbidity if self.linke_turbidity is not None
                      else lookup_linke_turbidity(time_index, lat, lon))
                cs = loc.get_clearsky(
                    time_index, model="ineichen",
                    linke_turbidity=lt,
                    airmass_absolute=airmass_abs,
                    perez_enhancement=True,
                )
            else:
                cs = loc.get_clearsky(time_index, model=self.model)

            # Aplicar máscara de horizonte: GHI = 0 cuando el sol está
            # por debajo del perfil de elevación del terreno circundante
            if horizon_azimuths is not None:
                elev_interp  = interp1d(horizon_azimuths, horizon_elevations,
                                        bounds_error=False, fill_value=0)
                horizonte_el = elev_interp(solpos["azimuth"])
                sol_el       = solpos["apparent_elevation"]
                mask         = sol_el > horizonte_el
                cs["ghi"]    = cs["ghi"].where(mask, 0)

            # Corrección de sesgo zenital: el modelo Ineichen subestima
            # la GHI cuando el ángulo cenital supera zenith_threshold.
            # Se aplica un factor f = 1 + α · max(0, θ − θ₀).
            if self.zenith_corr:
                zenith    = solpos["apparent_zenith"]
                factor    = 1 + self.zenith_alpha * (
                    zenith - self.zenith_threshold
                ).clip(lower=0)
                cs["ghi"] = cs["ghi"] * factor

            cs_matrix[idx, :] = cs["ghi"].values

        # --- Promediar los 25 puntos y agregar a resolución horaria ----
        df     = pd.DataFrame(cs_matrix.T, index=time_index)
        hourly = df.groupby(group_labels).mean()
        avg    = hourly.mean(axis=1)   # media espacial de los 25 puntos

        # --- Construir y guardar el dataset NetCDF ---------------------
        obs  = unique_hours.tz_localize(None).to_numpy()
        data = avg.values[:, None, None, None]
        ds = xr.Dataset(
            {"clear_sky_ghi": (("observation_time", "surface", "lat", "lon"), data)},
            coords={
                "observation_time": obs,
                "surface": [0],
                "lat": [self.lat_center],
                "lon": [self.lon_center],
            },
        )
        ds.to_netcdf(os.path.join(output_dir, output_filename))
        print(f"GHI de cielo despejado guardada en: {output_dir}/{output_filename}")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    gen = ClearSkyGridAverager(
        lat_center=LAT,
        lon_center=LON,
        start_date="2021-04-01 00:00",
        end_date="2025-06-01 00:00",
        tz="America/Bogota",
        altitude=ELEVATION,
        model="ineichen",
        linke_turbidity=None,   # None = usa tabla automática de pvlib
        seed=42,
        horizon_file=HORIZON_FILE,
        zenith_corr=ZENITH_CORR,           # importado de config.py
        zenith_alpha=ZENITH_ALPHA,         # importado de config.py
        zenith_threshold=ZENITH_THRESHOLD, # importado de config.py
    )
    gen.generate(
        output_dir=CSI_GHI_OUT_DIR,
        output_filename="CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct2.nc",
    )
    """
    Instrucciones de ejecución:
    1. Descargar el archivo TIF de elevación desde:
       https://portal.opentopography.org/apidocs/#/Public/getGlobalDem
       para la celda centrada en LAT=6.25, LON=-75.5 y guardarlo en
       HORIZON_FILE (ver config.py).
    2. Ajustar start_date y end_date al rango de datos disponibles.
    3. Configurar ZENITH_CORR, ZENITH_ALPHA y ZENITH_THRESHOLD en config.py
       según se desee activar o no la corrección de sesgo.
    4. Ejecutar el script (puede tardar varios minutos por la resolución
       minutal y los 25 puntos de muestreo por celda).
    """
