#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py  –  Configuración central del proyecto Solar Medellín (Emergente)
===========================================================================
Centraliza todas las rutas de archivos y parámetros globales del pipeline.
Organizado por sección según las 4 carpetas principales del proyecto.

Uso:
    from config import LAT, LON, SEQ_NPZ_FILE, ...
"""

# ===================================================================
# PARÁMETROS GEOGRÁFICOS Y TEMPORALES DE MEDELLÍN
# ===================================================================

# Coordenadas del punto de medición (rejilla GFS más cercana a Medellín)
LAT       = 6.25    # Latitud en grados norte
LON       = -75.5   # Longitud en grados oeste (negativo = hemisferio oeste)
ELEVATION = 1485    # Altitud sobre el nivel del mar en metros

# Zona horaria de Colombia respecto a UTC (sin cambio de horario de verano)
UTC_OFFSET = -5     # Colombia = UTC-5

# Horarios de lanzamiento del modelo GFS disponibles (hora local Colombia)
LAUNCH_TIMES = ["0100", "0700", "1300", "1900"]

# Ventana diurna para la máscara de día/noche (hora local)
FLAG_START = 19     # Hora en que empieza la noche local (19:00)
FLAG_END   = 6      # Hora en que termina la noche local (06:00)


# ===================================================================
# _1_data_acquisition  –  Descarga de datos GFS desde NOAA S3
# ===================================================================

# --- Radiación de onda corta (DSWRF) en intervalos acumulados ------
# Carpeta raíz de archivos brutos descargados por launch time
RAW_DSWRF_DIR      = "_1_data_acquisition/_01_raw_rad_data/dswrf"
RAW_DSWRF_0100_DIR = f"{RAW_DSWRF_DIR}/raw_dswrf_0100"   # Lanzamiento 01:00 h local
RAW_DSWRF_0700_DIR = f"{RAW_DSWRF_DIR}/raw_dswrf_0700"   # Lanzamiento 07:00 h local
RAW_DSWRF_1300_DIR = f"{RAW_DSWRF_DIR}/raw_dswrf_1300"   # Lanzamiento 13:00 h local
RAW_DSWRF_1900_DIR = f"{RAW_DSWRF_DIR}/raw_dswrf_1900"   # Lanzamiento 19:00 h local

# --- MultiGFS: todas las variables meteorológicas juntas -----------
# Carpeta raíz de archivos brutos MultiGFS por launch time
RAW_MULTGFS_DIR      = "_1_data_acquisition/_02_raw_MultGFS_data"
RAW_MULTGFS_0100_DIR = f"{RAW_MULTGFS_DIR}/raw_MultGFS_0100"   # Lanzamiento 01:00 h local
RAW_MULTGFS_0700_DIR = f"{RAW_MULTGFS_DIR}/raw_MultGFS_0700"   # Lanzamiento 07:00 h local
RAW_MULTGFS_1300_DIR = f"{RAW_MULTGFS_DIR}/raw_MultGFS_1300"   # Lanzamiento 13:00 h local
RAW_MULTGFS_1900_DIR = f"{RAW_MULTGFS_DIR}/raw_MultGFS_1900"   # Lanzamiento 19:00 h local

# --- Nombres de variables en los archivos del servidor S3 ----------
GFS_VAR_RAW = "sdswrf"   # Variable acumulada de onda corta en el servidor (media de intervalo)
GFS_VAR_OUT = "dswrf1"   # Variable horaria resultante tras desacumulación


# ===================================================================
# _2_data_postprocessing  –  Postprocesamiento de datos GFS
# ===================================================================

# --- Paso 1: Radiación bruta mergeada por día (antes de convertir a 1 h) ---
# Un archivo por día y launch time, mergeados de todos los pasos de pronóstico
MERGED_DSWRF_RAW_DIR      = "_2_data_postprocessing/_01_merged_Radrawdata/dswrf"
MERGED_DSWRF_RAW_0100_DIR = f"{MERGED_DSWRF_RAW_DIR}/merged_raw_dswrf_0100"   # LT 01:00 h
MERGED_DSWRF_RAW_0700_DIR = f"{MERGED_DSWRF_RAW_DIR}/merged_raw_dswrf_0700"   # LT 07:00 h
MERGED_DSWRF_RAW_1300_DIR = f"{MERGED_DSWRF_RAW_DIR}/merged_raw_dswrf_1300"   # LT 13:00 h
MERGED_DSWRF_RAW_1900_DIR = f"{MERGED_DSWRF_RAW_DIR}/merged_raw_dswrf_1900"   # LT 19:00 h

# --- Paso 2: Radiación horaria desacumulada (dswrf1) ---------------
# Un archivo por día y launch time, con la variable convertida a intervalos de 1 h
MERGED_DSWRF1_DIR      = "_2_data_postprocessing/_02_merged_Rad1data/dswrf"
MERGED_DSWRF1_0100_DIR = f"{MERGED_DSWRF1_DIR}/merged_raw_dswrf1_0100"   # LT 01:00 h
MERGED_DSWRF1_0700_DIR = f"{MERGED_DSWRF1_DIR}/merged_raw_dswrf1_0700"   # LT 07:00 h
MERGED_DSWRF1_1300_DIR = f"{MERGED_DSWRF1_DIR}/merged_raw_dswrf1_1300"   # LT 13:00 h
MERGED_DSWRF1_1900_DIR = f"{MERGED_DSWRF1_DIR}/merged_raw_dswrf1_1900"   # LT 19:00 h

# --- Paso 3: MultiGFS mergeado en 1 archivo por launch time --------
# Contiene todas las variables meteorológicas a lo largo de toda la serie temporal
MERGED_MULTGFS_DIR       = "_2_data_postprocessing/_03_Merged_MultGFS_data"
MERGED_MULTGFS_0100_FILE = f"{MERGED_MULTGFS_DIR}/MultGFS_0100_2.nc"   # LT 01:00 h
MERGED_MULTGFS_0700_FILE = f"{MERGED_MULTGFS_DIR}/MultGFS_0700_2.nc"   # LT 07:00 h
MERGED_MULTGFS_1300_FILE = f"{MERGED_MULTGFS_DIR}/MultGFS_1300_2.nc"   # LT 13:00 h
MERGED_MULTGFS_1900_FILE = f"{MERGED_MULTGFS_DIR}/MultGFS_1900_2.nc"   # LT 19:00 h

# Launch time por defecto para scripts que procesan de a uno por ejecución
LAUNCH_TIME_DEFAULT = "0100"


# ===================================================================
# _3_Data_preparation_for_LSTM  –  Preparación de datos para la red
# ===================================================================

# Directorio base de todos los datos preparados
PREP_DATA_DIR = "_3_Data_preparation_for_LSTM/Preparation_data"

# --- Radiación de referencia solar --------------------------------
# Radiación extraterrestre (techo físico máximo de radiación)
EXT_GHI_FILE = f"{PREP_DATA_DIR}/_01_CSI_EXT_radiation/Extraterrestrial_GHI/EXT_GHI_all.nc"
EXT_VAR_NAME = "extraterrestrial_ghi"   # Nombre de la variable dentro del NetCDF

# Radiación de cielo despejado Ineichen (con corrección de horizonte y sesgo)
CSI_GHI_FILE = (
    f"{PREP_DATA_DIR}/_01_CSI_EXT_radiation/Ineichen_GHI/"
    "CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct2.nc"
)
CSI_VAR_NAME = "clear_sky_ghi"   # Nombre de la variable de cielo despejado

# --- dswrf1: radiación GFS sin cortar, una serie temporal por LT --
DSWRF1_UNCLIPPED_DIR = f"{PREP_DATA_DIR}/_02_GFS_dswrf1/Unclipped_merged_dswrf1"
DSWRF1_0100_FILE     = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_0100.nc"   # LT 01:00 h
DSWRF1_0700_FILE     = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_0700.nc"   # LT 07:00 h
DSWRF1_1300_FILE     = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_1300.nc"   # LT 13:00 h
DSWRF1_1900_FILE     = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_1900.nc"   # LT 19:00 h

# --- dswrf1 cortada por Clear-Sky Index (CSI) ----------------------
# Valores que exceden el cielo despejado son reemplazados por interpolación
DSWRF1_CSI_DIR       = f"{PREP_DATA_DIR}/_02_GFS_dswrf1/GFS_merged_CSI_clipped"
DSWRF1_CSI_0100_FILE = f"{DSWRF1_CSI_DIR}/dswrf1_CSI_0100.nc"   # LT 01:00 h
DSWRF1_CSI_0700_FILE = f"{DSWRF1_CSI_DIR}/dswrf1_CSI_0700.nc"   # LT 07:00 h
DSWRF1_CSI_1300_FILE = f"{DSWRF1_CSI_DIR}/dswrf1_CSI_1300.nc"   # LT 13:00 h
DSWRF1_CSI_1900_FILE = f"{DSWRF1_CSI_DIR}/dswrf1_CSI_1900.nc"   # LT 19:00 h

# --- dswrf1 cortada por radiación extraterrestre (EXT) -------------
DSWRF1_EXT_DIR = f"{PREP_DATA_DIR}/_02_GFS_dswrf1/GFS_merged_EXT_clipped"

# --- Datos observados SIATA (irradiancia global horizontal medida) --
SIATA_GHI_FILE = f"{PREP_DATA_DIR}/_03_Siata_GHI/Netcdf_Siata_GHI/GHI_CSI_clipped.nc"
SIATA_GHI_VAR  = "GHI_clean"   # Nombre de la variable GHI observada

# --- Índices solares calculados ------------------------------------
# Clear-sky index (kc): GHI_GFS / GHI_ClearSky  →  normalizado 0-1
CSI_INDEX_DIR  = f"{PREP_DATA_DIR}/_04_indices/clear_sky_indices"
SIATA_CSI_FILE = f"{CSI_INDEX_DIR}/clearsky_index_Siata.nc"   # Índice CSI de SIATA (target)
SIATA_CSI_VAR  = "clearsky_index_Siata"

# Clearness index (ks): GHI_GFS / GHI_Extraterrestre  →  normalizado 0-1
CLEARNESS_INDEX_DIR = f"{PREP_DATA_DIR}/_04_indices/clearness_indices"

# --- Plantillas de rutas por feature (usar {LT} como placeholder) --
# Se rellenan en tiempo de ejecución con el launch time correspondiente
FEAT_KC_TEMPLATE          = f"{CSI_INDEX_DIR}/clearsky_index_GFS_{{LT}}.nc"
FEAT_KS_TEMPLATE          = f"{CLEARNESS_INDEX_DIR}/clearness_index_GFS_{{LT}}.nc"
FEAT_DSWRF1_TEMPLATE      = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_{{LT}}.nc"
FEAT_DLWRF_TEMPLATE       = f"{PREP_DATA_DIR}/_12_DLWRF/dlwrf1_{{LT}}.nc"
FEAT_TMP_TEMPLATE         = f"{PREP_DATA_DIR}/_05_Temp_surface/TMP_surface_{{LT}}.nc"
FEAT_RH_TEMPLATE          = f"{PREP_DATA_DIR}/_06_RH_2m/RH_2m_{{LT}}.nc"
FEAT_CAPE_TEMPLATE        = f"{PREP_DATA_DIR}/_10_CAPE_surface/CAPE_surface_{{LT}}.nc"
FEAT_HPBL_TEMPLATE        = f"{PREP_DATA_DIR}/_11_HPBL/HPBL_surface_{{LT}}.nc"
FEAT_PWAT_TEMPLATE        = f"{PREP_DATA_DIR}/_13_PWAT_ent/PWAT_ent_{{LT}}.nc"
FEAT_TCDC_TEMPLATE        = f"{PREP_DATA_DIR}/_07_CDC_ent/_01_TCDC/TCDC_ent_{{LT}}.nc"
FEAT_HCDC_TEMPLATE        = f"{PREP_DATA_DIR}/_07_CDC_ent/_02_HCDC/HCDC_high_{{LT}}.nc"
FEAT_MCDC_TEMPLATE        = f"{PREP_DATA_DIR}/_07_CDC_ent/_03_MCDC/MCDC_mid_{{LT}}.nc"
FEAT_LCDC_TEMPLATE        = f"{PREP_DATA_DIR}/_07_CDC_ent/_04_LCDC/LCDC_low_{{LT}}.nc"
FEAT_HGT_TEMPLATE         = f"{PREP_DATA_DIR}/_08_HGT_cloud_ceiling/HGT_cloud_ceiling_{{LT}}.nc"
FEAT_WIND10M_TEMPLATE     = f"{PREP_DATA_DIR}/_09_Wind10m/Wind10m_{{LT}}.nc"
FEAT_SUNSD_TEMPLATE       = f"{PREP_DATA_DIR}/_11_SUNSD/SUNSD_minutes_{{LT}}.nc"

# --- Parámetros de clipping de radiación ---------------------------
CLIP_MIN_DEN = 2    # Umbral mínimo (W/m²): valores por debajo se fijan a 0
CLIP_SEED    = 42   # Semilla aleatoria para el jitter al reemplazar excedencias


# ===================================================================
# _4_LSTM_modules  –  Red neuronal Bi-LSTM
# ===================================================================

LSTM_DIR = "_4_LSTM_modules"

# --- Archivos de secuencias preparadas (.npz) ----------------------
# Secuencias multi-feature con los 4 launch times (archivo principal)
SEQ_NPZ_FILE      = f"{LSTM_DIR}/Prepared_data/4launch_multfeat_sym24.npz"
# Archivo de secuencias de prueba (sin extensión; np.save añade .npy)
SEQ_NPZ_TEST_FILE = f"{LSTM_DIR}/Prepared_data/4launch_multfeat_test"

# --- Directorio de corridas de entrenamiento -----------------------
RUNS_DIR = f"{LSTM_DIR}/_runs/4launch_multfeat_sym"

# Mejor modelo entrenado (pesos del checkpoint con menor pérdida de validación)
BEST_MODEL_FILE = (
    f"{RUNS_DIR}/4launch_Multfeat_sym24_numl3_hidden96_20250701_153709/best_model.pt"
)

# --- Archivos de salida de evaluación ------------------------------
# CSV con predicciones desescaladas para todos los splits (train/val/test)
CSV_OUT = f"{LSTM_DIR}/Evaluation/Sym24_predictions_full.csv"

# --- Índices de test (timestamps del split de prueba) --------------
TEST_INDICES_DIR             = f"{LSTM_DIR}/test_indices"
# Índices del experimento principal multi-feature simétrico
TEST_INDICES_FILE            = f"{TEST_INDICES_DIR}/test_indices_4launch_multfeat_test"
# Índices para comparación baseline (GFS vs SIATA, 1 launch time causal)
TEST_INDICES_COMPARISON_FILE = f"{TEST_INDICES_DIR}/test_indices_multfeat_caus24_CSI_0100.npy"


# ===================================================================
# HIPERPARÁMETROS DE LA RED BI-LSTM
# ===================================================================

N_FEAT     = 69     # Número de features de entrada (4 LT × variables + meta)
HIDDEN     = 96     # Dimensión del estado oculto de la LSTM
NUM_LAYERS = 3      # Número de capas LSTM apiladas
DROPOUT    = 0.25   # Tasa de dropout entre capas (regularización)
BATCH_SIZE = 128    # Tamaño de batch para inferencia y entrenamiento


# ===================================================================
# PARÁMETROS DE ENTRENAMIENTO
# ===================================================================

LR_INIT     = 1e-3   # Tasa de aprendizaje inicial (AdamW)
MIN_LR      = 1e-6   # Tasa de aprendizaje mínima permitida por el scheduler
EPOCHS      = 50     # Número máximo de épocas de entrenamiento
L2_LAMBDA   = 10e-4  # Regularización L2 (weight decay en AdamW)
EARLY_STOP  = 25     # Paciencia de early stopping (épocas sin mejora)
LR_FACTOR   = 0.5    # Factor de reducción de LR en el scheduler ReduceLROnPlateau
LR_PATIENCE = 4      # Paciencia del scheduler antes de reducir LR


# ===================================================================
# PARÁMETROS DE SECUENCIA TEMPORAL
# ===================================================================

SEQ_MODE     = "symmetric"  # Estrategia de ventana: "symmetric" o "causal"
K_LEFT       = 24            # Horas hacia atrás en ventana simétrica
K_RIGHT      = 24            # Horas hacia adelante en ventana simétrica
K            = 24            # Horas de look-back en modo causal
OFF          = None          # Posición del target en ventana simétrica (None = K_LEFT)
VAL_SPLIT    = 0.15          # Fracción de datos para validación (15 %)
TEST_SPLIT   = 0.15          # Fracción de datos para prueba (15 %)
SHUFFLE_SEED = 16            # Semilla para el shuffle antes del split

# --- Canales meta adicionales en el vector de features -------------
INCLUDE_STEP    = True   # Diferencia (observation_time − launch_time) como feature
INCLUDE_DAYFLAG = False  # Indicador binario de día/noche como feature
ADD_HOD         = False  # Hora del día (Hour of Day) como feature
ADD_DOY         = False  # Día del año (Day of Year) como feature
ADD_ZENITH      = True   # Ángulo cenital solar calculado con pvlib como feature
