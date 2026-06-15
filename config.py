#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py  –  Central configuration for the Solar Medellín project (Emergente)
===============================================================================
Centralises all file paths and global pipeline parameters.
Organised by section according to the 4 main project folders.

Usage:
    from config import LAT, LON, SEQ_NPZ_FILE, ...
"""

# ===================================================================
# NETCDF VARIABLE NAMING CONVENTION
# ===================================================================
# This section centralises all variable names that appear inside the
# NetCDF files of the project, following the convention agreed for the
# Solar Medellín Pipeline (Proyecto Emergente).
#
# Prefix scheme:
#   gfs_dswrf_acum   → GFS radiation *accumulated* (raw server variable, sdswrf)
#   gfs_dswrf        → GFS radiation *de-accumulated* to 1-h intervals (generic)
#   gfs_dswrf_<LT>   → Same, specific to a launch time (0100/0700/1300/1900)
#   gfs_csi          → GFS Clear-Sky Index  = DSWRF_GFS / Clear-Sky Ineichen
#   gfs_kc           → GFS Clearness Index  = DSWRF_GFS / Extraterrestrial radiation
#   siata_ghi        → SIATA observed GHI, cleaned (no clear-sky exceedances)
#   siata_csi        → SIATA Clear-Sky Index = GHI_SIATA / Clear-Sky Ineichen
#   ref_clearsky_ghi → Clear-sky radiation computed with the Ineichen model
#   ref_ext_ghi      → Extraterrestrial radiation (physical upper limit of the atmosphere)
#
# NOTE: Variable names INSIDE existing .nc files are NOT modified to avoid
#       invalidating already-generated data. The VAR_* constants simply
#       document which physical concept each string corresponds to and serve
#       as the single source of truth across all Python scripts.
# ===================================================================

# — GFS radiation —
VAR_GFS_DSWRF_ACUM     = "sdswrf"                    # GFS accumulated (raw NOAA server variable)
VAR_GFS_DSWRF          = "dswrf1"                    # GFS de-accumulated generic (no specific LT)
VAR_GFS_DSWRF_TEMPLATE = "dswrf1_{LT}"               # Per-LT template: .format(LT="0100")
VAR_GFS_DSWRF_0100     = "dswrf1_0100"               # GFS de-accumulated, launch time 01:00 h
VAR_GFS_DSWRF_0700     = "dswrf1_0700"               # GFS de-accumulated, launch time 07:00 h
VAR_GFS_DSWRF_1300     = "dswrf1_1300"               # GFS de-accumulated, launch time 13:00 h
VAR_GFS_DSWRF_1900     = "dswrf1_1900"               # GFS de-accumulated, launch time 19:00 h
VAR_GFS_CSI_TEMPLATE   = "clearsky_index_GFS_{LT}"  # GFS CSI per LT (DSWRF / Clear-Sky Ineichen)
VAR_GFS_KC_TEMPLATE    = "clearness_index_GFS_{LT}" # GFS KC per LT (DSWRF / Extraterrestrial)

# — SIATA observed radiation —
VAR_SIATA_GHI = "GHI_clean"            # SIATA measured GHI, cleaned of exceedances
VAR_SIATA_CSI = "clearsky_index_Siata" # SIATA Clear-Sky Index (GHI_SIATA / Ineichen)

# — Solar references (physical models) —
VAR_REF_CLEARSKY_GHI = "clear_sky_ghi"        # Ineichen model with horizon and bias correction
VAR_REF_EXT_GHI      = "extraterrestrial_ghi" # Radiation at the top of the atmosphere


# ===================================================================
# GEOGRAPHIC AND TEMPORAL PARAMETERS FOR MEDELLÍN
# ===================================================================

# Coordinates of the measurement point (closest GFS grid cell to Medellín)
LAT       = 6.25    # Latitude in degrees north
LON       = -75.5   # Longitude in degrees west (negative = western hemisphere)
ELEVATION = 1485    # Altitude above sea level in metres

# Colombia time zone offset from UTC (no daylight saving time)
UTC_OFFSET = -5     # Colombia = UTC-5

# Available GFS model launch times (local Colombia time)
LAUNCH_TIMES = ["0100", "0700", "1300", "1900"]

# Diurnal window for the day/night mask (local time)
FLAG_START = 19     # Hour at which night begins locally (19:00)
FLAG_END   = 6      # Hour at which night ends locally (06:00)


# ===================================================================
# _1_data_acquisition  –  GFS data download from NOAA S3
# ===================================================================

# --- Short-wave radiation (DSWRF) in accumulated intervals ----------
# Root folder for raw files downloaded per launch time
RAW_DSWRF_DIR      = "_1_data_acquisition/_01_raw_rad_data/dswrf"
RAW_DSWRF_0100_DIR = f"{RAW_DSWRF_DIR}/raw_dswrf_0100"   # Launch 01:00 h local
RAW_DSWRF_0700_DIR = f"{RAW_DSWRF_DIR}/raw_dswrf_0700"   # Launch 07:00 h local
RAW_DSWRF_1300_DIR = f"{RAW_DSWRF_DIR}/raw_dswrf_1300"   # Launch 13:00 h local
RAW_DSWRF_1900_DIR = f"{RAW_DSWRF_DIR}/raw_dswrf_1900"   # Launch 19:00 h local

# --- MultiGFS: all meteorological variables together ---------------
# Root folder for raw MultiGFS files per launch time
RAW_MULTGFS_DIR      = "_1_data_acquisition/_02_raw_MultGFS_data"
RAW_MULTGFS_0100_DIR = f"{RAW_MULTGFS_DIR}/raw_MultGFS_0100"   # Launch 01:00 h local
RAW_MULTGFS_0700_DIR = f"{RAW_MULTGFS_DIR}/raw_MultGFS_0700"   # Launch 07:00 h local
RAW_MULTGFS_1300_DIR = f"{RAW_MULTGFS_DIR}/raw_MultGFS_1300"   # Launch 13:00 h local
RAW_MULTGFS_1900_DIR = f"{RAW_MULTGFS_DIR}/raw_MultGFS_1900"   # Launch 19:00 h local

# --- Variable names in the S3 server files -------------------------
GFS_VAR_RAW = VAR_GFS_DSWRF_ACUM   # Accumulated short-wave variable on the server (interval mean)
GFS_VAR_OUT = VAR_GFS_DSWRF        # Hourly variable produced after de-accumulation


# ===================================================================
# _2_data_postprocessing  –  GFS data post-processing
# ===================================================================

# --- Step 1: Raw merged radiation per day (before converting to 1 h) ---
# One file per day and launch time, merged across all forecast steps
MERGED_DSWRF_RAW_DIR      = "_2_data_postprocessing/_01_merged_Radrawdata/dswrf"
MERGED_DSWRF_RAW_0100_DIR = f"{MERGED_DSWRF_RAW_DIR}/merged_raw_dswrf_0100"   # LT 01:00 h
MERGED_DSWRF_RAW_0700_DIR = f"{MERGED_DSWRF_RAW_DIR}/merged_raw_dswrf_0700"   # LT 07:00 h
MERGED_DSWRF_RAW_1300_DIR = f"{MERGED_DSWRF_RAW_DIR}/merged_raw_dswrf_1300"   # LT 13:00 h
MERGED_DSWRF_RAW_1900_DIR = f"{MERGED_DSWRF_RAW_DIR}/merged_raw_dswrf_1900"   # LT 19:00 h

# --- Step 2: De-accumulated hourly radiation (dswrf1) ---------------
# One file per day and launch time, variable converted to 1-h intervals
MERGED_DSWRF1_DIR      = "_2_data_postprocessing/_02_merged_Rad1data/dswrf"
MERGED_DSWRF1_0100_DIR = f"{MERGED_DSWRF1_DIR}/merged_raw_dswrf1_0100"   # LT 01:00 h
MERGED_DSWRF1_0700_DIR = f"{MERGED_DSWRF1_DIR}/merged_raw_dswrf1_0700"   # LT 07:00 h
MERGED_DSWRF1_1300_DIR = f"{MERGED_DSWRF1_DIR}/merged_raw_dswrf1_1300"   # LT 13:00 h
MERGED_DSWRF1_1900_DIR = f"{MERGED_DSWRF1_DIR}/merged_raw_dswrf1_1900"   # LT 19:00 h

# --- Step 3: MultiGFS merged into one file per launch time ---------
# Contains all meteorological variables across the full time series
MERGED_MULTGFS_DIR       = "_2_data_postprocessing/_03_Merged_MultGFS_data"
MERGED_MULTGFS_0100_FILE = f"{MERGED_MULTGFS_DIR}/MultGFS_0100_2.nc"   # LT 01:00 h
MERGED_MULTGFS_0700_FILE = f"{MERGED_MULTGFS_DIR}/MultGFS_0700_2.nc"   # LT 07:00 h
MERGED_MULTGFS_1300_FILE = f"{MERGED_MULTGFS_DIR}/MultGFS_1300_2.nc"   # LT 13:00 h
MERGED_MULTGFS_1900_FILE = f"{MERGED_MULTGFS_DIR}/MultGFS_1900_2.nc"   # LT 19:00 h

# Default launch time for scripts that process one launch time per run
LAUNCH_TIME_DEFAULT = "0100"


# ===================================================================
# _3_Data_preparation_for_LSTM  –  Data preparation for the network
# ===================================================================

# Base directory for all prepared data
PREP_DATA_DIR = "_3_Data_preparation_for_LSTM/Preparation_data"

# --- Solar reference radiation ------------------------------------
# Extraterrestrial radiation (physical maximum ceiling of radiation)
EXT_GHI_FILE = f"{PREP_DATA_DIR}/_01_CSI_EXT_radiation/Extraterrestrial_GHI/EXT_GHI_all.nc"
EXT_VAR_NAME = VAR_REF_EXT_GHI   # Variable name inside the NetCDF

# Clear-sky Ineichen radiation (with horizon profile and bias correction)
CSI_GHI_FILE = (
    f"{PREP_DATA_DIR}/_01_CSI_EXT_radiation/Ineichen_GHI/"
    "CSI_GHI_grid25_avg_with_horizon_and_enhancement_with_bias_correct2.nc"
)
CSI_VAR_NAME = VAR_REF_CLEARSKY_GHI   # Clear-sky variable name

# --- dswrf1: GFS radiation without clipping, one time series per LT --
DSWRF1_UNCLIPPED_DIR = f"{PREP_DATA_DIR}/_02_GFS_dswrf1/Unclipped_merged_dswrf1"
DSWRF1_0100_FILE     = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_0100.nc"   # LT 01:00 h
DSWRF1_0700_FILE     = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_0700.nc"   # LT 07:00 h
DSWRF1_1300_FILE     = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_1300.nc"   # LT 13:00 h
DSWRF1_1900_FILE     = f"{DSWRF1_UNCLIPPED_DIR}/dswrf1_1900.nc"   # LT 19:00 h

# --- dswrf1 clipped by Clear-Sky Index (CSI) -----------------------
# Values exceeding the clear-sky reference are replaced by interpolation
DSWRF1_CSI_DIR       = f"{PREP_DATA_DIR}/_02_GFS_dswrf1/GFS_merged_CSI_clipped"
DSWRF1_CSI_0100_FILE = f"{DSWRF1_CSI_DIR}/dswrf1_CSI_0100.nc"   # LT 01:00 h
DSWRF1_CSI_0700_FILE = f"{DSWRF1_CSI_DIR}/dswrf1_CSI_0700.nc"   # LT 07:00 h
DSWRF1_CSI_1300_FILE = f"{DSWRF1_CSI_DIR}/dswrf1_CSI_1300.nc"   # LT 13:00 h
DSWRF1_CSI_1900_FILE = f"{DSWRF1_CSI_DIR}/dswrf1_CSI_1900.nc"   # LT 19:00 h

# --- dswrf1 clipped by extraterrestrial radiation (EXT) ------------
DSWRF1_EXT_DIR = f"{PREP_DATA_DIR}/_02_GFS_dswrf1/GFS_merged_EXT_clipped"

# --- SIATA observed data (measured global horizontal irradiance) ---
SIATA_GHI_FILE = f"{PREP_DATA_DIR}/_03_Siata_GHI/Netcdf_Siata_GHI/GHI_CSI_clipped.nc"
SIATA_GHI_VAR  = VAR_SIATA_GHI   # Observed GHI variable name (SIATA cleaned)

# --- Computed solar indices ----------------------------------------
# Clear-sky index (gfs_csi): GHI_GFS / GHI_ClearSky  →  normalised 0-1
CSI_INDEX_DIR  = f"{PREP_DATA_DIR}/_04_indices/clear_sky_indices"
SIATA_CSI_FILE  = f"{CSI_INDEX_DIR}/clearsky_index_Siata.nc"   # SIATA CSI (prediction target)
SIATA_CSI_VAR   = VAR_SIATA_CSI
SIATA_CLIM_FILE = f"{PREP_DATA_DIR}/siata_climatology.nc"       # Pre-computed SIATA climatology

# Clearness index (gfs_kc): GHI_GFS / GHI_Extraterrestrial  →  normalised 0-1
CLEARNESS_INDEX_DIR = f"{PREP_DATA_DIR}/_04_indices/clearness_indices"

# --- Per-feature path templates (use {LT} as placeholder) ----------
# Filled at runtime with the corresponding launch time
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

# --- Radiation clipping parameters ---------------------------------
CLIP_MIN_DEN = 2    # Minimum threshold (W/m²): values below this are set to 0
CLIP_SEED    = 42   # Random seed for jitter when replacing exceedances

# --- Elevation and horizon profile file ----------------------------
# TIF raster downloaded from OpenTopography for the Medellín grid cell
HORIZON_FILE    = "_3_Data_preparation_for_LSTM/Preparation_data/Elevation_data/Medellin.tif"
# Output directory for Ineichen clear-sky radiation
CSI_GHI_OUT_DIR = f"{PREP_DATA_DIR}/_01_CSI_EXT_radiation/Ineichen_GHI"

# --- Zenith bias correction parameters ----------------------------
# The Ineichen model underestimates GHI at high zenith angles (> 60°)
ZENITH_CORR      = True    # Enable zenith bias correction (True/False)
ZENITH_ALPHA     = 0.003   # Correction increment per degree of zenith (0.3 % per degree)
ZENITH_THRESHOLD = 60      # Zenith angle above which the correction is applied


# ===================================================================
# _4_LSTM_modules  –  Bi-LSTM neural network
# ===================================================================

LSTM_DIR = "_4_LSTM_modules"

# --- Prepared sequence files (.npz) --------------------------------
# Multi-feature sequences with all 4 launch times (main file)
SEQ_NPZ_FILE      = f"{LSTM_DIR}/Prepared_data/4launch_multfeat_sym18.npz"       # sym18 + attention run
SEQ_NPZ_CLIM_FILE = f"{LSTM_DIR}/Prepared_data/4launch_multfeat_sym18_clim.npz"  # + 10 climatology features
# Test sequence file (no extension; np.save appends .npy)
SEQ_NPZ_TEST_FILE = f"{LSTM_DIR}/Prepared_data/4launch_multfeat_test"

# --- Training run directory ----------------------------------------
RUNS_DIR = f"{LSTM_DIR}/_runs/4launch_multfeat_sym"

# Best trained model (checkpoint weights with the lowest validation loss)
BEST_MODEL_FILE = (
    f"{RUNS_DIR}/4launch_Multfeat_sym24_numl3_hidden96_20250701_153709/best_model.pt"
)

# --- Evaluation output files ---------------------------------------
# CSV with de-scaled predictions for all splits (train/val/test)
CSV_OUT = f"{LSTM_DIR}/Evaluation/Sym24_predictions_full.csv"

# --- Test indices (timestamps for the test split) ------------------
TEST_INDICES_DIR             = f"{LSTM_DIR}/test_indices"
# Indices for the main multi-feature symmetric experiment
TEST_INDICES_FILE            = f"{TEST_INDICES_DIR}/test_indices_4launch_multfeat_test"
# Indices for the climatology-enriched sequences (sym18 + 10 clim features)
TEST_INDICES_CLIM_FILE       = f"{TEST_INDICES_DIR}/test_indices_4launch_multfeat_clim"
# Indices for baseline comparison (GFS vs SIATA, 1 causal launch time)
TEST_INDICES_COMPARISON_FILE = f"{TEST_INDICES_DIR}/test_indices_multfeat_caus24_CSI_0100.npy"


# ===================================================================
# BI-LSTM NETWORK HYPERPARAMETERS
# ===================================================================

N_FEAT         = 69   # Number of input features (4 LT × variables + meta)
N_CLIM_FEATURES = 10  # Climatological features injected per time step (see compute_siata_climatology.py)
# Total features with climatology: N_FEAT + N_CLIM_FEATURES = 79
HIDDEN     = 96     # Hidden state dimension of the LSTM
NUM_LAYERS = 3      # Number of stacked LSTM layers
DROPOUT    = 0.25   # Dropout rate between layers (regularisation)
BATCH_SIZE = 128    # Batch size for inference and training


# ===================================================================
# TRAINING AND TRAINER BEHAVIOUR PARAMETERS
# ===================================================================

LR_INIT     = 1e-3   # Initial learning rate (AdamW)
MIN_LR      = 1e-6   # Minimum learning rate allowed by the scheduler
EPOCHS      = 50     # Maximum number of training epochs
L2_LAMBDA   = 10e-4  # L2 regularisation (weight decay in AdamW)
EARLY_STOP  = 25     # Early stopping patience (epochs without improvement)
LR_FACTOR   = 0.5    # LR reduction factor in the ReduceLROnPlateau scheduler
LR_PATIENCE = 4      # Scheduler patience before reducing LR

# Output activation function of the Bi-LSTM
# "sigmoid" → output bounded in (0, 1), suitable when the target is the normalised CSI
# "linear"  → no activation, suitable when the target is not normalised to [0, 1]
ACTIVATION = "sigmoid"

# De-scaling method to convert normalised predictions to W/m²
# "physical" → multiplies by the clear-sky radiation at the same instant
# "minmax"   → inverse of min-max normalisation
# "z_score"  → inverse of standardisation (mean=0, std=1)
# "average"  → inverse of mean normalisation
DESCALER_METHOD = "physical"

# Apply night mask during training
# True  → night-time predictions are set to 0 and do not contribute to the gradient
# False → the model learns all hours equally (including night-time)
USE_DAYMASK = True

# MSE of the GFS reference model (computed with Comparison_before_NN.py)
# Used to calculate the Skill Score: SS = 1 - MSE_model / MSE_GFS
MSE_BASELINE_R = 40761.472609


# ===================================================================
# TEMPORAL SEQUENCE PARAMETERS
# ===================================================================

SEQ_MODE     = "sym18"  # Window strategy: "symmetric" or "causal"
K_LEFT       = 18            # Hours look-back in symmetric window
K_RIGHT      = 18            # Hours look-ahead in symmetric window
K            = 24            # Look-back hours in causal mode
OFF          = None          # Target position in symmetric window (None = K_LEFT)
VAL_SPLIT    = 0.15          # Fraction of data for validation (15 %)
TEST_SPLIT   = 0.15          # Fraction of data for test (15 %)
SHUFFLE_SEED = 16            # Seed for the shuffle before splitting

# --- Additional meta channels in the feature vector ----------------
INCLUDE_STEP    = True   # Difference (observation_time − launch_time) as a feature
INCLUDE_DAYFLAG = False  # Binary day/night indicator as a feature
ADD_HOD         = False  # Hour of Day as a feature
ADD_DOY         = False  # Day of Year as a feature
ADD_ZENITH      = True   # Solar zenith angle computed with pvlib as a feature
