# Solar Radiation Forecasting — BiLSTM for Medellín
### Empresa Emergente · Valle de Aburrá, Colombia
**Responsible:** Isabela Velasco · **Last updated:** April 2026

---

## Executive Summary

This project develops a **BiLSTM (Bidirectional Long Short-Term Memory) neural network** that corrects the systematic errors of the GFS weather model when predicting solar radiation in Medellín. The network is trained on 4 years of historical data comparing GFS forecasts against real measurements from the SIATA ground station network.

| Item | Status |
|------|--------|
| Pipeline operational | ✅ Complete |
| Baseline training (50 epochs) | ✅ Complete |
| Window size analysis | ✅ Complete |
| Hyperparameter search (Optuna) | ✅ Complete |
| Bahdanau attention mechanism | ✅ Complete |
| Model beats GFS baseline | ⚠️ In progress — SkillScore = −0.311 |
| Architecture sensitivity analysis | 🔄 Next step |

---

## Table of Contents

1. [What this project does](#1-what-this-project-does)
2. [Data sources](#2-data-sources)
3. [Pipeline — how data flows](#3-pipeline--how-data-flows)
4. [Neural network architecture](#4-neural-network-architecture)
5. [Input features — the 69 variables](#5-input-features--the-69-variables)
6. [Results and diagnostics](#6-results-and-diagnostics)
7. [Why SkillScore is negative](#7-why-skillscore-is-negative)
8. [Historical reference — Leonard Merl 2025](#8-historical-reference--leonard-merl-2025)
9. [Data availability](#9-data-availability)
10. [Metrics reference](#10-metrics-reference)
11. [File types](#11-file-types)
12. [Glossary](#13-glossary)
13. [Scientific references](#14-scientific-references)

---

## 1. What This Project Does

### The problem

The **GFS (Global Forecast System)** from NOAA produces hourly solar radiation forecasts for the entire planet. However its spatial resolution of **~28 km** is too coarse to capture the local microclimate of the Aburrá Valley.

Medellín can be raining in El Poblado while it is sunny in Bello — the GFS cannot detect this. These errors directly affect Emergente's ability to accurately estimate solar panel production.

### The solution

A neural network that **learns the systematic relationship** between what the GFS predicts and what SIATA actually measures in Medellín. After training on 4 years of historical data, the network can correct GFS forecasts in real time.

```
GFS forecast (4 launch times) ──► BiLSTM + Attention ──► Corrected GHI (W/m²)
         ↑
   69 input variables
   37-hour time window
```

### Production use case

Every day the GFS launches 4 forecasts (at 01:00, 07:00, 13:00, 19:00 local time). The trained model takes all 4 forecasts as input and produces an improved hourly prediction of solar radiation for the following 24 hours — allowing Emergente to better plan energy production.

---

## 2. Data Sources

### GFS — Global Forecast System (NOAA)

| Property | Value |
|----------|-------|
| Provider | NOAA (USA) |
| Resolution | ~28 km spatial, 1 hour temporal |
| Launch times | 4 per day: 00, 06, 12, 18 UTC |
| Download method | Byte-range HTTP requests (extract only needed variables) |
| Format | GRIB2 → converted to NetCDF |
| Variables used | 15 meteorological variables per launch time |
| Period available | 2021 – 2024 (4 years) |

**What are launch times?**
The GFS runs 4 times per day. Each run ("launch") uses the most recent atmospheric observations and generates forecasts for the next 24 hours. This means for any given hour, there are up to 4 different GFS predictions — each computed from progressively more recent data. This project uses all 4 simultaneously as model inputs.

| UTC | Colombia local | Label in code |
|-----|---------------|--------------|
| 00 UTC | 19:00 prev. day | `0100` |
| 06 UTC | 01:00 | `0700` |
| 12 UTC | 07:00 | `1300` |
| 18 UTC | 13:00 | `1900` |

### SIATA — Sistema de Alerta Temprana del Valle de Aburrá

| Property | Value |
|----------|-------|
| Provider | SIATA (Medellín) |
| Resolution | 1 minute → aggregated to 1 hour |
| Format | CSV → converted to NetCDF |
| Variable | GHI (Global Horizontal Irradiance) in W/m² |
| Role | **Ground truth** — what the model learns to predict |

### Physical radiation references (calculated, not downloaded)

| Reference | Description | Library |
|-----------|-------------|---------|
| Extraterrestrial GHI | Maximum radiation above atmosphere | pvlib |
| Clear-Sky GHI (Ineichen) | Maximum radiation on a perfect clear day in Medellín | pvlib |

These physical ceilings are used to:
1. **Clip impossible values** in GFS and SIATA data
2. **Normalize** radiation to a 0-1 index (CSI/KC)
3. **De-scale** model predictions back to W/m²

---

## 3. Pipeline — How Data Flows

```
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 1 — Data Acquisition                                    │
│                                                                 │
│  NOAA S3 Servers                                                │
│       │                                                         │
│       │ Byte-range HTTP requests                                │
│       │ (download only needed variables)                        │
│       ▼                                                         │
│  GRIB2 fragments ──► cfgrib ──► xarray ──► Clip to Medellín    │
│       │                                    (6.25°N, 75.5°W)    │
│       ▼                                                         │
│  Individual .nc files per hour                                  │
│  (~35,040 files × 4 launch times × 15 variables)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 2 — Postprocessing                                      │
│                                                                 │
│  1. Merge hourly files → 4 continuous time series               │
│     (one per launch time, covering full 4-year period)          │
│                                                                 │
│  2. Deaccumulate DSWRF:                                         │
│     GFS stores cumulative 6-hour averages, not hourly values    │
│     Formula: value(h) = (h-b)×avg(h) - (h-1-b)×avg(h-1)       │
│                                                                 │
│  3. Calculate wind speed:                                       │
│     Wind10m = √(UGRD² + VGRD²)                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 3 — Feature & Target Preparation                        │
│                                                                 │
│  Physical reference:                                            │
│  ├── Extraterrestrial GHI (pvlib, astronomical geometry)        │
│  └── Clear-Sky GHI Ineichen (pvlib, turbidity + horizon)        │
│                                                                 │
│  GFS processing:                                                │
│  ├── Clip values exceeding physical ceilings                    │
│  ├── CSI_GFS = DSWRF / Clear-Sky  (0 to 1)  ← feature          │
│  └── KC_GFS  = DSWRF / Extraterr. (0 to 1)  ← feature          │
│                                                                 │
│  SIATA processing:                                              │
│  ├── Aggregate minutes → hours (special rules for dawn/dusk)    │
│  ├── Clip values exceeding Clear-Sky ceiling                    │
│  └── CSI_SIATA = GHI / Clear-Sky  (0 to 1)  ← TARGET           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODULE 4 — Neural Network                                      │
│                                                                 │
│  Build sequences:                                               │
│  ├── 37-step sliding window (18h past + current + 18h future)   │
│  ├── 69 features per step                                       │
│  ├── Filter NaN sequences                                       │
│  └── Split: 70% train / 15% val / 15% test                     │
│                                                                 │
│  Train BiLSTM:                                                  │
│  ├── Forward + backward LSTM reads the 37-step sequence         │
│  ├── Bahdanau attention weights the most informative steps      │
│  ├── Output: CSI prediction (0 to 1)                            │
│  └── De-scale: CSI × Clear-Sky GHI → W/m²                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Neural Network Architecture

### Layer-by-layer structure

```
INPUT
37 time steps × 69 features = 2,553 values per sequence
         │
    ┌────▼────────────────────────────────┐
    │           LayerNorm                  │
    │  Normalizes input distribution       │
    │  Prevents gradient instability       │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │              BiLSTM                  │
    │                                      │
    │  ──────────────────────────────►     │  Forward LSTM
    │  96 hidden units · 3 layers          │  (reads past → future)
    │                                      │
    │  ◄──────────────────────────────     │  Backward LSTM
    │  96 hidden units · 3 layers          │  (reads future → past)
    │                                      │
    │  Output: 37 vectors × 192 values     │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │        Bahdanau Attention            │
    │                                      │
    │  Learns which of the 37 time steps   │
    │  are most informative per prediction │
    │                                      │
    │  weights = softmax(tanh(W · h_t))    │
    │  context = Σ weights × h_t           │
    │                                      │
    │  Output: 1 vector × 192 values       │
    │  + attention weights (37 values)     │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │           Dropout 25%                │
    │  Randomly disables 25% of neurons    │
    │  during training to prevent          │
    │  overfitting (memorization)          │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │         Linear 192 → 1              │
    │  Final fully connected layer         │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │            Sigmoid                   │
    │  Forces output between 0 and 1       │
    │  (compatible with CSI range)         │
    └────────────────┬────────────────────┘
                     │
              CSI predicted
                     │
                     × Clear-Sky GHI (W/m²)
                     │
              GHI predicted (W/m²)  ← FINAL OUTPUT
```

### Architecture summary

| Component | Configuration | Parameters |
|-----------|--------------|-----------|
| Input | 37 steps × 69 features | — |
| LayerNorm | 69 features | 138 |
| BiLSTM Layer 1 | input=69, hidden=96, bidirectional | 127,488 |
| BiLSTM Layer 2 | input=192, hidden=96, bidirectional | 221,184 |
| BiLSTM Layer 3 | input=192, hidden=96, bidirectional | 221,184 |
| Bahdanau Attention | 192 → 1 | 193 |
| Dropout | p=0.25 | — |
| Linear output | 192 → 1 | 193 |
| Sigmoid | — | — |
| **Total** | | **574,220** |

### Training configuration

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| Epochs | 50 | Complete passes through training data |
| Batch size | 128 | Sequences per weight update |
| Learning rate | 0.001 | Step size for weight updates |
| Optimizer | AdamW | Adaptive learning with weight decay |
| Loss function | MSE | Mean Squared Error on CSI |
| Early stopping | 25 epochs | Stop if validation doesn't improve |
| LR scheduler | ReduceLROnPlateau | Halve LR when validation plateaus |
| Weight decay | 0.001 | L2 regularization |
| Night mask | True | Exclude nighttime hours from loss |

---

## 5. Input Features — The 69 Variables

For each of the **4 launch times** (0100, 0700, 1300, 1900), the following **17 variables** are used as features — totalling 68 variables plus 1 astronomical feature:

| # | Variable | Description | Unit | Normalization |
|---|----------|-------------|------|--------------|
| 1 | `kc_{LT}` | Clearness Index = DSWRF / Extraterrestrial | 0–1 | None |
| 2 | `ks_{LT}` | Clear-Sky Index = DSWRF / Clear-Sky | 0–1 | None |
| 3 | `dswrf1_{LT}` | Solar radiation (deaccumulated) | W/m² | Min-max |
| 4 | `dlwrf1_{LT}` | Downward longwave radiation | W/m² | Min-max |
| 5 | `TMP_surface_{LT}` | Surface temperature | K | Z-score |
| 6 | `RH_2m_{LT}` | Relative humidity at 2m | % | Min-max |
| 7 | `CAPE_surface_{LT}` | Convective Available Potential Energy | J/kg | Min-max |
| 8 | `HPBL_surface_{LT}` | Planetary boundary layer height | m | Min-max |
| 9 | `PWAT_ent_{LT}` | Precipitable water | kg/m² | Min-max |
| 10 | `TCDC_ent_{LT}` | Total cloud cover | % | Min-max |
| 11 | `HCDC_ent_{LT}` | High cloud cover | % | Min-max |
| 12 | `MCDC_ent_{LT}` | Medium cloud cover | % | Min-max |
| 13 | `LCDC_ent_{LT}` | Low cloud cover | % | Min-max |
| 14 | `HGT_cloud_ceiling_{LT}` | Cloud ceiling height | m | Min-max |
| 15 | `Wind10m_{LT}` | Wind speed at 10m = √(U²+V²) | m/s | Min-max |
| 16 | `SUNSD_minutes_{LT}` | Sunshine duration | min | Min-max |
| 17 | `step_{LT}` | Lead time (hours between launch and forecast) | h | Min-max |

**Plus 1 feature independent of launch time:**

| # | Variable | Description | Unit | Normalization |
|---|----------|-------------|------|--------------|
| 69 | `zenith` | Solar zenith angle | degrees | Min-max |

**Total: 17 × 4 launch times + 1 = 69 features**

### Why these variables?

Each variable group contributes different information about atmospheric conditions:

- **CSI and KC** — normalized radiation indices that directly encode how much of the physical maximum is being realized
- **Cloud cover (3 levels)** — the primary driver of radiation reduction at surface
- **CAPE** — energy available for convective cloud formation (thunderstorm indicator)
- **Precipitable water** — atmospheric moisture content, related to cloud formation
- **Temperature and humidity** — atmospheric stability indicators
- **Wind speed** — horizontal advection of cloud systems
- **Sunshine duration** — historical sunshine in the forecast period
- **Step (lead time)** — GFS accuracy degrades with longer lead times; the network learns to weight recent forecasts more
- **Zenith angle** — controls the geometric maximum of radiation regardless of atmospheric conditions

---

## 6. Results and Diagnostics

### Baseline training results (50 epochs, sym18 window)

| Metric | All hours | Daytime only | Interpretation |
|--------|-----------|-------------|----------------|
| RMSE | 171.1 W/m² | 231.2 W/m² | Average error magnitude |
| MAE | 82.5 W/m² | 154.1 W/m² | Typical error |
| R² | 0.671 | 0.472 | Variability explained |
| Correlation | 0.837 | — | Trend captured |
| SkillScore | — | −0.311 | vs GFS raw (daytime) |

> ⚠️ The all-hours RMSE of 171 W/m² includes nighttime where both model and GFS predict 0. The honest daytime-only RMSE is 231 W/m². Always use daytime-only metrics for fair comparison.

### Diagnostic plots

**Loss curve — training vs validation**

![Loss curve](docs/images/loss_curve.png)

*The training loss (blue) continues decreasing while validation (orange) stabilizes after epoch 20 — a classic sign of overfitting. The network memorizes training patterns instead of learning general rules.*

---

**Scatter plot — predicted vs observed (W/m²)**

![Scatter real](docs/images/scatter_real.png)

*Points should lie on the red diagonal for perfect predictions. Two problems are visible: (1) systematic overestimation in the 200-600 W/m² range (cloud points above diagonal); (2) horizontal cluster at y=0 where the network predicts night during daytime hours (dawn/dusk transitions).*

---

**Error histograms — GFS raw vs model (daytime hours only)**

![Error histograms](docs/images/error_histograms.png)

*The GFS raw errors (orange, left) are roughly symmetric around zero — it over and underestimates equally. The model errors (blue, right) show a heavy left tail — the model systematically overestimates radiation, especially on partially cloudy days.*

---

**Time series — 2-week sample**

![Time series](docs/images/timeseries_sample.png)

*Black = SIATA observed. Orange = GFS raw. Blue = model. The model closely follows the GFS instead of correcting it. During 10-14 May, when GFS predicted ~800 W/m² but SIATA measured ~200-300 W/m² (cloudy days), the model also failed to detect the cloudiness.*

---

**Monthly MSE comparison**

![Monthly MSE](docs/images/monthly_mse.png)

*Orange bars = GFS raw MSE. Blue bars = model MSE. The model MSE is higher than GFS in almost every month — confirming the negative SkillScore is consistent across time, not driven by outlier months.*

---

### Experiment comparison table

| Experiment | Window | Hidden | Layers | RMSE (all hrs) | SkillScore (daytime) |
|-----------|--------|--------|--------|----------------|---------------------|
| Baseline sym24 | 49 steps | 96 | 3 | 171.59 W/m² | −0.319 |
| Window sym12 | 25 steps | 96 | 3 | 180.36 W/m² | −0.457 |
| **Window sym18** | **37 steps** | **96** | **3** | **171.10 W/m²** | **−0.311** |
| Optuna best (20 ep) | 37 steps | 192 | 1 | 172.55 W/m² | — |
| sym18 + Attention | 37 steps | 96 | 3 | 171.11 W/m² | −0.311 |

**Selected configuration:** sym18 (37 steps) with 3 layers, hidden=96, Bahdanau attention.
**Reason:** Equivalent performance to sym24 with 25% less computation. Attention added negligible overhead (+193 parameters) with no degradation.

---

## 7. Why SkillScore is Negative

The SkillScore of −0.311 means the model currently predicts **31% worse than simply using the raw GFS forecast without correction**. This is the main challenge to resolve.

### Root cause analysis

**1. The model follows the GFS instead of correcting it**

Looking at the time series plot, the model (blue) and GFS raw (orange) are nearly identical. The network learned to copy its primary input rather than learn the correction. This happens because the GFS CSI and the SIATA CSI target are highly correlated — the network found the path of least resistance.

**2. Convective cloudiness is unpredictable from GFS**

Medellín's cloudiness is largely driven by local convective processes (afternoon thunderstorms, valley fog) that develop at scales smaller than the GFS resolution. No amount of GFS data can predict these events perfectly — there is an inherent limit to how much correction is possible from GFS inputs alone.

**3. Systematic overestimation bias**

The model consistently overestimates radiation on partially cloudy days. This creates a business risk: Emergente could commit to delivering more energy than the panels actually produce.

**4. Daytime RMSE hidden by nighttime zeros**

The all-hours RMSE of 171 W/m² appears reasonable but is diluted by ~45% nighttime hours where both model and truth are 0. The true daytime RMSE is 231 W/m² — significantly worse than GFS raw (201.9 W/m²).

### What is being done

| Action | Expected impact |
|--------|----------------|
| Architecture sensitivity analysis (5 layer counts × 5 hidden sizes) | Find optimal geometry that reduces overfitting |
| Daytime-only loss function | Train exclusively on hours that matter for the business |
| Second Optuna round (35-40 epochs per trial) | More reliable hyperparameter search |
| Causal vs bidirectional comparison | Test if forward-only LSTM reduces overfitting to future GFS |

### Business implication

Until SkillScore turns positive, the raw GFS forecast is more accurate than the corrected model for production planning. The GFS provides a reasonable baseline but still misses localized cloud events. The goal remains to train a model that reliably corrects these events — improving both accuracy and business reliability.

---

## 8. Historical Reference — Leonard Merl (2025)

The pipeline was originally developed by Leonard Merl. His experiments provide important context.

### Architecture differences

| Property | Leonard's model | Current model |
|----------|----------------|---------------|
| Launch times | 1 (0100 only) | 4 (all) |
| Direction | Causal (past only) | Bidirectional |
| Features | ~17 variables | 69 variables |
| Window | 24-36 steps | 37 steps |

### Leonard's results (from `_runs/` folders dated July 2025)

| Run name | Hidden | Layers | Window | Notes |
|----------|--------|--------|--------|-------|
| `numl3_hidden96` | 96 | 3 | sym24 | Closest to current config |
| `numl5_hidden128` | 128 | 5 | sym24 | Deeper architecture |
| `numl1_hidden64` | 64 | 1 | sym36 | Simpler, longer window |

### Why Leonard's SkillScore appeared positive

Leonard's `MSE_BASELINE_R = 22,322` was computed including nighttime hours. At night both GFS and model predict 0, contributing 0 error and diluting the baseline MSE. The daytime-only baseline is `40,761` — 82% higher.

```
Leonard's SS = 1 - MSE_model / 22,322  → appeared positive
Correct SS   = 1 - MSE_model / 40,761  → would be less positive or negative
```

This does not mean Leonard's work was wrong — it means the SkillScore metric was not computed consistently across experiments. A fair comparison requires running `verify_skillscore.py` on Leonard's `pred_real.csv` using the corrected daytime-only baseline.

---

## 9. Data Availability

### What is on disk

| Module | Content | Period | Size |
|--------|---------|--------|------|
| `_1_` | Raw GFS NetCDF (15 vars × 4 LT) | 2021–2024 | ~50 GB |
| `_2_` | Merged time series per variable/LT | 2021–2024 | ~8 GB |
| `_3_` | Clear-Sky, CSI indices, SIATA | 2021–2024 | ~2 GB |
| `_4_/Prepared_data/` | Training sequences .npz | 2021–2024 | ~150–450 MB |
| `_4_/_runs/` | Training results and predictions | 2025–2026 | ~500 MB |

### Train / Validation / Test split

```
Total valid sequences (sym18): 35,553
├── Train (70%):  24,889 sequences  ← used to update weights
├── Val   (15%):   5,332 sequences  ← used to monitor training
└── Test  (15%):   5,332 sequences  ← used only for final evaluation
```

The test set timestamps are fixed in `_4_LSTM_modules/test_indices/test_indices_4launch_multfeat_sym18.npy`. Any SkillScore calculation must use exactly these timestamps to be comparable across experiments.

---

## 10. Metrics Reference

| Metric | Formula | Units | Interpretation |
|--------|---------|-------|----------------|
| **RMSE** | √mean((pred−true)²) | W/m² | Average error magnitude. Sensitive to large errors |
| **MAE** | mean(\|pred−true\|) | W/m² | Typical error. Less sensitive to outliers than RMSE |
| **R²** | 1 − MSE_model/MSE_mean | — | Fraction of variability explained. 1=perfect, 0=no better than mean |
| **Correlation** | cov(pred,true)/(σ_pred·σ_true) | — | Whether predictions move with reality. High correlation ≠ low error |
| **SkillScore** | 1 − MSE_model/MSE_GFS_raw | — | Improvement over GFS. >0 = better than GFS, <0 = worse |

> ⚠️ **SkillScore must be computed on daytime hours only** (Clear-Sky GHI > 0). Including nighttime zeros inflates the GFS baseline MSE and produces an artificially negative SkillScore. Use `verify_skillscore.py` for correct calculation.

---

## 11. File Types

| Extension | Name | Used for | How to open |
|-----------|------|---------|-------------|
| `.nc` | NetCDF | GFS data, SIATA, radiation indices | `xr.open_dataset(path, engine='h5netcdf')` |
| `.npz` | NumPy compressed | Training sequences | `np.load(path)` |
| `.pt` | PyTorch model | Saved network weights | `torch.load(path)` |
| `.csv` | Comma separated | Predictions and results | `pd.read_csv(path)` |
| `.npy` | NumPy array | Test set timestamps | `np.load(path, allow_pickle=True)` |
| `.py` | Python script | Pipeline code | Any text editor / VS Code |
| `.md` | Markdown | Documentation | GitHub (rendered automatically) |

> ⚠️ **Windows path note:** Always use `engine='h5netcdf'` when opening `.nc` files with xarray. The default netCDF4 backend corrupts paths containing accented characters (e.g. `ó`).

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **GFS** | Global Forecast System — NOAA's global weather model |
| **DSWRF** | Downward Shortwave Radiation Flux — solar radiation at surface (W/m²) |
| **GHI** | Global Horizontal Irradiance — solar radiation on horizontal surface (W/m²) |
| **SIATA** | Ground station network measuring real conditions in Medellín |
| **CSI** | Clear-Sky Index = GHI / Clear-Sky GHI. Range 0–1 |
| **KC** | Clearness Index = GHI / Extraterrestrial GHI. Range 0–1 |
| **Clear-Sky GHI** | Maximum radiation on a perfectly clear day (no clouds) |
| **Extraterrestrial GHI** | Radiation above the atmosphere — absolute physical maximum |
| **Launch time** | When GFS runs its forecast (4 times per day) |
| **Feature** | Input variable to the neural network (one of 69) |
| **Target** | What the network predicts (SIATA CSI → W/m²) |
| **LSTM** | Long Short-Term Memory — neural network with temporal memory |
| **BiLSTM** | Bidirectional LSTM — processes sequence both forward and backward |
| **Attention** | Mechanism that learns which time steps matter most for each prediction |
| **Epoch** | One complete pass through all training sequences |
| **Batch** | Group of sequences processed together before updating weights |
| **Overfitting** | Model memorizes training data instead of learning general patterns |
| **SkillScore** | Improvement over GFS raw (positive = better than GFS) |
| **RMSE** | Root Mean Squared Error — average error in W/m² |
| **R²** | Coefficient of determination — fraction of variability explained |
| **De-scaling** | Converting CSI prediction (0–1) back to W/m² via × Clear-Sky |
| **PYTHONPATH** | Environment variable telling Python where to find project modules |
| **config.py** | Central file containing all paths and parameters |

---

## 13. Scientific References

| Method | Reference |
|--------|-----------|
| Ineichen Clear-Sky model | Ineichen & Perez (2002). *Solar Energy*, 73(3), 151–157 |
| Linke turbidity | Remund et al. (2003). *ISES Solar World Congress* |
| pvlib Python library | Holmgren et al. (2018). *Journal of Open Source Software* |
| LSTM architecture | Hochreiter & Schmidhuber (1997). *Neural Computation*, 9(8), 1735–1780 |
| Bahdanau Attention | Bahdanau et al. (2015). *ICLR 2015* |
| Optuna / TPE algorithm | Akiba et al. (2019). *KDD 2019* |
| Nash-Sutcliffe Efficiency | Nash & Sutcliffe (1970). *Journal of Hydrology*, 10(3), 282–290 |
| Solar forecasting review | Voyant et al. (2017). *Renewable Energy*, 105, 569–582 |

---

*Empresa Emergente · Valle de Aburrá, Colombia · April 2026*
*Responsible: Isabela Velasco · isvelascor1061@github*
