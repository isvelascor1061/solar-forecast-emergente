#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM_trainer.py
===============
Script unificado de entrenamiento de la red Bi-LSTM para predicción
de irradiancia solar en Medellín (Proyecto Emergente).

Unifica los 5 scripts anteriores:
    • first_LSTM_test_CSI.py
    • LSTM_test_descaler_linear.py
    • LSTM_test_descaler_sigmoid.py
    • LSTM_test_descalersigmoid_daymask.py
    • LSTM_test_descaler_sigmoid_dayflag.py

Comportamiento configurable desde config.py:
    • ACTIVATION      : "sigmoid" o "linear" — función de activación de salida
    • DESCALER_METHOD : "physical" | "minmax" | "z_score" | "average"
    • USE_DAYMASK     : True/False — máscara de noche durante entrenamiento

Flujo principal:
    1. Cargar splits (train/val/test) desde el archivo .npz
    2. Construir la Bi-LSTM con los hiperparámetros de config.py
    3. Entrenar con AdamW + ReduceLROnPlateau + early stopping
       (si USE_DAYMASK=True: predicciones nocturnas → 0, pérdida nocturna = 0)
    4. Evaluar en espacio normalizado y real (desescalado)
    5. Generar gráficas: curva de pérdida, scatter, histogramas de residuos
    6. Guardar CSV con predicciones, imagen resumen 3×3 y report.txt
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from tqdm import trange
from tqdm.auto import tqdm
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
)

# Arquitectura principal: Bi-LSTM bidireccional
from _4_LSTM_modules.NN_modules.BiLSTMRegressor import BiLSTMRegressor

# TensorBoard (opcional — solo si está instalado)
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD = True
except ImportError:
    _TENSORBOARD = False

# -----------------------------------------------------------------------
# Importar todos los hiperparámetros y rutas desde config.py
# -----------------------------------------------------------------------
from config import (
    SEQ_NPZ_FILE, RUNS_DIR,
    CSI_GHI_FILE, CSI_VAR_NAME,
    LR_INIT, MIN_LR, EPOCHS, BATCH_SIZE, HIDDEN, NUM_LAYERS, DROPOUT,
    L2_LAMBDA, EARLY_STOP, LR_FACTOR, LR_PATIENCE,
    ACTIVATION, DESCALER_METHOD, USE_DAYMASK,
    FLAG_START, FLAG_END, UTC_OFFSET,
    MSE_BASELINE_R,
)

# -----------------------------------------------------------------------
# CONFIGURACIÓN DE LA CORRIDA — único bloque que el usuario debe editar
# -----------------------------------------------------------------------
# Nombre descriptivo de la corrida (se añade timestamp automáticamente)
RUN_NAME = "4launch_Multfeat_sym24_BiLSTM"

# Ruta al archivo .npz con las secuencias preparadas
SEQ_NPZ = SEQ_NPZ_FILE

# Especificación del desescalado: (ruta NetCDF, variable)
# El método se controla con DESCALER_METHOD en config.py
DESCALER_FILE = CSI_GHI_FILE
DESCALER_VAR  = CSI_VAR_NAME

# -----------------------------------------------------------------------
# Crear directorio de la corrida con timestamp
# -----------------------------------------------------------------------
_NOW    = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUNS_DIR, f"{RUN_NAME}_{_NOW}")
os.makedirs(RUN_DIR, exist_ok=True)

# Diccionario con todas las rutas de artefactos de la corrida
PATHS = dict(
    model       = os.path.join(RUN_DIR, "best_model.pt"),
    csv_n       = os.path.join(RUN_DIR, "pred_norm.csv"),
    csv_r       = os.path.join(RUN_DIR, "pred_real.csv"),
    loss        = os.path.join(RUN_DIR, "loss_curve.png"),
    scatter_n   = os.path.join(RUN_DIR, "scatter_norm.png"),
    scatter_r   = os.path.join(RUN_DIR, "scatter_real.png"),
    hist_n      = os.path.join(RUN_DIR, "hist_norm.png"),
    hist_r      = os.path.join(RUN_DIR, "hist_real.png"),
    hist_n_zeros= os.path.join(RUN_DIR, "hist_norm_without_zeros.png"),
    hist_r_zeros= os.path.join(RUN_DIR, "hist_real_without_zeros.png"),
    summary     = os.path.join(RUN_DIR, "summary.png"),
    report      = os.path.join(RUN_DIR, "report.txt"),
)


# =======================================================================
# DATASET
# =======================================================================

class SeqDS(Dataset):
    """Dataset de secuencias 3D (muestras, pasos_temporales, features)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx   # devuelve el índice para la máscara


def load_splits(path: str):
    """Carga los splits train/val/test desde el archivo .npz."""
    d = np.load(path, allow_pickle=True)
    return (
        d["X_train"], d["y_train"], d["t_train"],
        d["X_val"],   d["y_val"],   d["t_val"],
        d["X_test"],  d["y_test"],  d["t_test"],
    )


# =======================================================================
# MÁSCARA DE DÍA/NOCHE
# =======================================================================

def compute_day_mask(
    timestamps,
    flag_start: int = FLAG_START,
    flag_end: int = FLAG_END,
    tz_offset: int = UTC_OFFSET,
) -> torch.Tensor:
    """
    Genera un tensor binario (1 = día, 0 = noche) para cada timestamp.

    Parámetros
    ----------
    timestamps  : array de timestamps (UTC) correspondientes al target.
    flag_start  : hora local de inicio de la noche (después de esta hora = noche).
    flag_end    : hora local de fin de la noche (antes de esta hora = noche).
    tz_offset   : desplazamiento UTC → hora local (horas).
    """
    ts_local = pd.to_datetime(timestamps) + pd.Timedelta(hours=tz_offset)
    horas    = ts_local.hour
    mascara  = ((horas > flag_start) & (horas < flag_end)).astype(float)
    vals     = mascara.values if hasattr(mascara, "values") else mascara
    return torch.tensor(vals, dtype=torch.float32)


# =======================================================================
# UTILIDADES DE DESESCALADO
# =======================================================================

def get_dataarray(path: str, var: str | None) -> xr.DataArray:
    """Abre el NetCDF de referencia y devuelve el DataArray de la variable indicada."""
    ds = xr.open_dataset(path)
    if var is None:
        var = list(ds.data_vars)[0]
    return ds[var].squeeze()


def build_stats(da: xr.DataArray, method: str) -> dict:
    """Pre-calcula estadísticos necesarios para el desescalado."""
    if method == "z_score":
        return {"mu": float(da.mean()), "sigma": float(da.std())}
    if method == "average":
        return {"mean": float(da.mean())}
    if method == "minmax":
        return {"min": float(da.min()), "max": float(da.max())}
    return {}   # "physical" y "none" no necesitan estadísticos pre-calculados


def descale(
    arr: np.ndarray,
    times: np.ndarray,
    method: str,
    da: xr.DataArray,
    stats: dict,
) -> np.ndarray:
    """
    Convierte predicciones del espacio normalizado al espacio real (W/m²).

    Métodos soportados
    ------------------
    physical : multiplica por el valor de referencia (GHI_cs) en cada instante.
    minmax   : invierte la normalización min-max.
    z_score  : invierte la estandarización (media=0, std=1).
    average  : invierte la normalización por media.
    none     : sin transformación (devuelve arr intacto).
    """
    if method == "physical":
        # Selecciona el valor de GHI_cs en cada instante de observación
        return arr * da.sel(observation_time=times).values
    elif method == "minmax":
        return arr * (stats["max"] - stats["min"]) + stats["min"]
    elif method == "z_score":
        return arr * stats["sigma"] + stats["mu"]
    elif method == "average":
        return arr * stats["mean"]
    elif method == "none":
        return arr
    else:
        raise ValueError(f"Método de desescalado desconocido: '{method}'")


# =======================================================================
# FORMATO DE NÚMEROS (estilo europeo con punto de miles y coma decimal)
# =======================================================================

def fmt_de(val, prec: int = 4) -> str:
    """Formatea un número como '1.234.567,8910' (estilo europeo)."""
    if isinstance(val, (float, np.floating)):
        s = f"{val:,.{prec}f}"
    else:
        s = f"{val:,}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# =======================================================================
# FUNCIONES DE VISUALIZACIÓN
# =======================================================================

def plot_loss(history: list, out: str) -> None:
    """Guarda la curva de pérdida (train vs val) por época."""
    ep, tl, vl = zip(*history)
    plt.figure()
    plt.plot(ep, tl, label="train")
    plt.plot(ep, vl, label="val")
    plt.title(f"{RUN_NAME} — Curva de pérdida")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def scatter(y_t: np.ndarray, y_p: np.ndarray, out: str, espacio: str) -> None:
    """
    Genera un scatter plot de valores reales vs predichos.
    espacio : "Norm" para espacio normalizado, "Real" para W/m².
    """
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    plt.figure(figsize=(8, 8))
    plt.scatter(y_t, y_p, s=8, alpha=0.5)
    plt.plot(lims, lims, "r--")
    if espacio == "Real":
        plt.title(f"{RUN_NAME} — Predicho vs Real (W/m²)")
        plt.xlabel(r"$y_{true}$ (W/m²)")
        plt.ylabel(r"$y_{pred}$ (W/m²)")
    else:
        plt.title(f"{RUN_NAME} — Predicho vs Real (normalizado)")
        plt.xlabel(r"$y_{true}$ (normalizado)")
        plt.ylabel(r"$y_{pred}$ (normalizado)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def hist_residuos(resid: np.ndarray, out: str, espacio: str) -> None:
    """Histograma de residuos (predicho − real)."""
    plt.figure()
    plt.hist(resid, bins=50, edgecolor="k", alpha=0.7)
    plt.title(f"{RUN_NAME} — Histograma de residuos")
    plt.xlabel("Residuo [W/m²]" if espacio == "Real" else "Residuo")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def hist_sin_ceros(resid: np.ndarray, out: str, espacio: str) -> None:
    """
    Histograma de residuos excluyendo ceros (las horas nocturnas generan
    muchos residuos = 0 que distorsionan la escala del histograma).
    """
    resid_nz = resid[resid != 0]
    plt.figure(figsize=(8, 6))
    plt.hist(resid_nz, bins=50, edgecolor="k", alpha=0.7)
    plt.title(f"{RUN_NAME} — Residuos sin ceros")
    plt.xlabel("Residuo [W/m²]" if espacio == "Real" else "Residuo")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def make_summary(paths: dict, metrics_n: dict, metrics_r: dict, hparams: dict) -> None:
    """
    Genera una imagen resumen con layout 3×3:
    ┌──────────────┬──────────────┬──────────────┐
    │ Scatter Norm │ Scatter Real │  Loss Curve  │ fila 0
    ├──────────────┼──────────────┼──────────────┤
    │  Hist Norm   │  Hist Real   │ Hiperparams  │ fila 1
    ├──────────────┼──────────────┼──────────────┤
    │ Hist Norm ∅0 │ Hist Real ∅0 │   Métricas   │ fila 2
    └──────────────┴──────────────┴──────────────┘
    """
    fig = plt.figure(figsize=(16, 15))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])

    # Fila 0 y parte de fila 1: imágenes de scatter, pérdida e histogramas
    imagenes = [
        (gs[0, 0], paths["scatter_n"],    "Scatter Norm"),
        (gs[0, 1], paths["scatter_r"],    "Scatter Real"),
        (gs[0, 2], paths["loss"],         "Curva de pérdida"),
        (gs[1, 0], paths["hist_n"],       "Histograma Norm"),
        (gs[1, 1], paths["hist_r"],       "Histograma Real"),
    ]
    for celda, fname, titulo in imagenes:
        ax = fig.add_subplot(celda)
        ax.imshow(plt.imread(fname))
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")

    # Fila 1, columna 2: tabla de hiperparámetros
    ax_hp = fig.add_subplot(gs[1, 2])
    ax_hp.axis("off")
    tbl = ax_hp.table(
        cellText=[[k, str(v)] for k, v in hparams.items()],
        colLabels=["Hiperparámetro", "Valor"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    ax_hp.set_title("Hiperparámetros", fontsize=11, pad=6)

    # Fila 2: histogramas sin ceros y tabla de métricas
    for celda, clave, titulo in [
        (gs[2, 0], "hist_n_zeros", "Hist Norm ∅0"),
        (gs[2, 1], "hist_r_zeros", "Hist Real ∅0"),
    ]:
        ax = fig.add_subplot(celda)
        ax.imshow(plt.imread(paths[clave]))
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")

    ax_met = fig.add_subplot(gs[2, 2])
    ax_met.axis("off")
    filas_met = ["MSE", "RMSE", "MAE", "R2", "Corr", "Residual_Variance", "SkillScore"]
    datos_met = [[r, f"{metrics_r[r]:.4f}"] for r in filas_met if r in metrics_r]
    tbl_met = ax_met.table(
        cellText=datos_met,
        colLabels=["Métrica", "Valor real"],
        loc="center", cellLoc="center",
    )
    tbl_met.auto_set_font_size(False)
    tbl_met.set_fontsize(9)
    tbl_met.scale(1.1, 1.5)
    ax_met.set_title("Métricas de evaluación (real)", fontsize=11, pad=6)

    fig.suptitle(RUN_NAME, fontsize=17, y=0.94)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(paths["summary"], dpi=300)
    plt.close(fig)


# =======================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# =======================================================================

def main():
    # -------------------------------------------------------------------
    # 1) Cargar splits train / val / test desde el archivo .npz
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Corrida: {RUN_NAME}")
    print(f"  Datos:   {SEQ_NPZ}")
    print(f"  Activación:  {ACTIVATION} | Desescalado: {DESCALER_METHOD} | Máscara noche: {USE_DAYMASK}")
    print(f"{'='*60}\n")

    X_tr, y_tr, t_tr, X_va, y_va, t_va, X_te, y_te, t_te = load_splits(SEQ_NPZ)
    print(f"Forma del vector de entrada (train): {X_tr.shape}")

    tr_dl = DataLoader(SeqDS(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(SeqDS(X_va, y_va), BATCH_SIZE)
    te_dl = DataLoader(SeqDS(X_te, y_te), BATCH_SIZE)

    seq_len, n_feat = X_tr.shape[1], X_tr.shape[2]

    # -------------------------------------------------------------------
    # 2) Pre-calcular máscaras de día/noche (se usan si USE_DAYMASK=True)
    # -------------------------------------------------------------------
    # Las máscaras tienen un valor por muestra (1=día, 0=noche)
    mascara_train = compute_day_mask(t_tr)
    mascara_val   = compute_day_mask(t_va)
    mascara_test  = compute_day_mask(t_te)

    # -------------------------------------------------------------------
    # 3) Construir el modelo Bi-LSTM con los hiperparámetros de config.py
    # -------------------------------------------------------------------
    modelo = BiLSTMRegressor(
        n_feat     = n_feat,
        hidden     = HIDDEN,
        seq_len    = seq_len,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
        activation = ACTIVATION,   # "sigmoid" o "linear" — desde config.py
    )
    print(f"Parámetros del modelo: {sum(p.numel() for p in modelo.parameters()):,}")

    # Optimizador AdamW con regularización L2
    optim     = torch.optim.AdamW(modelo.parameters(), lr=LR_INIT, weight_decay=L2_LAMBDA)
    # Scheduler que reduce la LR cuando la pérdida de validación se estanca
    scheduler = ReduceLROnPlateau(
        optim, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
        threshold=1e-4, min_lr=MIN_LR, verbose=True,
    )
    # Pérdida MSE sin reducción (para poder ponderar con la máscara)
    loss_fn = nn.MSELoss(reduction="none")

    # TensorBoard: registra el grafo del modelo si está disponible
    if _TENSORBOARD:
        writer    = SummaryWriter(log_dir=RUN_DIR)
        dummy_inp = torch.randn(1, seq_len, n_feat)
        writer.add_graph(modelo, dummy_inp)

    # -------------------------------------------------------------------
    # 4) Bucle de entrenamiento con early stopping
    # -------------------------------------------------------------------
    mejor_val, paciencia, historial = float("inf"), 0, []

    barra = trange(1, EPOCHS + 1, desc="Épocas", unit="ep", leave=True)
    for ep in barra:
        # ---- Fase de entrenamiento ------------------------------------
        modelo.train()
        perdida_train = 0.0
        for xb, yb, idx in tr_dl:
            optim.zero_grad()
            preds = modelo(xb)              # (B,)

            if USE_DAYMASK:
                # Aplicar máscara: predicciones nocturnas → 0
                mascara = mascara_train[idx]
                preds   = preds * mascara
                # La pérdida no se acumula en horas nocturnas
                perdida = (loss_fn(preds, yb) * mascara).mean()
            else:
                perdida = loss_fn(preds, yb).mean()

            perdida.backward()
            # Recorte de gradientes para estabilidad numérica
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optim.step()
            perdida_train += perdida.item() * len(xb)
        perdida_train /= len(tr_dl.dataset)

        # ---- Fase de validación ---------------------------------------
        modelo.eval()
        perdida_val = 0.0
        with torch.no_grad():
            for xb, yb, idx in va_dl:
                preds = modelo(xb)
                if USE_DAYMASK:
                    mascara = mascara_val[idx]
                    preds   = preds * mascara
                    perdida = (loss_fn(preds, yb) * mascara).mean()
                else:
                    perdida = loss_fn(preds, yb).mean()
                perdida_val += perdida.item() * len(xb)
        perdida_val /= len(va_dl.dataset)

        scheduler.step(perdida_val)
        tqdm.write(
            f"[{ep:3d}/{EPOCHS}] train {perdida_train:.5f} | "
            f"val {perdida_val:.5f} | lr {optim.param_groups[0]['lr']:.1e}"
        )
        historial.append((ep, perdida_train, perdida_val))

        # ---- Early stopping -------------------------------------------
        if perdida_val < mejor_val:
            mejor_val, paciencia = perdida_val, 0
            torch.save(modelo.state_dict(), PATHS["model"])
        else:
            paciencia += 1
            if paciencia >= EARLY_STOP:
                tqdm.write("Early stopping activado.")
                break

    plot_loss(historial, PATHS["loss"])

    # -------------------------------------------------------------------
    # 5) Evaluación en espacio normalizado (test set)
    # -------------------------------------------------------------------
    modelo.load_state_dict(torch.load(PATHS["model"]))
    modelo.eval()

    preds_lista = []
    with torch.no_grad():
        for xb, _, idx in te_dl:
            preds = modelo(xb)
            if USE_DAYMASK:
                # Durante la inferencia también se aplica el hard-clip nocturno
                preds = preds * mascara_test[idx]
            preds_lista.append(preds.numpy())
    y_pred = np.concatenate(preds_lista)   # (n_test,)

    # Calcular métricas en espacio normalizado
    residuos_n           = y_te - y_pred
    varianza_residuos_n  = float(np.var(residuos_n))
    metricas_n = dict(
        MSE              = mean_squared_error(y_te, y_pred),
        MAE              = mean_absolute_error(y_te, y_pred),
        R2               = r2_score(y_te, y_pred),
        Corr             = float(np.corrcoef(y_te.ravel(), y_pred.ravel())[0, 1]),
        Residual_Variance= varianza_residuos_n,
    )
    print("\n=== Evaluación — espacio normalizado ===")
    for k, v in metricas_n.items():
        print(f"{k:>20}: {fmt_de(v)}")

    # Guardar CSV normalizado y generar gráficas
    pd.DataFrame({"time": pd.to_datetime(t_te), "y_true": y_te, "y_pred": y_pred})\
        .to_csv(PATHS["csv_n"], index=False)
    scatter(y_te,    y_pred,           PATHS["scatter_n"],    "Norm")
    hist_residuos(y_pred - y_te,       PATHS["hist_n"],        "Norm")
    hist_sin_ceros(y_pred - y_te,      PATHS["hist_n_zeros"],  "Norm")

    # -------------------------------------------------------------------
    # 6) Desescalado: convertir predicciones normalizadas a W/m²
    # -------------------------------------------------------------------
    da_ref    = get_dataarray(DESCALER_FILE, DESCALER_VAR)
    stats_ref = build_stats(da_ref, DESCALER_METHOD)
    y_true_r  = descale(y_te,   t_te, DESCALER_METHOD, da_ref, stats_ref)
    y_pred_r  = descale(y_pred, t_te, DESCALER_METHOD, da_ref, stats_ref)

    # -------------------------------------------------------------------
    # 7) Evaluación en espacio real (W/m²)
    # -------------------------------------------------------------------
    residuos_r          = y_true_r - y_pred_r
    varianza_residuos_r = float(np.var(residuos_r))
    metricas_r = dict(
        MSE              = mean_squared_error(y_true_r, y_pred_r),
        RMSE             = root_mean_squared_error(y_true_r, y_pred_r),
        MAE              = mean_absolute_error(y_true_r, y_pred_r),
        R2               = r2_score(y_true_r, y_pred_r),
        Corr             = float(np.corrcoef(y_true_r.ravel(), y_pred_r.ravel())[0, 1]),
        Residual_Variance= varianza_residuos_r,
    )
    # Skill Score: cuánto mejora el modelo sobre el pronóstico GFS directo
    metricas_r["SkillScore"] = 1.0 - metricas_r["MSE"] / MSE_BASELINE_R

    print("\n=== Evaluación — espacio real (W/m²) ===")
    for k, v in metricas_r.items():
        print(f"{k:>20}: {fmt_de(v)}")

    # Guardar CSV real y generar gráficas
    pd.DataFrame({"time": pd.to_datetime(t_te), "y_true": y_true_r, "y_pred": y_pred_r})\
        .to_csv(PATHS["csv_r"], index=False)
    scatter(y_true_r, y_pred_r,            PATHS["scatter_r"],    "Real")
    hist_residuos(y_pred_r - y_true_r,     PATHS["hist_r"],        "Real")
    hist_sin_ceros(y_pred_r - y_true_r,    PATHS["hist_r_zeros"],  "Real")

    # -------------------------------------------------------------------
    # 8) Imagen resumen 3×3 y archivo report.txt
    # -------------------------------------------------------------------
    hparams = {
        "LR":          LR_INIT,
        "Épocas":      EPOCHS,
        "Batch Size":  BATCH_SIZE,
        "Hidden":      HIDDEN,
        "Dropout":     DROPOUT,
        "Num Layers":  NUM_LAYERS,
        "L2 Lambda":   L2_LAMBDA,
        "Early Stop":  EARLY_STOP,
        "LR Factor":   LR_FACTOR,
        "LR Patience": LR_PATIENCE,
        "Activación":  ACTIVATION,
        "Desescalado": DESCALER_METHOD,
        "Máscara noche": USE_DAYMASK,
    }
    make_summary(PATHS, metricas_n, metricas_r, hparams)

    # Escribir reporte de texto con todos los detalles de la corrida
    with open(PATHS["report"], "w", encoding="utf-8") as f:
        f.write(f"# Reporte de corrida Bi-LSTM — {RUN_NAME}\n")
        f.write(f"Timestamp: {_NOW}\n\n")

        f.write("## Hiperparámetros\n")
        for k, v in hparams.items():
            f.write(f"{k}: {v}\n")

        f.write("\n## Configuración de la corrida\n")
        f.write(f"Archivo NPZ: {SEQ_NPZ}\n")
        f.write(f"Descaler: {json.dumps({'method': DESCALER_METHOD, 'file': DESCALER_FILE, 'variable': DESCALER_VAR}, indent=2)}\n")

        f.write("\n## Métricas — espacio normalizado\n")
        for k, v in metricas_n.items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\n## Métricas — espacio real (W/m²)\n")
        for k, v in metricas_r.items():
            f.write(f"{k}: {v:.6f}\n")

    if _TENSORBOARD:
        writer.close()

    print(f"\nTodos los artefactos guardados en: {RUN_DIR}")


# =======================================================================
if __name__ == "__main__":
    main()
