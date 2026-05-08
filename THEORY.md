# THEORY.md — Documentación Teórica Completa
## Predicción de Radiación Solar con BiLSTM — Empresa Emergente

> **Para quién es este documento:** Es tu referencia personal de estudio profundo. Explica cada concepto usado en este proyecto desde cero — sin asumir conocimiento previo de machine learning ni meteorología. Léelo sección por sección, en orden, para construir un modelo mental completo del proyecto.

---

## Tabla de Contenidos

1. [La física de la radiación solar](#1-la-física-de-la-radiación-solar)
2. [Modelos meteorológicos y el GFS](#2-modelos-meteorológicos-y-el-gfs)
3. [Formatos de datos](#3-formatos-de-datos)
4. [El pipeline — módulo por módulo](#4-el-pipeline--módulo-por-módulo)
5. [Normalización e índices de radiación](#5-normalización-e-índices-de-radiación)
6. [¿Qué es una red neuronal?](#6-qué-es-una-red-neuronal)
7. [¿Qué es una LSTM?](#7-qué-es-una-lstm)
8. [¿Qué es una BiLSTM?](#8-qué-es-una-bilstm)
9. [Mecanismo de atención](#9-mecanismo-de-atención)
10. [Entrenamiento de una red neuronal](#10-entrenamiento-de-una-red-neuronal)
11. [Sobreajuste y regularización](#11-sobreajuste-y-regularización)
12. [Las 69 features de entrada — por qué importa cada una](#12-las-69-features-de-entrada--por-qué-importa-cada-una)
13. [Secuencias y la ventana temporal](#13-secuencias-y-la-ventana-temporal)
14. [Métricas — explicación profunda](#14-métricas--explicación-profunda)
15. [SkillScore — qué significa y por qué es negativo](#15-skillscore--qué-significa-y-por-qué-es-negativo)
16. [Experimentos — qué probamos y qué aprendimos](#16-experimentos--qué-probamos-y-qué-aprendimos)
17. [El config.py — cada parámetro explicado](#17-el-configpy--cada-parámetro-explicado)

---

## 1. La Física de la Radiación Solar

### ¿Qué es la radiación solar?

La radiación solar es la energía electromagnética emitida por el sol que llega a la Tierra. Para la producción de paneles solares, lo que importa es la **GHI — Irradiancia Horizontal Global**: la energía solar total que llega a una superficie horizontal por unidad de área por unidad de tiempo, medida en **W/m²** (vatios por metro cuadrado).

Piénsalo así: ¿cuántos vatios de energía solar están golpeando cada metro cuadrado del suelo en este momento?

```
Valores típicos en Medellín:
  Noche:                    0 W/m²
  Día muy nublado:        50–200 W/m²
  Parcialmente nublado:  200–600 W/m²
  Día soleado despejado: 600–900 W/m²
  Máximo posible:          ~950 W/m²
```

### Los tres techos de radiación

Este proyecto usa tres niveles de "máximo posible" de radiación:

```
MÁXIMO ABSOLUTO (sin atmósfera)
Radiación Extraterrestre: ~1,300 W/m² en equinoccio
    │
    │ La atmósfera absorbe ~30% incluso en días despejados
    ▼
MÁXIMO REALISTA (cielo despejado, con atmósfera)
Clear-Sky GHI (Ineichen): ~700–950 W/m² en Medellín
    │
    │ Las nubes absorben y reflejan radiación adicional
    ▼
MEDICIÓN REAL
SIATA GHI: 0 a ~950 W/m²
```

### Radiación extraterrestre — el techo absoluto

Es la radiación que llegaría a una superficie horizontal en Medellín **si no existiera atmósfera**. Depende únicamente de:

1. **Posición del sol** — calculada desde la latitud (6.25°N), fecha y hora usando geometría astronómica
2. **Distancia Tierra-Sol** — varía a lo largo del año porque la órbita terrestre es elíptica (más cerca en enero, más lejos en julio)

```python
# pvlib lo calcula por nosotros:
import pvlib

location = pvlib.location.Location(
    latitude=6.25,
    longitude=-75.5,
    altitude=1582,
    tz='America/Bogota'
)

# Obtener posición solar para un momento específico
solar_pos = location.get_solarposition(times)
# ángulo cenital: 0° = sol directamente arriba, 90° = sol en el horizonte

# Radiación extraterrestre
dni_extra = pvlib.irradiance.get_extra_radiation(times)
```

La radiación extraterrestre sigue una curva suave y perfectamente predecible — no tiene aleatoriedad. La misma fecha del próximo año tendrá la misma radiación extraterrestre.

### Radiación de cielo despejado — el techo realista

Es la radiación que llegaría a la superficie si **el cielo estuviera perfectamente despejado** — sin nubes, pero con absorción atmosférica normal. Es menor que la extraterrestre porque:

- **Dispersión de Rayleigh** — las moléculas de aire dispersan la luz azul
- **Absorción por aerosoles** — partículas de polvo, polen y contaminación absorben radiación
- **Absorción por vapor de agua** — la humedad atmosférica absorbe radiación infrarroja

El **modelo de Ineichen** (usado en este proyecto) calcula el Clear-Sky GHI considerando:

```python
# Entradas clave al modelo Ineichen:
tl = pvlib.clearsky.lookup_linke_turbidity(times, latitude=6.25, longitude=-75.5)
# Turbidez de Linke: mide qué tan "sucia" está la atmósfera
# Rango: ~2 (aire de montaña muy limpio) a ~7 (ciudad muy contaminada)
# Varía por mes — Medellín tiene patrones estacionales

altitud = 1582  # metros — atmósfera más delgada = menos absorción

# Perfil del horizonte — las montañas circundantes bloquean la radiación 
# cuando el sol está bajo
# Medellín está rodeada de montañas de hasta 2,500+ metros
```

**Corrección del ángulo cenital:** Cuando el sol está muy bajo en el horizonte (ángulo cenital > 60°), el modelo Ineichen tiende a *subestimar* la radiación. Esto ocurre porque la luz viaja a través de una capa de atmósfera mucho más gruesa en ángulos bajos y las aproximaciones del modelo pierden precisión. Se aplica una corrección de sesgo:

```python
if angulo_cenital > ZENITH_THRESHOLD:  # 60 grados
    correccion = 1 + ZENITH_ALPHA * (angulo_cenital - ZENITH_THRESHOLD)
    # Ejemplo: zenith=70°, alpha=0.003
    # correccion = 1 + 0.003 × (70-60) = 1.03  (aumento del 3%)
    clear_sky_ghi *= correccion
```

Esta corrección se derivó empíricamente comparando la salida del modelo Ineichen contra mediciones de SIATA en días despejados — la misma metodología que calibrar cualquier instrumento físico.

---

## 2. Modelos Meteorológicos y el GFS

### ¿Qué es la predicción numérica del tiempo?

Un modelo meteorológico es un programa de computador que simula la atmósfera resolviendo las ecuaciones de dinámica de fluidos y termodinámica en una malla 3D que cubre la Tierra. Parte del estado actual observado de la atmósfera y lo proyecta hacia adelante en el tiempo.

Las ecuaciones clave son:
- Conservación del momento (Segunda ley de Newton para fluidos)
- Conservación de masa
- Conservación de energía
- Ley del gas ideal
- Ecuaciones de transferencia de radiación

### El GFS — Sistema Global de Pronóstico

El GFS es el modelo meteorológico global operacional de la NOAA. Datos clave:

| Propiedad | Valor |
|-----------|-------|
| Operador | NOAA (Administración Nacional Oceánica y Atmosférica, EE.UU.) |
| Resolución espacial | 0.25° × 0.25° (~28 km) |
| Niveles verticales | 127 niveles de presión desde la superficie hasta 80 km |
| Resolución temporal | 1 hora (primeras 120 horas), luego 3 horas |
| Alcance máximo del pronóstico | 16 días |
| Frecuencia de ejecución | 4 veces al día (00, 06, 12, 18 UTC) |
| Disponibilidad de datos | Pública en servidores S3 de NOAA |

### Los 4 launch times diarios

Cada una de las 4 ejecuciones diarias se llama "lanzamiento" o "ciclo". Cada una usa las observaciones atmosféricas más recientes (de globos sonda, satélites, aviones, etc.) para inicializar el modelo.

```
Línea de tiempo del día (hora local Colombia, UTC-5):

Día anterior 19:00  ←── lanzamiento 00 UTC  →  pronósticos próximas 24h
             01:00  ←── lanzamiento 06 UTC  →  pronósticos próximas 24h
             07:00  ←── lanzamiento 12 UTC  →  pronósticos próximas 24h
             13:00  ←── lanzamiento 18 UTC  →  pronósticos próximas 24h
```

**¿Por qué este proyecto usa los 4 launch times?**

Cada lanzamiento usa observaciones progresivamente más recientes. El lanzamiento de las 13:00 tiene 6 horas más de datos observacionales que el de las 07:00 — haciéndolo potencialmente más preciso para la misma hora pronosticada.

Pero cada lanzamiento también tiene diferentes **tiempos de adelanto** (con cuánta anticipación está prediciendo):

```
Ejemplo: prediciendo el martes a las 15:00

Desde lanzamiento 01:00: tiempo adelanto = 14 horas
Desde lanzamiento 07:00: tiempo adelanto = 8 horas
Desde lanzamiento 13:00: tiempo adelanto = 2 horas
Desde lanzamiento 19:00: tiempo adelanto = -4 horas (este lanzamiento es 
                                            DESPUÉS de la hora pronosticada)
```

La feature `step` captura esto — la red aprende que los lanzamientos con menor tiempo de adelanto (más recientes) son generalmente más confiables.

### Por qué el GFS es imperfecto para Medellín

La resolución del GFS de 28 km significa que una celda de la malla cubre un área más grande que toda el área urbana de Medellín. Dentro de esa única celda, el modelo asume condiciones uniformes — no puede representar:

1. **Efectos de valle** — el Valle de Aburrá crea sus propios patrones de circulación con brisas de montaña y niebla de valle
2. **Nubosidad convectiva** — tormentas eléctricas de la tarde que se forman localmente en 5-10 km
3. **Gradientes de elevación** — diferentes barrios a diferentes altitudes experimentan nubosidad muy diferente
4. **Isla de calor urbana** — el centro de la ciudad es más caliente que las montañas circundantes, afectando la convección

### El problema de acumulación del DSWRF

El GFS no genera valores instantáneos de radiación horaria directamente. En cambio genera **promedios acumulados** sobre bloques de 6 horas de pronóstico. Esto significa:

- Valor en hora de pronóstico 3 = promedio de radiación sobre las horas 1, 2, 3
- Valor en hora de pronóstico 6 = promedio de radiación sobre las horas 1, 2, 3, 4, 5, 6

Para recuperar el valor horario real, aplicamos la inversa matemática:

```
Para la hora h dentro de un bloque que inicia en la hora b:
valor(h) = (h - b) × promedio(h) - (h - 1 - b) × promedio(h - 1)

Ejemplo:
El bloque inicia en b=1. Queremos la hora h=4:
promedio(h=3) = 400 W/m²  (promedio de horas 1, 2, 3)
promedio(h=4) = 350 W/m²  (promedio de horas 1, 2, 3, 4)

valor(4) = (4-1) × 350 - (4-1-1) × 400
         = 3 × 350 - 2 × 400
         = 1050 - 800
         = 250 W/m²
```

Cualquier resultado negativo se fuerza a cero (la radiación no puede ser negativa).

---

## 3. Formatos de Datos

### NetCDF — Network Common Data Form

NetCDF es el formato científico estándar para datos multidimensionales con coordenadas. Piénsalo como un Excel más inteligente: en lugar de números de fila, cada dimensión tiene etiquetas significativas (marcas de tiempo, latitud, longitud).

```python
import xarray as xr

# Abrir un archivo NetCDF
ds = xr.open_dataset("clearsky_index_GFS_0100.nc", engine="h5netcdf")

# Un Dataset contiene múltiples Variables
print(ds)
# <xarray.Dataset>
# Dimensions:  (time: 35040, lat: 1, lon: 1)
# Coordinates:
#   * time     (time) datetime64[ns] 2021-01-01 ... 2024-12-31
#   * lat      (lat) float64 6.25
#   * lon      (lon) float64 -75.5
# Data variables:
#     clearsky_index_GFS_0100  (time, lat, lon) float32 ...

# Acceder a una variable y convertir a pandas Series
csi = ds["clearsky_index_GFS_0100"].squeeze().to_series()
print(csi["2023-05-15 12:00"])  # Valor en tiempo específico: 0.82
```

**¿Por qué `engine='h5netcdf'`?**

Los archivos NetCDF pueden usar dos bibliotecas C diferentes para leerlos. La predeterminada (`netCDF4`) pasa las rutas de archivos a través de la API ANSI de Windows, que no puede manejar la letra `ó` en "Código RS". La ruta se corrompe en un carácter ilegible. El backend `h5netcdf` usa la E/S propia de Python con soporte Unicode, que maneja cualquier carácter correctamente.

### GRIB2 — GRIdded Binary

GRIB2 es el formato estándar usado por agencias meteorológicas en todo el mundo. Un archivo GRIB2 de NOAA contiene datos para **todo el planeta** para todas las variables en un tiempo de pronóstico — esto hace que los archivos sean muy grandes (2-5 GB cada uno).

Las **solicitudes byte-range** resuelven esto: en lugar de descargar el archivo completo, leemos el índice GRIB2 para encontrar las posiciones exactas de bytes de las variables que necesitamos, luego descargamos solo esos bytes usando cabeceras HTTP Range:

```python
# Ejemplo conceptual:
import requests

# Paso 1: Leer el archivo índice para encontrar la posición de bytes del DSWRF
# El DSWRF comienza en el byte 1,234,567 y tiene 45,678 bytes de longitud

# Paso 2: Descargar solo esos bytes
headers = {"Range": "bytes=1234567-1280244"}
response = requests.get(url, headers=headers)

# Paso 3: Convertir fragmento a xarray usando cfgrib
# cfgrib sabe cómo decodificar el formato binario GRIB2
```

Esto reduce el tamaño de descarga de ~3 GB a ~50 KB por variable por hora de pronóstico — una reducción del 99.998%.

### NPZ — Archivo Comprimido NumPy

Un archivo NPZ es un archivo `.zip` que contiene múltiples arrays NumPy guardados como archivos `.npy`. Es el formato principal de datos para las secuencias de entrenamiento.

```python
import numpy as np

# Guardar múltiples arrays
np.savez_compressed("secuencias.npz",
    X_train=X_train,  # forma (24889, 37, 69)
    y_train=y_train,  # forma (24889,)
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

# Cargar
data = np.load("secuencias.npz")
X_train = data["X_train"]
print(X_train.shape)  # (24889, 37, 69)
# Leer como: 24,889 secuencias de entrenamiento, cada una de 37 pasos 
# temporales de largo, cada paso con 69 features
```

---

## 4. El Pipeline — Módulo por Módulo

### Módulo 1 — Adquisición de datos

**Entrada:** Conexión a internet a los servidores de NOAA
**Salida:** Archivos `.nc` individuales por hora, por launch time, por variable

El descargador maneja:
1. Generar la lista de todas las horas a descargar (4 años × 4 launch times × 24 horas)
2. Construir URLs de S3 de NOAA para cada combinación
3. Leer el índice GRIB2 para encontrar los desplazamientos de bytes
4. Descargar solo los bytes necesarios
5. Convertir GRIB2 → xarray usando cfgrib
6. Recortar espacialmente a Medellín (6.25°N, 75.5°W)
7. Guardar como archivos `.nc` individuales

El descargador usa un **pool de procesos** — múltiples descargas ocurriendo simultáneamente para maximizar el uso del ancho de banda.

### Módulo 2 — Postprocesamiento

**Entrada:** ~35,040 × 4 archivos `.nc` individuales por hora
**Salida:** 4 archivos NetCDF de series temporales continuas (uno por launch time)

Operaciones clave:

**Paso 1 — Fusión:** Agrupar todos los archivos horarios del mismo launch time en un archivo continuo que cubre el período completo de 4 años.

**Paso 2 — Desacumulación del DSWRF:** Aplicar la inversa matemática de la acumulación del GFS (fórmula mostrada en la Sección 2).

**Paso 3 — Corregir velocidad del viento:** El GFS proporciona componentes U (este-oeste) y V (norte-sur) del viento. Velocidad total:
```
Wind10m = √(UGRD_10m² + VGRD_10m²)
```
Esto es el teorema de Pitágoras aplicado a componentes vectoriales.

**Paso 4 — Corregir duración del brillo solar:** El GFS proporciona SUNSD en segundos. Convertir a minutos y manejar el mismo problema de acumulación que el DSWRF.

### Módulo 3 — Preparación de Features y Target

**Entrada:** Series temporales fusionadas del Módulo 2 + archivos CSV de SIATA
**Salida:** Índices CSI para los 4 launch times (features) + CSI de SIATA (target)

**Cálculos de referencia física:**

```python
# Radiación extraterrestre (sin atmósfera)
# Depende solo de: latitud, fecha, hora
# Completamente predecible — sin aleatoriedad

# Radiación de cielo despejado (cielo claro, con atmósfera)
# Depende de: latitud, fecha, hora, turbidez, altitud, horizonte
# Aún determinista para un estado atmosférico dado
```

**Recorte de datos (clipping):**

Los valores que exceden el techo físico son físicamente imposibles — indican errores de medición o artefactos del modelo. Los reemplazamos con el promedio histórico para esa combinación mes/día/hora, añadiendo una pequeña perturbación aleatoria:

```python
# Clipping del GFS (dos etapas):
if valor_gfs > extraterrestre:      # Absolutamente imposible
    valor_gfs = promedio_mensual × uniforme(0.97, 1.00)

if valor_gfs > clear_sky:           # Implausible para Medellín
    valor_gfs = promedio_mensual × uniforme(0.97, 1.00)

# Clipping de SIATA (una etapa):
if valor_siata > clear_sky:
    valor_siata = clear_sky × uniforme(0.97, 0.995)
# La pequeña perturbación evita valores artificialmente "perfectos"
```

**Reglas de agregación de SIATA:**

SIATA proporciona mediciones de 1 minuto. Para obtener valores horarios:

```
Para horas nocturnas (20:00–05:00):
    → Forzar a 0 (físicamente correcto — no hay sol)

Para horas de transición (06:00 y 19:00 — amanecer/atardecer):
    Si más del 75% de los minutos son NaN:
        → Marcar hora como NaN (datos insuficientes)
    Si no:
        → Reemplazar minutos NaN con 0 (la radiación está comenzando/terminando)
        → Calcular promedio de los 60 minutos
    Razón: Al amanecer/atardecer la radiación es muy baja;
           los minutos faltantes probablemente son verdaderamente 0

Para horas diurnas (07:00–18:00):
    Si más del 75% de los minutos son NaN:
        → Marcar hora como NaN (datos insuficientes)
    Si no:
        → Calcular promedio solo de los minutos válidos (no NaN)
    Razón: Los minutos faltantes durante el día probablemente indican 
           problemas del sensor, no radiación cero; promediar solo valores 
           válidos evita subestimar la radiación
```

**Cálculo de índices CSI y KC:**

```python
# Índice de Cielo Despejado (CSI) — qué tan nublado relativo a un día perfecto
CSI_GFS = DSWRF_GFS / Clear_Sky_GHI       # Feature de entrada
CSI_SIATA = GHI_SIATA / Clear_Sky_GHI     # Variable objetivo (target)

# Índice de Claridad (KC) — fracción del máximo teórico
KC_GFS = DSWRF_GFS / Extraterrestre_GHI   # Feature de entrada

# ¿Por qué normalizar así?
# Un CSI de 0.8 siempre significa "80% de un día perfecto despejado"
# sin importar si son las 8am o mediodía, enero o julio
# Esto elimina el ciclo diario/estacional de los datos
```

---

## 5. Normalización e Índices de Radiación

### ¿Por qué normalizar las features?

Diferentes variables tienen rangos numéricos muy diferentes:

```
Temperatura:     270–310 K
Humedad relativa:  0–100 %
Velocidad viento:  0–30 m/s
Cobertura nubes:   0–100 %
DSWRF:             0–1000 W/m²
CAPE:              0–5000 J/kg
```

Si alimentamos estos valores brutos a la red neuronal, las variables con valores grandes (CAPE, DSWRF) dominarían el proceso de aprendizaje — la red les prestaría más atención simplemente porque sus números son más grandes, no porque sean más informativas.

La normalización pone todas las variables en la misma escala para que la red pueda evaluar justamente la importancia de cada una.

### Tres métodos de normalización usados

**Normalización Min-Max:** Escala al rango [0, 1]
```python
normalizado = (valor - mínimo) / (máximo - mínimo)

# Ejemplo: Velocidad del viento rango 0–30 m/s
# viento = 15 m/s → normalizado = (15 - 0) / (30 - 0) = 0.5
# viento = 30 m/s → normalizado = (30 - 0) / (30 - 0) = 1.0
```
Usado para: la mayoría de variables meteorológicas

**Normalización Z-score:** Escala a media=0, desviación estándar=1
```python
normalizado = (valor - media) / desviacion_estandar

# Ejemplo: Temperatura media=295K, desv=5K
# temp = 300K → normalizado = (300 - 295) / 5 = 1.0
# temp = 290K → normalizado = (290 - 295) / 5 = -1.0
```
Usado para: temperatura (que puede tener anomalías negativas)

**Sin normalización:** Valores ya en [0, 1]
```
CSI, KC: ya entre 0 y 1 por definición
```

**Importante:** Los parámetros de normalización (mínimo, máximo, media, desviación estándar) se calculan solo del **conjunto de entrenamiento** y luego se aplican a los conjuntos de validación y prueba. Esto previene la fuga de datos — no podemos usar información futura para normalizar datos pasados.

### El CSI como objetivo del modelo

La red predice el **CSI** (Índice de Cielo Despejado) en lugar de W/m² directamente. Esto tiene tres ventajas:

**1. Restricción de rango:** El CSI siempre está entre 0 y 1, compatible con la activación sigmoid de la salida.

**2. Elimina el ciclo diario/estacional:** Un CSI de 0.8 significa lo mismo a las 8am en enero y al mediodía en julio — 80% del máximo posible para ese momento específico.

**3. Físicamente significativo:** El CSI representa directamente el estado de nubosidad, que es lo que la red realmente aprende a predecir.

**Descalado:** Después de la predicción, multiplicar por el Clear-Sky para obtener W/m²:
```python
ghi_predicho = csi_predicho × clear_sky_ghi_en_esa_hora
```

---

## 6. ¿Qué es una Red Neuronal?

### La unidad básica: una neurona

Una neurona artificial recibe múltiples entradas, multiplica cada una por un **peso**, las suma todas, añade un término de **sesgo** y pasa el resultado por una **función de activación**:

```
entradas:  x1=0.8, x2=0.3, x3=0.6
pesos:     w1=0.5, w2=-0.2, w3=0.4
sesgo:     b=0.1

suma ponderada = (0.8×0.5) + (0.3×-0.2) + (0.6×0.4) + 0.1
               = 0.40 - 0.06 + 0.24 + 0.10
               = 0.68

salida = funcion_activacion(0.68)
       = sigmoid(0.68) = 0.664  (sigmoid mapea cualquier número a 0-1)
```

Los pesos y el sesgo son los **parámetros aprendibles** — comienzan aleatorios y se ajustan durante el entrenamiento para minimizar el error de predicción.

### De neurona a red

Una red neuronal apila muchas neuronas en **capas**:

```
Capa de entrada:  toma las features brutas (69 valores)
Capas ocultas:    transforman las features mediante patrones aprendidos
Capa de salida:   produce la predicción final (1 valor: CSI)
```

Cada neurona en una capa se conecta con cada neurona en la siguiente — esto es una capa **completamente conectada** o **densa**.

### ¿Por qué funciona?

Un teorema matemático llamado el **Teorema de Aproximación Universal** establece que una red neuronal con suficientes neuronas puede aproximar cualquier función continua con cualquier precisión deseada. En la práctica esto significa: dados suficientes datos y la arquitectura correcta, una red neuronal puede aprender cualquier relación entre entradas y salidas — incluyendo la relación compleja y no lineal entre los pronósticos del GFS y la radiación real en Medellín.

### Funciones de activación

Sin funciones de activación, apilar múltiples capas equivaldría a una sola transformación lineal — las capas no añadirían poder expresivo.

**Sigmoid:**
```
sigmoid(x) = 1 / (1 + e^(-x))

Valores: siempre entre 0 y 1
Usada en: capa de salida (el CSI debe estar en 0–1)

x=-3: sigmoid = 0.05
x= 0: sigmoid = 0.50
x=+3: sigmoid = 0.95
```

**Tanh:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Valores: siempre entre -1 y 1
Usada en: dentro de las puertas LSTM
```

**ReLU (Unidad Lineal Rectificada):**
```
ReLU(x) = max(0, x)

Valores: 0 para x negativo, x para x positivo
Usada en: la mayoría de redes feedforward modernas (no usada aquí)
```

---

## 7. ¿Qué es una LSTM?

### El problema con las redes neuronales estándar para series de tiempo

Una red neuronal estándar trata cada entrada de forma independiente — no tiene memoria de lo que vino antes. Para predecir la radiación solar, esto es una limitación fundamental: lo que ocurrió en las horas pasadas es muy informativo sobre lo que ocurrirá a continuación.

Si a las 8am el cielo estaba despejado (CSI=0.9), a las 9am seguía despejado (CSI=0.85) y a las 10am apareció una nube (CSI=0.5), una red estándar procesa cada hora de forma aislada. Una LSTM puede recordar ese patrón y usarlo para predecir las 11am.

### La celda LSTM — una unidad de memoria

Cada neurona (celda) LSTM tiene un **estado de memoria** interno (llamado estado de celda, `c_t`) que persiste a través de los pasos temporales. En cada paso decide:

1. **Qué olvidar** de la memoria anterior
2. **Qué información nueva almacenar**
3. **Qué producir como salida** basándose en la memoria actual

Estas tres decisiones las toman **puertas** — funciones sigmoid que producen valores entre 0 (bloquear completamente) y 1 (pasar completamente):

```
┌──────────────────────────────────────────────────────────────┐
│                    CELDA LSTM                                 │
│                                                              │
│  Memoria anterior ──────────────────────────────────────►   │
│         c_{t-1}                                     c_t      │
│                                                              │
│  Salida anterior ─►│                                         │
│         h_{t-1}    │  OLVIDO    ENTRADA   CELDA   SALIDA    │
│                    │  PUERTA    PUERTA    PUERTA  PUERTA     │
│  Entrada actual ──►│                                         │
│         x_t        │  "Qué     "Qué       "Nueva  "Qué      │
│                    │  borrar"   añadir"   memoria" mostrar"  │
│                    │                                         │
│                    └────────────────────────────────────►    │
│                                                     h_t       │
└──────────────────────────────────────────────────────────────┘
```

### Las cuatro puertas en detalle

**Puerta de olvido** — decide qué borrar de la memoria:
```python
f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)
# Salida: 0 = olvidar todo, 1 = recordar todo
# Ejemplo: si el modelo ha estado rastreando "estuvo nublado por 3 horas"
# y ahora de repente el GFS muestra cielo despejado, la puerta de olvido
# borra el patrón nublado
```

**Puerta de entrada** — decide qué información nueva almacenar:
```python
i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)
# Salida: 0 = no actualizar, 1 = actualizar completamente
```

**Puerta de celda** — crea el contenido candidato de nueva memoria:
```python
g_t = tanh(W_g × [h_{t-1}, x_t] + b_g)
# Salida: -1 a 1 (nuevos valores candidatos de memoria)
```

**Nuevo estado de celda** — combina olvido y entrada:
```python
c_t = f_t × c_{t-1} + i_t × g_t
# Olvidar algo de la memoria antigua, añadir algo de la nueva memoria
```

**Puerta de salida** — decide qué producir:
```python
o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)
h_t = o_t × tanh(c_t)
# Salida: versión filtrada del estado de memoria actual
```

### ¿Por qué la LSTM recibe tanto x_t como h_{t-1}?

La entrada a cada puerta es la **concatenación** de las entradas externas actuales (`x_t` — las 69 features del GFS en este paso temporal) y la salida anterior (`h_{t-1}` — lo que la neurona produjo en el último paso).

Esto es lo que da a la LSTM su "memoria" — la salida anterior de cada neurona se convierte en parte de la entrada para el siguiente paso temporal. La red literalmente propaga información hacia adelante a través del tiempo.

**Por eso la fórmula de conteo de parámetros es:**
```
Parámetros por puerta = (features_entrada + neuronas_ocultas) × neuronas_ocultas
# No solo features_entrada × neuronas_ocultas

# Con 69 features y 96 neuronas ocultas:
# Cada puerta: (69 + 96) × 96 = 15,840 parámetros
# 4 puertas: 4 × 15,840 = 63,360 parámetros por capa LSTM por dirección
```

### Apilar capas LSTM

Se pueden apilar múltiples capas LSTM — la salida de la capa 1 se convierte en la entrada de la capa 2. Cada capa adicional aprende patrones temporales más abstractos:

```
Capa 1: aprende patrones hora a hora
         "la cobertura de nubes aumentó un 20% en la última hora"

Capa 2: aprende patrones de medio día
         "este es el típico aumento de nubosidad de la tarde"

Capa 3: aprende patrones del día completo
         "hoy ha sido consistentemente más nublado de lo normal — 
          probablemente un sistema frontal"
```

Este proyecto usa 3 capas LSTM apiladas.

---

## 8. ¿Qué es una BiLSTM?

### La limitación de la LSTM solo hacia adelante

Una LSTM estándar lee la secuencia de izquierda a derecha — desde el paso temporal más antiguo hasta el más reciente. Para predecir la radiación en la hora 0 (el centro de nuestra ventana de 37 pasos), usa:

```
Información disponible:
← horas -18, -17, -16, ... -1, 0  (solo contexto del pasado)
```

Pero tenemos pronósticos del GFS para las horas **FUTURAS** (+1 hasta +18). Saber lo que el GFS predice para la tarde puede ser muy informativo para predecir la mañana.

### BiLSTM — procesar ambas direcciones

Una **LSTM Bidireccional** ejecuta dos redes LSTM independientes simultáneamente:

```
LSTM hacia adelante:  lee  hora-18 → hora-17 → ... → hora 0 → ... → hora+18
LSTM hacia atrás:     lee  hora+18 → hora+17 → ... → hora 0 → ... → hora-18
```

En cada paso temporal, las salidas de ambas LSTMs se **concatenan**:

```
Salida hacia adelante en hora 0:   [v1, v2, ..., v96]  (96 valores)
Salida hacia atrás en hora 0:      [u1, u2, ..., u96]  (96 valores)
Salida combinada en hora 0:        [v1...v96, u1...u96] (192 valores)
```

Esto le da al modelo información sobre lo que ocurrió antes Y lo que el GFS predice que ocurrirá después — un contexto más rico para hacer predicciones.

### ¿Cuándo es apropiada la BiLSTM?

La BiLSTM es apropiada cuando:
1. La información futura está disponible en el momento de la predicción ✅ (los pronósticos del GFS para las próximas 18 horas están disponibles)
2. La información futura es genuinamente útil para la predicción actual ✅ (los patrones de nubosidad de la tarde pueden informar las predicciones de la mañana)

La BiLSTM NO es apropiada para predicción en tiempo real verdadero donde los datos futuros no están disponibles. Este proyecto usa pronósticos del GFS — que existen para el futuro — por lo que la BiLSTM es válida.

---

## 9. Mecanismo de Atención

### El problema con usar solo el último estado oculto

Antes de añadir atención, la BiLSTM se usaba de la forma tradicional: solo el estado oculto del ÚLTIMO paso temporal se usaba para la predicción. Esto significa:

```
Secuencia de 37 pasos procesada por BiLSTM:
Paso 1:  salida h_1  (descartada)
Paso 2:  salida h_2  (descartada)
...
Paso 36: salida h_36 (descartada)
Paso 37: salida h_37 (USADA para la predicción)
```

¡Todas las salidas intermedias — 36 de los 37 pasos temporales — fueron ignoradas! El último estado oculto debería contener teóricamente un resumen de toda la secuencia (porque la información se propagó hacia adelante), pero en la práctica algo de información se pierde o diluye, especialmente para secuencias largas.

### Atención de Bahdanau — aprender qué pasos importan

La atención permite que el modelo mire las 37 salidas simultáneamente y aprenda cuáles son más relevantes para cada predicción específica:

```python
class AtenciónBahdanau(nn.Module):
    def __init__(self, hidden_size):
        # Un vector de pesos aprendible: 192 valores + 1 sesgo
        self.W = nn.Linear(hidden_size, 1)

    def forward(self, salida_lstm):
        # salida_lstm forma: (batch, 37, 192)
        # Tenemos 37 vectores de salida, cada uno de 192 valores

        # Paso 1: Puntuar cada paso temporal
        # ¿Qué tan "importante" es cada paso para esta predicción?
        energia = torch.tanh(self.W(salida_lstm))
        # energia forma: (batch, 37, 1) — una puntuación por paso temporal

        # Paso 2: Convertir puntuaciones en pesos que suman 1
        pesos = torch.softmax(energia, dim=1)
        # pesos forma: (batch, 37, 1)
        # Ejemplo: [0.01, 0.02, ..., 0.35, 0.28, ..., 0.01]
        #           hora-18           hora0  hora+6    hora+18
        # El modelo aprendió que las horas 0 y +6 son más informativas

        # Paso 3: Suma ponderada — el "vector de contexto"
        contexto = (pesos * salida_lstm).sum(dim=1)
        # contexto forma: (batch, 192)
        # Un solo vector que resume toda la secuencia,
        # con más peso en los pasos temporales importantes

        return contexto, pesos
```

### Qué nos dicen los pesos de atención

Después del entrenamiento, podemos visualizar los pesos de atención para entender qué aprendió el modelo:

```python
import numpy as np
import matplotlib.pyplot as plt

pesos = np.load("attn_weights_test.npy")  # forma: (5332, 37)

# Atención promedio por posición de paso temporal
pesos_promedio = pesos.mean(axis=0)  # forma: (37,)

# Pasos temporales: -18, -17, ..., 0, ..., +17, +18
pasos = range(-18, 19)

# Si el modelo aprendió bien, podríamos ver:
# - Alta atención cerca del paso 0 (la hora del pronóstico misma)
# - Alta atención en pasos recientes del pasado (-1, -2, -3)
# - Alta atención en pasos futuros cercanos (+1, +2, +3)
# - Baja atención en pasado/futuro distante
```

### Por qué la atención no mejoró los resultados en este proyecto

La atención de Bahdanau añadió solo 193 parámetros y produjo métricas idénticas al baseline. Esto probablemente se debe a que:

1. La BiLSTM de 3 capas ya era suficientemente sofisticada para aprender qué pasos temporales importan — la atención fue redundante
2. El problema fundamental (el modelo sigue al GFS en lugar de corregirlo) no es un problema de ponderación temporal sino de aprendizaje de features

---

## 10. Entrenamiento de una Red Neuronal

### El bucle de entrenamiento — paso a paso

```python
for epoca in range(50):
    model.train()  # Activar modo entrenamiento (activa el dropout)

    for X_batch, y_batch in train_loader:
        # X_batch forma: (128, 37, 69) — 128 secuencias, 37 pasos, 69 features
        # y_batch forma: (128,) — 128 valores CSI objetivo

        # PASO 1: PROPAGACIÓN HACIA ADELANTE
        # El modelo procesa el batch y produce predicciones
        csi_predicho = model(X_batch)
        # csi_predicho forma: (128,) — valores entre 0 y 1

        # PASO 2: CALCULAR LA PÉRDIDA
        # ¿Qué tan equivocadas estaban las predicciones?
        perdida = error_cuadratico_medio(csi_predicho, y_batch)
        # perdida: un solo número (error cuadrático promedio del batch)

        if USE_DAYMASK:
            # Solo contar errores durante horas diurnas
            perdida = perdida * mascara_dia_batch

        # PASO 3: RETROPROPAGACIÓN
        # Calcular cuánto contribuyó cada uno de los 574,220 parámetros
        # al error
        optimizer.zero_grad()   # Limpiar gradientes anteriores
        perdida.backward()      # Calcular gradientes

        # PASO 4: ACTUALIZAR PESOS
        # Mover cada parámetro ligeramente en la dirección que reduce el error
        optimizer.step()
        # Cantidad movida = tasa_aprendizaje × gradiente

    # DESPUÉS DE CADA ÉPOCA: evaluar en el conjunto de validación
    model.eval()  # Desactivar dropout para evaluación
    with torch.no_grad():  # No calcular gradientes (ahorra memoria)
        predicciones_val = model(X_val)
        perdida_val = mse(predicciones_val, y_val)

    # Verificación de early stopping
    if perdida_val < mejor_perdida_val:
        mejor_perdida_val = perdida_val
        torch.save(model.state_dict(), "mejor_modelo.pt")
        contador_paciencia = 0
    else:
        contador_paciencia += 1
        if contador_paciencia >= EARLY_STOP:
            print(f"Early stopping en época {epoca}")
            break

    # Scheduler de tasa de aprendizaje
    scheduler.step(perdida_val)
    # Si perdida_val no mejoró por LR_PATIENCE épocas,
    # multiplicar LR por LR_FACTOR (0.5) — dar pasos más pequeños
```

### Retropropagación — cómo se calculan los gradientes

La retropropagación es el algoritmo que calcula cuánto contribuyó cada peso al error total. Funciona aplicando la **regla de la cadena del cálculo** hacia atrás a través de la red:

```
El error total depende de:
  → Pesos de la capa de salida
      → que dependen de las salidas de la capa oculta 3
          → que dependen de los pesos de la capa oculta 3
              → que dependen de las salidas de la capa oculta 2
                  → ... y así hacia atrás hasta la entrada

Regla de la cadena: d(error)/d(peso) = d(error)/d(salida) × d(salida)/d(peso)
```

PyTorch maneja todo esto automáticamente — solo llamas `perdida.backward()` y calcula todos los gradientes.

### El optimizador — AdamW

AdamW (Adam con Decaimiento de Peso) es el algoritmo que usa gradientes para actualizar pesos:

```python
# Regla de actualización simplificada de Adam:
# Para cada peso w:
m = beta1 × m + (1 - beta1) × gradiente      # Momento (gradiente suavizado)
v = beta2 × v + (1 - beta2) × gradiente²     # Velocidad (cuadrado suavizado)

# Paso adaptativo: pesos que ya cambiaron mucho reciben actualizaciones menores
w = w - lr × m / (√v + epsilon)

# Decaimiento de pesos (regularización L2):
w = w - lr × decaimiento_peso × w
# Esto penaliza pesos grandes, previniendo el sobreajuste
```

**¿Por qué AdamW sobre el descenso de gradiente simple?**
- **Tasa de aprendizaje adaptativa por parámetro:** Algunos pesos pueden necesitar actualizaciones grandes, otros pequeñas. AdamW ajusta automáticamente
- **Momento:** Ayuda a escapar de mínimos locales superficiales y suavizar gradientes ruidosos
- **Decaimiento de pesos:** Regularización incorporada

### El scheduler de tasa de aprendizaje — ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    factor=0.5,      # Multiplicar LR por 0.5 cuando se activa
    patience=4,      # Activar después de 4 épocas sin mejora
    min_lr=1e-6      # Nunca bajar de esto
)

# Ejemplo de línea de tiempo:
# Épocas 1-15:  LR = 0.001, validación mejorando
# Épocas 16-19: LR = 0.001, validación estancada (contador paciencia: 1,2,3,4)
# Época 20:     LR = 0.0005, scheduler activado
# Épocas 21-24: LR = 0.0005, todavía estancado
# Época 25:     LR = 0.00025, activado de nuevo
```

La intuición: cuando los pasos grandes ya no ayudan (la validación se estabiliza), cambiar a pasos más pequeños y precisos.

---

## 11. Sobreajuste y Regularización

### ¿Qué es el sobreajuste?

El sobreajuste ocurre cuando el modelo aprende los datos de entrenamiento **demasiado específicamente** — memoriza los ejemplos exactos en lugar de aprender los patrones subyacentes.

Analogía visual usando un gráfico de dispersión de puntos de datos:

```
SUBAJUSTE:                   BUEN AJUSTE:              SOBREAJUSTE:
Una línea recta              Una curva suave que        Un zigzag que pasa
que falla en la              sigue la tendencia         por cada punto exacto
mayoría de puntos            general                    

Error entrenamiento: alto    Error entrenamiento: bajo  Error entrenamiento: muy bajo
Error validación: alto       Error validación: bajo     Error validación: alto
```

**Señales de sobreajuste en este proyecto:**
- La pérdida de entrenamiento sigue disminuyendo (0.029 → 0.012 en 50 épocas)
- La pérdida de validación deja de mejorar después de ~época 20 (se estabiliza en ~0.020)
- Gran brecha entre el RMSE de entrenamiento y el RMSE de prueba

### Por qué ocurre el sobreajuste aquí

El modelo tiene **574,220 parámetros** pero solo **24,889 secuencias de entrenamiento**. Esto es aproximadamente 23 parámetros por ejemplo de entrenamiento — dando al modelo enorme capacidad para memorizar específicos en lugar de generalizar.

Adicionalmente, el CSI del GFS y el CSI de SIATA están muy correlacionados — el modelo encontró la solución fácil de simplemente copiar la entrada del GFS en lugar de aprender la corrección.

### Técnicas de regularización usadas

**1. Dropout (p=0.25)**

Durante cada paso de entrenamiento, el 25% de las neuronas se establece aleatoriamente en 0. Esto obliga a la red a aprender representaciones redundantes — ninguna neurona puede volverse indispensable:

```
Sin dropout:
Neurona A siempre maneja "patrones de nubes de la mañana"
Neurona B siempre maneja "convección de la tarde"
→ La red se vuelve frágil — si A o B falla, la predicción sufre

Con dropout:
A veces A está desactivada, a veces B, a veces ambas, a veces ninguna
→ La red aprende a distribuir la tarea entre muchas neuronas
→ Más robusta, menos propensa a memorizar patrones específicos
```

En producción (modo de evaluación), el dropout se desactiva — todas las neuronas están activas.

**2. Decaimiento de pesos (regularización L2, λ=0.001)**

Añade una penalización a la función de pérdida proporcional a la suma de pesos al cuadrado:

```
Pérdida total = Pérdida_MSE + λ × Σ(w²)

Esto desalienta que cualquier peso individual se vuelva muy grande.
Pesos grandes = la red depende fuertemente de features específicas,
lo cual es señal de memorización en lugar de generalización.
```

**3. Early Stopping (paciencia=25)**

Monitorea la pérdida de validación después de cada época. Si no mejora por 25 épocas consecutivas, detiene el entrenamiento y restaura el mejor modelo (en el mínimo de pérdida de validación):

```
Línea de tiempo de pérdida de validación:
Época 1:  0.0230  ← nuevo mejor, guardar modelo
Época 5:  0.0215  ← nuevo mejor, guardar modelo
Época 15: 0.0200  ← nuevo mejor, guardar modelo
Época 20: 0.0202  ← sin mejora (contador: 1)
Época 21: 0.0205  ← sin mejora (contador: 2)
...
Época 40: 0.0198  ← ¡nuevo mejor! reiniciar contador, guardar modelo
...
Época 65: igual que época 40 por 25 épocas → DETENER
```

**4. LayerNorm**

Aplicada a la entrada antes de la BiLSTM. Normaliza la distribución de activaciones:

```python
# Fórmula LayerNorm:
normalizado = (x - media(x)) / (desv_std(x) + ε)

# Previene el "cambio de covariable interno":
# Sin LayerNorm, la distribución de entradas a cada capa cambia
# a medida que los pesos de la capa anterior cambian durante el entrenamiento.
# Esto hace el entrenamiento inestable y lento.
```

---

## 12. Las 69 Features de Entrada — Por Qué Importa Cada Una

### Los 4 launch times como perspectivas paralelas

Tener los 4 launch times le da a la red 4 vistas simultáneas del mismo pronóstico, cada una con diferentes tiempos de adelanto y calculadas a partir de observaciones progresivamente más recientes:

```
Para predecir la radiación del martes a las 15:00:

Lanzamiento 0100 (adelanto: 14h): "Hace 14 horas, la atmósfera lucía así..."
Lanzamiento 0700 (adelanto:  8h): "Hace 8 horas, el pronóstico actualizado decía..."
Lanzamiento 1300 (adelanto:  2h): "Hace 2 horas, el pronóstico más reciente decía..."
Lanzamiento 1900 (adelanto: -4h): "Este lanzamiento es después de las 15:00 
                                    — usa info del día siguiente"
```

La feature `step` (tiempo de adelanto en horas) le dice a la red qué lanzamientos son más recientes y probablemente más confiables.

### Variables de radiación (CSI y KC)

**CSI (Índice de Cielo Despejado):** Este es el predictor más directo. Si el GFS predice CSI=0.9, está prediciendo un día principalmente despejado. La red refina esto aprendiendo los errores sistemáticos específicos de Medellín.

**KC (Índice de Claridad):** Una normalización ligeramente diferente usando radiación extraterrestre en lugar de Clear-Sky. Los dos índices juntos le dan a la red información redundante que ayuda a detectar inconsistencias del modelo — si CSI y KC dan señales de nubosidad muy diferentes, algo inusual está ocurriendo.

**DSWRF (vatios brutos):** El valor de radiación no normalizado proporciona información de magnitud absoluta que CSI/KC no capturan.

**DLWRF (radiación de onda larga):** Radiación térmica de nubes y atmósfera. DLWRF alto indica aire cálido y húmedo, a menudo correlacionado con condiciones nubladas. Complementa la señal de radiación de onda corta.

### Variables de cobertura de nubes (TCDC, HCDC, MCDC, LCDC)

El GFS predice cobertura de nubes en cuatro niveles:
- **Total (TCDC):** Fracción total de nubes — predictor directo de nubosidad
- **Altas (HCDC):** Nubes cirrus — delgadas, usualmente no bloquean mucha radiación
- **Medias (MCDC):** Alto-stratus — bloquean parcialmente la radiación
- **Bajas (LCDC):** Stratus, niebla — más impactantes, pueden bloquear el 80%+ de radiación

La combinación de los cuatro niveles ayuda a distinguir entre:
- **Cirrus delgado:** HCDC alta, MCDC/LCDC baja — reducción pequeña de radiación
- **Nublado denso:** TCDC + LCDC alta — reducción grande de radiación
- **Cúmulos de buen tiempo:** TCDC moderada pero cobertura baja por tipo — irregular

**HGT_cloud_ceiling:** La altura de la base de las nubes. Techo de nubes bajo (< 1000 m) indica niebla o nubes stratus que reducen fuertemente la radiación. Techo alto indica cirrus alto con impacto mínimo.

### Variables termodinámicas

**CAPE (Energía Potencial Convectiva Disponible):** Esta es una variable clave específicamente para Medellín. El CAPE mide la energía disponible para la convección — cuanto mayor el CAPE, más probable es que se desarrollen tormentas eléctricas por la tarde (nubes cumulonimbus). Medellín frecuentemente experimenta nubosidad convectiva por la tarde impulsada por el calentamiento del valle, y el CAPE es el principal indicador termodinámico.

```
Interpretación del CAPE:
  0–500 J/kg:    Bajo potencial convectivo
  500–1000 J/kg: Moderado — posibles aguaceros de tarde
  1000–2500 J/kg: Alto — probable tormentas
  > 2500 J/kg:  Extremo — tormentas severas
```

**PWAT (Agua Precipitable):** Total de vapor de agua en la columna atmosférica. Alto contenido de humedad aumenta la probabilidad de formación de nubes y mejora la absorción de radiación solar incluso en días nominalmente despejados.

**HPBL (Altura de la Capa Límite Planetaria):** La profundidad de la capa de mezcla turbulenta cerca de la superficie. HPBL bajo a menudo indica condiciones estables y neblinosas. HPBL alto indica mezcla vigorosa y actividad convectiva.

### Otras variables

**TMP (Temperatura):** La temperatura superficial afecta la evaporación, el inicio de convección y el desarrollo de la capa límite. Altas temperaturas en la tarde impulsan la nubosidad convectiva típica de Medellín.

**RH (Humedad Relativa a 2m):** Humedad cerca de la superficie. Alta HR combinada con temperaturas en aumento es un disparador para la formación de nubes.

**Wind10m:** La velocidad del viento afecta la advección horizontal de sistemas de nubes — qué tan rápido se mueven los sistemas de nubes por el área. Vientos fuertes pueden cambiar rápidamente las condiciones.

**SUNSD (Duración del Brillo Solar):** Minutos de sol acumulados dentro del período temporal del GFS. Una medida directa de nubosidad integrada en el tiempo — si el GFS acumuló solo 20 minutos de sol en la última hora, espera densa cobertura de nubes.

**step (Tiempo de adelanto):** El número de horas entre cuando se lanzó el GFS y la hora de pronóstico. Crítico para ponderar: los pronósticos de 2 horas son generalmente mucho más precisos que los de 14 horas. La red aprende a confiar más en los lanzamientos recientes.

**zenith (Ángulo cenital):** El ángulo cenital solar (ángulo desde la vertical hasta la posición del sol). En zenith=0°, el sol está directamente arriba (máxima radiación). En zenith=90°, el sol está en el horizonte (mínima radiación). Esta única variable geométrica captura todo el ciclo diario de radiación sin que la red necesite inferirlo de los datos.

---

## 13. Secuencias y la Ventana Temporal

### ¿Por qué usar secuencias en lugar de puntos temporales individuales?

La radiación solar tiene fuerte autocorrelación temporal — lo que ocurrió en el pasado es informativo sobre lo que ocurrirá a continuación. Una red neuronal simple que solo ve las features del GFS de la hora actual pierde este contexto temporal.

Patrones temporales clave que las secuencias capturan:
- **Aumento de nubes de la mañana:** Si las últimas 3 horas muestran cobertura de nubes en aumento, la siguiente hora probablemente continuará la tendencia
- **Temporización convectiva:** Medellín típicamente desarrolla nubes convectivas en la tarde (12:00-16:00). Conocer la hora del día (codificada en zenith) y la evolución reciente de nubes ayuda a predecir el inicio
- **Paso de frentes:** Los sistemas meteorológicos sinópticos tardan horas en cruzar la región — las condiciones pasadas revelan la trayectoria del sistema

### La ventana simétrica de 37 pasos (sym18)

```
Estructura de la ventana:
[hora-18] [hora-17] ... [hora-1] [HORA 0] [hora+1] ... [hora+17] [hora+18]
    ←──────────── PASADO (18 horas) ──────────────► ←── FUTURO (18 horas) ──►

Cada posición: 69 features del GFS (de los 4 launch times + zenith)
Objetivo: CSI de SIATA en la HORA 0
```

**¿Por qué 18 horas a cada lado (sym18) en lugar de 24 (sym24)?**

Del experimento de comparación de ventanas:
- sym12 (12h cada lado): RMSE = 180.36 W/m² — muy poco contexto
- **sym18 (18h cada lado): RMSE = 171.10 W/m²** ← seleccionado
- sym24 (24h cada lado): RMSE = 171.59 W/m² — mismo rendimiento, más cómputo

sym18 proporciona la ventana de contexto óptima: suficiente información temporal (18 horas cubre la mayoría de patrones sinópticos relevantes) sin cómputo innecesario (25% menos pasos que sym24).

### Construcción de secuencias — la ventana deslizante

```
Serie temporal horaria de 4 años: 35,040 horas

Deslizar una ventana de 37 pasos a través de la serie temporal:
Posición 1: horas 1–37    → predecir hora 19 (centro)
Posición 2: horas 2–38    → predecir hora 20
Posición 3: horas 3–39    → predecir hora 21
...
Posición N: horas N–N+36  → predecir hora N+18

Total de posiciones: 35,040 - 36 = 35,004
Menos efectos de borde (primeras/últimas 18 horas): ≈ 35,004
Menos secuencias con valores NaN: ~451 secuencias eliminadas
Secuencias válidas finales: ~35,553
```

**¿Por qué se eliminan algunas secuencias por NaN?**

Si cualquiera de las 69 features en cualquiera de los 37 pasos temporales es NaN (datos faltantes), toda la secuencia se elimina. Las redes neuronales no pueden procesar valores NaN — se propagarían a través de todos los cálculos y producirían salidas NaN.

### La división 70/15/15

```python
secuencias = mezclar(todas_35553_secuencias)  # Orden aleatorio

n_train = int(0.70 × 35553) = 24887
n_val   = int(0.15 × 35553) = 5332
n_test  = int(0.15 × 35553) = 5332 (+ 2 por redondeo)

train = secuencias[:24887]
val   = secuencias[24887:30219]
test  = secuencias[30219:]
```

**¿Por qué mezclar antes de dividir?**

Si dividiéramos cronológicamente (train = primer 70% del tiempo), el conjunto de entrenamiento contendría 2021-2023 y el de prueba 2024. El modelo se evaluaría en un año diferente con patrones climáticos potencialmente distintos.

La mezcla aleatoria asegura que los tres conjuntos tengan distribuciones similares de estaciones, patrones meteorológicos y eventos extremos.

---

## 14. Métricas — Explicación Profunda

### El protocolo de evaluación

Todas las métricas finales se calculan en el **conjunto de prueba solamente** — las 5,332 secuencias que el modelo nunca vio durante el entrenamiento o la selección de hiperparámetros.

Para el SkillScore específicamente, solo se incluyen las **horas diurnas** (Clear-Sky GHI > 0). Incluir ceros nocturnos inflaría el MSE del baseline y haría el SkillScore artificialmente negativo.

### RMSE — Raíz del Error Cuadrático Medio

**Fórmula:**
```
RMSE = √( (1/N) × Σᵢ (predicho_i - real_i)² )
```

**Paso a paso:**
```
Para cada una de las N horas de prueba:
  error_i = GHI_predicho - GHI_real

  Errores de ejemplo:
  Hora 1: predicho=400, real=420 → error = -20
  Hora 2: predicho=800, real=500 → error = +300 (¡malo!)
  Hora 3: predicho=50,  real=60  → error = -10

Elevar al cuadrado cada error (hace todos positivos, penaliza errores grandes):
  Hora 1: (-20)²  = 400
  Hora 2: (300)²  = 90,000  ← ¡este domina!
  Hora 3: (-10)²  = 100

Promediar los errores cuadráticos (MSE):
  MSE = (400 + 90,000 + 100) / 3 = 30,167

Tomar la raíz cuadrada para volver a las unidades originales:
  RMSE = √30,167 = 173.7 W/m²
```

**El problema de dilución nocturna:**
```
Cálculo del RMSE de todas las horas:
  5,332 secuencias de prueba
  2,411 horas nocturnas → error = 0 para modelo y GFS
  2,921 horas diurnas   → errores pueden ser grandes

MSE de todas horas = (suma de todos los errores al cuadrado) / 5,332
                   = (0 × 2411 + errores_diurnos) / 5,332
                   = MSE_diurno × (2921/5332)
                   = 53,447 × 0.548
                   = 29,281
                   → RMSE = √29,281 = 171.1 W/m²

RMSE solo diurno = √53,447 = 231.2 W/m²

¡El RMSE de todas las horas de 171 W/m² parece mucho mejor que 
el RMSE diurno honesto de 231 W/m²!
```

### MAE — Error Absoluto Medio

**Fórmula:**
```
MAE = (1/N) × Σᵢ |predicho_i - real_i|
```

**Comparación con RMSE:**

```
Mismo ejemplo:
Hora 1: error = -20 → |error| = 20
Hora 2: error = +300 → |error| = 300
Hora 3: error = -10 → |error| = 10

MAE = (20 + 300 + 10) / 3 = 110 W/m²
RMSE = 173.7 W/m²

RMSE >> MAE → hay algunos errores muy grandes inflando el RMSE
```

En este proyecto: MAE = 82.5 W/m² (todas las horas), RMSE = 171.1 W/m²
La relación RMSE/MAE ≈ 2.1 indica eventos de error grande significativos.

### R² — Coeficiente de Determinación

**Fórmula:**
```
R² = 1 - (MSE_modelo / MSE_ingenuo)

donde MSE_ingenuo = varianza de los valores reales
                  = promedio de (real_i - media_real)²
```

**Intuición:**

```
MSE_ingenuo es lo que obtendrías prediciendo siempre el promedio histórico.
Si no sabes nada sobre la radiación solar excepto el promedio histórico
(digamos 300 W/m²) y siempre predices 300 W/m², tu MSE sería
la varianza de los datos de radiación.

R² mide cuánto mejor es tu modelo relativo a esta línea base:
  R² = 0.0: tu modelo es exactamente tan bueno como predecir el promedio
  R² = 0.5: tu modelo explica la mitad de la variabilidad
  R² = 1.0: tu modelo es perfecto
  R² < 0:   tu modelo es PEOR que predecir el promedio
```

**Por qué R² difiere entre espacios:**

```
R² de todas las horas = 0.671  (incluye noche)
R² solo diurno        = 0.472  (solo horas diurnas)

De noche, real = 0 y predicho = 0 para todas las horas.
Estos pares de ceros inflan artificialmente la correlación porque
el modelo trivialmente "predice" la noche perfectamente.
Incluir la noche hace que R² aparezca mejor de lo que realmente es.
```

### Correlación de Pearson

**Fórmula:**
```
r = Σ[(predicho_i - media_pred) × (real_i - media_real)] /
    [N × desv_pred × desv_real]
```

**Idea clave: correlación alta ≠ predicciones precisas**

```
Ejemplo: el modelo siempre predice exactamente 2× el valor real
  real:      100, 200, 500, 300, 700
  predicho:  200, 400, 1000, 600, 1400

  Correlación r = 1.0 (perfecta — se mueven perfectamente juntos)
  RMSE = (100 + 200 + 500 + 300 + 700) / 5 = 360 W/m² (¡terrible!)
  R² = negativo (el modelo es peor que predecir la media)
```

En este proyecto, el modelo tiene correlación = 0.84 pero SkillScore = -0.31. Esto confirma que el modelo captura la tendencia general del GFS (cuando el GFS sube, el modelo sube) pero con sesgo sistemático y errores individuales grandes.

### SkillScore

**Fórmula:**
```
SS = 1 - (MSE_modelo / MSE_GFS_crudo)

donde ambos MSE se calculan en el MISMO conjunto de prueba, solo horas diurnas
```

**El problema del MSE_BASELINE_R:**

```
Valor de config de Leonard:  MSE_BASELINE_R = 22,322
Valor diurno correcto:       MSE_BASELINE_R = 40,761

¿Por qué la diferencia?
22,322 = MSE_diurno × (fracción_diurna)
       = 40,761 × (2,921/5,332)
       = 40,761 × 0.548
       = 22,337 ≈ 22,322 ✓

Leonard calculó el MSE del baseline del GFS sobre TODAS las horas (incluida 
la noche), lo que diluyó el denominador en ~45%, haciendo que el SkillScore 
apareciera más positivo (o menos negativo) que el cálculo honesto solo diurno.
```

**Situación actual:**
```
MSE_modelo    = 53,447 (diurno)
MSE_GFS_crudo = 40,761 (diurno)
SkillScore    = 1 - 53,447/40,761 = 1 - 1.311 = -0.311

El modelo es 31.1% peor que el GFS crudo en horas diurnas.
```

---

## 15. SkillScore — Qué Significa y Por Qué Es Negativo

### Qué significa SkillScore = -0.311 en la práctica

Para cada hora diurna en el conjunto de prueba:
- **MSE del GFS crudo:** error cuadrático promedio de 40,761 W²/m⁴ → RMSE = 201.9 W/m²
- **MSE del modelo:** error cuadrático promedio de 53,447 W²/m⁴ → RMSE = 231.2 W/m²

El modelo añade aproximadamente **29 W/m² de error** encima del pronóstico GFS crudo. Sería mejor para Emergente usar el GFS directamente sin ninguna corrección.

### Análisis de causa raíz

**Causa 1: El modelo aprendió a copiar el GFS**

Mirando el gráfico de series temporales de diagnóstico, el modelo (azul) y el GFS (naranja) son casi idénticos. La red aprendió un mapeo casi de identidad de CSI_GFS → CSI_SIATA, que es el camino de menor resistencia — minimiza la pérdida de entrenamiento sin aprender realmente las correcciones.

¿Por qué ocurre esto? El CSI del GFS y el CSI de SIATA están correlacionados (r ≈ 0.77). El modelo encontró que la manera más simple de reducir el MSE de entrenamiento es aproximar esta relación lineal en lugar de aprender las correcciones complejas específicas de la ubicación.

**Causa 2: La nubosidad convectiva es inherentemente impredecible desde el GFS**

Una fracción significativa de la nubosidad de Medellín está impulsada por:
- Circulación local del valle (niebla matutina, convección de la tarde)
- Sistemas convectivos de mesoescala que se desarrollan en horas
- Efectos orográficos de las montañas circundantes

Estos fenómenos ocurren a escalas de 1-10 km — muy por debajo de la resolución del GFS de 28 km. Ninguna arquitectura de red neuronal sofisticada puede extraer información sobre nubes a escala de 5 km de un modelo a resolución de 28 km.

**Causa 3: La pérdida de entrenamiento de todas las horas ocultó el problema real**

La pérdida de entrenamiento incluía horas nocturnas (donde modelo y verdad = 0, contribuyendo cero pérdida). Esto infló el rendimiento aparente del modelo. El criterio de early stopping monitoreaba esta métrica diluida, no el rendimiento solo diurno que realmente importa.

### Qué probar a continuación

**1. Función de pérdida solo diurna**

```python
# Implementación actual:
perdida = mse(predicciones * mascara_dia, objetivos * mascara_dia)
# Horas nocturnas: ambos lados se vuelven 0, error = 0 pero aún contado

# Mejor implementación:
idx_diurno = mascara_dia > 0
perdida = mse(predicciones[idx_diurno], objetivos[idx_diurno])
# Horas nocturnas completamente excluidas del cálculo de pérdida
# Esto fuerza al modelo a enfocarse enteramente en mejorar las predicciones diurnas
```

**2. Análisis de sensibilidad de arquitectura**

Probar si arquitecturas más profundas/anchas pueden aprender patrones de corrección más complejos:
- 5 diferentes conteos de capas: 1, 2, 3, 5, 7, 10
- 5 diferentes tamaños ocultos: 32, 64, 96, 128, 192, 256

**3. Arquitectura causal (LSTM solo hacia adelante)**

El modelo de Leonard con 1 launch time y LSTM causal (solo pasado) aparentemente tuvo mejor desempeño. Una arquitectura causal no puede "hacer trampa" mirando pronósticos futuros del GFS, potencialmente forzándola a aprender correcciones más robustas de las condiciones actuales y pasadas.

**4. Variable objetivo alternativa**

En lugar de predecir el CSI directamente, predecir el **término de corrección**:
```
corrección = CSI_SIATA - CSI_GFS
objetivo = corrección (positivo = GFS subestima, negativo = sobreestima)
```
Esto fuerza explícitamente al modelo a aprender correcciones en lugar de valores absolutos.

---

## 16. Experimentos — Qué Probamos y Qué Aprendimos

### Experimento 1: Entrenamiento baseline (sym24, hiperparámetros predeterminados)

**Configuración:**
```python
SEQ_MODE = "sym24"   # Ventana de 49 pasos
HIDDEN = 96
NUM_LAYERS = 3
LR = 0.001
BATCH_SIZE = 128
DROPOUT = 0.25
EARLY_STOP = 25
EPOCHS = 50
```

**Resultados:**
```
RMSE todas las horas: 171.59 W/m²
RMSE solo diurno:     231.2 W/m²
R² (diurno):          0.472
SkillScore:          -0.319
```

**Qué aprendimos:** El modelo entrena exitosamente pero no supera el baseline del GFS. Sobreajuste visible desde la época 20 (la pérdida de entrenamiento continúa disminuyendo, la validación se estabiliza).

---

### Experimento 2: Comparación de tamaño de ventana

**Hipótesis:** La ventana de 49 pasos (sym24) puede incluir demasiado contexto distante que añade ruido. Ventanas más cortas pueden capturar patrones relevantes más limpiamente.

**Método:** Mismos hiperparámetros, tres tamaños de ventana.

**Resultados:**
| Ventana | Pasos | RMSE | SkillScore |
|---------|-------|------|-----------|
| sym12 | 25 | 180.36 W/m² | −0.457 |
| **sym18** | **37** | **171.10 W/m²** | **−0.311** |
| sym24 | 49 | 171.59 W/m² | −0.319 |

**Qué aprendimos:**
- 12h de contexto es insuficiente — el RMSE aumenta 9 W/m²
- 18h y 24h son esencialmente equivalentes
- sym18 seleccionado: mismo rendimiento con 25% menos cómputo
- **La ventana no es el cuello de botella** — el problema está en otro lugar

---

### Experimento 3: Búsqueda de hiperparámetros con Optuna

**Hipótesis:** Diferentes hiperparámetros pueden encontrar un mínimo de pérdida mejor.

**Método:** 30 trials, 20 épocas cada uno, 6 hiperparámetros buscados.

**Mejor trial encontrado:**
```python
LR = 0.000102    (10× más pequeño que el baseline)
HIDDEN = 192     (2× más grande que el baseline)
NUM_LAYERS = 1   (3× menos capas que el baseline)
DROPOUT = 0.128  (la mitad del baseline)
BATCH_SIZE = 32  (4× más pequeño que el baseline)
```

**Resultado del entrenamiento completo (50 épocas):**
```
RMSE: 172.55 W/m²  — PEOR que el baseline (171.10)
```

**Qué aprendimos:**
- 20 épocas insuficientes para que el modelo más grande hidden=192 converja
- La arquitectura de 3 capas supera la de 1 capa — la jerarquía más profunda es beneficiosa
- Los hiperparámetros predeterminados ya estaban cerca del óptimo para esta arquitectura
- **Los hiperparámetros no son el cuello de botella** — el problema es estructural

---

### Experimento 4: Atención de Bahdanau

**Hipótesis:** El modelo ignora la mayoría de los 37 pasos temporales al usar solo el último estado oculto. La atención podría ayudarlo a enfocarse en los pasos más informativos.

**Método:** Añadir capa de AtenciónBahdanau (+193 parámetros), por lo demás idéntico al baseline sym18.

**Resultados:**
| Métrica | Baseline | Con Atención | Cambio |
|---------|---------|-------------|--------|
| RMSE | 171.10 | 171.11 | ≈ 0 |
| R² | 0.6708 | 0.6708 | = |
| SkillScore | −0.311 | −0.311 | ≈ 0 |

**Qué aprendimos:**
- La BiLSTM de 3 capas ya captura suficiente información temporal
- La atención es redundante cuando la LSTM subyacente ya es suficientemente poderosa
- El problema fundamental no es la ponderación temporal sino el aprendizaje de features
- **El mecanismo de atención no es el cuello de botella** — el problema está en lo que aprende el modelo

### Resumen de conclusiones

Las tres modificaciones — tamaño de ventana, hiperparámetros, atención — produjeron esencialmente el mismo SkillScore de aproximadamente −0.311. Esta fuerte consistencia sugiere que el problema es **estructural**, no ajustable:

1. El modelo aprende a aproximar el GFS en lugar de corregirlo
2. Los eventos de nubosidad convectiva que causan los errores más grandes simplemente no son predecibles desde datos del GFS a 28 km de resolución
3. La métrica de pérdida de entrenamiento (MSE de todas las horas) no está alineada con la métrica de evaluación (SkillScore solo diurno)

Las direcciones más prometedoras para mejorar son cambios arquitecturales (pérdida solo diurna, objetivo de corrección, arquitectura causal) en lugar de ajustar la configuración existente.

---

## 17. El config.py — Cada Parámetro Explicado

```python
# ─────────────────────────────────────────────────────
# PARÁMETROS GEOGRÁFICOS
# ─────────────────────────────────────────────────────

LAT = 6.25
# Latitud de Medellín en grados decimales (Norte positivo)
# Usada por pvlib para calcular posición del sol y techos de radiación
# Usada por cfgrib para recortar datos del GFS al punto de malla más cercano

LON = -75.5
# Longitud de Medellín en grados decimales (Oeste negativo)
# Mismos usos que LAT

ELEVATION = 1582
# Altitud sobre el nivel del mar en metros
# Afecta el cálculo del Clear-Sky — atmósfera más delgada en altitud
# significa menos absorción atmosférica → valores más altos de Clear-Sky

UTC_OFFSET = -5
# Colombia es UTC-5, sin horario de verano
# Usado para convertir marcas de tiempo UTC del GFS a etiquetas de hora local
# (00 UTC → 19:00 local → etiqueta "0100" para el siguiente día del calendario)

LAUNCH_TIMES = ["0100", "0700", "1300", "1900"]
# Etiquetas para los 4 lanzamientos diarios del GFS en hora local Colombia
# Corresponden a 06, 12, 18, 00 UTC respectivamente
# Usados en todo el código para nombrar archivos y variables

FLAG_START = 19
# Hora (hora local) cuando comienza la noche
# Después de las 19:00 → la agregación de SIATA fuerza la radiación a 0

FLAG_END = 6
# Hora (hora local) cuando comienza el día
# Antes de las 06:00 → la agregación de SIATA fuerza la radiación a 0
# Nota: diurno = FLAG_END a FLAG_START (06:00 a 19:00)

# ─────────────────────────────────────────────────────
# CONVENCIÓN DE NOMBRES DE VARIABLES (VAR_*)
# Todos los nombres de variables para archivos NetCDF están centralizados aquí
# Los scripts importan estas constantes en lugar de codificar cadenas directamente
# ─────────────────────────────────────────────────────

VAR_GFS_DSWRF_ACUM = "sdswrf"
# Nombre de la variable DSWRF acumulada en archivos GRIB2 crudos del GFS
# "sdswrf" = flujo de radiación de onda corta descendente superficial (acumulado)

VAR_GFS_DSWRF = "dswrf1"
# Nombre después de la desacumulación (Módulo 2)
# El "1" indica que es el valor instantáneo de 1 hora

VAR_GFS_DSWRF_TEMPLATE = "dswrf1_{LT}"
# Cadena de plantilla — reemplazar {LT} con el launch time:
# "dswrf1_{LT}".format(LT="0100") → "dswrf1_0100"

VAR_GFS_CSI = "clearsky_index_GFS"
# Índice de Cielo Despejado = DSWRF / Clear_Sky_GHI
# Rango 0-1, usado como feature de entrada

VAR_GFS_KC = "clearness_index_GFS"
# Índice de Claridad = DSWRF / Extraterrestre_GHI
# Rango 0-1, usado como feature de entrada

VAR_SIATA_GHI = "GHI_clean"
# Radiación de SIATA en W/m² después de limpieza y recorte
# "clean" indica que ha pasado por control de calidad

VAR_SIATA_CSI = "clearsky_index_Siata"
# Índice de Cielo Despejado de SIATA = GHI_clean / Clear_Sky_GHI
# Rango 0-1 — ESTE ES EL OBJETIVO DEL MODELO

VAR_REF_CLEARSKY = "clear_sky_ghi"
# Nombre de la variable de salida del modelo Clear-Sky de Ineichen
# Usado para recorte, cálculo de CSI y descalado

VAR_REF_EXT = "extraterrestrial_ghi"
# Nombre de la variable de radiación extraterrestre
# Usado para el cálculo de KC y como techo físico absoluto

# ─────────────────────────────────────────────────────
# PARÁMETROS DE SECUENCIA
# ─────────────────────────────────────────────────────

SEQ_MODE = "sym18"
# Tipo de secuencia: "sym12", "sym18" o "sym24"
# Determina qué archivo .npz cargar para entrenamiento

K_LEFT = 18
# Horas de contexto pasado en la ventana de secuencia
# Con K_LEFT=18: la secuencia incluye 18 horas antes de la hora de pronóstico

K_RIGHT = 18
# Horas de contexto futuro en la ventana de secuencia
# Con K_RIGHT=18: la secuencia incluye 18 horas de pronósticos futuros del GFS
# Ventana total: K_LEFT + 1 + K_RIGHT = 37 pasos temporales

VAL_SPLIT = 0.15
# Fracción de secuencias para validación (15%)
# Usado para monitorear el entrenamiento y aplicar early stopping

TEST_SPLIT = 0.15
# Fracción de secuencias para prueba (15%)
# Nunca visto durante el entrenamiento — usado solo para evaluación final

# ─────────────────────────────────────────────────────
# HIPERPARÁMETROS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────

EPOCHS = 50
# Número máximo de pasadas completas por los datos de entrenamiento
# El entrenamiento real puede detenerse antes si se activa el early stopping
# Cambiar a 3 para pruebas de humo, 50 para entrenamiento completo

BATCH_SIZE = 128
# Número de secuencias procesadas juntas antes de actualizar pesos
# Mayor = gradientes más estables, actualizaciones menos frecuentes, más memoria
# Menor = gradientes más ruidosos, actualizaciones más frecuentes, menos memoria
# Actual: 128 (elegido empíricamente)

HIDDEN = 96
# Número de unidades ocultas LSTM por dirección
# La BiLSTM tiene 96 × 2 = 192 unidades en total
# Más unidades = más capacidad pero más parámetros y riesgo de sobreajuste
# Actual: 96 (~574K parámetros totales con 3 capas)

NUM_LAYERS = 3
# Número de capas LSTM apiladas
# Más capas = jerarquía temporal más profunda
# Optuna confirmó que 3 capas > 1 capa para este conjunto de datos
# Actual: 3

DROPOUT = 0.25
# Probabilidad de poner a cero cada neurona durante el entrenamiento
# Aplicado después de la BiLSTM, antes de la capa de salida
# Mayor = más regularización, menos sobreajuste
# Actual: 0.25 (25% de neuronas desactivadas por paso de entrenamiento)

LR = 0.001
# Tasa de aprendizaje inicial para el optimizador AdamW
# Controla el tamaño del paso para las actualizaciones de pesos
# Muy alta: entrenamiento inestable, sobrepasa los mínimos
# Muy baja: entrenamiento muy lento
# Actual: 0.001 (será reducida por el scheduler)

LR_FACTOR = 0.5
# Multiplicador para la reducción de la tasa de aprendizaje
# Cuando el scheduler se activa: nuevo_LR = LR × LR_FACTOR = LR × 0.5
# Actual: divide el LR a la mitad cuando la validación se estanca

LR_PATIENCE = 4
# Épocas sin mejora antes de reducir el LR
# Después de 4 épocas consecutivas sin mejora de validación,
# reducir LR por LR_FACTOR
# Actual: 4

MIN_LR = 1e-6
# Tasa de aprendizaje mínima (piso)
# El scheduler no reducirá el LR por debajo de este valor
# Actual: 0.000001

EARLY_STOP = 25
# Paciencia para el early stopping (épocas)
# Después de 25 épocas consecutivas sin mejora de validación,
# detener el entrenamiento y restaurar el mejor modelo
# Actual: 25

ACTIVATION = "sigmoid"
# Función de activación de salida
# "sigmoid": fuerza la salida a [0,1] — apropiada para el objetivo CSI
# "linear": salida sin límites — apropiada si el objetivo no está normalizado

DESCALER_METHOD = "physical"
# Método para convertir predicción CSI → W/m²
# "physical": multiplicar por el Clear-Sky GHI de Ineichen (físicamente correcto)
# "average": multiplicar por la radiación media histórica
# "z_score": transformación z-score inversa
# Actual: "physical" (más significativo físicamente)

USE_DAYMASK = True
# Si aplicar máscara nocturna a la pérdida de entrenamiento
# True: horas nocturnas (FLAG_START a FLAG_END) contribuyen 0 a la pérdida
# False: todas las horas contribuyen igualmente a la pérdida
# Nota: la implementación actual pone a cero las predicciones de noche
# pero aún las incluye en el denominador de la pérdida
# Una pérdida verdaderamente solo diurna excluiría la noche del denominador

L2_LAMBDA = 0.001
# Coeficiente de decaimiento de pesos para AdamW
# Añade penalización: pérdida_total = pérdida_mse + L2_LAMBDA × suma(pesos²)
# Desalienta pesos grandes → menos sobreajuste

MSE_BASELINE_R = 40761.472609
# Error Cuadrático Medio del GFS crudo en el conjunto de prueba, 
# solo horas diurnas
# Usado para calcular el SkillScore:
# SS = 1 - MSE_modelo / MSE_BASELINE_R
# IMPORTANTE: debe calcularse con verify_skillscore.py
# usando el MISMO conjunto de prueba que el modelo, solo horas diurnas
# Valor anterior (22322.349260) estaba diluido por ceros nocturnos
```

---

*THEORY.md — Empresa Emergente · Proyecto de Predicción de Radiación Solar*
*Autora: Isabela Velasco · Abril 2026*
*Este documento es tu referencia teórica completa para todo lo que hay en este proyecto.*
