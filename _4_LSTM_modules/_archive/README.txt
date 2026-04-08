CARPETA _archive — _4_LSTM_modules
====================================

Esta carpeta contiene módulos de la red neuronal y scripts de preparación de
datos que fueron reemplazados por versiones mejoradas durante el desarrollo
del modelo Bi-LSTM final. Se conservan por referencia histórica pero NO
forman parte del flujo activo del proyecto.

MOTIVO DEL ARCHIVO
------------------
El pipeline activo usa:
  - Arquitectura:   BiLSTMRegressor.py  (red bidireccional, salida Sigmoid)
  - Preparación:    multi_feature_target_converter.py + _01_prepare_sequence_shuffle.py
  - Pérdida:        Losses.py  (MixedHuberMSELoss, RampMSELoss, WeightedMSELoss)

Los scripts archivados corresponden a versiones anteriores de estos módulos:

- Feature_target_converter.py
    Primera versión del convertidor feature-target. Sustituida por
    multi_feature_target_converter.py, que soporta múltiples launch times
    simultáneos, normalización configurable por variable y cálculo de ángulo
    cenital solar como feature adicional.

- Feature_target_converter_normalizer.py
    Versión intermedia que añadía normalización básica al convertidor anterior.
    La lógica de normalización fue integrada directamente en
    multi_feature_target_converter.py con soporte para múltiples métodos
    (none, min_max, z_score, average, auto, physical).

- prepare_sequence.py
    Versión anterior del constructor de secuencias. Reemplazada por
    _01_prepare_sequence_shuffle.py, que soporta ventanas simétricas y
    causales, shuffle con semilla fija y persistencia de índices de test.

- first_LSTM_Model_sigmoid.py
    Arquitectura LSTM unidireccional con salida Sigmoid. Reemplazada por
    BiLSTMRegressor.py, que usa capas bidireccionales (Bi-LSTM) para
    aprovechar contexto futuro en la ventana simétrica, logrando mayor
    capacidad de representación con el mismo número de parámetros.

- LSTM_Model_linear.py
    Variante del modelo con capa de salida lineal (sin Sigmoid). Descartada
    porque la normalización de la variable objetivo (clear-sky index en [0,1])
    requiere acotar la salida, lo que la activación Sigmoid garantiza.

- LSTM_Model_sigmoid_dayflag.py
    Variante que incorporaba una máscara de día/noche (dayflag) para ponderar
    la pérdida durante horas nocturnas. Descartada porque la máscara se
    gestiona ahora a nivel de secuencia en el preparador de datos, y el modelo
    final no requiere esta lógica interna.

- Nash_sutcliffe_loss.py
    Implementación del coeficiente de Nash-Sutcliffe como función de pérdida.
    Descartada tras experimentación: el entrenamiento con NSE resultó
    inestable en comparación con la combinación MSE/Huber implementada
    en Losses.py, que converge de forma más robusta.
