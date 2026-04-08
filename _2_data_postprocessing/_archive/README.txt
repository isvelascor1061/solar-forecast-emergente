CARPETA _archive — _2_data_postprocessing
==========================================

Esta carpeta contiene scripts de postprocesamiento que fueron reemplazados
por versiones mejoradas durante el desarrollo del pipeline. Se conservan
por referencia histórica pero NO forman parte del flujo activo del proyecto.

MOTIVO DEL ARCHIVO
------------------
El pipeline de postprocesamiento activo está compuesto por:
  - 01_S3_GFSDataDownloader_1file_per_launchtime.py  (merge por launch time)
  - 02_S3_intervall_to_1hour_transformer.py           (desacumulación a 1 h)
  - S3_MultGFS_merger.py                              (merge de variables MultiGFS)

Los scripts archivados corresponden a enfoques anteriores:

- S3_RAD_1hour_interval_transformer.py
    Versión anterior del transformador de intervalos de radiación. Fue
    reemplazado por 02_S3_intervall_to_1hour_transformer.py, que implementa
    la lógica de desacumulación en bloques de 6 horas de forma más robusta
    y con nombres de variable configurables (VAR_IN / VAR_OUT).

- SW3_data_extractor_nc_file_xarray.py
    Script de extracción de datos desde el servidor SW3 (fuente alternativa
    descartada). Se abandonó junto con el conjunto de scripts SW3 cuando se
    migró definitivamente a NOAA S3 como fuente de datos principal.
