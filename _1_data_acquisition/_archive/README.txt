CARPETA _archive — _1_data_acquisition
=======================================

Esta carpeta contiene scripts de descarga de datos que fueron descartados
durante el desarrollo del pipeline final. Se conservan por referencia histórica
pero NO forman parte del flujo de procesamiento activo del proyecto.

MOTIVO DEL ARCHIVO
------------------
Los scripts del directorio _1_data_acquisition activo usan las fuentes de datos
definitivas: NOAA S3 (acceso público sin autenticación) con los endpoints
definidos en S3_intervall_Data_Downloader_processpool_final.py y
S3_direct_MultGFS_DataDownloader_processpool_final.py.

Los scripts aquí archivados fueron prototipos o intentos con fuentes
alternativas que resultaron inadecuadas:

- untitled0.py
    Script de prueba sin nombre, sin funcionalidad definida.

- Noaa_tester.py / Noaa_tester1.py
    Pruebas tempranas de conexión a servidores NOAA. Sustituidos por los
    descargadores definitivos con multiprocesamiento (processpool).

- thredds_tester.py / coordinate_query_xarray2.py / data_downloader_xarray2.py
    Intentos de descarga via servidor THREDDS de NOAA. Se abandonó este
    enfoque porque los datos de dswrf con resolución de intervalo de 1 h
    no estaban disponibles en THREDDS con la cobertura temporal requerida.

- SW3_tester.py / sw3_variable_tester.py / SW3_GFSDataDownloader.py /
  SW3_DataDownloader_parallel.py / SW3_data_downloader_dswrf1-6_xarray_test.py /
  SW3_GFSDataDownloader_dynamicaly.py
    Scripts de descarga usando el servidor S3 de Southwest (SW3). Esta fuente
    fue sustituida por el servidor S3 público de NOAA, que ofrece mayor
    cobertura temporal y variables adicionales necesarias para el modelo.

- NOMADS_coordinate_query_xarray.py / NOMADS_data_downloader_xarray.py
    Pruebas con el servidor NOMADS de NOAA. Se descartó porque el acceso
    via S3 es más eficiente y no requiere autenticación.
