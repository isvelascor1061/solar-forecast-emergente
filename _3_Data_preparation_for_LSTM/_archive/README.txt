CARPETA _archive — _3_Data_preparation_for_LSTM
================================================

Esta carpeta contiene scripts de preparación de datos que fueron descartados
o sustituidos durante el desarrollo del pipeline. Se conservan por referencia
histórica pero NO forman parte del flujo activo del proyecto.

MOTIVO DEL ARCHIVO
------------------
El pipeline de preparación activo incluye los scripts en:
  - _Format_and_Adjustment_Code/_02_Intervall_format_and_adjustment/
  - _Format_and_Adjustment_Code/_03_Direct_format_and_adjusment/

Los scripts archivados corresponden a enfoques anteriores o utilitarios
de análisis puntuales:

- Test_day_year_clip.py
    Script experimental para probar el recorte de radiación GFS usando
    el día del año como referencia. Fue reemplazado por la clase
    NetCDFDataCleaner en 02_Clip_GFS_day_of_year.py, que implementa una
    lógica más completa: recorte por radiación de referencia (extraterrestre
    o clear-sky), reemplazo de excedencias con media interanual y jitter.

- SIATA_monthly_plotter.py
    Script de visualización mensual de los datos SIATA (irradiancia global
    horizontal observada). Utilizado durante la fase exploratoria del proyecto
    para verificar la calidad de los datos, pero no es parte del pipeline
    de preparación de secuencias para el modelo.
