# Proyecto 4: Análisis de Movimientos Migratorios con Big Data

**Tags:** `Big Data`, `Apache Spark`, `PySpark`, `Spark SQL`, `Computación Distribuida`, `Python`

## Objetivo del Proyecto

Este proyecto demuestra la capacidad de analizar datasets a gran escala (Big Data) para extraer insights sobre patrones de migración globales. Dada la magnitud de los datos, que excede la capacidad de una sola máquina, se utilizó **Apache Spark** para procesar y analizar la información en un entorno de computación distribuida. El objetivo es identificar los principales corredores migratorios y tendencias a lo largo del tiempo.

## Metodología y Herramientas

El núcleo de este proyecto es el framework de Apache Spark, orquestado a través de su API de Python (PySpark):

1.  **Procesamiento de Datos Distribuido:**
    * Se utilizó un `SparkSession` para iniciar la aplicación.
    * Los datos se cargaron en un **DataFrame de Spark**, una estructura de datos distribuida e inmutable que permite el procesamiento en paralelo a través de un clúster.
2.  **Análisis y Transformaciones:**
    * Se realizaron transformaciones (`filter`, `groupBy`, `agg`) para limpiar y agregar los datos a escala.
    * Se utilizó **Spark SQL** para ejecutar consultas declarativas sobre los DataFrames, simplificando la lógica de análisis complejos y aprovechando el optimizador de consultas de Spark.
3.  **Extracción de Insights:** Se realizaron agregaciones para calcular el volumen de migrantes por país de origen y destino, identificando así los flujos más significativos.

## Resultados Clave

El análisis distribuido con Spark permitió procesar eficientemente el dataset masivo, revelando los **tres principales corredores migratorios** a nivel mundial. Un hallazgo significativo fue [Menciona un hallazgo, ej: "el crecimiento exponencial de la migración del Sudeste Asiático hacia Europa en la última década"]. Este tipo de análisis a gran escala es fundamental para que organizaciones internacionales y gobiernos puedan crear políticas informadas y asignar recursos de manera efectiva.

## Cómo Utilizar

1.  Clona este repositorio: `git clone https://github.com/patriciolavin/fundamentos_big_data.git`
2.  **Requisito:** Se necesita un entorno con Apache Spark y PySpark configurado.
3.  Ejecuta la Jupyter Notebook o el script `.py` a través de `spark-submit` para lanzar el trabajo de análisis.
