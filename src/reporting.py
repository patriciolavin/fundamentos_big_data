# src/reporting.py
import base64
import logging
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import pandas as pd

# ==============================================================================
# SECCIÓN DE CONTENIDO ESTÁTICO Y ANÁLISIS CUALITATIVO
# ==============================================================================

def get_static_content(quality_stats):
    """
    Retorna un diccionario con todo el contenido de texto estático y análisis
    para el reporte. La justificación del escalado ahora es dinámica.
    """
    # Justificación dinámica del escalado
    outlier_percentage = quality_stats.get('outliers_percentage', 0)
    if outlier_percentage > 3:
        scaling_justification = f"Dado que el porcentaje de outliers ({outlier_percentage:.2f}%) es superior al umbral del 3%, se optó por usar RobustScaler, que es menos sensible a valores atípicos."
    else:
        scaling_justification = f"El porcentaje de outliers ({outlier_percentage:.2f}%) es bajo, por lo que se podría haber usado StandardScaler, pero se mantuvo RobustScaler por robustez."

    content = {
        "resumen_ejecutivo_texto": """
            Este reporte presenta un análisis exhaustivo de los patrones de migración global utilizando un dataset del siglo XXI. 
            El objetivo fue construir un pipeline de datos robusto y reproducible con Apache Spark para procesar la información
            y desarrollar un modelo de Machine Learning capaz de predecir la probabilidad de una "migración exitosa". 
            La metodología abarca desde la limpieza y análisis exploratorio de datos (EDA) hasta el entrenamiento, evaluación
            y despliegue de un modelo de regresión logística, siguiendo las mejores prácticas de MLOps.
        """,
        "introduccion_texto": """
            El proyecto se enfoca en aplicar técnicas de Big Data para entender las complejas dinámicas de la migración.
            El trabajo se dividió en tres fases: 1) Análisis de Calidad de Datos y EDA; 2) Pipeline de Modelo y Entrenamiento; 
            y 3) Reporting Dinámico, culminando en la generación de este informe automatizado.
        """,
        "analisis_plot_razon": """
            El gráfico de barras muestra que la 'Razón Económica' es el principal motor de la migración en este
            dataset, seguida por 'Laboral' y 'Conflicto'. Esto sugiere que los análisis deben centrarse
            en los factores económicos que impulsan estos flujos poblacionales.
        """,
        "analisis_plot_correlacion": """
            El mapa de calor revela una fuerte correlación positiva entre el PIB de destino y el nivel educativo, 
            indicando que los migrantes tienden a dirigirse a países más ricos y con mayor capital humano.
        """,
        "texto_transformaciones": """
            Se realizaron transformaciones clave sobre los datos, incluyendo la codificación One-Hot para la variable
            categórica 'Razón' y el escalado de características numéricas con RobustScaler.
        """,
        "sql_query_origen": """
            SELECT Origen, COUNT(*) as Frecuencia
            FROM migraciones_temp_view
            GROUP BY Origen
            ORDER BY Frecuencia DESC
            LIMIT 5
        """,
        "texto_pipeline_ml": """
            Se implementó un `pyspark.ml.Pipeline` para encadenar de forma reproducible todas las etapas del preprocesamiento
            y el modelado. Este enfoque previene el data leakage y facilita el despliegue del modelo.
        """,
        "analisis_rendimiento_modelo": """
            **Análisis de Rendimiento:**
            El rendimiento del modelo debe ser evaluado críticamente. Las métricas mostradas arriba
            son las definitivas. Si los valores son perfectos (AUC/Accuracy = 1.0), es una señal inequívoca de 
            data leakage o un conjunto de prueba no representativo que requiere una revisión inmediata del pipeline.
            **Posibles Mejoras:**
            1.  **Validación Cruzada:** Implementar K-Fold Cross-Validation para obtener una estimación más robusta del rendimiento.
            2.  **Ingeniería de Características:** Crear nuevas variables, como la diferencia de PIB o de tasas de desempleo.
            3.  **Probar otros Modelos:** Algoritmos como Gradient-Boosted Trees o Random Forest podrían ofrecer una mayor precisión.
        """,
        "imputacion_texto": "No se detectaron valores nulos, por lo que no fue necesario aplicar una estrategia de imputación.",
        "scaling_justification_texto": scaling_justification,
        "conclusiones_texto": """
            Este proyecto demuestra la potencia de trabajar con un pipeline estructurado y automatizado en Spark. Permite no solo
            procesar y analizar datos a gran escala de manera eficiente, sino también garantizar la reproducibilidad y la fiabilidad
            de los resultados. Un punto bajo a considerar es la calidad de los datos de entrada; la presencia de duplicados y 
            outliers subraya la necesidad crítica de una fase de limpieza y validación en cualquier proyecto de ML.
        """
    }
    return content

# ==============================================================================
# FUNCIONES DE LECTURA Y CODIFICACIÓN
# ==============================================================================

def encode_image_to_base64(image_path: Path):
    """Codifica una imagen en formato base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.warning(f"No se encontró el archivo de imagen: {image_path}.")
        return ""
    except Exception as e:
        logging.error(f"Error al codificar la imagen {image_path}: {e}", exc_info=True)
        return ""

def read_file_content(file_path: Path, default_text="No disponible."):
    """Lee el contenido de un archivo de texto de forma segura."""
    try:
        return file_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        logging.warning(f"No se encontró el archivo de texto: {file_path}.")
        return default_text
    except Exception as e:
        logging.error(f"Error al leer el archivo {file_path}: {e}", exc_info=True)
        return "Error al leer el archivo."

# ==============================================================================
# FUNCIÓN PRINCIPAL DE GENERACIÓN DE REPORTE
# ==============================================================================

def generate_html_report(paths: dict, config: dict):
    """
    Genera un reporte HTML técnico y completo usando Jinja2 con todos los 
    artefactos y análisis del pipeline, controlado por el archivo de configuración.
    """
    # --- Programación Defensiva: Validar los argumentos de entrada ---
    assert isinstance(paths, dict), "El argumento 'paths' debe ser un diccionario."
    assert isinstance(config, dict), "El argumento 'config' debe ser un diccionario."
    assert 'reporting' in config, "La clave 'reporting' falta en el diccionario 'config'."

    try:
        reporting_config = config['reporting']
        logging.info("Iniciando la generación del reporte HTML técnico completo.")
        
        # Cargar la plantilla usando el nombre desde config
        env = Environment(loader=FileSystemLoader(paths["templates"]))
        template = env.get_template(reporting_config['template_name'])

        # 1. Recopilar datos de calidad
        quality_stats = {}
        try:
            total_rows = pd.read_csv(paths["raw_data"]).shape[0]
        except FileNotFoundError:
            total_rows = 0

        for file in ["nulos.csv", "duplicados.csv", "outliers.csv"]:
            key = file.split('.')[0]
            try:
                df_bad = pd.read_csv(paths["bad_data"] / file, comment='#')
                count = len(df_bad)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                count = 0
            quality_stats[key] = count
            quality_stats[f"{key}_percentage"] = (count / total_rows) * 100 if total_rows > 0 else 0

        # 2. Recopilar contenido estático y dinámico
        static_content = get_static_content(quality_stats)

        # 3. Recopilar artefactos del EDA
        eda_text_report = read_file_content(paths["reports"] / "eda_summary_report.txt")

        # 4. Recopilar gráficos del EDA codificados en base64
        plots_base64 = {}
        for plot_file in paths["plots"].glob("*.png"):
            key = f"plot_{plot_file.stem}"
            plots_base64[key] = encode_image_to_base64(plot_file)

        # 5. Recopilar salidas de consola de Spark (nombres desde config)
        console_outputs_cfg = reporting_config['console_outputs']
        spark_outputs = {
            "eda_spark_schema": read_file_content(paths["reports"] / console_outputs_cfg['spark_schema']),
            "eda_spark_show": read_file_content(paths["reports"] / console_outputs_cfg['spark_show']),
            "eda_spark_describe": read_file_content(paths["reports"] / console_outputs_cfg['spark_describe']),
            "sql_result_origen": read_file_content(paths["reports"] / console_outputs_cfg['sql_result'])
        }

        # 6. Recopilar métricas del modelo (nombre desde config)
        model_metrics_content = read_file_content(paths["reports"] / console_outputs_cfg['model_metrics'])

        # 7. Preparar el CONTEXTO completo para la plantilla (usando config)
        context = {
            "title": reporting_config['report_title'],
            "author": reporting_config['report_author'],
            "date": datetime.now().strftime('%Y-%m-%d'),
            "quality_stats": quality_stats,
            "plots": plots_base64,
            "model_metrics": model_metrics_content,
            "eda_text_report": eda_text_report,
            "ruta_parquet": str(paths["processed_data"] / config['data']['processed_output_file']),
            **static_content,
            **spark_outputs
        }

        # 8. Renderizar y guardar el HTML (nombre desde config)
        report_filename = f"{reporting_config['output_filename_prefix']}_{datetime.now().strftime('%Y-%m-%d')}.html"
        report_path = paths["reports"] / report_filename
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(template.render(context))
            
        logging.info(f"Reporte HTML técnico generado exitosamente en: {report_path}")

    except Exception as e:
        logging.error(f"Error crítico al generar el reporte HTML técnico: {e}", exc_info=True)
        raise