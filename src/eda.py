# src/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import io

# =================================================================
# <-- INICIO DE LA CORRECCIÓN
# Se actualiza la firma para aceptar el diccionario 'paths' (Solución Nivel 2).
# =================================================================
def perform_eda(df: pd.DataFrame, paths: dict):
# =================================================================
# <-- FIN DE LA CORRECCIÓN
# =================================================================
    """
    Realiza un Análisis Exploratorio de Datos (EDA) completo y guarda tanto los
    gráficos como un reporte de texto con todas las estadísticas requeridas.
    """
    try:
        logging.info("Iniciando Análisis Exploratorio de Datos (EDA) completo.")
        plots_path = paths["plots"]
        reports_path = paths["reports"]
        eda_report_path = reports_path / "eda_summary_report.txt"

        # =================================================================
        # <-- INICIO DE LA CORRECCIÓN (Solución Nivel 3)
        # Guarda de validación para DataFrames pequeños.
        # =================================================================
        if len(df) < 10:
            logging.warning(f"El DataFrame para el EDA es muy pequeño ({len(df)} filas). Algunas estadísticas o gráficos pueden no ser significativos o generar errores.")
        # =================================================================
        # <-- FIN DE LA CORRECCIÓN
        # =================================================================

        # --- Captura de Estadísticas en un String ---
        report_buffer = io.StringIO()
        
        # a) Información general del dataset
        report_buffer.write("--- a) Información General del Dataset ---\n\n")
        report_buffer.write(f"Dimensiones (Shape): {df.shape}\n")
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        report_buffer.write(f"Consumo de Memoria: {memory_usage_mb:.2f} MB\n\n")
        report_buffer.write("Tipos de Datos (df.info()):\n")
        df.info(buf=report_buffer)
        report_buffer.write("\n\n")

        # b) Estadísticas descriptivas adicionales
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        report_buffer.write("--- b) Estadísticas Descriptivas Adicionales ---\n\n")
        stats_df = pd.DataFrame({
            'Varianza': numeric_df.var(),
            'Asimetría (Skew)': numeric_df.skew(),
            'Curtosis (Kurt)': numeric_df.kurt()
        }).T
        report_buffer.write(stats_df.to_string())
        report_buffer.write("\n\n")
        
        # Guardar el reporte de texto
        with open(eda_report_path, 'w', encoding='utf-8') as f:
            f.write(report_buffer.getvalue())
        logging.info(f"Reporte de texto del EDA guardado en: {eda_report_path}")

        # --- Generación de Gráficos ---
        sns.set_theme(style="whitegrid")

        # Gráfico 1: Distribución para la variable categórica 'Razón'
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(y='Razón', data=df, hue='Razón', order=df['Razón'].value_counts().index, palette="viridis", legend=False)
        plt.title('Distribución de Razones de Migración', fontsize=16)
        plt.xlabel('Frecuencia', fontsize=12)
        plt.ylabel('Razón', fontsize=12)
        ax.bar_label(ax.containers[0])
        plt.tight_layout()
        plt.savefig(plots_path / "razon_migracion_distribucion.png")
        plt.close()

        # Gráfico 2: Heatmap de Correlación
        # Solo generar heatmap si hay más de una columna numérica para evitar errores
        if numeric_df.shape[1] > 1:
            corr_df = numeric_df.drop(columns=['ID', 'Año'], errors='ignore')
            plt.figure(figsize=(14, 10))
            sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
            plt.title('Mapa de Calor de Correlaciones', fontsize=16)
            plt.tight_layout()
            plt.savefig(plots_path / "correlation_heatmap.png")
            plt.close()
        else:
            logging.warning("No se generó el heatmap de correlación porque hay menos de 2 columnas numéricas.")


        # Gráfico 3 (Nuevo): Box plot para 'PIB_Destino'
        if 'PIB_Destino' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df['PIB_Destino'], color='skyblue')
            sns.stripplot(x=df['PIB_Destino'], color='black', alpha=0.3, jitter=0.2)
            plt.title('Distribución del PIB en Países de Destino (Box Plot)', fontsize=16)
            plt.xlabel('PIB per cápita (Destino)', fontsize=12)
            plt.tight_layout()
            plt.savefig(plots_path / "pib_destino_boxplot.png")
            plt.close()
            logging.info("Gráfico Box Plot generado.")

        # Gráfico 4 (Nuevo): Scatter plot
        if 'PIB_Origen' in df.columns and 'PIB_Destino' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='PIB_Origen', y='PIB_Destino', hue='Razón', palette="viridis", s=100)
            plt.title('Relación entre PIB de Origen y Destino', fontsize=16)
            plt.xlabel('PIB per cápita (Origen)', fontsize=12)
            plt.ylabel('PIB per cápita (Destino)', fontsize=12)
            plt.tight_layout()
            plt.savefig(plots_path / "pib_origen_destino_scatter.png")
            plt.close()
            logging.info("Gráfico Scatter Plot generado.")

        logging.info("EDA completado. Todos los artefactos han sido generados.")

    except Exception as e:
        logging.error(f"Ocurrió un error durante el EDA: {e}", exc_info=True)
        raise