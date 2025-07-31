# src/data_processing.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path

def save_bad_data(df, path, filename, reason):
    """
    Guarda los datos de mala calidad en un CSV con metadatos.
    """
    try:
        count = len(df)
        if count > 0:
            metadata = f"# Archivo: {filename}\n# Razón de segregación: {reason}\n# Cantidad de registros: {count}\n"
            with open(path / filename, 'w', encoding='utf-8') as f:
                f.write(metadata)
            df.to_csv(path / filename, mode='a', index=False)
            logging.warning(f"Se encontraron y guardaron {count} registros en {filename}")
        else:
            (path / filename).touch()
            logging.info(f"No se encontraron registros para '{reason}'. Se creó archivo de constancia.")
    except Exception as e:
        logging.error(f"Error al guardar datos de mala calidad en {filename}: {e}", exc_info=True)
        raise

# =================================================================
# <-- INICIO DE LA CORRECCIÓN (Solución Nivel 3)
# Se actualiza la firma y el cuerpo de la función para usar los diccionarios 'paths' y 'config'.
# Se añaden aserciones y type hints para mayor robustez.
# =================================================================
def process_data_quality(paths: dict, config: dict) -> pd.DataFrame:
    """
    Carga los datos crudos, identifica duplicados y outliers, los guarda en
    la carpeta bad_data y devuelve un DataFrame limpio.
    """
    # --- Programación Defensiva: Validar los argumentos de entrada ---
    assert isinstance(paths, dict), "El argumento 'paths' debe ser un diccionario."
    assert 'raw_data' in paths, "La clave 'raw_data' falta en el diccionario 'paths'."
    assert 'bad_data' in paths, "La clave 'bad_data' falta en el diccionario 'paths'."

    raw_data_path = paths['raw_data']
    bad_data_path = paths['bad_data']

    try:
        logging.info(f"Cargando datos crudos desde: {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        original_rows = len(df)
        logging.info(f"Dataset cargado con {original_rows} filas y {len(df.columns)} columnas.")

        # 1. Detección y manejo de Nulos
        nulos = df[df.isnull().any(axis=1)]
        save_bad_data(nulos, bad_data_path, "nulos.csv", "Contiene valores nulos")

        # 2. Detección y manejo de Duplicados
        duplicados = df[df.duplicated(keep=False)]
        save_bad_data(duplicados, bad_data_path, "duplicados.csv", "Registros duplicados exactos")
        
        df_no_duplicates = df.drop_duplicates().copy()

        # 3. Detección de Outliers (Método IQR)
        numeric_cols = df_no_duplicates.select_dtypes(include=np.number).columns
        Q1 = df_no_duplicates[numeric_cols].quantile(0.25)
        Q3 = df_no_duplicates[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_condition = ((df_no_duplicates[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_no_duplicates[numeric_cols] > (Q3 - 1.5 * IQR))).any(axis=1)
        outliers = df_no_duplicates[outlier_condition]
        save_bad_data(outliers, bad_data_path, "outliers.csv", "Valores atípicos detectados por método IQR")

        # 4. Devolver el DataFrame limpio (solo sin duplicados) para las siguientes fases
        clean_df = df.drop_duplicates().reset_index(drop=True)
        logging.info(f"Procesamiento de calidad finalizado. Se retorna un DataFrame con {len(clean_df)} filas.")
        
        return clean_df

    except FileNotFoundError:
        logging.error(f"El archivo de datos no se encontró en la ruta: {raw_data_path}")
        raise
    except Exception as e:
        logging.error(f"Error durante el procesamiento de calidad de datos: {e}", exc_info=True)
        raise
# =================================================================
# <-- FIN DE LA CORRECCIÓN
# =================================================================