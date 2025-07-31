# utils/config_loader.py
import yaml
from pathlib import Path
import logging

def load_config(config_path: Path):
    """
    Carga un archivo de configuración YAML de forma segura.

    Args:
        config_path (Path): La ruta al archivo config.yaml.

    Returns:
        dict: Un diccionario con la configuración cargada.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuración cargada exitosamente desde {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"El archivo de configuración no se encontró en: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error al parsear el archivo YAML: {e}")
        raise
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado al cargar la configuración: {e}")
        raise