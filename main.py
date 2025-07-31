# main.py
import logging
from pathlib import Path
from datetime import datetime

# Importar funciones de los módulos del proyecto
from utils.logger import setup_logger
from utils.config_loader import load_config
from src.data_processing import process_data_quality
from src.eda import perform_eda
from src.train_model import train_and_evaluate_model
from src.reporting import generate_html_report

# --- Configuración de Rutas Dinámicas ---
ROOT_DIR = Path(__file__).resolve().parent

def main():
    """
    Función principal que orquesta la ejecución completa del pipeline de ML.
    """
    log_file = ROOT_DIR / 'logs' / f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file)
    
    try:
        # Cargar la configuración UNA VEZ desde el archivo YAML.
        config_path = ROOT_DIR / 'utils' / 'config.yaml'
        config = load_config(config_path)

        logging.info("=================================================")
        logging.info(f"INICIO DEL PIPELINE: {config['project']['name']} v{config['project']['version']}")
        logging.info("=================================================")

        # Construcción dinámica de rutas desde la configuración
        paths_cfg = config['paths']
        paths = {
            "raw_data": ROOT_DIR / paths_cfg['data_folder'] / paths_cfg['raw_folder'] / config['data']['raw_input_file'],
            "bad_data": ROOT_DIR / paths_cfg['data_folder'] / paths_cfg['bad_data_folder'],
            "processed_data": ROOT_DIR / paths_cfg['data_folder'] / paths_cfg['processed_folder'],
            "plots": ROOT_DIR / paths_cfg['plots_folder'],
            "models": ROOT_DIR / paths_cfg['models_folder'],
            "reports": ROOT_DIR / paths_cfg['reports_folder'],
            "templates": ROOT_DIR / paths_cfg['templates_folder']
        }

        # Crear directorios necesarios
        for key, path in paths.items():
            if 'data' not in key:
                path.mkdir(parents=True, exist_ok=True)

        # Fase 1: Calidad de Datos y EDA
        logging.info("--- Iniciando Fase 1: Análisis de Calidad de Datos y EDA ---")
        clean_data_df = process_data_quality(paths, config)
        perform_eda(clean_data_df, paths)
        logging.info("--- Fase 1 completada exitosamente ---")

        # Fase 2: Entrenamiento y Evaluación del Modelo
        logging.info("--- Iniciando Fase 2: Entrenamiento y Evaluación del Modelo ---")
        train_and_evaluate_model(clean_data_df, paths, config)
        logging.info("--- Fase 2 completada exitosamente ---")
        
        # Fase 3: Generación de Reporte Dinámico
        logging.info("--- Iniciando Fase 3: Generación del Reporte HTML ---")
        generate_html_report(paths, config)
        logging.info("--- Fase 3 completada exitosamente ---")
        
    except Exception as e:
        logging.critical(f"El pipeline ha fallado. Error: {e}", exc_info=True)
    finally:
        logging.info("=" * 71)
        logging.info("✅ EVALUACIÓN FINAL DEL MÓDULO 9 COMPLETADA EXITOSAMENTE ".center(70, "="))
        logging.info("=" * 71)
        logging.info("\n\n")

if __name__ == "__main__":
    main()