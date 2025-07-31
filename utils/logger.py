# utils/logger.py
import logging
import sys

class ColorFormatter(logging.Formatter):
    """
    Formatter que añade colores a los niveles de los logs para una mejor legibilidad en consola.
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # Formato: [NIVEL] [FECHA HORA] [MODULO:FUNCION:LINEA] - MENSAJE
    format_str = "[%(levelname)s] [%(asctime)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(log_file_path):
    """
    Configura y devuelve un logger raíz con handlers para consola y archivo.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Evitar duplicación de handlers si la función se llama varias veces
    if logger.hasHandlers():
        logger.handlers.clear()

    # Handler para la consola con colores
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(ColorFormatter())

    # Handler para el archivo de log (sin colores)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "[%(levelname)s] [%(asctime)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logger configurado. Los logs se guardarán en: {log_file_path}")