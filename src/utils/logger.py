import logging
import sys
from pathlib import Path

def setup_logger(log_level: int = logging.INFO) -> None:
    """
    Configura el sistema de logging global para toda la aplicación.
    Debe llamarse una sola vez al arrancar el programa (ej. en main.py).
    """
    # Crear la carpeta de logs si no existe
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Formato limpio y profesional para los mensajes
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(log_format)

    # Configurar el root logger
    root_logger = logging.getLogger()
    # root_logger.setLevel(log_level)
    root_logger.setLevel(logging.DEBUG)

    # Evitar duplicidad de logs si se llama a la función varias veces
    if root_logger.handlers:
        root_logger.handlers.clear()

    # 1. Handler para la consola (salida estándar)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level) # Si es INFO, ignorará DEBUG en pantalla
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Handler para archivo (guarda todo lo que pase en un archivo rotativo o diario)
    file_handler = logging.FileHandler(log_dir / "bot_execution.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # En archivo guardamos todo el detalle
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)