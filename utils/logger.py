"""
Sistema de Logging para el Análisis Emocional Multimodal
========================================================

Proporciona funciones de logging centralizadas para todo el sistema.
"""

import logging
import os
from datetime import datetime
from typing import Optional

# Configuración global de logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

def setup_logger(name: str, log_file: Optional[str] = None, level: int = LOG_LEVEL) -> logging.Logger:
    """
    Configura un logger con formato estándar.
    
    Args:
        name (str): Nombre del logger
        log_file (str): Archivo de log (opcional)
        level (int): Nivel de logging
        
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Evitar duplicar handlers
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler si se especifica
    if log_file:
        # Crear directorio de logs si no existe
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_error(message: str, logger_name: str = "sistema"):
    """
    Registra un error en el log.
    
    Args:
        message (str): Mensaje de error
        logger_name (str): Nombre del logger
    """
    logger = setup_logger(logger_name)
    logger.error(message)

def log_info(message: str, logger_name: str = "sistema"):
    """
    Registra información en el log.
    
    Args:
        message (str): Mensaje informativo
        logger_name (str): Nombre del logger
    """
    logger = setup_logger(logger_name)
    logger.info(message)

def log_warning(message: str, logger_name: str = "sistema"):
    """
    Registra una advertencia en el log.
    
    Args:
        message (str): Mensaje de advertencia
        logger_name (str): Nombre del logger
    """
    logger = setup_logger(logger_name)
    logger.warning(message)

def create_session_logger(session_id: str) -> logging.Logger:
    """
    Crea un logger específico para una sesión de análisis.
    
    Args:
        session_id (str): ID de la sesión
        
    Returns:
        logging.Logger: Logger de sesión
    """
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"session_{session_id}.log")
    return setup_logger(f"session_{session_id}", log_file)

# Configurar logging básico al importar el módulo
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler()
    ]
)

# Logger por defecto para el sistema
default_logger = setup_logger("sistema_emocional")

