"""
Backend del Sistema de Análisis Emocional Multimodal
====================================================

Módulo principal de backend que contiene toda la lógica de procesamiento
y análisis para el sistema de análisis emocional especializado en niños
con discapacidad.

Componentes principales:
- pipeline.py: Orquestador principal del sistema
- detector_emociones.py: Detección y análisis de emociones faciales
- analizador_audio.py: Análisis de audio y comunicación verbal
- api_recomendaciones.py: Sistema de recomendaciones personalizadas
- generador_informes.py: Generación de reportes y visualizaciones
- emotion_ensemble.py: Ensemble de modelos de detección emocional
- recomendaciones.py: Lógica de recomendaciones específicas

Autor: Sistema de Análisis Emocional
Versión: 2.0
"""

import os
import sys
import logging

# Configurar logging para el backend
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Agregar directorio actual al path para importaciones
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importaciones principales del backend
try:
    from .pipeline import PipelineAnalisisEmocional, ejecutar_pipeline
    from .detector_emociones import DetectorEmociones
    from .analizador_audio import AudioAnalyzer
    from .api_recomendaciones import ApiRecomendaciones
    from .generador_informes import GeneradorInformes
    from .emotion_ensemble import EmotionEnsemble
    from .recomendaciones import generar_recomendaciones
    
    __all__ = [
        'PipelineAnalisisEmocional',
        'ejecutar_pipeline',
        'DetectorEmociones', 
        'AudioAnalyzer',
        'ApiRecomendaciones',
        'GeneradorInformes',
        'EmotionEnsemble',
        'generar_recomendaciones'
    ]
    
    # Logger para el backend
    logger = logging.getLogger(__name__)
    logger.info("✅ Backend del sistema de análisis emocional inicializado correctamente")
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Error importando componentes del backend: {e}")
    
    # Definir componentes vacíos para evitar errores
    __all__ = []

def validar_backend():
    """
    Valida que todos los componentes del backend estén disponibles.
    
    Returns:
        dict: Estado de validación de componentes
    """
    componentes = {
        'PipelineAnalisisEmocional': False,
        'DetectorEmociones': False,
        'AudioAnalyzer': False,
        'ApiRecomendaciones': False,
        'GeneradorInformes': False,
        'EmotionEnsemble': False,
        'generar_recomendaciones': False
    }
    
    for componente in componentes.keys():
        try:
            globals()[componente]
            componentes[componente] = True
        except KeyError:
            pass
    
    return componentes

def obtener_info_backend():
    """
    Obtiene información detallada del backend.
    
    Returns:
        dict: Información del backend
    """
    return {
        "version": "2.0",
        "componentes_disponibles": validar_backend(),
        "directorio": current_dir,
        "descripcion": "Sistema de análisis emocional multimodal para niños con discapacidad"
    }
