"""
Componentes del Frontend
========================

Componentes reutilizables para la interfaz de usuario del sistema
de análisis emocional multimodal.

Componentes disponibles:
- TimelineEmotions: Visualización interactiva de timeline emocional
- create_simple_timeline: Timeline simplificado para otros componentes
"""

from .timeline_emotions import TimelineEmotions, create_simple_timeline

__all__ = [
    'TimelineEmotions',
    'create_simple_timeline'
]
