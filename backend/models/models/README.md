# Modelos de IA para Análisis Emocional

Esta carpeta debe contener los modelos pre-entrenados necesarios para el análisis emocional.

## Modelos Requeridos:

### 1. FER_model.h5
- **Descripción**: Modelo principal de reconocimiento emocional
- **Framework**: Keras/TensorFlow
- **Tamaño**: ~50 MB
- **Fuente**: Entrenado en dataset FER2013

### 2. fer2013_mini_XCEPTION.99-0.65.hdf5
- **Descripción**: Modelo mini-XCEPTION optimizado
- **Framework**: Keras/TensorFlow
- **Tamaño**: ~100 MB
- **Fuente**: Arquitectura Xception adaptada para FER2013

### 3. haarcascade_frontalface_default.xml
- **Descripción**: Clasificador Haar Cascade para detección facial
- **Framework**: OpenCV
- **Tamaño**: ~1 MB
- **Fuente**: OpenCV oficial

## Cómo obtener los modelos:

### Opción 1: Descargar Pre-entrenados
Busca modelos compatibles en:
- GitHub: https://github.com/oarriaga/face_classification
- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

### Opción 2: Modo Fallback
El sistema puede funcionar sin modelos usando análisis basado en reglas.
La precisión será menor pero permite probar el sistema.

## Verificación:
Ejecuta: `python setup_models.py` para verificar modelos instalados.
