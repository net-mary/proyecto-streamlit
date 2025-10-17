# Análisis Emocional Multimodal

Sistema avanzado de análisis emocional multimodal para niños con discapacidad, que utiliza detección de emociones faciales y análisis de audio en tiempo real.

## Características

- Detección de 7 emociones (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- Análisis de audio y transcripción de palabras
- Recomendaciones personalizadas por diagnóstico
- Interfaz web con Streamlit
- Generación de reportes y visualizaciones
- Ensemble de 3 modelos de IA pre-entrenados

## Requisitos Previos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- FFmpeg
- Git (opcional)

## Instalación Rápida

### 1. Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd proyecto_streamlit
```

### 2. Crear Entorno Virtual

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Si no tienes requirements.txt:

```bash
pip install streamlit pandas numpy opencv-python tensorflow keras matplotlib seaborn plotly pydub SpeechRecognition pillow scikit-learn scipy requests
```

### 4. Instalar FFmpeg

**Windows:**
```bash
choco install ffmpeg
```

O descarga desde: https://www.gyan.dev/ffmpeg/builds/

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### 5. Descargar Modelos de IA

Coloca los siguientes archivos en la carpeta `./models/`:

| Archivo | Tamaño | Fuente |
|---------|--------|--------|
| `FER_model.h5` | 15.39 MB | GitHub: oarriaga/face_classification |
| `fer2013_mini_XCEPTION.99-0.65.hdf5` | 0.83 MB | Kaggle: msambare/fer2013 |
| `emotion_model.hdf5` | 0.83 MB | Kaggle: msambare/fer2013 |

## Estructura del Proyecto

```
proyecto_streamlit/
├── frontend/
│   ├── interface.py              # Interfaz Streamlit principal
│   └── components/               # Componentes reutilizables
├── backend/
│   ├── emotion_ensemble.py       # Ensemble de modelos
│   ├── detector_emociones.py     # Detector facial
│   ├── pipeline.py               # Pipeline de análisis
│   ├── analizador_audio.py       # Análisis de audio
│   ├── generador_informes.py     # Generación de reportes
│   └── recomendaciones.py        # Generador de recomendaciones
├── models/                        # Modelos pre-entrenados
├── resultados/                    # Reportes y visualizaciones
├── requirements.txt               # Dependencias
└── README.md
```

## Uso

### Ejecutar la Aplicación

```bash
streamlit run frontend/interface.py
```

Accede en: `http://localhost:8501`

### Flujo de Uso

1. Subir un video (MP4, AVI, MOV, MKV)
2. Configurar parámetros de análisis
3. Ingresar información del participante
4. Ejecutar análisis completo
5. Revisar resultados y descargar reportes

## Configuración Recomendada

| Diagnóstico | Intervalo (ms) | Umbral Confianza | Notas |
|-------------|---|---|---|
| Autismo/TEA | 2000 | 0.05 | Análisis más detallado |
| TDAH | 1500 | 0.05 | Ritmo moderado |
| Síndrome Down | 2500 | 0.05 | Mayor sensibilidad |
| Parálisis Cerebral | 3000 | 0.05 | Análisis más lento |
| General/Default | 1000 | 0.05 | Configuración estándar |

## Requisitos del Sistema

| Componente | Mínimo | Recomendado |
|-----------|---|---|
| RAM | 4 GB | 8 GB |
| Procesador | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| Almacenamiento | 2 GB libres | 5 GB libres |
| Internet | No requerido* | Recomendado para instalación inicial |

*El sistema funciona sin conexión después de la instalación

## Troubleshooting

### "No se pudo abrir el video"

- Verifica que el archivo es un video válido
- Intenta con otro formato (MP4 recomendado)
- Asegúrate de que el archivo no está dañado

### "Modelo no encontrado"

- Verifica que los archivos están en `./models/`
- Comprueba los nombres exactos de los archivos
- Descarga nuevamente si es necesario

### "Error de audio"

- Instala FFmpeg correctamente
- Verifica que el video tiene audio
- Intenta con otro archivo de video

### "Error de memoria"

- Aumenta el intervalo de análisis (1000ms o más)
- Cierra otras aplicaciones
- Reinicia Streamlit

## Archivos Generados

La aplicación crea automáticamente:

| Directorio | Contenido |
|-----------|----------|
| `./resultados/fotogramas_detectados/` | Rostros detectados |
| `./resultados/informes_*/` | Visualizaciones y gráficos |
| `./resultados/sesiones/` | Reportes completos |
| `./audio_extraido/` | Archivos de audio procesados |

## Desarrollo

### Ejecutar Diagnóstico

```bash
python diagnostico_modelos.py
```

Verifica que todos los componentes estén instalados correctamente.

### Agregar Nuevas Funcionalidades

- Backend: `backend/`
- Frontend: `frontend/`
- Componentes: `frontend/components/`

## API y Funciones Principales

### EmotionEnsemble

```python
from backend.emotion_ensemble import EmotionEnsemble

ensemble = EmotionEnsemble(models_dir="./models")
emotion, confidence = ensemble.predict_emotion(face_image)
distribution = ensemble.get_emotion_distribution(face_image)
```

### DetectorEmociones

```python
from backend.detector_emociones import DetectorEmociones

detector = DetectorEmociones()
resultados = detector.analizar_video(
    video_path="video.mp4",
    intervalo_ms=1000,
    guardar_frames=True
)
```

### Pipeline Completo

```python
from backend.pipeline import PipelineAnalisisEmocional

pipeline = PipelineAnalisisEmocional()
resultado = pipeline.ejecutar_pipeline(
    video_path="video.mp4",
    lang="es-ES",
    datos_personales={"diagnostico": "autismo"},
    configuracion_personalizada={"umbral_confianza": 0.05}
)
```

## Emociones Detectadas

| Emoción | Inglés | Código |
|---------|--------|--------|
| Enojado | Angry | 0 |
| Disgusto | Disgust | 1 |
| Miedo | Fear | 2 |
| Feliz | Happy | 3 |
| Triste | Sad | 4 |
| Sorpresa | Surprise | 5 |
| Neutral | Neutral | 6 |

## Contribución

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Reportar Problemas

Crea un issue incluendo:
- Sistema operativo
- Versión de Python
- Mensaje de error completo
- Pasos para reproducir el problema

## Licencia


## Autores



---

**Última actualización:** Octubre 2025
**Versión:** 1.0.0
