# 🔧 Corrección Rápida del Error de Logger

## ❌ Problema
```
'GeneradorInformes' object has no attribute 'logger'
```

## ✅ Solución

El problema es que en **TODOS** los archivos del backend, el logger debe inicializarse en el `__init__` de cada clase.

### Archivos que debes revisar y corregir:

#### 1. `backend/generador_informes.py`
En la función `__init__`, asegúrate de tener:

```python
def __init__(self, carpeta_resultados: str = "./resultados"):
    self.carpeta_resultados = carpeta_resultados
    
    # ESTA LÍNEA DEBE ESTAR ANTES DE setup_directories()
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # Ahora sí llamar a setup
    self.setup_directories()
    
    # ... resto del código
```

#### 2. `backend/detector_emociones.py`
```python
def __init__(self, cascade_path: str = None, save_frames_path: str = "./fotogramas_detectados"):
    # Configurar logging PRIMERO
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # Luego el resto
    # ...
```

#### 3. `backend/analizador_audio.py`
```python
def __init__(self, lang="es-ES", audio_output_dir="./audio_extraido"):
    # Configurar logging PRIMERO
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # ...
```

#### 4. `backend/emotion_ensemble.py`
```python
def __init__(self, models_dir: str = "./models"):
    # Configurar logging PRIMERO
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # ...
```

#### 5. `backend/api_recomendaciones.py`
```python
def __init__(self, base_url: str = None, token: str = None, max_retries: int = 3):
    # Configurar logging PRIMERO
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # ...
```

#### 6. `backend/pipeline.py`
```python
def __init__(self, models_dir: str = "./models", resultados_dir: str = "./resultados"):
    # Configurar logging PRIMERO
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # ...
```

## 🚀 Pasos para Corregir:

1. **Abre cada archivo** mencionado arriba
2. **Busca la función `__init__`** de cada clase
3. **Asegúrate que estas 2 líneas estén AL PRINCIPIO del `__init__`:**
   ```python
   logging.basicConfig(level=logging.INFO)
   self.logger = logging.getLogger(self.__class__.__name__)
   ```
4. **IMPORTANTE:** Estas líneas deben estar ANTES de cualquier llamada a `self.logger`

## 🧪 Verificar que funciona:

Ejecuta el script de verificación:
```bash
python verificar_sistema.py
```

Deberías ver:
```
✅ DetectorEmociones: logger inicializado
✅ AudioAnalyzer: logger inicializado  
✅ GeneradorInformes: logger inicializado
✅ EmotionEnsemble: logger inicializado
✅ ApiRecomendaciones: logger inicializado
✅ PipelineAnalisisEmocional: logger inicializado
```

## 💡 Tip: Corrección Automática

Si quieres, puedo darte un script Python que corrija automáticamente todos los archivos. ¿Te gustaría eso?

## ⚠️ Nota Importante

El problema ocurre porque en algunos archivos, el código llama a métodos que usan `self.logger` ANTES de que el logger se haya inicializado. Por ejemplo:

```python
def __init__(self):
    self.setup_directories()  # ❌ Este método usa self.logger
    self.logger = logging.getLogger()  # ⚠️ Se inicializa DESPUÉS
```

Debe ser:

```python
def __init__(self):
    self.logger = logging.getLogger(self.__class__.__name__)  # ✅ PRIMERO
    self.setup_directories()  # ✅ Ahora sí puede usar self.logger
```