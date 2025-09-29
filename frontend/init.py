"""
Sistema de Análisis Emocional Multimodal - Frontend
===================================================

Módulo principal del frontend para el sistema de análisis emocional
especializado en niños con discapacidad.

Componentes principales:
- interface.py: Interfaz principal de Streamlit
- timeline_emotions.py: Componente de timeline emocional interactivo

Autor: Sistema de Análisis Emocional
Versión: 2.0
"""

import os
import sys
import logging
from typing import Optional

# Configurar logging para el frontend
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def iniciar_aplicacion():
    """
    Función principal para iniciar la aplicación de análisis emocional.
    
    Esta función configura el entorno, valida dependencias y ejecuta la interfaz
    principal de Streamlit.
    """
    try:
        logger.info("🚀 Iniciando Sistema de Análisis Emocional Multimodal")
        
        # Validar entorno
        _validar_entorno()
        
        # Configurar rutas
        _configurar_rutas()
        
        # Importar y ejecutar interfaz principal
        from .interface import main as main_interface
        
        logger.info("✅ Sistema iniciado correctamente")
        main_interface()
        
    except ImportError as e:
        error_msg = f"Error importando módulos requeridos: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        _mostrar_ayuda_instalacion()
        
    except Exception as e:
        error_msg = f"Error crítico iniciando la aplicación: {e}"
        logger.error(error_msg)
        print(f"💥 {error_msg}")
        sys.exit(1)

def _validar_entorno():
    """Valida que el entorno tenga las dependencias necesarias."""
    dependencias_requeridas = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'opencv-python',
        'tensorflow',
        'matplotlib',
        'seaborn'
    ]
    
    dependencias_faltantes = []
    
    for dep in dependencias_requeridas:
        try:
            # Mapeo especial para nombres de módulos
            module_name = dep
            if dep == 'opencv-python':
                module_name = 'cv2'
            elif dep == 'pillow':
                module_name = 'PIL'
                
            __import__(module_name)
        except ImportError:
            dependencias_faltantes.append(dep)
    
    if dependencias_faltantes:
        logger.warning(f"⚠️ Dependencias faltantes: {dependencias_faltantes}")
        _mostrar_ayuda_instalacion(dependencias_faltantes)
    else:
        logger.info("✅ Todas las dependencias están instaladas")

def _configurar_rutas():
    """Configura las rutas necesarias para el sistema."""
    # Obtener directorio raíz del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Agregar rutas al sys.path si no están
    paths_to_add = [
        project_root,
        os.path.join(project_root, 'backend'),
        os.path.join(project_root, 'frontend'),
        os.path.join(project_root, 'frontend', 'components')
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)
    
    # Crear directorios necesarios
    directorios_necesarios = [
        os.path.join(project_root, 'models'),
        os.path.join(project_root, 'resultados'),
        os.path.join(project_root, 'temp_uploaded_videos'),
        os.path.join(project_root, 'logs')
    ]
    
    for directorio in directorios_necesarios:
        if not os.path.exists(directorio):
            os.makedirs(directorio, exist_ok=True)
            logger.info(f"📁 Directorio creado: {directorio}")

def _mostrar_ayuda_instalacion(dependencias_faltantes: Optional[list] = None):
    """
    Muestra ayuda para instalar dependencias faltantes.
    
    Args:
        dependencias_faltantes (list): Lista de dependencias que faltan
    """
    print("\n" + "="*60)
    print("🔧 CONFIGURACIÓN DEL ENTORNO")
    print("="*60)
    
    if dependencias_faltantes:
        print("❌ Dependencias faltantes detectadas:")
        for dep in dependencias_faltantes:
            print(f"   • {dep}")
        
        print("\n💡 Para instalar las dependencias faltantes, ejecuta:")
        deps_str = " ".join(dependencias_faltantes)
        print(f"   pip install {deps_str}")
    
    print("\n📦 Para instalar todas las dependencias del proyecto:")
    print("   pip install -r requirements.txt")
    
    print("\n🚀 Para ejecutar la aplicación:")
    print("   streamlit run frontend/interface.py")
    
    print("\n📚 Documentación adicional disponible en el README.md")
    print("="*60 + "\n")

def verificar_estado_sistema():
    """
    Verifica el estado actual del sistema y sus componentes.
    
    Returns:
        dict: Estado de los componentes del sistema
    """
    estado = {
        "frontend": {"status": "ok", "detalles": []},
        "backend": {"status": "ok", "detalles": []},
        "dependencias": {"status": "ok", "detalles": []},
        "directorios": {"status": "ok", "detalles": []}
    }
    
    try:
        # Verificar frontend
        try:
            from .interface import main
            from .components.timeline_emotions import TimelineEmotions
            estado["frontend"]["detalles"].append("✅ Interfaz principal disponible")
            estado["frontend"]["detalles"].append("✅ Componente timeline disponible")
        except ImportError as e:
            estado["frontend"]["status"] = "error"
            estado["frontend"]["detalles"].append(f"❌ Error frontend: {e}")
        
        # Verificar backend
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_path = os.path.join(os.path.dirname(current_dir), 'backend')
            sys.path.append(backend_path)
            
            from backend.pipeline import PipelineAnalisisEmocional
            from backend.detector_emociones import DetectorEmociones
            from backend.analizador_audio import AudioAnalyzer
            
            estado["backend"]["detalles"].append("✅ Pipeline principal disponible")
            estado["backend"]["detalles"].append("✅ Detector de emociones disponible")
            estado["backend"]["detalles"].append("✅ Analizador de audio disponible")
        except ImportError as e:
            estado["backend"]["status"] = "error"  
            estado["backend"]["detalles"].append(f"❌ Error backend: {e}")
        
        # Verificar dependencias críticas
        dependencias_criticas = {
            'streamlit': 'Interfaz web',
            'cv2': 'Procesamiento de video', 
            'tensorflow': 'Modelos de IA',
            'plotly': 'Visualizaciones interactivas',
            'pandas': 'Manejo de datos',
            'numpy': 'Computación numérica'
        }
        
        for dep, descripcion in dependencias_criticas.items():
            try:
                __import__(dep)
                estado["dependencias"]["detalles"].append(f"✅ {descripcion} ({dep})")
            except ImportError:
                estado["dependencias"]["status"] = "warning"
                estado["dependencias"]["detalles"].append(f"⚠️ {descripcion} ({dep}) - No disponible")
        
        # Verificar directorios
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        directorios_criticos = {
            'models': 'Modelos de IA',
            'resultados': 'Almacenamiento de resultados',
            'backend': 'Lógica de procesamiento',
            'frontend': 'Interfaz de usuario'
        }
        
        for dir_name, descripcion in directorios_criticos.items():
            dir_path = os.path.join(project_root, dir_name)
            if os.path.exists(dir_path):
                estado["directorios"]["detalles"].append(f"✅ {descripcion} ({dir_name}/)")
            else:
                estado["directorios"]["status"] = "warning"
                estado["directorios"]["detalles"].append(f"⚠️ {descripcion} ({dir_name}/) - No encontrado")
        
    except Exception as e:
        logger.error(f"Error verificando estado del sistema: {e}")
        estado["sistema"] = {"status": "error", "detalles": [f"Error general: {e}"]}
    
    return estado

def generar_reporte_estado():
    """Genera un reporte detallado del estado del sistema."""
    print("\n" + "="*60)
    print("🔍 REPORTE DE ESTADO DEL SISTEMA")
    print("="*60)
    
    estado = verificar_estado_sistema()
    
    # Mostrar estado de cada componente
    for componente, info in estado.items():
        status_icon = {
            "ok": "✅",
            "warning": "⚠️", 
            "error": "❌"
        }.get(info["status"], "❓")
        
        print(f"\n{status_icon} {componente.upper()}: {info['status'].upper()}")
        for detalle in info["detalles"]:
            print(f"   {detalle}")
    
    # Resumen general
    print(f"\n📊 RESUMEN:")
    total_componentes = len(estado)
    componentes_ok = sum(1 for info in estado.values() if info["status"] == "ok")
    componentes_warning = sum(1 for info in estado.values() if info["status"] == "warning")
    componentes_error = sum(1 for info in estado.values() if info["status"] == "error")
    
    print(f"   • Componentes OK: {componentes_ok}/{total_componentes}")
    if componentes_warning > 0:
        print(f"   • Componentes con advertencias: {componentes_warning}")
    if componentes_error > 0:
        print(f"   • Componentes con errores: {componentes_error}")
    
    # Recomendaciones
    if componentes_error > 0:
        print(f"\n🚨 ACCIÓN REQUERIDA:")
        print(f"   • Revisar errores críticos antes de ejecutar")
        print(f"   • Ejecutar: pip install -r requirements.txt")
    elif componentes_warning > 0:
        print(f"\n💡 RECOMENDACIONES:")
        print(f"   • Revisar advertencias para óptimo funcionamiento")
        print(f"   • Crear directorios faltantes si es necesario")
    else:
        print(f"\n🎉 SISTEMA LISTO:")
        print(f"   • Todos los componentes funcionan correctamente")
        print(f"   • Ejecutar: streamlit run frontend/interface.py")
    
    print("="*60 + "\n")
    
    return estado

def crear_estructura_proyecto():
    """Crea la estructura de directorios necesaria para el proyecto."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    estructura = {
        'models': 'Almacena los modelos de IA pre-entrenados',
        'resultados': 'Guarda resultados de análisis',
        'resultados/sesiones': 'Sesiones individuales de análisis',
        'resultados/cache': 'Cache temporal del sistema',
        'resultados/logs': 'Logs del sistema',
        'temp_uploaded_videos': 'Videos temporales subidos por usuarios',
        'fotogramas_detectados': 'Fotogramas con rostros detectados',
        'logs': 'Logs generales de la aplicación'
    }
    
    print("📁 Creando estructura de directorios...")
    
    directorios_creados = []
    directorios_existentes = []
    
    for directorio, descripcion in estructura.items():
        dir_path = os.path.join(project_root, directorio)
        
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                directorios_creados.append((directorio, descripcion))
                print(f"   ✅ Creado: {directorio}/ - {descripcion}")
            except Exception as e:
                print(f"   ❌ Error creando {directorio}/: {e}")
        else:
            directorios_existentes.append((directorio, descripcion))
    
    print(f"\n📊 Resumen:")
    print(f"   • Directorios creados: {len(directorios_creados)}")
    print(f"   • Directorios existentes: {len(directorios_existentes)}")
    
    if directorios_existentes:
        print(f"\n📂 Directorios que ya existían:")
        for directorio, descripcion in directorios_existentes:
            print(f"   • {directorio}/ - {descripcion}")
    
    # Crear archivos de configuración básicos
    _crear_archivos_config(project_root)

def _crear_archivos_config(project_root: str):
    """Crea archivos de configuración básicos si no existen."""
    
    # .gitignore
    gitignore_path = os.path.join(project_root, '.gitignore')
    if not os.path.exists(gitignore_path):
        gitignore_content = """# Archivos de Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Entornos virtuales
venv/
env/
ENV/

# Archivos del sistema
.DS_Store
Thumbs.db

# Archivos temporales del proyecto
temp_uploaded_videos/
fotogramas_detectados/
resultados/
logs/
*.log

# Modelos de IA (muy grandes)
models/*.h5
models/*.hdf5
models/*.pb

# IDE
.vscode/
.idea/
*.swp
*.swo
"""
        
        try:
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(gitignore_content)
            print(f"   ✅ Creado: .gitignore")
        except Exception as e:
            print(f"   ⚠️ Error creando .gitignore: {e}")
    
    # README básico si no existe
    readme_path = os.path.join(project_root, 'README.md')
    if not os.path.exists(readme_path):
        readme_content = """# Sistema de Análisis Emocional Multimodal

Sistema avanzado para análisis de emociones y comunicación en niños con discapacidad.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
streamlit run frontend/interface.py
```

## Estructura del Proyecto

- `backend/`: Lógica de procesamiento y análisis
- `frontend/`: Interfaz de usuario con Streamlit  
- `models/`: Modelos de IA pre-entrenados
- `resultados/`: Almacenamiento de resultados

## Funcionalidades

- ✅ Análisis emocional facial en tiempo real
- ✅ Transcripción y análisis de audio
- ✅ Recomendaciones personalizadas por diagnóstico
- ✅ Reportes completos y visualizaciones
- ✅ Interfaz web profesional y responsive

"""
        
        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            print(f"   ✅ Creado: README.md")
        except Exception as e:
            print(f"   ⚠️ Error creando README.md: {e}")

# Función principal de compatibilidad (mantiene nombre original)
def main_interface():
    """Función de compatibilidad que ejecuta la interfaz principal."""
    iniciar_aplicacion()

# Ejecutar si se llama directamente
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema de Análisis Emocional Multimodal")
    parser.add_argument("--setup", action="store_true", help="Crear estructura de proyecto")
    parser.add_argument("--check", action="store_true", help="Verificar estado del sistema")
    parser.add_argument("--run", action="store_true", help="Ejecutar aplicación")
    
    args = parser.parse_args()
    
    if args.setup:
        crear_estructura_proyecto()
    elif args.check:
        generar_reporte_estado()
    elif args.run:
        iniciar_aplicacion()
    else:
        # Si no se especifica argumento, mostrar ayuda
        print("🧠 Sistema de Análisis Emocional Multimodal")
        print("="*50)
        print("Opciones disponibles:")
        print("  --setup    Crear estructura de directorios")
        print("  --check    Verificar estado del sistema")
        print("  --run      Ejecutar la aplicación")
        print()
        print("Ejemplo: python -m frontend --run")
