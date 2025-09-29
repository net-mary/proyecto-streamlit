#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Verificación del Sistema
===================================

Verifica que todos los componentes del sistema estén correctamente configurados
y que no haya problemas de importación o inicialización.

Uso:
    python verificar_sistema.py
"""

import sys
import os
import traceback

def test_logger_imports():
    """Prueba las importaciones de logging."""
    print("\n" + "="*60)
    print("1. VERIFICANDO IMPORTACIONES DE LOGGING")
    print("="*60)
    
    try:
        import logging
        print("✅ Módulo logging disponible")
        return True
    except ImportError as e:
        print(f"❌ Error importando logging: {e}")
        return False

def test_backend_imports():
    """Prueba las importaciones del backend."""
    print("\n" + "="*60)
    print("2. VERIFICANDO IMPORTACIONES DEL BACKEND")
    print("="*60)
    
    # Agregar ruta del backend
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    
    componentes = {
        'DetectorEmociones': 'backend.detector_emociones',
        'AudioAnalyzer': 'backend.analizador_audio',
        'GeneradorInformes': 'backend.generador_informes',
        'EmotionEnsemble': 'backend.emotion_ensemble',
        'ApiRecomendaciones': 'backend.api_recomendaciones',
        'PipelineAnalisisEmocional': 'backend.pipeline',
    }
    
    resultados = {}
    
    for nombre, modulo in componentes.items():
        try:
            mod = __import__(modulo, fromlist=[nombre])
            clase = getattr(mod, nombre)
            print(f"✅ {nombre} importado correctamente")
            resultados[nombre] = True
        except Exception as e:
            print(f"❌ Error importando {nombre}: {e}")
            traceback.print_exc()
            resultados[nombre] = False
    
    return all(resultados.values())

def test_logger_initialization():
    """Prueba la inicialización de loggers en cada clase."""
    print("\n" + "="*60)
    print("3. VERIFICANDO INICIALIZACIÓN DE LOGGERS")
    print("="*60)
    
    # Agregar ruta del backend
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    
    tests = []
    
    # Test DetectorEmociones
    try:
        from backend.detector_emociones import DetectorEmociones
        detector = DetectorEmociones()
        if hasattr(detector, 'logger'):
            print("✅ DetectorEmociones: logger inicializado")
            tests.append(True)
        else:
            print("❌ DetectorEmociones: logger NO encontrado")
            tests.append(False)
    except Exception as e:
        print(f"❌ DetectorEmociones: Error - {e}")
        tests.append(False)
    
    # Test AudioAnalyzer
    try:
        from backend.analizador_audio import AudioAnalyzer
        analyzer = AudioAnalyzer()
        if hasattr(analyzer, 'logger'):
            print("✅ AudioAnalyzer: logger inicializado")
            tests.append(True)
        else:
            print("❌ AudioAnalyzer: logger NO encontrado")
            tests.append(False)
    except Exception as e:
        print(f"❌ AudioAnalyzer: Error - {e}")
        tests.append(False)
    
    # Test GeneradorInformes
    try:
        from backend.generador_informes import GeneradorInformes
        generador = GeneradorInformes()
        if hasattr(generador, 'logger'):
            print("✅ GeneradorInformes: logger inicializado")
            tests.append(True)
        else:
            print("❌ GeneradorInformes: logger NO encontrado")
            tests.append(False)
    except Exception as e:
        print(f"❌ GeneradorInformes: Error - {e}")
        tests.append(False)
    
    # Test EmotionEnsemble
    try:
        from backend.emotion_ensemble import EmotionEnsemble
        ensemble = EmotionEnsemble()
        if hasattr(ensemble, 'logger'):
            print("✅ EmotionEnsemble: logger inicializado")
            tests.append(True)
        else:
            print("❌ EmotionEnsemble: logger NO encontrado")
            tests.append(False)
    except Exception as e:
        print(f"❌ EmotionEnsemble: Error - {e}")
        tests.append(False)
    
    # Test ApiRecomendaciones
    try:
        from backend.api_recomendaciones import ApiRecomendaciones
        api = ApiRecomendaciones()
        if hasattr(api, 'logger'):
            print("✅ ApiRecomendaciones: logger inicializado")
            tests.append(True)
        else:
            print("❌ ApiRecomendaciones: logger NO encontrado")
            tests.append(False)
    except Exception as e:
        print(f"❌ ApiRecomendaciones: Error - {e}")
        tests.append(False)
    
    # Test PipelineAnalisisEmocional
    try:
        from backend.pipeline import PipelineAnalisisEmocional
        pipeline = PipelineAnalisisEmocional()
        if hasattr(pipeline, 'logger'):
            print("✅ PipelineAnalisisEmocional: logger inicializado")
            tests.append(True)
        else:
            print("❌ PipelineAnalisisEmocional: logger NO encontrado")
            tests.append(False)
    except Exception as e:
        print(f"❌ PipelineAnalisisEmocional: Error - {e}")
        traceback.print_exc()
        tests.append(False)
    
    return all(tests)

def test_dependencies():
    """Verifica que las dependencias estén instaladas."""
    print("\n" + "="*60)
    print("4. VERIFICANDO DEPENDENCIAS")
    print("="*60)
    
    dependencias = {
        'streamlit': 'Streamlit',
        'cv2': 'OpenCV',
        'tensorflow': 'TensorFlow',
        'plotly': 'Plotly',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'pydub': 'Pydub',
        'speech_recognition': 'SpeechRecognition'
    }
    
    results = []
    
    for module, nombre in dependencias.items():
        try:
            __import__(module)
            print(f"✅ {nombre} instalado")
            results.append(True)
        except ImportError:
            print(f"❌ {nombre} NO instalado")
            results.append(False)
    
    return all(results)

def test_directories():
    """Verifica que los directorios necesarios existan."""
    print("\n" + "="*60)
    print("5. VERIFICANDO ESTRUCTURA DE DIRECTORIOS")
    print("="*60)
    
    directorios = [
        'backend',
        'frontend',
        'frontend/components',
        'models',
        'resultados',
        'utils'
    ]
    
    results = []
    
    for directorio in directorios:
        if os.path.exists(directorio):
            print(f"✅ {directorio}/ existe")
            results.append(True)
        else:
            print(f"⚠️  {directorio}/ NO existe (se creará automáticamente)")
            results.append(True)  # No es crítico
    
    return all(results)

def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("INICIANDO VERIFICACIÓN DEL SISTEMA")
    print("="*60)
    
    tests = [
        ("Importaciones Logging", test_logger_imports),
        ("Importaciones Backend", test_backend_imports),
        ("Inicialización Loggers", test_logger_initialization),
        ("Dependencias", test_dependencies),
        ("Estructura Directorios", test_directories)
    ]
    
    resultados = []
    
    for nombre, test_func in tests:
        try:
            resultado = test_func()
            resultados.append((nombre, resultado))
        except Exception as e:
            print(f"\n❌ Error en {nombre}: {e}")
            traceback.print_exc()
            resultados.append((nombre, False))
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE VERIFICACIÓN")
    print("="*60)
    
    for nombre, resultado in resultados:
        status = "✅ PASS" if resultado else "❌ FAIL"
        print(f"{status} - {nombre}")
    
    total_tests = len(resultados)
    tests_passed = sum(1 for _, r in resultados if r)
    
    print(f"\nTests pasados: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 ¡SISTEMA VERIFICADO CORRECTAMENTE!")
        print("Puedes ejecutar la aplicación con: streamlit run frontend/interface.py")
        return True
    else:
        print("\n⚠️  SISTEMA CON PROBLEMAS")
        print("Revisa los errores anteriores y corrígelos antes de ejecutar.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)