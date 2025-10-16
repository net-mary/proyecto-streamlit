#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnóstico de Modelos de Detección Emocional
==============================================

Script para diagnosticar problemas con la detección de emociones.
"""

import os
import sys
import numpy as np

# Agregar rutas
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_emotion_ensemble():
    """Prueba el EmotionEnsemble."""
    print("\n" + "="*60)
    print("DIAGNÓSTICO DEL EMOTION ENSEMBLE")
    print("="*60 + "\n")
    
    try:
        from backend.emotion_ensemble import EmotionEnsemble
        
        print("1. Inicializando EmotionEnsemble...")
        ensemble = EmotionEnsemble(models_dir="./models")
        
        print(f"\n2. Estado del Ensemble:")
        print(f"   - Modelos cargados: {len(ensemble.models)}")
        print(f"   - Modo fallback: {ensemble.fallback_mode}")
        print(f"   - Input shapes: {ensemble.input_shapes}")
        
        if ensemble.models:
            print(f"\n3. Información de modelos:")
            for info in ensemble.model_info:
                print(f"   📦 {info['filename']}")
                print(f"      Descripción: {info['description']}")
                print(f"      Peso: {info['weight']}")
                print(f"      Input shape: {info['input_shape']}")
        else:
            print(f"\n⚠️ NO HAY MODELOS CARGADOS - Usando modo fallback")
        
        print(f"\n4. Probando predicción con imágenes de prueba VARIADAS...")
        
        # CORREGIDO: Crear imágenes DIFERENTES en cada iteración
        emociones_detectadas = {}
        
        for i in range(10):
            # Generar imagen NUEVA en cada iteración con valores aleatorios diferentes
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            emocion, confianza = ensemble.predict_emotion(test_image)
            emociones_detectadas[emocion] = emociones_detectadas.get(emocion, 0) + 1
            print(f"   Prueba {i+1}: {emocion} (confianza: {confianza:.3f})")
        
        print(f"\n5. Resumen de predicciones:")
        print(f"   Emociones únicas detectadas: {len(emociones_detectadas)}")
        for emocion, count in sorted(emociones_detectadas.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {emocion}: {count} veces")
        
        if len(emociones_detectadas) == 1:
            print(f"\n⚠️ ADVERTENCIA: Solo se detectó una emoción")
            print(f"   Esto puede indicar:")
            print(f"   - El modelo está en modo fallback")
            print(f"   - Las imágenes aleatorias son muy similares")
            print(f"   - El modelo necesita calibración")
        elif len(emociones_detectadas) >= 3:
            print(f"\n✅ El ensemble funciona correctamente (buena variedad de emociones)")
        else:
            print(f"\n⚠️ Variedad limitada de emociones detectadas")
            print(f"   El modelo funciona pero podría beneficiarse de más calibración")
        
        # Test adicional con imágenes específicas
        print(f"\n6. Test con patrones específicos:")
        
        # Imagen oscura (debería tender a Sad)
        dark_image = np.ones((100, 100, 3), dtype=np.uint8) * 50
        dark_emotion, dark_conf = ensemble.predict_emotion(dark_image)
        print(f"   Imagen oscura: {dark_emotion} ({dark_conf:.3f})")
        
        # Imagen brillante (debería tender a Happy/Surprise)
        bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        bright_emotion, bright_conf = ensemble.predict_emotion(bright_image)
        print(f"   Imagen brillante: {bright_emotion} ({bright_conf:.3f})")
        
        # Imagen neutral (gris medio)
        neutral_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        neutral_emotion, neutral_conf = ensemble.predict_emotion(neutral_image)
        print(f"   Imagen neutral: {neutral_emotion} ({neutral_conf:.3f})")
        
        return ensemble
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_detector_emociones():
    """Prueba el DetectorEmociones."""
    print("\n" + "="*60)
    print("DIAGNÓSTICO DEL DETECTOR EMOCIONES")
    print("="*60 + "\n")
    
    try:
        from backend.detector_emociones import DetectorEmociones
        import cv2
        
        print("1. Inicializando DetectorEmociones...")
        detector = DetectorEmociones()
        
        print(f"\n2. Verificando cascade facial:")
        if detector.face_cascade.empty():
            print(f"   ❌ Cascade facial NO cargado correctamente")
        else:
            print(f"   ✅ Cascade facial cargado")
        
        print(f"\n3. Verificando EmotionEnsemble integrado:")
        if hasattr(detector.emotion_model, 'models'):
            print(f"   Modelos en ensemble: {len(detector.emotion_model.models)}")
            print(f"   Modo fallback: {detector.emotion_model.fallback_mode}")
        
        print(f"\n4. Probando detección en imagen de prueba...")
        
        # Crear frame de prueba con "rostro" simulado
        test_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Intentar detectar rostros
        rostros = detector.detectar_rostros(test_frame)
        print(f"   Rostros detectados: {len(rostros)}")
        
        if len(rostros) > 0:
            for i, (x, y, w, h) in enumerate(rostros):
                print(f"   Rostro {i}: posición=({x},{y}), tamaño=({w}x{h})")
        else:
            print(f"   (Esto es normal, la imagen de prueba es aleatoria)")
        
        return detector
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def verificar_archivos_modelos():
    """Verifica que los archivos de modelos existan."""
    print("\n" + "="*60)
    print("VERIFICACIÓN DE ARCHIVOS DE MODELOS")
    print("="*60 + "\n")
    
    models_dir = "./models"
    
    # CORREGIDO: Usar los nombres reales de tus archivos
    archivos_esperados = {
        "FER_model.h5": "Modelo FER principal",
        "fer2013_mini_XCEPTION.99-0.65.hdf5": "Modelo mini-XCEPTION",
        "emotion_model.hdf5": "Modelo de emociones",  # CORREGIDO
        "haarcascade_frontalface_default.xml": "Cascade facial"
    }
    
    if not os.path.exists(models_dir):
        print(f"❌ Directorio {models_dir}/ NO EXISTE")
        print(f"   Creando directorio...")
        os.makedirs(models_dir, exist_ok=True)
        return False
    
    print(f"✅ Directorio {models_dir}/ existe\n")
    
    modelos_encontrados = 0
    
    for archivo, descripcion in archivos_esperados.items():
        ruta = os.path.join(models_dir, archivo)
        if os.path.exists(ruta):
            tamaño_mb = os.path.getsize(ruta) / (1024 * 1024)
            print(f"✅ {descripcion}")
            print(f"   Archivo: {archivo}")
            print(f"   Tamaño: {tamaño_mb:.2f} MB\n")
            modelos_encontrados += 1
        else:
            print(f"❌ {descripcion}")
            print(f"   Archivo faltante: {archivo}\n")
    
    print(f"Resumen: {modelos_encontrados}/{len(archivos_esperados)} archivos encontrados")
    
    return modelos_encontrados == len(archivos_esperados)

def main():
    """Función principal de diagnóstico."""
    print("\n" + "="*60)
    print("🔍 DIAGNÓSTICO COMPLETO DE DETECCIÓN EMOCIONAL")
    print("="*60)
    
    # 1. Verificar archivos
    archivos_ok = verificar_archivos_modelos()
    
    # 2. Probar EmotionEnsemble
    ensemble = test_emotion_ensemble()
    
    # 3. Probar DetectorEmociones
    detector = test_detector_emociones()
    
    # 4. Resumen final
    print("\n" + "="*60)
    print("RESUMEN DEL DIAGNÓSTICO")
    print("="*60 + "\n")
    
    if not archivos_ok:
        print("❌ PROBLEMA: Faltan archivos de modelos")
        print("   Solución: Ejecuta 'python setup_models.py' para descargar")
        print("   O coloca los modelos manualmente en ./models/")
        print()
    
    if ensemble:
        if ensemble.fallback_mode or len(ensemble.models) == 0:
            print("⚠️  PROBLEMA: EmotionEnsemble en modo FALLBACK")
            print("   El sistema funciona pero con precisión limitada")
            print("   Causa: Modelos FER no encontrados o no se cargaron")
            print("   Solución: Coloca los modelos .h5/.hdf5 en ./models/")
            print()
        else:
            print("✅ EmotionEnsemble funcionando correctamente")
            print(f"   Modelos activos: {len(ensemble.models)}")
            print()
    
    if detector:
        print("✅ DetectorEmociones inicializado correctamente")
        print()
    
    # Recomendaciones finales
    print("📋 RECOMENDACIONES:")
    print()
    
    if not archivos_ok or (ensemble and ensemble.fallback_mode):
        print("1. Descargar modelos FER:")
        print("   • Ejecuta: python setup_models.py")
        print("   • O descarga manualmente de:")
        print("     - https://github.com/oarriaga/face_classification")
        print("     - https://www.kaggle.com/datasets/msambare/fer2013")
        print()
        print("2. Coloca los archivos en ./models/:")
        print("   • FER_model.h5")
        print("   • fer2013_mini_XCEPTION.99-0.65.hdf5")
        print("   • emotion_model.hdf5")
        print()
        print("3. Reinicia la aplicación")
        print()
    else:
        print("✅ Todo está configurado correctamente")
        print("   Si sigues teniendo problemas:")
        print("   - Verifica que el video tenga buena iluminación")
        print("   - Asegúrate de que los rostros sean visibles")
        print("   - Reduce el umbral de confianza en configuración avanzada")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()