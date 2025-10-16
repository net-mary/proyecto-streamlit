#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test simple del pipeline SIN Streamlit
Ejecuta: python test_video_simple.py
"""

import sys
import os

# Agregar rutas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from backend.pipeline import PipelineAnalisisEmocional

def main():
    print("=" * 80)
    print("TEST DIRECTO DEL PIPELINE")
    print("=" * 80)
    
    # ¡¡¡ CAMBIA ESTA RUTA A UN VIDEO QUE TENGAS !!!
    video_path = "C:/Users/User/Videos/sample.mp4"  # ← REEMPLAZA CON TU VIDEO
    
    print(f"\nBuscando video en: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"\n❌ ERROR: Video NO encontrado")
        print(f"\nBúscate un video en tu PC y actualiza la ruta en este script")
        print(f"Ejemplos de rutas válidas:")
        print(f"  - C:/Users/User/Desktop/video.mp4")
        print(f"  - C:/Users/User/Videos/mi_video.mp4")
        print(f"  - ./videos/mi_video.mp4 (si está en ./videos/)")
        return
    
    print(f"✅ Video encontrado")
    print(f"   Tamaño: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    print(f"\n" + "=" * 80)
    print("Inicializando pipeline...")
    print("=" * 80)
    
    try:
        pipeline = PipelineAnalisisEmocional(models_dir="./models")
        print("✅ Pipeline inicializado")
        
    except Exception as e:
        print(f"❌ Error inicializando pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n" + "=" * 80)
    print("Ejecutando análisis...")
    print("=" * 80 + "\n")
    
    try:
        resultado = pipeline.ejecutar_pipeline(
            video_path=video_path,
            lang="es-ES",
            datos_personales={
                "nombre": "Test",
                "diagnostico": "autismo"
            },
            configuracion_personalizada={
                "umbral_confianza": 0.05
            }
        )
        
    except Exception as e:
        print(f"\n❌ Error ejecutando pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Mostrar resultados
    print("\n" + "=" * 80)
    print("RESULTADOS")
    print("=" * 80)
    
    if "error" in resultado:
        print(f"\n❌ ERROR: {resultado['error']}")
    else:
        emociones = resultado.get('emociones', [])
        audio = resultado.get('audio', {})
        
        print(f"\n✅ ANÁLISIS COMPLETADO")
        print(f"\n📊 EMOCIONES:")
        print(f"   - Frames analizados: {len(emociones)}")
        
        if emociones:
            total_emos = sum(len(f.get('emociones', [])) for f in emociones)
            print(f"   - Total emociones: {total_emos}")
            
            # Mostrar distribución
            conteo = {}
            for frame in emociones:
                for emo in frame.get('emociones', []):
                    emocion = emo.get('emotion', 'Unknown')
                    conteo[emocion] = conteo.get(emocion, 0) + 1
            
            print(f"   - Distribución:")
            for emo, count in sorted(conteo.items(), key=lambda x: x[1], reverse=True):
                print(f"      • {emo}: {count}")
        else:
            print(f"   ❌ NO SE DETECTARON EMOCIONES")
        
        print(f"\n🎤 AUDIO:")
        print(f"   - Palabras: {audio.get('palabras_totales', 0)}")
        
        print(f"\n📋 INFORMACIÓN:")
        session_info = resultado.get('session_info', {})
        print(f"   - ID Sesión: {session_info.get('session_id')}")
        print(f"   - Tiempo: {session_info.get('tiempo_procesamiento')}")
        print(f"   - Etapas: {len(session_info.get('etapas_completadas', []))}/6")
        
        if session_info.get('errores'):
            print(f"\n⚠️ ERRORES:")
            for error in session_info['errores']:
                print(f"   - {error}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()