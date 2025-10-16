import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from backend.pipeline import PipelineAnalisisEmocional

# Test directo sin Streamlit
video_path = "./temp_uploaded_videos/WhatsApp Video 2025-08-20 at 14.23.53.mp4"

if not os.path.exists(video_path):
    print(f"ERROR: Video no existe en {video_path}")
    print(f"Archivos disponibles: {os.listdir('./temp_uploaded_videos') if os.path.exists('./temp_uploaded_videos') else 'Directorio no existe'}")
else:
    print(f"Ejecutando pipeline directamente con: {video_path}")
    
    pipeline = PipelineAnalisisEmocional()
    resultado = pipeline.ejecutar_pipeline(
        video_path=video_path,
        lang="es-ES",
        datos_personales={"diagnostico": "autismo"},
        configuracion_personalizada={"umbral_confianza": 0.05}
    )
    
    print("\n" + "="*60)
    print("RESULTADO:")
    print("="*60)
    
    if "error" in resultado:
        print(f"ERROR: {resultado['error']}")
    else:
        emociones = resultado.get('emociones', [])
        print(f"Emociones detectadas: {len(emociones)}")
        
        if emociones:
            print(f"Frame 0 emociones: {len(emociones[0].get('emociones', []))}")
        else:
            print("PROBLEMA: 0 emociones retornadas")