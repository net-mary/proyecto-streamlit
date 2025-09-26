from .detector_emociones import DetectorEmociones
from .analizador_audio import AudioAnalyzer
from .reporte import generar_histograma, generar_reporte
from .recomendaciones import generar_recomendaciones  # Si tienes este módulo

def ejecutar_pipeline(video_path, models_dir="./models", lang="es-ES"):
    detector = DetectorEmociones()
    emociones_resultados = detector.analizar_video(video_path, intervalo_ms=1000)
    audio_analyzer = AudioAnalyzer(lang)
    audio_path = audio_analyzer.extraer_audio(video_path)
    audio_resultados = audio_analyzer.transcribir_audio(audio_path)
    
    recomendaciones = generar_recomendaciones(emociones_resultados, audio_resultados) if 'generar_recomendaciones' in globals() else []

    hist_path = generar_histograma(emociones_resultados)
    rep_path = generar_reporte(emociones_resultados, audio_resultados)
    
    return {
        "emociones": emociones_resultados,
        "audio": audio_resultados,
        "recomendaciones": recomendaciones,
        "histograma": hist_path,
        "reporte": rep_path
    }

