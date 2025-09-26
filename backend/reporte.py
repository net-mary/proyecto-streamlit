import matplotlib.pyplot as plt

def generar_histograma(emociones, filename="histograma.png"):
    emotion_counts = {}
    for r in emociones:
        for e in r['emociones']:
            emotion = e['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    plt.bar(emotion_counts.keys(), emotion_counts.values())
    plt.title("Distribución de emociones")
    plt.savefig(filename)
    plt.close()
    return filename

def generar_reporte(emociones, audio, output="reporte.txt"):
    with open(output, "w", encoding="utf-8") as f:
        f.write("REPORTE DE ANALISIS EMOCIONAL\n")
        f.write("=============================\n\n")
        f.write(f"Frames analizados: {len(emociones)}\n")
        for r in emociones:
            f.write(f"Frame {r['frame']} - Rostros: {r['num_faces']}\n")
            for e in r['emociones']:
                f.write(f"   Emoción: {e['emotion']} (Confianza: {e['confidence']:.2f})\n")
        f.write("\nTRANSCRIPCION AUDIO Y PALABRAS\n")
        f.write(f"Palabras detectadas: {audio.get('palabras_detectadas', [])}\n")
        f.write(f"Intentos de palabra: {audio.get('intentos', 0)}\n")
        f.write(f"Transcripción: {audio.get('transcription','')}\n")
    return output
