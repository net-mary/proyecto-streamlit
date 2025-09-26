def generar_recomendaciones(emociones, audio, gestos=None):
    recomendaciones = []

    # Regla simple basada en cantidad de emociones tristes consecutivas
    triste_frames = sum(
        1 for r in emociones for e in r['emociones'] if e['emotion'] == 'Sad' and e['confidence'] > 0.6
    )
    if triste_frames > 5:
        recomendaciones.append("Se detectó tristeza frecuente; se recomienda acompañamiento emocional cercano.")

    # Regla para intentos bajos en audio
    if audio.get('intentos', 0) < 3:
        recomendaciones.append("Baja frecuencia de intentos verbales, se sugiere estimular comunicación verbal.")

    # Ejemplo regla gestos (opcional)
    if gestos and gestos.get('autocalmantes', 0) > 5:
        recomendaciones.append("Gestos de auto-regulación elevados, puede indicar estrés o ansiedad.")

    return recomendaciones
