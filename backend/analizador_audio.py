import os
from pydub import AudioSegment
import speech_recognition as sr

class AudioAnalyzer:
    def __init__(self, lang="es-ES"):
        self.recognizer = sr.Recognizer()
        self.lang = lang

    def extraer_audio(self, video_path, audio_salida="audio_extraido.wav"):
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_salida, format="wav")
        return audio_salida

    def transcribir_audio(self, audio_path):
        with sr.AudioFile(audio_path) as source:
            audio = self.recognizer.record(source)
        try:
            texto = self.recognizer.recognize_google(audio, language=self.lang)
            palabras = texto.split()
            intentos = len([w for w in palabras if len(w) > 1])
            return {"transcription": texto, "palabras_detectadas": palabras, "intentos": intentos}
        except Exception as e:
            return {"error": str(e), "transcription": "", "palabras_detectadas": [], "intentos": 0}



