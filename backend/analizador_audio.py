import os
import logging
from datetime import datetime
from pydub import AudioSegment
import speech_recognition as sr
import numpy as np

class AudioAnalyzer:
    """
    Analizador de audio avanzado para detección de palabras y patrones de comunicación
    en niños con discapacidad. Incluye análisis de calidad de audio y métricas detalladas.
    """
    
    def __init__(self, lang="es-ES", audio_output_dir="./audio_extraido"):
        """
        Inicializa el analizador de audio.
        
        Args:
            lang (str): Idioma para reconocimiento de voz (default: es-ES)
            audio_output_dir (str): Directorio para guardar archivos de audio extraídos
        """
        self.recognizer = sr.Recognizer()
        self.lang = lang
        self.audio_output_dir = audio_output_dir
        
        # Crear directorio si no existe
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # Configurar parámetros del reconocedor para mejor precisión
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Palabras comunes para niños (para análisis contextual)
        self.palabras_infantiles = {
            'mama', 'papa', 'agua', 'quiero', 'no', 'si', 'hola', 'adios', 
            'dame', 'mio', 'mas', 'poco', 'mucho', 'grande', 'pequeño', 
            'bueno', 'malo', 'me gusta', 'no me gusta', 'ayuda', 'por favor'
        }

    def extraer_audio(self, video_path):
        """
        Extrae audio de un archivo de video con marca temporal.
        
        Args:
            video_path (str): Ruta del archivo de video
            
        Returns:
            dict: Información del audio extraído
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"audio_extraido_{timestamp}.wav"
            audio_output_path = os.path.join(self.audio_output_dir, audio_filename)
            
            # Extraer y procesar audio
            audio = AudioSegment.from_file(video_path)
            
            # Normalizar audio para mejor reconocimiento
            normalized_audio = audio.normalize()
            
            # Convertir a mono si es estéreo
            if normalized_audio.channels > 1:
                normalized_audio = normalized_audio.set_channels(1)
            
            # Asegurar frecuencia de muestreo adecuada
            normalized_audio = normalized_audio.set_frame_rate(16000)
            
            # Guardar audio procesado
            normalized_audio.export(audio_output_path, format="wav")
            
            # Calcular métricas del audio
            duracion = len(normalized_audio) / 1000.0  # en segundos
            volumen_promedio = normalized_audio.dBFS
            
            self.logger.info(f"Audio extraído: {audio_output_path}")
            
            return {
                "ruta_audio": audio_output_path,
                "duracion_segundos": duracion,
                "volumen_promedio_db": volumen_promedio,
                "frecuencia_muestreo": 16000,
                "canales": 1,
                "timestamp": timestamp,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error extrayendo audio: {str(e)}")
            return {
                "error": f"Error extrayendo audio: {str(e)}",
                "success": False
            }

    def analizar_segmentos_audio(self, audio_path, duracion_segmento=5):
        """
        Analiza el audio en segmentos para mejor detección de palabras.
        
        Args:
            audio_path (str): Ruta del archivo de audio
            duracion_segmento (int): Duración de cada segmento en segundos
            
        Returns:
            list: Lista de resultados por segmento
        """
        try:
            audio = AudioSegment.from_wav(audio_path)
            duracion_total = len(audio) / 1000.0
            
            resultados_segmentos = []
            
            for inicio in range(0, int(duracion_total), duracion_segmento):
                fin = min(inicio + duracion_segmento, duracion_total)
                
                # Extraer segmento
                segmento = audio[inicio*1000:fin*1000]
                
                # Guardar segmento temporal
                temp_path = f"temp_segmento_{inicio}_{fin}.wav"
                segmento.export(temp_path, format="wav")
                
                # Transcribir segmento
                resultado_segmento = self.transcribir_audio(temp_path)
                resultado_segmento["tiempo_inicio"] = inicio
                resultado_segmento["tiempo_fin"] = fin
                resultado_segmento["volumen_segmento"] = segmento.dBFS
                
                resultados_segmentos.append(resultado_segmento)
                
                # Limpiar archivo temporal
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            return resultados_segmentos
            
        except Exception as e:
            self.logger.error(f"Error analizando segmentos: {str(e)}")
            return []

    def transcribir_audio(self, audio_path):
        """
        Transcribe audio a texto con análisis detallado de palabras.
        
        Args:
            audio_path (str): Ruta del archivo de audio
            
        Returns:
            dict: Resultados de transcripción y análisis
        """
        try:
            with sr.AudioFile(audio_path) as source:
                # Ajustar para ruido ambiente
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            # Intentar transcripción con Google
            try:
                texto = self.recognizer.recognize_google(
                    audio_data, 
                    language=self.lang,
                    show_all=False
                )
                
                # Análisis detallado del texto
                palabras_detectadas = texto.lower().split()
                
                # Filtrar palabras muy cortas o ruido
                palabras_validas = [p for p in palabras_detectadas if len(p) > 1]
                
                # Contar intentos de comunicación (palabras significativas)
                intentos_comunicacion = len(palabras_validas)
                
                # Detectar palabras infantiles comunes
                palabras_infantiles_detectadas = [
                    p for p in palabras_validas 
                    if any(pi in p for pi in self.palabras_infantiles)
                ]
                
                # Calcular métricas de comunicación
                longitud_promedio_palabra = (
                    sum(len(p) for p in palabras_validas) / len(palabras_validas)
                    if palabras_validas else 0
                )
                
                # Determinar calidad de comunicación
                calidad_comunicacion = self._evaluar_calidad_comunicacion(
                    intentos_comunicacion, 
                    palabras_infantiles_detectadas,
                    longitud_promedio_palabra
                )
                
                return {
                    "transcription": texto,
                    "palabras_detectadas": palabras_validas,
                    "palabras_totales": len(palabras_validas),
                    "intentos_comunicacion": intentos_comunicacion,
                    "palabras_infantiles": palabras_infantiles_detectadas,
                    "longitud_promedio_palabra": round(longitud_promedio_palabra, 2),
                    "calidad_comunicacion": calidad_comunicacion,
                    "confidence": "alta",
                    "success": True
                }
                
            except sr.UnknownValueError:
                # Audio no reconocible
                return {
                    "transcription": "",
                    "palabras_detectadas": [],
                    "palabras_totales": 0,
                    "intentos_comunicacion": 0,
                    "palabras_infantiles": [],
                    "longitud_promedio_palabra": 0,
                    "calidad_comunicacion": "sin_deteccion",
                    "confidence": "nula",
                    "success": True,
                    "nota": "Audio no reconocible - puede indicar comunicación no verbal"
                }
                
            except sr.RequestError as e:
                raise Exception(f"Error del servicio de reconocimiento: {e}")
                
        except Exception as e:
            self.logger.error(f"Error transcribiendo audio: {str(e)}")
            return {
                "error": str(e),
                "transcription": "",
                "palabras_detectadas": [],
                "palabras_totales": 0,
                "intentos_comunicacion": 0,
                "palabras_infantiles": [],
                "longitud_promedio_palabra": 0,
                "calidad_comunicacion": "error",
                "confidence": "nula",
                "success": False
            }

    def _evaluar_calidad_comunicacion(self, intentos, palabras_infantiles, longitud_promedio):
        """
        Evalúa la calidad de comunicación basada en métricas.
        
        Args:
            intentos (int): Número de intentos de comunicación
            palabras_infantiles (list): Palabras infantiles detectadas
            longitud_promedio (float): Longitud promedio de palabras
            
        Returns:
            str: Nivel de calidad de comunicación
        """
        if intentos == 0:
            return "sin_comunicacion_verbal"
        elif intentos < 3:
            return "comunicacion_limitada"
        elif intentos < 8:
            if len(palabras_infantiles) > 2:
                return "comunicacion_basica_apropiada"
            else:
                return "comunicacion_basica"
        else:
            if longitud_promedio > 4 and len(palabras_infantiles) > 0:
                return "comunicacion_desarrollada"
            else:
                return "comunicacion_activa"

    def generar_reporte_audio(self, resultados_analisis, info_personal=None):
        """
        Genera un reporte detallado del análisis de audio.
        
        Args:
            resultados_analisis (dict): Resultados del análisis
            info_personal (dict): Información personal del niño
            
        Returns:
            dict: Reporte estructurado
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        reporte = {
            "timestamp": timestamp,
            "resumen_comunicacion": {
                "nivel": resultados_analisis.get("calidad_comunicacion", "no_evaluado"),
                "palabras_totales": resultados_analisis.get("palabras_totales", 0),
                "intentos_comunicacion": resultados_analisis.get("intentos_comunicacion", 0),
                "palabras_infantiles_detectadas": len(resultados_analisis.get("palabras_infantiles", []))
            },
            "transcripcion_completa": resultados_analisis.get("transcription", ""),
            "palabras_detectadas": resultados_analisis.get("palabras_detectadas", []),
            "recomendaciones": self._generar_recomendaciones_audio(resultados_analisis)
        }
        
        if info_personal:
            reporte["info_personal"] = info_personal
            
        return reporte

    def _generar_recomendaciones_audio(self, resultados):
        """
        Genera recomendaciones basadas en el análisis de audio.
        
        Args:
            resultados (dict): Resultados del análisis
            
        Returns:
            list: Lista de recomendaciones
        """
        recomendaciones = []
        
        calidad = resultados.get("calidad_comunicacion", "")
        intentos = resultados.get("intentos_comunicacion", 0)
        
        if calidad == "sin_comunicacion_verbal":
            recomendaciones.extend([
                "Considerar evaluación de comunicación alternativa y aumentativa (CAA)",
                "Fomentar comunicación gestual y señalamiento",
                "Implementar sistemas pictográficos de comunicación"
            ])
        elif calidad == "comunicacion_limitada":
            recomendaciones.extend([
                "Estimular comunicación verbal con actividades lúdicas",
                "Usar técnicas de modelado del lenguaje",
                "Crear rutinas de comunicación estructuradas"
            ])
        elif calidad in ["comunicacion_basica", "comunicacion_basica_apropiada"]:
            recomendaciones.extend([
                "Expandir vocabulario con actividades temáticas",
                "Fomentar construcción de frases simples",
                "Implementar lectura compartida diaria"
            ])
        
        return recomendaciones