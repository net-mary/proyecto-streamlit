import requests
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

class ApiRecomendaciones:
    """
    Cliente API para obtener recomendaciones personalizadas usando Gemini AI.
    Maneja autenticación, reintentos, cache y análisis contextual avanzado.
    """
    
    def __init__(self, base_url: str = None, token: str = None, max_retries: int = 3):
        """
        Inicializa el cliente de API de recomendaciones.
        
        Args:
            base_url (str): URL base de la API (opcional, usa simulación si no se proporciona)
            token (str): Token de autenticación
            max_retries (int): Número máximo de reintentos
        """
        self.base_url = base_url
        self.token = token
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Configurar headers predeterminados
        if self.token:
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            })
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache simple para evitar llamadas repetitivas
        self.cache = {}
        self.cache_duration = 300  # 5 minutos
        
        # Plantillas de contexto por diagnóstico
        self.plantillas_contexto = {
            "autismo": {
                "areas_foco": ["comunicacion_social", "patrones_repetitivos", "sensibilidad_sensorial"],
                "estrategias_base": ["rutinas_estructuradas", "comunicacion_visual", "refuerzo_positivo"]
            },
            "tdah": {
                "areas_foco": ["atencion_sostenida", "hiperactividad", "impulsividad"],
                "estrategias_base": ["descansos_frecuentes", "tareas_cortas", "ambiente_estructurado"]
            },
            "sindrome_down": {
                "areas_foco": ["desarrollo_cognitivo", "habilidades_motoras", "comunicacion"],
                "estrategias_base": ["aprendizaje_visual", "repeticion_estructurada", "motivacion_social"]
            },
            "paralisis_cerebral": {
                "areas_foco": ["movilidad", "comunicacion", "independencia"],
                "estrategias_base": ["adaptaciones_fisicas", "tecnologia_asistiva", "terapia_ocupacional"]
            }
        }

    def _generar_cache_key(self, datos: Dict) -> str:
        """Genera clave única para cache basada en los datos de entrada."""
        key_data = {
            "diagnostico": datos.get("diagnostico", ""),
            "edad": datos.get("edad", ""),
            "contexto_hash": hash(str(datos.get("contexto_usuario", "")))
        }
        return f"rec_{hash(str(key_data))}"

    def _verificar_cache(self, cache_key: str) -> Optional[Dict]:
        """Verifica si existe una entrada válida en cache."""
        if cache_key in self.cache:
            entrada = self.cache[cache_key]
            if time.time() - entrada["timestamp"] < self.cache_duration:
                self.logger.info("Recomendaciones obtenidas desde cache")
                return entrada["data"]
            else:
                # Cache expirado
                del self.cache[cache_key]
        return None

    def _guardar_en_cache(self, cache_key: str, data: Dict):
        """Guarda datos en cache con timestamp."""
        self.cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }

    def obtener_recomendaciones(self, diagnostico: str, contexto_usuario: Dict, 
                              resultados_emociones: List = None, 
                              resultados_audio: Dict = None) -> Dict:
        """
        Obtiene recomendaciones personalizadas basadas en contexto completo.
        
        Args:
            diagnostico (str): Diagnóstico del niño
            contexto_usuario (Dict): Contexto del usuario y situación
            resultados_emociones (List): Resultados del análisis emocional
            resultados_audio (Dict): Resultados del análisis de audio
            
        Returns:
            Dict: Recomendaciones estructuradas con diferentes niveles
        """
        try:
            # Preparar datos completos para la API
            datos_completos = self._preparar_datos_contexto(
                diagnostico, contexto_usuario, resultados_emociones, resultados_audio
            )
            
            # Verificar cache
            cache_key = self._generar_cache_key(datos_completos)
            recomendaciones_cache = self._verificar_cache(cache_key)
            if recomendaciones_cache:
                return recomendaciones_cache
            
            # Si no hay API configurada, usar simulación inteligente
            if not self.base_url:
                recomendaciones = self._generar_recomendaciones_simuladas(datos_completos)
            else:
                # Llamar API real con reintentos
                recomendaciones = self._llamar_api_con_reintentos(datos_completos)
            
            # Enriquecer con recomendaciones específicas por contexto
            recomendaciones_enriquecidas = self._enriquecer_recomendaciones(
                recomendaciones, datos_completos
            )
            
            # Guardar en cache
            self._guardar_en_cache(cache_key, recomendaciones_enriquecidas)
            
            return recomendaciones_enriquecidas
            
        except Exception as e:
            self.logger.error(f"Error obteniendo recomendaciones: {e}")
            return self._generar_recomendaciones_emergencia(diagnostico)

    def _preparar_datos_contexto(self, diagnostico: str, contexto: Dict, 
                               emociones: List, audio: Dict) -> Dict:
        """Prepara datos contextuales completos para la API."""
        
        datos = {
            "diagnostico": diagnostico.lower() if diagnostico else "",
            "timestamp": datetime.now().isoformat(),
            "contexto_usuario": contexto,
            "resumen_emocional": self._resumir_emociones(emociones) if emociones else {},
            "resumen_audio": self._resumir_audio(audio) if audio else {}
        }
        
        # Agregar plantilla específica del diagnóstico
        diagnostico_norm = diagnostico.lower() if diagnostico else ""
        for key in self.plantillas_contexto:
            if key in diagnostico_norm:
                datos["plantilla_diagnostico"] = self.plantillas_contexto[key]
                break
        
        return datos

    def _resumir_emociones(self, emociones: List) -> Dict:
        """Crea resumen estadístico de las emociones detectadas."""
        if not emociones:
            return {}
        
        # Contar emociones por tipo
        conteo_emociones = {}
        total_frames = len(emociones)
        confianza_promedio = 0
        
        for frame_data in emociones:
            for emocion_data in frame_data.get("emociones", []):
                emocion = emocion_data.get("emotion", "Unknown")
                confianza = emocion_data.get("confidence", 0)
                
                if emocion not in conteo_emociones:
                    conteo_emociones[emocion] = {"count": 0, "confidence_sum": 0}
                
                conteo_emociones[emocion]["count"] += 1
                conteo_emociones[emocion]["confidence_sum"] += confianza
        
        # Calcular porcentajes y confianza promedio
        resumen = {}
        for emocion, datos in conteo_emociones.items():
            resumen[emocion] = {
                "porcentaje": round((datos["count"] / total_frames) * 100, 1),
                "confianza_promedio": round(datos["confidence_sum"] / datos["count"], 2)
            }
        
        # Identificar emoción predominante
        emocion_predominante = max(conteo_emociones, key=lambda x: conteo_emociones[x]["count"])
        
        return {
            "total_frames_analizados": total_frames,
            "distribucion_emociones": resumen,
            "emocion_predominante": emocion_predominante,
            "variabilidad_emocional": len(conteo_emociones)
        }

    def _resumir_audio(self, audio: Dict) -> Dict:
        """Crea resumen del análisis de audio."""
        if not audio:
            return {}
        
        return {
            "calidad_comunicacion": audio.get("calidad_comunicacion", "no_evaluado"),
            "palabras_totales": audio.get("palabras_totales", 0),
            "intentos_comunicacion": audio.get("intentos_comunicacion", 0),
            "palabras_infantiles": len(audio.get("palabras_infantiles", [])),
            "tiene_transcripcion": bool(audio.get("transcription", "")),
            "nivel_comunicativo": self._evaluar_nivel_comunicativo(audio)
        }

    def _evaluar_nivel_comunicativo(self, audio: Dict) -> str:
        """Evalúa el nivel comunicativo basado en métricas de audio."""
        intentos = audio.get("intentos_comunicacion", 0)
        palabras_infantiles = len(audio.get("palabras_infantiles", []))
        
        if intentos == 0:
            return "no_verbal"
        elif intentos < 3:
            return "pre_verbal"
        elif intentos < 8 and palabras_infantiles > 0:
            return "verbal_emergente"
        else:
            return "verbal_funcional"

    def _llamar_api_con_reintentos(self, datos: Dict) -> Dict:
        """Llama a la API real con sistema de reintentos."""
        url = f"{self.base_url}/recomendaciones"
        
        for intento in range(self.max_retries):
            try:
                response = self.session.post(
                    url, 
                    json=datos, 
                    timeout=30,
                    verify=True
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout en intento {intento + 1}")
                if intento == self.max_retries - 1:
                    raise Exception("API no responde - timeout múltiple")
                time.sleep(2 ** intento)  # Backoff exponencial
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error API en intento {intento + 1}: {e}")
                if intento == self.max_retries - 1:
                    raise Exception(f"Error de conexión API: {e}")
                time.sleep(2 ** intento)
                
        raise Exception("Máximo número de reintentos alcanzado")

    def _generar_recomendaciones_simuladas(self, datos: Dict) -> Dict:
        """
        Genera recomendaciones simuladas inteligentes basadas en IA para desarrollo/testing.
        """
        diagnostico = datos.get("diagnostico", "")
        contexto = datos.get("contexto_usuario", {})
        emociones = datos.get("resumen_emocional", {})
        audio = datos.get("resumen_audio", {})
        
        recomendaciones = {
            "recomendaciones_generales": [],
            "recomendaciones_especificas": [],
            "actividades_sugeridas": [],
            "objetivos_terapeuticos": [],
            "alertas": []
        }
        
        # Recomendaciones basadas en diagnóstico
        if "autismo" in diagnostico:
            recomendaciones["recomendaciones_generales"].extend([
                "🔄 Implementar rutinas visuales estructuradas con pictogramas",
                "🎯 Usar refuerzo positivo inmediato para comunicación espontánea",
                "🌈 Crear un entorno sensorial predecible y calmante"
            ])
            
        elif "tdah" in diagnostico:
            recomendaciones["recomendaciones_generales"].extend([
                "⏰ Dividir tareas en segmentos de 10-15 minutos máximo",
                "🏃 Incorporar descansos activos cada 20 minutos",
                "📍 Usar recordatorios visuales y auditivos para transiciones"
            ])
            
        # Recomendaciones basadas en emociones
        emocion_predominante = emociones.get("emocion_predominante", "")
        if emocion_predominante == "Sad":
            porcentaje_tristeza = emociones.get("distribucion_emociones", {}).get("Sad", {}).get("porcentaje", 0)
            if porcentaje_tristeza > 40:
                recomendaciones["alertas"].append("⚠️ Nivel alto de tristeza detectado - considerar evaluación emocional")
                recomendaciones["actividades_sugeridas"].extend([
                    "🎵 Sesiones de musicoterapia 3 veces por semana",
                    "🎨 Actividades de expresión artística libre",
                    "🤗 Tiempo de juego estructurado con adulto significativo"
                ])
                
        elif emocion_predominante == "Angry":
            recomendaciones["recomendaciones_especificas"].extend([
                "🧘 Enseñar técnicas de autorregulación apropiadas para la edad",
                "📚 Implementar historias sociales sobre manejo de frustración"
            ])
            
        # Recomendaciones basadas en comunicación
        nivel_comunicativo = audio.get("nivel_comunicativo", "")
        if nivel_comunicativo == "no_verbal":
            recomendaciones["objetivos_terapeuticos"].extend([
                "🗣️ Evaluar necesidad de sistema de comunicación aumentativa (CAA)",
                "👋 Fomentar gestos funcionales y señalamiento",
                "📱 Considerar aplicaciones de comunicación por intercambio de imágenes"
            ])
            
        elif nivel_comunicativo == "verbal_emergente":
            recomendaciones["actividades_sugeridas"].extend([
                "📖 Lectura interactiva diaria con preguntas simples",
                "🎭 Juegos de imitación vocal y gestual",
                "🔤 Expansión de vocabulario temático semanal"
            ])
            
        # Recomendaciones basadas en contexto del usuario
        rol_usuario = contexto.get("rol", "")
        if rol_usuario == "padre" or rol_usuario == "madre":
            recomendaciones["recomendaciones_generales"].append(
                "👨‍👩‍👧 Involucrar a toda la familia en estrategias consistentes"
            )
        elif rol_usuario == "terapeuta":
            recomendaciones["objetivos_terapeuticos"].append(
                "📊 Documentar progresos semanalmente con métricas objetivas"
            )
            
        return recomendaciones

    def _enriquecer_recomendaciones(self, recomendaciones: Dict, datos: Dict) -> Dict:
        """Enriquece las recomendaciones con información contextual adicional."""
        
        recomendaciones_enriquecidas = recomendaciones.copy()
        
        # Agregar metadata útil
        recomendaciones_enriquecidas["metadata"] = {
            "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "diagnostico_procesado": datos.get("diagnostico", ""),
            "nivel_confianza": self._calcular_confianza_recomendaciones(datos),
            "proxima_revision": "2 semanas"
        }
        
        # Agregar recursos adicionales
        recomendaciones_enriquecidas["recursos"] = self._sugerir_recursos(datos)
        
        # Priorizar recomendaciones
        recomendaciones_enriquecidas = self._priorizar_recomendaciones(recomendaciones_enriquecidas, datos)
        
        return recomendaciones_enriquecidas

    def _calcular_confianza_recomendaciones(self, datos: Dict) -> str:
        """Calcula nivel de confianza de las recomendaciones basado en datos disponibles."""
        factores_confianza = 0
        
        if datos.get("diagnostico"):
            factores_confianza += 1
        if datos.get("resumen_emocional", {}).get("total_frames_analizados", 0) > 10:
            factores_confianza += 1
        if datos.get("resumen_audio", {}).get("tiene_transcripcion"):
            factores_confianza += 1
        if datos.get("contexto_usuario", {}).get("edad"):
            factores_confianza += 1
            
        if factores_confianza >= 3:
            return "alta"
        elif factores_confianza >= 2:
            return "media"
        else:
            return "baja"

    def _sugerir_recursos(self, datos: Dict) -> List[str]:
        """Sugiere recursos adicionales basados en el contexto."""
        recursos = []
        diagnostico = datos.get("diagnostico", "")
        
        if "autismo" in diagnostico:
            recursos.extend([
                "📚 Guía de comunicación visual para familias",
                "🎓 Curso online: Estrategias para TEA en casa",
                "👥 Grupo de apoyo para padres de niños con autismo"
            ])
        elif "tdah" in diagnostico:
            recursos.extend([
                "⏱️ App de gestión de tiempo para niños",
                "🎯 Técnicas de mindfulness adaptadas para TDAH",
                "📋 Lista de verificación diaria para rutinas"
            ])
            
        return recursos

    def _priorizar_recomendaciones(self, recomendaciones: Dict, datos: Dict) -> Dict:
        """Prioriza recomendaciones basándose en urgencia y relevancia."""
        
        # Si hay alertas, moverlas al principio
        alertas = recomendaciones.get("alertas", [])
        if alertas:
            recomendaciones["prioridad_alta"] = alertas
            
        return recomendaciones

    def _generar_recomendaciones_emergencia(self, diagnostico: str) -> Dict:
        """Genera recomendaciones básicas en caso de falla de API."""
        return {
            "recomendaciones_generales": [
                "🔍 Observar patrones de comportamiento diariamente",
                "📝 Mantener registro de actividades y respuestas",
                "👨‍⚕️ Consultar con profesional especializado"
            ],
            "nota": "Recomendaciones básicas - API no disponible",
            "timestamp": datetime.now().isoformat()
        }

    def limpiar_cache(self):
        """Limpia el cache de recomendaciones."""
        self.cache.clear()
        self.logger.info("Cache de recomendaciones limpiado")

    def obtener_estadisticas_uso(self) -> Dict:
        """Obtiene estadísticas de uso de la API."""
        return {
            "entradas_cache": len(self.cache),
            "api_configurada": bool(self.base_url),
            "reintentos_maximos": self.max_retries
        }