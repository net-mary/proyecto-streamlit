import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from .detector_emociones import DetectorEmociones
from .analizador_audio import AudioAnalyzer
from .generador_informes import GeneradorInformes
from .api_recomendaciones import ApiRecomendaciones
from .recomendaciones import generar_recomendaciones
import numpy as np # Necesario para las funciones de cálculo

class PipelineAnalisisEmocional:
    """
    Pipeline principal para análisis emocional multimodal.
    Orchestraa todos los componentes del sistema de análisis.
    """
    
    def __init__(self, models_dir: str = "./models", resultados_dir: str = "./resultados"):
        """
        Inicializa el pipeline de análisis emocional.
        
        Args:
            models_dir (str): Directorio de modelos
            resultados_dir (str): Directorio de resultados
        """
        # [CORRECCIÓN/OPTIMIZACIÓN]: Se elimina logging.basicConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configurar directorios
        self.models_dir = models_dir
        self.resultados_dir = resultados_dir
        self.ensure_directories()
        
        # Inicializar componentes
        try:
            self.detector_emociones = DetectorEmociones(save_frames_path=os.path.join(resultados_dir, "fotogramas_detectados"))
            # Esta línea ya no falla si GeneradorInformes.__init__ está corregido.
            self.generador_informes = GeneradorInformes(carpeta_resultados=resultados_dir)
            self.api_recomendaciones = ApiRecomendaciones()  # Simulación por defecto
            self.logger.info("✓ Pipeline inicializado correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando pipeline: {e}")
            raise
        
        # Métricas del pipeline
        self.pipeline_metrics = {
            'sesiones_procesadas': 0,
            'videos_analizados': 0,
            'errores_totales': 0,
            'tiempo_total_procesamiento': 0,
            'inicio_pipeline': datetime.now()
        }
        
        # Cache de configuraciones por diagnóstico
        self.configuraciones_diagnostico = self._cargar_configuraciones_diagnostico()

    def ensure_directories(self):
        """Asegura que existan todos los directorios necesarios."""
        directories = [
            self.models_dir,
            self.resultados_dir,
            os.path.join(self.resultados_dir, "sesiones"),
            os.path.join(self.resultados_dir, "cache"),
            os.path.join(self.resultados_dir, "logs")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _cargar_configuraciones_diagnostico(self) -> Dict:
        """Carga configuraciones específicas por diagnóstico."""
        return {
            "autismo": {
                "intervalo_analisis_ms": 2000, 
                "umbral_confianza": 0.6,
                "priorizar_emociones": ["Neutral", "Happy", "Fear"],
                "alertas_especiales": ["Angry", "Sad"]
            },
            "tdah": {
                "intervalo_analisis_ms": 1500,
                "umbral_confianza": 0.5,
                "priorizar_emociones": ["Happy", "Surprise", "Neutral"],
                "alertas_especiales": ["Angry"]
            },
            "sindrome_down": {
                "intervalo_analisis_ms": 2500,
                "umbral_confianza": 0.7,
                "priorizar_emociones": ["Happy", "Surprise"],
                "alertas_especiales": ["Sad", "Fear"]
            },
            "paralisis_cerebral": {
                "intervalo_analisis_ms": 3000,
                "umbral_confianza": 0.4, 
                "priorizar_emociones": ["Happy", "Neutral"],
                "alertas_especiales": ["Disgust", "Fear", "Sad"]
            },
            "default": {
                "intervalo_analisis_ms": 1000,
                "umbral_confianza": 0.5,
                "priorizar_emociones": [],
                "alertas_especiales": []
            }
        }

    def ejecutar_pipeline(self, video_path: str, lang: str = "es-ES", 
                          datos_personales: Optional[Dict] = None,
                          configuracion_personalizada: Optional[Dict] = None) -> Dict:
        """
        Ejecuta el pipeline completo de análisis emocional.
        """
        inicio_procesamiento = datetime.now()
        session_id = f"sesion_{inicio_procesamiento.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"🚀 Iniciando análisis para sesión: {session_id}")
            self.logger.info(f"📹 Video: {os.path.basename(video_path)}")
            
            # Validar archivo de video
            if not self._validar_video(video_path):
                raise ValueError(f"Archivo de video inválido: {video_path}")
            
            # Obtener configuración específica
            configuracion = self._obtener_configuracion(datos_personales, configuracion_personalizada)
            
            # Crear directorio de sesión
            session_dir = os.path.join(self.resultados_dir, "sesiones", session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Inicializar resultados
            resultados_completos = {
                "session_id": session_id,
                "timestamp_inicio": inicio_procesamiento.isoformat(),
                "video_analizado": os.path.basename(video_path),
                "configuracion_usada": configuracion,
                "datos_personales": datos_personales or {},
                "etapas_completadas": [],
                "errores": []
            }
            
            # ETAPA 1: Análisis de emociones faciales
            self.logger.info("📊 Etapa 1: Analizando emociones faciales...")
            try:
                emociones_resultados = self.detector_emociones.analizar_video(
                    video_path, 
                    intervalo_ms=configuracion["intervalo_analisis_ms"],
                    guardar_frames=True
                )
                
                # Filtrar resultados por confianza
                emociones_filtradas = self._filtrar_por_confianza(
                    emociones_resultados, 
                    configuracion["umbral_confianza"]
                )
                
                resultados_completos["emociones"] = emociones_filtradas
                resultados_completos["estadisticas_emociones"] = self._calcular_estadisticas_emociones(emociones_filtradas)
                resultados_completos["etapas_completadas"].append("analisis_emociones")
                
                self.logger.info(f"✓ Emociones analizadas: {len(emociones_resultados)} frames procesados")
                
            except Exception as e:
                error_msg = f"Error en análisis emocional: {str(e)}"
                self.logger.error(error_msg)
                resultados_completos["errores"].append(error_msg)
                resultados_completos["emociones"] = []
                resultados_completos["estadisticas_emociones"] = {}
            
            # ETAPA 2: Análisis de audio
            self.logger.info("🎤 Etapa 2: Analizando audio y comunicación...")
            try:
                audio_analyzer = AudioAnalyzer(lang=lang)
                
                # Extraer audio
                info_audio = audio_analyzer.extraer_audio(video_path)
                
                if info_audio.get("success"):
                    # Análisis detallado por segmentos
                    resultados_segmentos = audio_analyzer.analizar_segmentos_audio(
                        info_audio["ruta_audio"]
                    )
                    
                    # Transcripción completa
                    audio_resultados = audio_analyzer.transcribir_audio(info_audio["ruta_audio"])
                    
                    # Combinar resultados
                    audio_resultados.update({
                        "info_extraccion": info_audio,
                        "analisis_segmentos": resultados_segmentos,
                        "metricas_audio": self._calcular_metricas_audio(resultados_segmentos)
                    })
                    
                    resultados_completos["audio"] = audio_resultados
                    resultados_completos["etapas_completadas"].append("analisis_audio")
                    
                    self.logger.info(f"✓ Audio analizado: {audio_resultados.get('palabras_totales', 0)} palabras detectadas")
                else:
                    raise Exception(f"Fallo en extracción de audio: {info_audio.get('error', 'Error desconocido')}")
                    
            except Exception as e:
                error_msg = f"Error en análisis de audio: {str(e)}"
                self.logger.error(error_msg)
                resultados_completos["errores"].append(error_msg)
                resultados_completos["audio"] = {"error": str(e)}
            
            # ETAPA 3: Generación de recomendaciones básicas
            self.logger.info("💡 Etapa 3: Generando recomendaciones...")
            try:
                diagnostico = datos_personales.get("diagnostico", "") if datos_personales else ""
                
                # Recomendaciones genéricas
                recomendaciones_genericas = generar_recomendaciones(
                    resultados_completos.get("emociones", []),
                    resultados_completos.get("audio", {}),
                    diagnostico
                )
                
                resultados_completos["recomendaciones_genericas"] = recomendaciones_genericas
                resultados_completos["etapas_completadas"].append("recomendaciones_genericas")
                
                self.logger.info(f"✓ Generadas {len(recomendaciones_genericas)} recomendaciones genéricas")
                
            except Exception as e:
                error_msg = f"Error generando recomendaciones: {str(e)}"
                self.logger.error(error_msg)
                resultados_completos["errores"].append(error_msg)
                resultados_completos["recomendaciones_genericas"] = []
            
            # ETAPA 4: Recomendaciones avanzadas con IA
            self.logger.info("🤖 Etapa 4: Generando recomendaciones avanzadas...")
            try:
                recomendaciones_ia = self.api_recomendaciones.obtener_recomendaciones(
                    diagnostico=datos_personales.get("diagnostico", "") if datos_personales else "",
                    contexto_usuario=datos_personales or {},
                    resultados_emociones=resultados_completos.get("emociones", []),
                    resultados_audio=resultados_completos.get("audio", {})
                )
                
                resultados_completos["recomendaciones_ia"] = recomendaciones_ia
                resultados_completos["etapas_completadas"].append("recomendaciones_ia")
                
                self.logger.info("✓ Recomendaciones avanzadas generadas")
                
            except Exception as e:
                error_msg = f"Error en recomendaciones IA: {str(e)}"
                self.logger.error(error_msg)
                resultados_completos["errores"].append(error_msg)
                resultados_completos["recomendaciones_ia"] = {}
            
            # ETAPA 5: Generación de informes y visualizaciones
            self.logger.info("📈 Etapa 5: Generando informes y visualizaciones...")
            try:
                # El GeneradorInformes solo tiene un método de visualización: generar_dashboard_visual
                
                # Dashboard completo (Reemplaza Histograma y Timeline)
                dashboard_path = self.generador_informes.generar_dashboard_visual(
                    resultados_completos,
                    f"dashboard_{session_id}.png"
                )
                resultados_completos["dashboard_path"] = dashboard_path
                # Mantener compatibilidad con las claves anteriores para Histograma y Timeline
                resultados_completos["histograma_path"] = dashboard_path 
                resultados_completos["timeline_path"] = dashboard_path
                
                # Reporte completo en texto (Se elimina el parámetro 'datos_personales' no necesario)
                reporte_path = self.generador_informes.generar_reporte_completo(
                    resultados_completos,
                    f"reporte_{session_id}.txt"
                )
                resultados_completos["reporte_path"] = reporte_path
                
                # Exportación JSON (Se elimina el parámetro 'datos_personales' no necesario)
                json_path = self.generador_informes.exportar_reporte_json(
                    resultados_completos,
                    f"reporte_{session_id}.json"
                )
                resultados_completos["json_path"] = json_path
                
                # Nota: El método exportar_datos_csv fue omitido ya que no existe en GeneradorInformes
                resultados_completos["csv_path"] = ""
                
                resultados_completos["etapas_completadas"].append("generacion_informes")
                
                self.logger.info("✓ Informes y visualizaciones generados")
                
            except Exception as e:
                error_msg = f"Error generando informes: {str(e)}"
                self.logger.error(error_msg)
                resultados_completos["errores"].append(error_msg)
            
            # ETAPA 6: Análisis de alertas y seguimiento
            self.logger.info("⚠️ Etapa 6: Evaluando alertas...")
            try:
                alertas = self._evaluar_alertas(resultados_completos, configuracion)
                resultados_completos["alertas"] = alertas
                resultados_completos["nivel_prioridad"] = self._determinar_prioridad(alertas)
                resultados_completos["etapas_completadas"].append("evaluacion_alertas")
                
                if alertas:
                    self.logger.warning(f"⚠️ {len(alertas)} alertas detectadas")
                else:
                    self.logger.info("✓ No se detectaron alertas")
                
            except Exception as e:
                error_msg = f"Error evaluando alertas: {str(e)}"
                self.logger.error(error_msg)
                resultados_completos["errores"].append(error_msg)
                resultados_completos["alertas"] = []
            
            # Finalizar procesamiento
            tiempo_procesamiento = datetime.now() - inicio_procesamiento
            resultados_completos["timestamp_fin"] = datetime.now().isoformat()
            resultados_completos["tiempo_procesamiento"] = str(tiempo_procesamiento)
            resultados_completos["tiempo_procesamiento_segundos"] = tiempo_procesamiento.total_seconds()
            
            # Guardar resultados completos
            self._guardar_resultados_sesion(resultados_completos, session_dir)
            
            # Actualizar métricas del pipeline
            self._actualizar_metricas_pipeline(tiempo_procesamiento, len(resultados_completos["errores"]))
            
            # Combinar todas las recomendaciones
            todas_recomendaciones = []
            todas_recomendaciones.extend(resultados_completos.get("recomendaciones_genericas", []))
            
            recomendaciones_ia_dict = resultados_completos.get("recomendaciones_ia", {})
            for categoria in ["recomendaciones_generales", "recomendaciones_especificas", "actividades_sugeridas"]:
                todas_recomendaciones.extend(recomendaciones_ia_dict.get(categoria, []))
            
            resultados_completos["recomendaciones"] = todas_recomendaciones
            
            # Resultado final para compatibilidad
            resultado_final = {
                "emociones": resultados_completos.get("emociones", []),
                "audio": resultados_completos.get("audio", {}),
                "recomendaciones": todas_recomendaciones,
                "histograma": resultados_completos.get("histograma_path", ""),
                "reporte": resultados_completos.get("reporte_path", ""),
                "session_info": {
                    "session_id": session_id,
                    "tiempo_procesamiento": str(tiempo_procesamiento),
                    "etapas_completadas": resultados_completos["etapas_completadas"],
                    "errores": resultados_completos["errores"],
                    "alertas": resultados_completos.get("alertas", [])
                },
                "archivos_generados": {
                    "dashboard": resultados_completos.get("dashboard_path", ""),
                    "timeline": resultados_completos.get("timeline_path", ""),
                    "csv": resultados_completos.get("csv_path", ""),
                    "json": resultados_completos.get("json_path", "")
                }
            }
            
            self.logger.info(f"🎉 Pipeline completado exitosamente en {tiempo_procesamiento}")
            self.logger.info(f"📊 Etapas completadas: {len(resultados_completos['etapas_completadas'])}/6")
            
            return resultado_final
            
        except Exception as e:
            self.logger.error(f"💥 Error crítico en pipeline: {str(e)}")
            self.pipeline_metrics['errores_totales'] += 1
            
            return {
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "pipeline_status": "failed"
            }

    def _validar_video(self, video_path: str) -> bool:
        """Valida que el archivo de video sea accesible."""
        try:
            if not os.path.exists(video_path):
                self.logger.error(f"Archivo no encontrado: {video_path}")
                return False
            
            # Verificar que no esté vacío
            if os.path.getsize(video_path) == 0:
                self.logger.error(f"Archivo vacío: {video_path}")
                return False
            
            # Verificar extensión
            ext = os.path.splitext(video_path)[1].lower()
            if ext not in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
                self.logger.warning(f"Extensión de video inusual: {ext}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando video: {e}")
            return False

    def _obtener_configuracion(self, datos_personales: Optional[Dict], 
                              config_personalizada: Optional[Dict]) -> Dict:
        """Obtiene configuración específica basada en diagnóstico."""
        
        # Configuración por defecto
        config = self.configuraciones_diagnostico["default"].copy()
        
        # Aplicar configuración por diagnóstico
        if datos_personales and "diagnostico" in datos_personales:
            diagnostico = datos_personales["diagnostico"].lower()
            
            for key in self.configuraciones_diagnostico:
                if key in diagnostico and key != "default":
                    config.update(self.configuraciones_diagnostico[key])
                    self.logger.info(f"Aplicando configuración para: {key}")
                    break
        
        # Aplicar configuración personalizada
        if config_personalizada:
            config.update(config_personalizada)
            self.logger.info("Configuración personalizada aplicada")
        
        return config

    def _filtrar_por_confianza(self, resultados_emociones: List[Dict], umbral: float) -> List[Dict]:
        """Filtra resultados de emociones por umbral de confianza."""
        resultados_filtrados = []
        
        for frame_result in resultados_emociones:
            emociones_filtradas = []
            
            for emocion_data in frame_result.get("emociones", []):
                if emocion_data.get("confidence", 0.0) >= umbral:
                    emociones_filtradas.append(emocion_data)
            
            if emociones_filtradas:  # Solo incluir frames con emociones válidas
                frame_result_filtrado = frame_result.copy()
                frame_result_filtrado["emociones"] = emociones_filtradas
                frame_result_filtrado["emociones_filtradas"] = len(frame_result.get("emociones", [])) - len(emociones_filtradas)
                resultados_filtrados.append(frame_result_filtrado)
        
        total_original = sum(len(f.get("emociones", [])) for f in resultados_emociones)
        total_filtrado = sum(len(f.get("emociones", [])) for f in resultados_filtrados)
        
        self.logger.info(f"Filtrado por confianza: {total_filtrado}/{total_original} detecciones mantenidas")
        
        return resultados_filtrados

    def _calcular_estadisticas_emociones(self, emociones_resultados: List[Dict]) -> Dict:
        """Calcula estadísticas detalladas de las emociones."""
        if not emociones_resultados:
            return {}
        
        # Contadores y métricas
        conteo_emociones = {}
        confianzas_por_emocion = {}
        frames_con_emociones = 0
        total_detecciones = 0
        
        for frame_result in emociones_resultados:
            emociones_frame = frame_result.get("emociones", [])
            if emociones_frame:
                frames_con_emociones += 1
            
            for emocion_data in emociones_frame:
                emocion = emocion_data.get("emotion", "Unknown")
                confianza = emocion_data.get("confidence", 0.0)
                
                # Contar emociones
                conteo_emociones[emocion] = conteo_emociones.get(emocion, 0) + 1
                
                # Agrupar confianzas
                if emocion not in confianzas_por_emocion:
                    confianzas_por_emocion[emocion] = []
                confianzas_por_emocion[emocion].append(confianza)
                
                total_detecciones += 1
        
        # Calcular estadísticas
        estadisticas = {
            "frames_analizados": len(emociones_resultados),
            "frames_con_detecciones": frames_con_emociones,
            "total_detecciones": total_detecciones,
            "promedio_detecciones_por_frame": total_detecciones / len(emociones_resultados) if emociones_resultados else 0,
            "distribucion_emociones": conteo_emociones
        }
        
        # Emoción predominante
        if conteo_emociones:
            emocion_predominante = max(conteo_emociones, key=conteo_emociones.get)
            estadisticas["emocion_predominante"] = {
                "emocion": emocion_predominante,
                "count": conteo_emociones[emocion_predominante],
                "porcentaje": (conteo_emociones[emocion_predominante] / total_detecciones) * 100
            }
        
        # Estadísticas de confianza por emoción
        estadisticas["confianza_por_emocion"] = {}
        for emocion, confianzas in confianzas_por_emocion.items():
            estadisticas["confianza_por_emocion"][emocion] = {
                "promedio": np.mean(confianzas),
                "mediana": np.median(confianzas),
                "std": np.std(confianzas),
                "min": np.min(confianzas),
                "max": np.max(confianzas)
            }
        
        return estadisticas

    def _calcular_metricas_audio(self, resultados_segmentos: List[Dict]) -> Dict:
        """Calcula métricas detalladas del análisis de audio."""
        if not resultados_segmentos:
            return {}
        
        # Métricas de comunicación
        total_palabras = 0
        total_intentos = 0
        segmentos_con_audio = 0
        calidades = []
        volumenes = []
        
        for segmento in resultados_segmentos:
            palabras = segmento.get("palabras_totales", 0)
            intentos = segmento.get("intentos_comunicacion", 0)
            volumen = segmento.get("volumen_segmento", 0)
            
            total_palabras += palabras
            total_intentos += intentos
            
            if palabras > 0 or intentos > 0:
                segmentos_con_audio += 1
            
            if segmento.get("calidad_comunicacion"):
                calidades.append(segmento["calidad_comunicacion"])
            
            if volumen != 0:
                volumenes.append(volumen)
        
        # Calcular métricas
        metricas = {
            "segmentos_analizados": len(resultados_segmentos),
            "segmentos_con_comunicacion": segmentos_con_audio,
            "total_palabras_detectadas": total_palabras,
            "total_intentos_comunicativos": total_intentos,
            "promedio_palabras_por_segmento": total_palabras / len(resultados_segmentos),
            "porcentaje_segmentos_activos": (segmentos_con_audio / len(resultados_segmentos)) * 100
        }
        
        # Análisis de volumen
        if volumenes:
            metricas["analisis_volumen"] = {
                "volumen_promedio_db": np.mean(volumenes),
                "volumen_max_db": np.max(volumenes),
                "volumen_min_db": np.min(volumenes),
                "variabilidad_volumen": np.std(volumenes)
            }
        
        return metricas

    def _evaluar_alertas(self, resultados: Dict, configuracion: Dict) -> List[Dict]:
        """Evalúa y genera alertas basadas en los resultados."""
        alertas = []
        
        # Alertas emocionales
        estadisticas_emociones = resultados.get("estadisticas_emociones", {})
        distribucion = estadisticas_emociones.get("distribucion_emociones", {})
        total_detecciones = estadisticas_emociones.get("total_detecciones", 0)
        
        if total_detecciones > 0:
            # Evaluar emociones negativas predominantes
            emociones_negativas = ["Sad", "Angry", "Fear", "Disgust"]
            total_negativas = sum(distribucion.get(emo, 0) for emo in emociones_negativas)
            porcentaje_negativas = (total_negativas / total_detecciones) * 100
            
            if porcentaje_negativas > 60:
                alertas.append({
                    "tipo": "emocional",
                    "nivel": "alto",
                    "mensaje": f"Predominio de emociones negativas ({porcentaje_negativas:.1f}%)",
                    "recomendacion": "Evaluación psicoemocional urgente recomendada",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Alertas específicas por diagnóstico
            alertas_especiales = configuracion.get("alertas_especiales", [])
            for emocion_alerta in alertas_especiales:
                count_emocion = distribucion.get(emocion_alerta, 0)
                porcentaje_emocion = (count_emocion / total_detecciones) * 100
                
                if porcentaje_emocion > 30:
                    alertas.append({
                        "tipo": "diagnostico_especifico",
                        "nivel": "medio",
                        "mensaje": f"Alta frecuencia de {emocion_alerta} ({porcentaje_emocion:.1f}%)",
                        "recomendacion": f"Monitoreo específico para {emocion_alerta.lower()} requerido",
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Alertas de comunicación
        audio_data = resultados.get("audio", {})
        calidad_comunicacion = audio_data.get("calidad_comunicacion", "")
        
        if calidad_comunicacion == "sin_comunicacion_verbal":
            alertas.append({
                "tipo": "comunicacion",
                "nivel": "alto",
                "mensaje": "Ausencia total de comunicación verbal detectada",
                "recomendacion": "Evaluación de comunicación alternativa urgente",
                "timestamp": datetime.now().isoformat()
            })
        elif calidad_comunicacion == "comunicacion_limitada":
            intentos = audio_data.get("intentos_comunicacion", 0)
            if intentos < 2:
                alertas.append({
                    "tipo": "comunicacion",
                    "nivel": "medio",
                    "mensaje": "Comunicación verbal muy limitada",
                    "recomendacion": "Estimulación del lenguaje prioritaria",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Alertas de calidad de datos
        errores = resultados.get("errores", [])
        if len(errores) > 2:
            alertas.append({
                "tipo": "tecnico",
                "nivel": "medio",
                "mensaje": f"Múltiples errores en el análisis ({len(errores)})",
                "recomendacion": "Revisar calidad del video y condiciones de grabación",
                "timestamp": datetime.now().isoformat()
            })
        
        return alertas

    def _determinar_prioridad(self, alertas: List[Dict]) -> str:
        """Determina el nivel de prioridad global basado en las alertas."""
        if not alertas:
            return "normal"
        
        niveles = [alerta.get("nivel", "bajo") for alerta in alertas]
        
        if "alto" in niveles:
            return "critico"
        elif "medio" in niveles:
            return "moderado"
        else:
            return "bajo"

    def _extraer_conteo_emociones(self, resultados_emociones: List[Dict]) -> Dict:
        """Extrae conteo simple de emociones para compatibilidad."""
        conteo = {}
        for frame_result in resultados_emociones:
            for emocion_data in frame_result.get("emociones", []):
                emocion = emocion_data.get("emotion", "Unknown")
                conteo[emocion] = conteo.get(emocion, 0) + 1
        return conteo

    def _guardar_resultados_sesion(self, resultados: Dict, session_dir: str):
        """Guarda los resultados completos de la sesión."""
        try:
            results_path = os.path.join(session_dir, "resultados_completos.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                # Usamos default=str para manejar objetos como datetime.timedelta
                json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"✓ Resultados guardados en: {results_path}")
            
        except Exception as e:
            self.logger.error(f"Error guardando resultados de sesión: {e}")

    def _actualizar_metricas_pipeline(self, tiempo_procesamiento, num_errores):
        """Actualiza las métricas del pipeline."""
        self.pipeline_metrics['sesiones_procesadas'] += 1
        self.pipeline_metrics['videos_analizados'] += 1
        self.pipeline_metrics['errores_totales'] += num_errores
        self.pipeline_metrics['tiempo_total_procesamiento'] += tiempo_procesamiento.total_seconds()

    def obtener_estadisticas_pipeline(self) -> Dict:
        """Obtiene estadísticas completas del pipeline."""
        tiempo_total = datetime.now() - self.pipeline_metrics['inicio_pipeline']
        
        estadisticas = self.pipeline_metrics.copy()
        estadisticas.update({
            'tiempo_operacion_total': str(tiempo_total),
            'promedio_tiempo_por_sesion': (
                self.pipeline_metrics['tiempo_total_procesamiento'] / 
                max(self.pipeline_metrics['sesiones_procesadas'], 1)
            ),
            'tasa_error': (
                self.pipeline_metrics['errores_totales'] / 
                max(self.pipeline_metrics['sesiones_procesadas'], 1)
            ),
            'componentes_activos': {
                'detector_emociones': hasattr(self, 'detector_emociones'),
                'generador_informes': hasattr(self, 'generador_informes'),
                'api_recomendaciones': hasattr(self, 'api_recomendaciones')
            }
        })
        
        return estadisticas

    def configurar_api_recomendaciones(self, base_url: str, token: str = None):
        """Configura la API real de recomendaciones."""
        try:
            self.api_recomendaciones = ApiRecomendaciones(base_url=base_url, token=token)
            self.logger.info(f"✓ API de recomendaciones configurada: {base_url}")
        except Exception as e:
            self.logger.error(f"Error configurando API: {e}")

    def limpiar_cache_pipeline(self):
        """Limpia cache y archivos temporales del pipeline."""
        try:
            cache_dir = os.path.join(self.resultados_dir, "cache")
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
            
            # Limpiar cache de recomendaciones
            if hasattr(self.api_recomendaciones, 'limpiar_cache'):
                self.api_recomendaciones.limpiar_cache()
            
            self.logger.info("✓ Cache del pipeline limpiado")
            
        except Exception as e:
            self.logger.error(f"Error limpiando cache: {e}")

    def validar_componentes(self) -> Dict[str, bool]:
        """Valida que todos los componentes estén funcionando correctamente."""
        validacion = {}
        
        try:
            # Validar detector de emociones
            validacion['detector_emociones'] = hasattr(self.detector_emociones, 'analizar_video')
            
            # Validar generador de informes
            validacion['generador_informes'] = hasattr(self.generador_informes, 'generar_dashboard_visual') # Corregido
            
            # Validar API de recomendaciones
            validacion['api_recomendaciones'] = hasattr(self.api_recomendaciones, 'obtener_recomendaciones')
            
            # Validar directorios
            validacion['directorios'] = all(os.path.exists(d) for d in [
                self.models_dir, self.resultados_dir
            ])
            
            self.logger.info(f"Validación de componentes: {validacion}")
            
        except Exception as e:
            self.logger.error(f"Error en validación: {e}")
            validacion['error'] = str(e)
        
        return validacion

# Función de compatibilidad con la interfaz original
def ejecutar_pipeline(video_path: str, models_dir: str = "./models", 
                      lang: str = "es-ES", datos_personales: Optional[Dict] = None) -> Dict:
    """
    Función de compatibilidad con la interfaz original.
    """
    try:
        # Se asume que el directorio de resultados es './resultados' por defecto.
        pipeline = PipelineAnalisisEmocional(models_dir=models_dir, resultados_dir="./resultados") 
        return pipeline.ejecutar_pipeline(video_path, lang, datos_personales)
    except Exception as e:
        logging.error(f"Error en pipeline de compatibilidad: {e}")
        return {
            "error": str(e),
            "emociones": [],
            "audio": {},
            "recomendaciones": [],
            "histograma": "",
            "reporte": ""
        }

# Función de simulación de Gemini mejorada (compatible con versión anterior)
def consultar_gemini(diagnostico: str, emociones: List[Dict], audio: Dict) -> List[str]:
    """
    Simulación mejorada de consulta a Gemini API (compatible con versión anterior).
    """
    try:
        # Usar el nuevo sistema de recomendaciones
        api_rec = ApiRecomendaciones()
        
        # Preparar contexto
        contexto = {"diagnostico": diagnostico}
        
        # Obtener recomendaciones
        recomendaciones_ia = api_rec.obtener_recomendaciones(
            diagnostico=diagnostico,
            contexto_usuario=contexto,
            resultados_emociones=emociones,
            resultados_audio=audio
        )
        
        # Extraer todas las recomendaciones en formato de lista
        todas_recomendaciones = []
        
        for categoria in ["recomendaciones_generales", "recomendaciones_especificas", "actividades_sugeridas"]:
            todas_recomendaciones.extend(recomendaciones_ia.get(categoria, []))
        
        return todas_recomendaciones
        
    except Exception as e:
        logging.error(f"Error en simulación Gemini: {e}")
        return [
            "[Gemini] Error en consulta, aplicando recomendaciones básicas",
            "[Gemini] Continuar con protocolo de seguimiento estándar"
        ]


