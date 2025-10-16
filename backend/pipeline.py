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

class PipelineAnalisisEmocional:
    """Pipeline principal para análisis emocional multimodal."""
    
    def __init__(self, models_dir: str = "./models", resultados_dir: str = "./resultados"):
        """Inicializa el pipeline de análisis emocional."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.models_dir = models_dir
        self.resultados_dir = resultados_dir
        self.ensure_directories()
        
        try:
            self.detector_emociones = DetectorEmociones(save_frames_path=os.path.join(resultados_dir, "fotogramas_detectados"))
            self.generador_informes = GeneradorInformes(carpeta_resultados=resultados_dir)
            self.api_recomendaciones = ApiRecomendaciones()
            self.logger.info("✓ Pipeline inicializado correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando pipeline: {e}")
            raise
        
        self.pipeline_metrics = {
            'sesiones_procesadas': 0,
            'videos_analizados': 0,
            'errores_totales': 0,
            'tiempo_total_procesamiento': 0,
            'inicio_pipeline': datetime.now()
        }
        
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
                "umbral_confianza": 0.05,
                "priorizar_emociones": ["Neutral", "Happy", "Fear"],
                "alertas_especiales": ["Angry", "Sad"]
            },
            "tdah": {
                "intervalo_analisis_ms": 1500,
                "umbral_confianza": 0.05,
                "priorizar_emociones": ["Happy", "Surprise", "Neutral"],
                "alertas_especiales": ["Angry"]
            },
            "sindrome_down": {
                "intervalo_analisis_ms": 2500,
                "umbral_confianza": 0.05,
                "priorizar_emociones": ["Happy", "Surprise"],
                "alertas_especiales": ["Sad", "Fear"]
            },
            "paralisis_cerebral": {
                "intervalo_analisis_ms": 3000,
                "umbral_confianza": 0.05,
                "priorizar_emociones": ["Happy", "Neutral"],
                "alertas_especiales": ["Disgust", "Fear", "Sad"]
            },
            "default": {
                "intervalo_analisis_ms": 1000,
                "umbral_confianza": 0.05,
                "priorizar_emociones": [],
                "alertas_especiales": []
            }
        }

    def ejecutar_pipeline(self, video_path: str, lang: str = "es-ES", 
                          datos_personales: Optional[Dict] = None,
                          configuracion_personalizada: Optional[Dict] = None) -> Dict:
        """Ejecuta el pipeline completo de análisis emocional."""
        inicio_procesamiento = datetime.now()
        session_id = f"sesion_{inicio_procesamiento.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"🚀 INICIANDO PIPELINE - Sesión: {session_id}")
            self.logger.info(f"📹 Video: {os.path.basename(video_path)}")
            
            if not self._validar_video(video_path):
                raise ValueError(f"Archivo de video inválido: {video_path}")
            
            configuracion = self._obtener_configuracion(datos_personales, configuracion_personalizada)
            
            self.logger.info(f"⚙️ Configuración:")
            self.logger.info(f"   Intervalo: {configuracion['intervalo_analisis_ms']}ms")
            self.logger.info(f"   Umbral confianza: {configuracion['umbral_confianza']}")
            
            session_dir = os.path.join(self.resultados_dir, "sesiones", session_id)
            os.makedirs(session_dir, exist_ok=True)
            
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
            self.logger.info("=" * 60)
            self.logger.info("ETAPA 1: Analizando emociones faciales...")
            self.logger.info("=" * 60)
            
            try:
                self.logger.info(f"DEBUG: Llamando a detector.analizar_video()")
                self.logger.info(f"  video_path: {video_path}")
                self.logger.info(f"  intervalo_ms: {configuracion['intervalo_analisis_ms']}")
                
                emociones_resultados = self.detector_emociones.analizar_video(
                    video_path, 
                    intervalo_ms=configuracion["intervalo_analisis_ms"],
                    guardar_frames=True
                )
                
                self.logger.info(f"✓ Detector retornó {len(emociones_resultados)} frames")
                
                if emociones_resultados:
                    frame_0 = emociones_resultados[0]
                    self.logger.info(f"DEBUG: Frame 0 claves: {frame_0.keys()}")
                    self.logger.info(f"DEBUG: Frame 0 tiene 'rostros': {'rostros' in frame_0}")
                    self.logger.info(f"DEBUG: Frame 0 tiene 'emociones': {'emociones' in frame_0}")
                    
                    if 'rostros' in frame_0:
                        self.logger.info(f"DEBUG: Frame 0 rostros count: {len(frame_0.get('rostros', []))}")
                        if frame_0.get('rostros'):
                            r0 = frame_0['rostros'][0]
                            self.logger.info(f"DEBUG: Rostro 0 claves: {r0.keys()}")
                            self.logger.info(f"DEBUG: Rostro 0 emociones: {len(r0.get('emociones', []))}")
                
                # Contar emociones ANTES de filtrar
                total_antes = 0
                for frame in emociones_resultados:
                    for rostro in frame.get('rostros', []):
                        total_antes += len(rostro.get('emociones', []))
                
                self.logger.info(f"📊 Emociones ANTES del filtro: {total_antes}")
                
                # Filtrar
                emociones_filtradas = self._filtrar_por_confianza(
                    emociones_resultados, 
                    configuracion["umbral_confianza"]
                )
                
                # Contar emociones DESPUÉS de filtrar
                total_despues = 0
                for frame in emociones_filtradas:
                    total_despues += len(frame.get('emociones', []))
                
                self.logger.info(f"📊 Emociones DESPUÉS del filtro: {total_despues}")
                
                resultados_completos["emociones"] = emociones_filtradas
                resultados_completos["estadisticas_emociones"] = self._calcular_estadisticas_emociones(emociones_filtradas)
                resultados_completos["etapas_completadas"].append("analisis_emociones")
                
                self.logger.info(f"✅ Etapa 1 completada: {len(emociones_filtradas)} frames con emociones")
                
            except Exception as e:
                error_msg = f"Error en análisis emocional: {str(e)}"
                self.logger.error(f"❌ {error_msg}")
                import traceback
                self.logger.error(traceback.format_exc())
                resultados_completos["errores"].append(error_msg)
                resultados_completos["emociones"] = []
                resultados_completos["estadisticas_emociones"] = {}
            
            # ETAPA 2: Análisis de audio
            self.logger.info("=" * 60)
            self.logger.info("ETAPA 2: Analizando audio...")
            self.logger.info("=" * 60)
            
            try:
                audio_analyzer = AudioAnalyzer(lang=lang)
                info_audio = audio_analyzer.extraer_audio(video_path)
                
                if info_audio.get("success"):
                    resultados_segmentos = audio_analyzer.analizar_segmentos_audio(
                        info_audio["ruta_audio"]
                    )
                    
                    audio_resultados = audio_analyzer.transcribir_audio(info_audio["ruta_audio"])
                    
                    audio_resultados.update({
                        "info_extraccion": info_audio,
                        "analisis_segmentos": resultados_segmentos,
                        "metricas_audio": self._calcular_metricas_audio(resultados_segmentos)
                    })
                    
                    resultados_completos["audio"] = audio_resultados
                    resultados_completos["etapas_completadas"].append("analisis_audio")
                    
                    self.logger.info(f"✅ Audio analizado: {audio_resultados.get('palabras_totales', 0)} palabras")
                else:
                    raise Exception(f"Fallo extracción audio: {info_audio.get('error')}")
                    
            except Exception as e:
                error_msg = f"Error en análisis de audio: {str(e)}"
                self.logger.error(f"⚠️ {error_msg}")
                resultados_completos["errores"].append(error_msg)
                resultados_completos["audio"] = {"error": str(e)}
            
            # ETAPA 3: Recomendaciones
            self.logger.info("=" * 60)
            self.logger.info("ETAPA 3: Generando recomendaciones...")
            self.logger.info("=" * 60)
            
            try:
                diagnostico = datos_personales.get("diagnostico", "") if datos_personales else ""
                
                recomendaciones_genericas = generar_recomendaciones(
                    resultados_completos.get("emociones", []),
                    resultados_completos.get("audio", {}),
                    diagnostico
                )
                
                resultados_completos["recomendaciones_genericas"] = recomendaciones_genericas
                resultados_completos["etapas_completadas"].append("recomendaciones_genericas")
                
                self.logger.info(f"✅ {len(recomendaciones_genericas)} recomendaciones generadas")
                
            except Exception as e:
                self.logger.error(f"⚠️ Error en recomendaciones: {str(e)}")
                resultados_completos["errores"].append(str(e))
                resultados_completos["recomendaciones_genericas"] = []
            
            # ETAPA 4: IA
            self.logger.info("=" * 60)
            self.logger.info("ETAPA 4: Recomendaciones IA...")
            self.logger.info("=" * 60)
            
            try:
                recomendaciones_ia = self.api_recomendaciones.obtener_recomendaciones(
                    diagnostico=datos_personales.get("diagnostico", "") if datos_personales else "",
                    contexto_usuario=datos_personales or {},
                    resultados_emociones=resultados_completos.get("emociones", []),
                    resultados_audio=resultados_completos.get("audio", {})
                )
                
                resultados_completos["recomendaciones_ia"] = recomendaciones_ia
                resultados_completos["etapas_completadas"].append("recomendaciones_ia")
                self.logger.info(f"✅ Recomendaciones IA generadas")
                
            except Exception as e:
                self.logger.error(f"⚠️ Error en IA: {str(e)}")
                resultados_completos["errores"].append(str(e))
                resultados_completos["recomendaciones_ia"] = {}
            
            # ETAPA 5: Informes
            self.logger.info("=" * 60)
            self.logger.info("ETAPA 5: Generando informes...")
            self.logger.info("=" * 60)
            
            try:
                conteo_emociones = self._extraer_conteo_emociones(resultados_completos.get("emociones", []))
                if conteo_emociones:
                    histograma_path = self.generador_informes.generar_histograma_emociones(
                        conteo_emociones, 
                        f"histograma_{session_id}.png"
                    )
                    resultados_completos["histograma_path"] = histograma_path
                
                if resultados_completos.get("emociones"):
                    timeline_path = self.generador_informes.generar_timeline_emocional(
                        resultados_completos["emociones"],
                        f"timeline_{session_id}.png"
                    )
                    resultados_completos["timeline_path"] = timeline_path
                
                dashboard_path = self.generador_informes.generar_dashboard_visual(
                    resultados_completos,
                    f"dashboard_{session_id}.png"
                )
                resultados_completos["dashboard_path"] = dashboard_path
                
                reporte_path = self.generador_informes.generar_reporte_completo(
                    resultados_completos,
                    datos_personales,
                    f"reporte_{session_id}.txt"
                )
                resultados_completos["reporte_path"] = reporte_path
                
                csv_path = self.generador_informes.exportar_datos_csv(
                    resultados_completos,
                    f"datos_{session_id}.csv"
                )
                resultados_completos["csv_path"] = csv_path
                
                json_path = self.generador_informes.exportar_reporte_json(
                    resultados_completos,
                    datos_personales,
                    f"reporte_{session_id}.json"
                )
                resultados_completos["json_path"] = json_path
                
                resultados_completos["etapas_completadas"].append("generacion_informes")
                self.logger.info(f"✅ Informes generados")
                
            except Exception as e:
                self.logger.error(f"⚠️ Error en informes: {str(e)}")
                resultados_completos["errores"].append(str(e))
            
            # ETAPA 6: Alertas
            self.logger.info("=" * 60)
            self.logger.info("ETAPA 6: Evaluando alertas...")
            self.logger.info("=" * 60)
            
            try:
                alertas = self._evaluar_alertas(resultados_completos, configuracion)
                resultados_completos["alertas"] = alertas
                resultados_completos["nivel_prioridad"] = self._determinar_prioridad(alertas)
                resultados_completos["etapas_completadas"].append("evaluacion_alertas")
                
                if alertas:
                    self.logger.warning(f"⚠️ {len(alertas)} alertas detectadas")
                else:
                    self.logger.info("✅ Sin alertas")
                
            except Exception as e:
                self.logger.error(f"⚠️ Error en alertas: {str(e)}")
                resultados_completos["errores"].append(str(e))
                resultados_completos["alertas"] = []
            
            # Finalizar
            tiempo_procesamiento = datetime.now() - inicio_procesamiento
            resultados_completos["timestamp_fin"] = datetime.now().isoformat()
            resultados_completos["tiempo_procesamiento"] = str(tiempo_procesamiento)
            resultados_completos["tiempo_procesamiento_segundos"] = tiempo_procesamiento.total_seconds()
            
            self._guardar_resultados_sesion(resultados_completos, session_dir)
            self._actualizar_metricas_pipeline(tiempo_procesamiento, len(resultados_completos["errores"]))
            
            # Combinar recomendaciones
            todas_recomendaciones = []
            todas_recomendaciones.extend(resultados_completos.get("recomendaciones_genericas", []))
            
            recomendaciones_ia_dict = resultados_completos.get("recomendaciones_ia", {})
            for categoria in ["recomendaciones_generales", "recomendaciones_especificas", "actividades_sugeridas"]:
                todas_recomendaciones.extend(recomendaciones_ia_dict.get(categoria, []))
            
            resultados_completos["recomendaciones"] = todas_recomendaciones
            
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
                    "alertas": resultados_completos.get("alertas", []),
                    "recomendaciones_ia": resultados_completos.get("recomendaciones_ia", {})
                },
                "archivos_generados": {
                    "dashboard": resultados_completos.get("dashboard_path", ""),
                    "timeline": resultados_completos.get("timeline_path", ""),
                    "csv": resultados_completos.get("csv_path", ""),
                    "json": resultados_completos.get("json_path", "")
                }
            }
            
            self.logger.info("=" * 60)
            self.logger.info(f"🎉 PIPELINE COMPLETADO EN {tiempo_procesamiento}")
            self.logger.info("=" * 60)
            
            return resultado_final
            
        except Exception as e:
            self.logger.error(f"💥 ERROR CRÍTICO: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "session_id": session_id,
                "pipeline_status": "failed"
            }

    def _validar_video(self, video_path: str) -> bool:
        if not os.path.exists(video_path):
            self.logger.error(f"Video no encontrado: {video_path}")
            return False
        if os.path.getsize(video_path) == 0:
            self.logger.error(f"Video vacío: {video_path}")
            return False
        return True

    def _obtener_configuracion(self, datos_personales: Optional[Dict], 
                               config_personalizada: Optional[Dict]) -> Dict:
        config = self.configuraciones_diagnostico["default"].copy()
        
        if datos_personales and "diagnostico" in datos_personales:
            diagnostico = datos_personales["diagnostico"].lower()
            for key in self.configuraciones_diagnostico:
                if key in diagnostico and key != "default":
                    config.update(self.configuraciones_diagnostico[key])
                    break
        
        if config_personalizada:
            config.update(config_personalizada)
        
        return config

    def _filtrar_por_confianza(self, resultados_emociones: List[Dict], umbral: float) -> List[Dict]:
        resultados_filtrados = []
        
        for frame_result in resultados_emociones:
            rostros_data = frame_result.get('rostros', [])
            emociones_frame = []
            
            for rostro_idx, rostro_data in enumerate(rostros_data):
                emociones_rostro = rostro_data.get('emociones', [])
                
                for emocion_data in emociones_rostro:
                    if emocion_data.get('confidence', 0.0) >= umbral:
                        emocion_completa = emocion_data.copy()
                        emocion_completa['rostro_id'] = rostro_idx
                        emocion_completa['bbox'] = rostro_data.get('bbox')
                        emociones_frame.append(emocion_completa)
            
            if emociones_frame:
                frame_compatible = {
                    'frame_id': frame_result.get('frame_id'),
                    'timestamp': frame_result.get('timestamp'),
                    'tiempo_video': frame_result.get('tiempo_video'),
                    'num_faces': len(rostros_data),
                    'emociones': emociones_frame,
                    'rostros': rostros_data,
                    'frame_path': frame_result.get('frame_path'),
                    'total_emociones_detectadas': len(emociones_frame)
                }
                resultados_filtrados.append(frame_compatible)
        
        self.logger.info(f"Filtrado: {len(resultados_filtrados)}/{len(resultados_emociones)} frames, umbral={umbral}")
        
        return resultados_filtrados

    def _calcular_estadisticas_emociones(self, emociones_resultados: List[Dict]) -> Dict:
        if not emociones_resultados:
            return {}
        
        conteo_emociones = {}
        total_detecciones = 0
        
        for frame_result in emociones_resultados:
            for emocion_data in frame_result.get('emociones', []):
                emocion = emocion_data.get('emotion', 'Unknown')
                conteo_emociones[emocion] = conteo_emociones.get(emocion, 0) + 1
                total_detecciones += 1
        
        return {
            "frames_analizados": len(emociones_resultados),
            "total_detecciones": total_detecciones,
            "distribucion_emociones": conteo_emociones
        }

    def _calcular_metricas_audio(self, resultados_segmentos: List[Dict]) -> Dict:
        return {}

    def _evaluar_alertas(self, resultados: Dict, configuracion: Dict) -> List[Dict]:
        return []

    def _determinar_prioridad(self, alertas: List[Dict]) -> str:
        return "normal"

    def _extraer_conteo_emociones(self, resultados_emociones: List[Dict]) -> Dict:
        conteo = {}
        for frame_result in resultados_emociones:
            for emocion_data in frame_result.get('emociones', []):
                emocion = emocion_data.get('emotion', 'Unknown')
                conteo[emocion] = conteo.get(emocion, 0) + 1
        return conteo

    def _guardar_resultados_sesion(self, resultados: Dict, session_dir: str):
        try:
            results_path = os.path.join(session_dir, "resultados_completos.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"Error guardando resultados: {e}")

    def _actualizar_metricas_pipeline(self, tiempo_procesamiento, num_errores):
        self.pipeline_metrics['sesiones_procesadas'] += 1
        self.pipeline_metrics['videos_analizados'] += 1
        self.pipeline_metrics['errores_totales'] += num_errores

def ejecutar_pipeline(video_path: str, models_dir: str = "./models", 
                      lang: str = "es-ES", datos_personales: Optional[Dict] = None) -> Dict:
    try:
        pipeline = PipelineAnalisisEmocional(models_dir=models_dir)
        return pipeline.ejecutar_pipeline(video_path, lang, datos_personales)
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"error": str(e)}