import os
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
# Asegúrate de que este módulo sea accesible en tu entorno
from .emotion_ensemble import EmotionEnsemble

class DetectorEmociones:
    """
    Detector avanzado de emociones faciales con análisis temporal,
    guardado automático de fotogramas etiquetados y métricas detalladas.
    """
    
    def __init__(self, cascade_path: str = None, save_frames_path: str = "./fotogramas_detectados"):
        """
        Inicializa el detector de emociones.
        
        Args:
            cascade_path (str): Ruta del clasificador Haar Cascade
            save_frames_path (str): Directorio para guardar fotogramas detectados
        """
        
        # INICIO DE LA CORRECCIÓN CLAVE
        # Configurar logging (MOVIDO AL PRINCIPIO)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        # FIN DE LA CORRECCIÓN CLAVE
        
        # Configurar ruta del clasificador
        if cascade_path is None:
            # Intentar ubicaciones comunes
            possible_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                './haarcascade_frontalface_default.xml',
                './models/haarcascade_frontalface_default.xml'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    cascade_path = path
                    break
            
            if cascade_path is None:
                raise FileNotFoundError("No se encontró el archivo haarcascade_frontalface_default.xml")
        
        # Inicializar componentes
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.emotion_model = EmotionEnsemble()
        
        # Configurar directorios
        self.save_frames_path = save_frames_path
        self.setup_directories()
        
        # Parámetros de detección
        self.detection_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 5,
            'minSize': (30, 30),
            'maxSize': (300, 300)
        }
        
        # Parámetros de calidad
        self.quality_thresholds = {
            'min_face_size': 50,
            'max_face_size': 400,
            'confidence_threshold': 0.3
        }
        
        # Métricas de sesión
        self.session_metrics = {
            'frames_processed': 0,
            'faces_detected': 0,
            'emotions_analyzed': 0,
            'high_confidence_detections': 0,
            'session_start': datetime.now()
        }

    def setup_directories(self):
        """Configura directorios necesarios con estructura por fecha."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear estructura de directorios
        self.session_dir = os.path.join(self.save_frames_path, f"sesion_{timestamp}")
        self.frames_dir = os.path.join(self.session_dir, "frames")
        self.faces_dir = os.path.join(self.session_dir, "rostros")
        self.reports_dir = os.path.join(self.session_dir, "reportes")
        
        for directory in [self.session_dir, self.frames_dir, self.faces_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Directorios configurados en: {self.session_dir}")

    def detectar_rostros(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostros en un frame con parámetros optimizados.
        
        Args:
            frame (np.ndarray): Frame de video en formato BGR
            
        Returns:
            List[Tuple[int, int, int, int]]: Lista de coordenadas de rostros (x, y, w, h)
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Aplicar ecualización de histograma para mejorar contraste
            gray = cv2.equalizeHist(gray)
            
            # Detectar rostros
            rostros = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.detection_params['scaleFactor'],
                minNeighbors=self.detection_params['minNeighbors'],
                minSize=self.detection_params['minSize'],
                maxSize=self.detection_params['maxSize'],
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Filtrar rostros por tamaño
            rostros_validos = []
            for (x, y, w, h) in rostros:
                if (self.quality_thresholds['min_face_size'] <= w <= self.quality_thresholds['max_face_size'] and
                    self.quality_thresholds['min_face_size'] <= h <= self.quality_thresholds['max_face_size']):
                    rostros_validos.append((x, y, w, h))
            
            return rostros_validos
            
        except Exception as e:
            self.logger.error(f"Error detectando rostros: {e}")
            return []

    def analizar_emocion(self, rostro_img: np.ndarray, frame_id: int, face_id: int) -> Dict:
        """
        Analiza emoción en una imagen de rostro con métricas de calidad.
        
        Args:
            rostro_img (np.ndarray): Imagen del rostro
            frame_id (int): ID del frame
            face_id (int): ID del rostro en el frame
            
        Returns:
            Dict: Análisis emocional con métricas de calidad
        """
        try:
            # Verificar calidad de la imagen
            quality_score = self._assess_image_quality(rostro_img)
            
            # Realizar predicción
            emocion, confianza = self.emotion_model.predict_emotion(rostro_img)
            
            # Determinar si es detección de alta confianza
            high_confidence = confianza > 0.7
            if high_confidence:
                self.session_metrics['high_confidence_detections'] += 1
            
            # Crear resultado estructurado
            resultado = {
                "emotion": emocion,
                "confidence": float(confianza),
                "quality_score": quality_score,
                "high_confidence": high_confidence,
                "frame_id": frame_id,
                "face_id": face_id,
                "timestamp": datetime.now().isoformat(),
                "image_size": rostro_img.shape[:2] if len(rostro_img.shape) >= 2 else None
            }
            
            # Actualizar métricas
            self.session_metrics['emotions_analyzed'] += 1
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Error analizando emoción: {e}")
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "error": str(e),
                "frame_id": frame_id,
                "face_id": face_id
            }

    def _assess_image_quality(self, img: np.ndarray) -> float:
        """
        Evalúa la calidad de una imagen de rostro.
        
        Args:
            img (np.ndarray): Imagen del rostro
            
        Returns:
            float: Puntuación de calidad (0.0 a 1.0)
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Evaluar nitidez usando varianza de Laplaciano
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000.0, 1.0)  # Normalizar
            
            # Evaluar contraste
            contrast = gray.std()
            contrast_score = min(contrast / 64.0, 1.0)  # Normalizar
            
            # Evaluar brillo (evitar imágenes muy oscuras o muy brillantes)
            brightness = gray.mean()
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # Puntuación combinada
            quality_score = (sharpness_score * 0.4 + contrast_score * 0.3 + brightness_score * 0.3)
            
            return round(quality_score, 3)
            
        except Exception as e:
            self.logger.error(f"Error evaluando calidad de imagen: {e}")
            return 0.0

    def guardar_frame_completo(self, frame: np.ndarray, frame_id: int, 
                                 rostros_info: List[Dict], timestamp: str) -> str:
        """
        Guarda el frame completo con rostros marcados.
        
        Args:
            frame (np.ndarray): Frame original
            frame_id (int): ID del frame
            rostros_info (List[Dict]): Información de rostros detectados
            timestamp (str): Timestamp del análisis
            
        Returns:
            str: Ruta del archivo guardado
        """
        try:
            # Crear copia del frame para marcar rostros
            frame_marked = frame.copy()
            
            # Marcar rostros detectados
            for i, info in enumerate(rostros_info):
                emocion = info.get('emotion', 'Unknown')
                confianza = info.get('confidence', 0.0)
                
                # Obtener coordenadas del rostro (asumiendo que están en el info)
                if 'bbox' in info:
                    x, y, w, h = info['bbox']
                    
                    # Color basado en confianza
                    if confianza > 0.7:
                        color = (0, 255, 0)  # Verde para alta confianza
                    elif confianza > 0.4:
                        color = (0, 255, 255)  # Amarillo para media confianza
                    else:
                        color = (0, 0, 255)  # Rojo para baja confianza
                    
                    # Dibujar rectángulo y etiqueta
                    cv2.rectangle(frame_marked, (x, y), (x + w, y + h), color, 2)
                    
                    # Etiqueta con emoción y confianza
                    label = f"{emocion}: {confianza:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Fondo para la etiqueta
                    cv2.rectangle(frame_marked, (x, y - label_size[1] - 10), 
                                  (x + label_size[0], y), color, -1)
                    
                    # Texto de la etiqueta
                    cv2.putText(frame_marked, label, (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Guardar frame marcado
            filename = f"frame_{frame_id:06d}_{timestamp}.jpg"
            filepath = os.path.join(self.frames_dir, filename)
            
            # Usar calidad alta para preservar detalles
            cv2.imwrite(filepath, frame_marked, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error guardando frame completo: {e}")
            return ""

    def guardar_rostro_individual(self, rostro_img: np.ndarray, frame_id: int, 
                                 face_id: int, emocion_info: Dict) -> str:
        """
        Guarda imagen individual del rostro con información de emoción.
        
        Args:
            rostro_img (np.ndarray): Imagen del rostro
            frame_id (int): ID del frame
            face_id (int): ID del rostro
            emocion_info (Dict): Información de la emoción detectada
            
        Returns:
            str: Ruta del archivo guardado
        """
        try:
            emocion = emocion_info.get('emotion', 'Unknown')
            confianza = emocion_info.get('confidence', 0.0)
            quality_score = emocion_info.get('quality_score', 0.0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Crear nombre descriptivo
            filename = (f"rostro_f{frame_id:06d}_r{face_id:02d}_{emocion}_"
                        f"conf{confianza:.2f}_qual{quality_score:.2f}_{timestamp}.jpg")
            
            filepath = os.path.join(self.faces_dir, filename)
            
            # Guardar con alta calidad
            cv2.imwrite(filepath, rostro_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error guardando rostro individual: {e}")
            return ""

    def analizar_video(self, video_path: str, intervalo_ms: int = 1000, 
                         guardar_frames: bool = True) -> List[Dict]:
        """
        Analiza video completo con detección emocional y guardado automático.
        
        Args:
            video_path (str): Ruta del archivo de video
            intervalo_ms (int): Intervalo entre análisis en milisegundos
            guardar_frames (bool): Si guardar frames y rostros
            
        Returns:
            List[Dict]: Resultados del análisis por frame
        """
        try:
            # Abrir video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"No se pudo abrir el video: {video_path}")
            
            # Obtener propiedades del video
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calcular frames a procesar
            skip_frames = max(1, int((intervalo_ms / 1000) * fps))
            
            self.logger.info(f"Analizando video: {video_path}")
            self.logger.info(f"FPS: {fps}, Duración: {duration:.2f}s, Total frames: {total_frames}")
            self.logger.info(f"Procesando cada {skip_frames} frames")
            
            frame_id = 0
            resultados = []
            timestamp_session = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame según intervalo
                if frame_id % skip_frames == 0:
                    timestamp_frame = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    
                    # Detectar rostros
                    rostros = self.detectar_rostros(frame)
                    self.session_metrics['faces_detected'] += len(rostros)
                    
                    # Analizar emociones en cada rostro
                    emociones = []
                    rostros_info = []
                    
                    for face_idx, (x, y, w, h) in enumerate(rostros):
                        # Extraer rostro
                        rostro_img = frame[y:y+h, x:x+w]
                        
                        # Analizar emoción
                        emocion_result = self.analizar_emocion(rostro_img, frame_id, face_idx)
                        
                        # Agregar coordenadas para referencia
                        emocion_result['bbox'] = (x, y, w, h)
                        emocion_result['area'] = w * h
                        
                        emociones.append(emocion_result)
                        rostros_info.append(emocion_result)
                        
                        # Guardar rostro individual si está habilitado
                        if guardar_frames:
                            rostro_path = self.guardar_rostro_individual(
                                rostro_img, frame_id, face_idx, emocion_result
                            )
                            emocion_result['saved_face_path'] = rostro_path
                    
                    # Guardar frame completo si está habilitado
                    frame_path = ""
                    if guardar_frames and rostros_info:
                        frame_path = self.guardar_frame_completo(
                            frame, frame_id, rostros_info, timestamp_frame
                        )
                    
                    # Crear resultado del frame
                    frame_result = {
                        'frame_id': frame_id,
                        'timestamp': timestamp_frame,
                        'tiempo_video': frame_id / fps if fps > 0 else 0,
                        'num_faces': len(rostros),
                        'emociones': emociones,
                        'frame_path': frame_path,
                        'calidad_promedio': np.mean([e.get('quality_score', 0) for e in emociones]) if emociones else 0
                    }
                    
                    resultados.append(frame_result)
                    
                    # Log progreso cada 10 frames procesados
                    if len(resultados) % 10 == 0:
                        progreso = (frame_id / total_frames) * 100 if total_frames > 0 else 0
                        self.logger.info(f"Progreso: {progreso:.1f}% - Frames procesados: {len(resultados)}")
                
                frame_id += 1
                self.session_metrics['frames_processed'] += 1
            
            # Cerrar video
            cap.release()
            
            # Generar reporte de sesión
            if resultados:
                self.generar_reporte_sesion(resultados, video_path, timestamp_session)
            
            self.logger.info(f"Análisis completado. Frames procesados: {len(resultados)}")
            
            return resultados
            
        except Exception as e:
            self.logger.error(f"Error analizando video: {e}")
            if 'cap' in locals():
                cap.release()
            return []

    def generar_reporte_sesion(self, resultados: List[Dict], video_path: str, timestamp: str):
        """
        Genera reporte detallado de la sesión de análisis.
        
        Args:
            resultados (List[Dict]): Resultados del análisis
            video_path (str): Ruta del video analizado
            timestamp (str): Timestamp de la sesión
        """
        try:
            # Calcular estadísticas
            total_frames = len(resultados)
            total_faces = sum(r['num_faces'] for r in resultados)
            
            # Contar emociones
            emotion_counts = {}
            confidence_scores = []
            quality_scores = []
            
            for frame_result in resultados:
                for emocion in frame_result['emociones']:
                    emotion = emocion.get('emotion', 'Unknown')
                    confidence = emocion.get('confidence', 0.0)
                    quality = emocion.get('quality_score', 0.0)
                    
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    confidence_scores.append(confidence)
                    quality_scores.append(quality)
            
            # Crear reporte
            reporte = {
                "sesion_info": {
                    "timestamp": timestamp,
                    "video_analizado": os.path.basename(video_path),
                    "duracion_analisis": str(datetime.now() - self.session_metrics['session_start']),
                },
                "estadisticas": {
                    "frames_analizados": total_frames,
                    "rostros_detectados": total_faces,
                    "emociones_analizadas": len(confidence_scores),
                    "detecciones_alta_confianza": self.session_metrics['high_confidence_detections'],
                },
                "distribucion_emociones": emotion_counts,
                "metricas_calidad": {
                    "confianza_promedio": np.mean(confidence_scores) if confidence_scores else 0,
                    "calidad_promedio": np.mean(quality_scores) if quality_scores else 0,
                    "confianza_maxima": max(confidence_scores) if confidence_scores else 0,
                    "confianza_minima": min(confidence_scores) if confidence_scores else 0,
                },
                "archivos_generados": {
                    "directorio_sesion": self.session_dir,
                    "directorio_frames": self.frames_dir,
                    "directorio_rostros": self.faces_dir,
                }
            }
            
            # Guardar reporte en JSON
            import json
            reporte_path = os.path.join(self.reports_dir, f"reporte_sesion_{timestamp}.json")
            with open(reporte_path, 'w', encoding='utf-8') as f:
                json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Reporte de sesión guardado en: {reporte_path}")
            
        except Exception as e:
            self.logger.error(f"Error generando reporte de sesión: {e}")

    def obtener_estadisticas_sesion(self) -> Dict:
        """
        Obtiene estadísticas actuales de la sesión.
        
        Returns:
            Dict: Estadísticas de la sesión
        """
        duracion_sesion = datetime.now() - self.session_metrics['session_start']
        
        return {
            **self.session_metrics,
            "duracion_sesion": str(duracion_sesion),
            "directorio_sesion": self.session_dir
        }

    def configurar_parametros_deteccion(self, **kwargs):
        """
        Configura parámetros de detección facial.
        
        Args:
            **kwargs: Parámetros a configurar (scaleFactor, minNeighbors, etc.)
        """
        for key, value in kwargs.items():
            if key in self.detection_params:
                self.detection_params[key] = value
                self.logger.info(f"Parámetro {key} actualizado a: {value}")
            elif key in self.quality_thresholds:
                self.quality_thresholds[key] = value
                self.logger.info(f"Umbral {key} actualizado a: {value}")

    def reiniciar_metricas(self):
        """Reinicia las métricas de sesión."""
        self.session_metrics = {
            'frames_processed': 0,
            'faces_detected': 0,
            'emotions_analyzed': 0,
            'high_confidence_detections': 0,
            'session_start': datetime.now()
        }
        self.logger.info("Métricas de sesión reiniciadas")



