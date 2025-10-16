import os
import cv2
import numpy as np
import logging
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

try:
    from .emotion_ensemble import EmotionEnsemble
except ImportError:
    EmotionEnsemble = None
    print("⚠️ EmotionEnsemble no disponible, usando predicción simulada")

class DetectorEmociones:
    """
    Detector de emociones faciales FUNCIONAL.
    Usa EmotionEnsemble si está disponible, sino fallback simulado.
    """
    
    EMOTION_MAP = {
        'happy': 'Feliz', 'Happy': 'Feliz',
        'sad': 'Triste', 'Sad': 'Triste',
        'angry': 'Enojado', 'Angry': 'Enojado',
        'neutral': 'Neutro', 'Neutral': 'Neutro',
        'surprise': 'Sorprendido', 'Surprise': 'Sorprendido',
        'disgust': 'Disgustado', 'Disgust': 'Disgustado',
        'fear': 'Temeroso', 'Fear': 'Temeroso'
    }
    
    EMOTIONS_LIST = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'disgust', 'fear']
    EMOTION_WEIGHTS = [0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1]
    
    def __init__(self, cascade_path: str = None, save_frames_path: str = "./fotogramas_detectados", models_dir: str = "./models"):
        """Inicializa el detector de emociones."""
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Inicializando DetectorEmociones...")
        
        # Cargar el clasificador de rostros
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            self.logger.error(f"Error: No se pudo cargar el clasificador desde: {cascade_path}")
            raise Exception("Cascade facial falló al cargar")
        
        self.logger.info(f"Cascade facial cargado correctamente")
        
        self.save_frames_path = save_frames_path
        self.models_dir = models_dir
        self.setup_directories()
        
        # Intentar cargar EmotionEnsemble
        self.emotion_ensemble = None
        self.use_ensemble = False
        
        if EmotionEnsemble is not None:
            try:
                self.logger.info("Intentando cargar EmotionEnsemble...")
                self.emotion_ensemble = EmotionEnsemble(models_dir=models_dir)
                self.use_ensemble = True
                self.logger.info(f"✓ EmotionEnsemble cargado correctamente")
            except Exception as e:
                self.logger.warning(f"No se pudo cargar EmotionEnsemble: {e}")
                self.logger.info("Usando predicción simulada como fallback")
                self.use_ensemble = False
        else:
            self.logger.info("EmotionEnsemble no disponible, usando predicción simulada")
        
        self.detection_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 4,
            'minSize': (30, 30),
            'maxSize': (800, 800)
        }
        
        self.quality_thresholds = {
            'min_face_size': 30,
            'max_face_size': 800,
            'confidence_threshold': 0.05,
            'min_emotion_confidence': 0.05
        }
        
        self.session_metrics = {
            'frames_processed': 0,
            'frames_analyzed': 0,
            'frames_with_faces': 0,
            'faces_detected': 0,
            'emotions_analyzed': 0,
            'emotions_detected_total': 0,
            'session_start': datetime.now()
        }

    def setup_directories(self):
        """Configura directorios necesarios."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_dir = os.path.join(self.save_frames_path, f"sesion_{timestamp}")
        self.frames_dir = os.path.join(self.session_dir, "frames")
        self.faces_dir = os.path.join(self.session_dir, "rostros")
        
        for directory in [self.session_dir, self.frames_dir, self.faces_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Directorios configurados: {self.session_dir}")

    def detectar_rostros(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detecta rostros en el frame."""
        try:
            # Asegurar formato correcto
            if len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    frame_bgr = frame
            elif len(frame.shape) == 2:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros
            rostros = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.detection_params['scaleFactor'],
                minNeighbors=self.detection_params['minNeighbors'],
                minSize=self.detection_params['minSize'],
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Filtrar rostros válidos
            rostros_validos = []
            for (x, y, w, h) in rostros:
                if (self.quality_thresholds['min_face_size'] <= w <= self.quality_thresholds['max_face_size'] and
                    self.quality_thresholds['min_face_size'] <= h <= self.quality_thresholds['max_face_size']):
                    rostros_validos.append((x, y, w, h))
            
            return rostros_validos
            
        except Exception as e:
            self.logger.error(f"Error detectando rostros: {e}")
            return []

    def predecir_emocion_ensemble(self, rostro_img: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Predice emoción usando EmotionEnsemble.
        Retorna: (emoción_principal, confianza, distribución_completa)
        """
        try:
            if not self.emotion_ensemble:
                return None
            
            # Obtener distribución completa
            distribution = self.emotion_ensemble.get_emotion_distribution(rostro_img)
            
            # Encontrar emoción dominante
            dominant = max(distribution.items(), key=lambda x: x[1])
            emotion = dominant[0]  # En inglés (Happy, Sad, etc.)
            confidence = dominant[1]
            
            return emotion, confidence, distribution
            
        except Exception as e:
            self.logger.warning(f"Error en ensemble: {e}")
            return None

    def predecir_emocion_simulada(self, rostro_img: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Simula la predicción de emociones (fallback).
        """
        import random
        
        dominant_emotion = random.choices(self.EMOTIONS_LIST, weights=self.EMOTION_WEIGHTS, k=1)[0]
        confidence = random.uniform(0.7, 0.95)
        
        # Generar distribución simulada
        distribution = {}
        for e in self.EMOTIONS_LIST:
            if e == dominant_emotion:
                distribution[e] = confidence
            else:
                distribution[e] = random.uniform(0.01, confidence * 0.5)
        
        # Normalizar
        total = sum(distribution.values())
        distribution = {k: v/total for k, v in distribution.items()}
        
        return dominant_emotion, confidence, distribution

    def analizar_emocion(self, rostro_img: np.ndarray, frame_id: int, face_id: int) -> Dict:
        """Analiza emociones en un rostro."""
        try:
            # Intentar con ensemble primero
            if self.use_ensemble:
                result = self.predecir_emocion_ensemble(rostro_img)
                if result:
                    emotion, confidence, distribution = result
                else:
                    emotion, confidence, distribution = self.predecir_emocion_simulada(rostro_img)
            else:
                emotion, confidence, distribution = self.predecir_emocion_simulada(rostro_img)
            
            # Emociones por encima del umbral
            min_confidence = self.quality_thresholds['min_emotion_confidence']
            emociones_detectadas = []
            
            for emocion, confianza in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                if confianza >= min_confidence:
                    # Convertir a minúsculas para consistencia
                    emocion_lower = emocion.lower()
                    emociones_detectadas.append({
                        "emotion": emocion_lower,
                        "emotion_es": self.EMOTION_MAP.get(emocion_lower, emocion),
                        "confidence": float(confianza),
                        "high_confidence": confianza > 0.7,
                        "frame_id": frame_id,
                        "face_id": face_id,
                        "timestamp": datetime.now().isoformat()
                    })
            
            emocion_principal = emociones_detectadas[0] if emociones_detectadas else None
            
            self.session_metrics['emotions_analyzed'] += 1
            self.session_metrics['emotions_detected_total'] += len(emociones_detectadas)
            
            resultado = {
                "emociones": emociones_detectadas,
                "emocion_principal": emocion_principal,
                "num_emociones_detectadas": len(emociones_detectadas),
                "frame_id": frame_id,
                "face_id": face_id,
                "timestamp": datetime.now().isoformat(),
                "image_size": rostro_img.shape[:2] if len(rostro_img.shape) >= 2 else None,
                "distribucion_completa": distribution
            }
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Error analizando emoción: {e}")
            import traceback
            traceback.print_exc()
            return {
                "emociones": [],
                "emocion_principal": None,
                "num_emociones_detectadas": 0,
                "error": str(e),
                "frame_id": frame_id,
                "face_id": face_id
            }

    def guardar_frame_completo(self, frame: np.ndarray, frame_id: int, 
                                 rostros_info: List[Dict], timestamp: str) -> str:
        """Guarda frame con rostros marcados."""
        try:
            frame_marked = frame.copy()
            
            for i, info in enumerate(rostros_info):
                if 'bbox' in info:
                    x, y, w, h = info['bbox']
                    
                    emocion_principal = info.get('emocion_principal')
                    if emocion_principal:
                        emocion = emocion_principal.get('emotion_es', 'Unknown')
                        confianza = emocion_principal.get('confidence', 0.0)
                    else:
                        emocion = 'Unknown'
                        confianza = 0.0
                    
                    # Color según confianza
                    if confianza > 0.7:
                        color = (0, 255, 0)
                    elif confianza > 0.4:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                    
                    cv2.rectangle(frame_marked, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"{emocion}: {confianza:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    cv2.rectangle(frame_marked, (x, y - label_size[1] - 10), 
                                     (x + label_size[0], y), color, -1)
                    
                    cv2.putText(frame_marked, label, (x, y - 5),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            filename = f"frame_{frame_id:06d}_{timestamp}.jpg"
            filepath = os.path.join(self.frames_dir, filename)
            cv2.imwrite(filepath, frame_marked, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error guardando frame: {e}")
            return ""

    def analizar_video(self, video_path: str, intervalo_ms: int = 1000, 
                         guardar_frames: bool = True) -> List[Dict]:
        """Analiza video completo."""
        try:
            # Abrir video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception(f"No se pudo abrir el video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Cálculo de frames a saltar
            skip_frames = max(1, int((intervalo_ms / 1000) * fps))
            
            self.logger.info(f"Video: {video_path}")
            self.logger.info(f"FPS: {fps}, Duración: {duration:.2f}s, Total: {total_frames}")
            self.logger.info(f"Procesando cada {skip_frames} frames (~{intervalo_ms}ms)")
            self.logger.info(f"Modo: {'Ensemble' if self.use_ensemble else 'Simulado'}")
            
            frame_id = 0
            resultados = []
            timestamp_session = datetime.now().strftime("%Y%m%d_%H%M%S")
            frames_analizados = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.session_metrics['frames_processed'] += 1
                
                # Analizar solo cada N frames
                if frame_id % skip_frames == 0:
                    timestamp_frame = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frames_analizados += 1
                    self.session_metrics['frames_analyzed'] += 1
                    
                    # Detectar rostros
                    rostros = self.detectar_rostros(frame)
                    
                    if len(rostros) > 0:
                        self.session_metrics['frames_with_faces'] += 1
                        self.session_metrics['faces_detected'] += len(rostros)
                        self.logger.info(f"Frame {frame_id}: {len(rostros)} rostro(s) detectado(s)")
                    
                    rostros_info = []
                    
                    # Analizar cada rostro
                    for face_idx, (x, y, w, h) in enumerate(rostros):
                        # Extraer rostro con márgen
                        margen = int(min(w, h) * 0.1)
                        x_start = max(0, x - margen)
                        y_start = max(0, y - margen)
                        x_end = min(frame.shape[1], x + w + margen)
                        y_end = min(frame.shape[0], y + h + margen)
                        
                        rostro_img = frame[y_start:y_end, x_start:x_end]
                        
                        if rostro_img.shape[0] < 10 or rostro_img.shape[1] < 10:
                            continue
                        
                        # Analizar emoción
                        emocion_result = self.analizar_emocion(rostro_img, frame_id, face_idx)
                        emocion_result['bbox'] = (x, y, w, h)
                        emocion_result['area'] = w * h
                        
                        rostros_info.append(emocion_result)
                        
                        emocion_principal = emocion_result.get('emocion_principal')
                        if emocion_principal:
                            self.logger.info(
                                f"  Rostro {face_idx}: {emocion_principal['emotion_es']} "
                                f"({emocion_principal['confidence']:.2f})"
                            )
                    
                    # Guardar frame si hay rostros
                    frame_path = ""
                    if guardar_frames and rostros_info:
                        frame_path = self.guardar_frame_completo(
                            frame, frame_id, rostros_info, timestamp_frame
                        )
                    
                    # Crear resultado del frame
                    if rostros_info:
                        frame_result = {
                            'frame_id': frame_id,
                            'timestamp': timestamp_frame,
                            'tiempo_video': frame_id / fps if fps > 0 else 0,
                            'num_faces': len(rostros),
                            'rostros': rostros_info,
                            'frame_path': frame_path,
                            'total_emociones_detectadas': sum(r.get('num_emociones_detectadas', 0) for r in rostros_info)
                        }
                        
                        resultados.append(frame_result)
                    
                    if frames_analizados % 10 == 0:
                        progreso = (frame_id / total_frames) * 100 if total_frames > 0 else 0
                        self.logger.info(
                            f"Progreso: {progreso:.1f}% - Frames analizados: {frames_analizados}, "
                            f"Con rostros: {self.session_metrics['frames_with_faces']}"
                        )
                
                frame_id += 1
            
            cap.release()
            
            self.logger.info("\n" + "="*60)
            self.logger.info("ANÁLISIS COMPLETADO:")
            self.logger.info(f"  Frames totales procesados: {frame_id}")
            self.logger.info(f"  Frames analizados: {frames_analizados}")
            self.logger.info(f"  Frames con rostros: {self.session_metrics['frames_with_faces']}")
            self.logger.info(f"  Rostros detectados: {self.session_metrics['faces_detected']}")
            self.logger.info(f"  Emociones totales: {self.session_metrics['emotions_detected_total']}")
            self.logger.info(f"  Resultados con emociones: {len(resultados)}")
            self.logger.info("="*60 + "\n")
            
            if len(resultados) == 0:
                self.logger.warning("⚠️ NO SE DETECTARON EMOCIONES EN TODO EL VIDEO")
            
            return resultados
            
        except Exception as e:
            self.logger.error(f"Error analizando video: {e}")
            import traceback
            traceback.print_exc()
            if 'cap' in locals():
                cap.release()
            return []

    def obtener_estadisticas_sesion(self) -> Dict:
        """Obtiene estadísticas de sesión."""
        duracion_sesion = datetime.now() - self.session_metrics['session_start']
        
        stats = self.session_metrics.copy()
        stats.update({
            "duracion_sesion": str(duracion_sesion),
            "directorio_sesion": self.session_dir,
            "tasa_deteccion_rostros": (
                self.session_metrics['frames_with_faces'] / 
                max(self.session_metrics['frames_analyzed'], 1)
            ) * 100,
            "promedio_emociones_por_rostro": (
                self.session_metrics['emotions_detected_total'] / 
                max(self.session_metrics['faces_detected'], 1)
            ),
            "modo": "Ensemble" if self.use_ensemble else "Simulado"
        })
        
        return stats

    def configurar_parametros_deteccion(self, **kwargs):
        """Configura parámetros de detección."""
        for key, value in kwargs.items():
            if key in self.detection_params:
                self.detection_params[key] = value
                self.logger.info(f"Parámetro {key} = {value}")
            elif key in self.quality_thresholds:
                self.quality_thresholds[key] = value
                self.logger.info(f"Umbral {key} = {value}")


