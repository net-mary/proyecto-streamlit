import os
import numpy as np
import cv2
import logging
from typing import Tuple, List, Dict, Optional
from datetime import datetime

# Importaciones de Keras/TensorFlow con manejo de errores
try:
    from keras.models import load_model
    from keras.optimizers import Adam
    import tensorflow as tf
    
    # Configurar TensorFlow para usar menos verbosidad
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
except ImportError as e:
    print(f"Error importando TensorFlow/Keras: {e}")
    print("Instale con: pip install tensorflow keras")
    raise

class EmotionEnsemble:
    """
    Ensemble avanzado de modelos de reconocimiento emocional facial.
    Combina múltiples modelos pre-entrenados para mayor precisión y robustez.
    """
    
    def __init__(self, models_dir: str = "./models"):
        """
        Inicializa el ensemble de modelos emocionales.
        
        Args:
            models_dir (str): Directorio que contiene los modelos pre-entrenados
        """
        self.models_dir = models_dir
        self.models = []
        self.input_shapes = []
        self.model_weights = []  # Pesos para el ensemble
        self.model_info = []  # Información de cada modelo
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Definir emociones estándar (FER-2013)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Mapeo a emociones en español para reportes
        self.emotions_spanish = {
            'Angry': 'Enojado',
            'Disgust': 'Disgusto', 
            'Fear': 'Miedo',
            'Happy': 'Feliz',
            'Sad': 'Triste',
            'Surprise': 'Sorpresa',
            'Neutral': 'Neutral'
        }
        
        # Configuración de modelos disponibles
        self.available_models = {
            "FER_model.h5": {
                "weight": 0.4,
                "description": "Modelo FER principal",
                "expected_input": (48, 48, 1)
            },
            "fer2013_mini_XCEPTION.99-0.65.hdf5": {
                "weight": 0.6,
                "description": "Modelo mini-XCEPTION optimizado",
                "expected_input": (64, 64, 1)
            },
            "emotion_model_v2.h5": {
                "weight": 0.3,
                "description": "Modelo de emociones v2",
                "expected_input": (48, 48, 1)
            }
        }
        
        # Inicializar cascade para detección facial como backup
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            self.logger.warning(f"No se pudo cargar cascade facial: {e}")
            self.face_cascade = None
        
        # Cargar modelos disponibles
        self.load_models()
        
        # Métricas de rendimiento
        self.performance_metrics = {
            'predictions_made': 0,
            'high_confidence_predictions': 0,
            'processing_times': [],
            'model_errors': 0,
            'session_start': datetime.now()
        }

    def load_models(self):
        """Carga todos los modelos disponibles en el directorio."""
        
        if not os.path.exists(self.models_dir):
            self.logger.warning(f"Directorio de modelos no encontrado: {self.models_dir}")
            self.logger.info("Creando modelo de fallback simple...")
            self._create_fallback_model()
            return
        
        loaded_count = 0
        
        for model_filename, model_config in self.available_models.items():
            model_path = os.path.join(self.models_dir, model_filename)
            
            if os.path.exists(model_path):
                try:
                    self.logger.info(f"Cargando modelo: {model_filename}")
                    
                    # Cargar modelo
                    model = load_model(model_path, compile=False)
                    
                    # Recompilar con configuración optimizada
                    model.compile(
                        optimizer=Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Verificar forma de entrada
                    input_shape = model.input_shape[1:]  # Remover batch dimension
                    expected_shape = model_config["expected_input"]
                    
                    if input_shape != expected_shape:
                        self.logger.warning(
                            f"Forma de entrada inesperada en {model_filename}: "
                            f"esperada {expected_shape}, obtenida {input_shape}"
                        )
                    
                    # Agregar a la lista de modelos
                    self.models.append(model)
                    self.input_shapes.append(input_shape)
                    self.model_weights.append(model_config["weight"])
                    self.model_info.append({
                        "filename": model_filename,
                        "description": model_config["description"],
                        "input_shape": input_shape,
                        "weight": model_config["weight"],
                        "loaded_at": datetime.now().isoformat()
                    })
                    
                    loaded_count += 1
                    self.logger.info(f"✓ Modelo cargado: {model_config['description']}")
                    
                except Exception as e:
                    self.logger.error(f"Error cargando {model_filename}: {e}")
                    continue
            else:
                self.logger.info(f"Modelo no encontrado: {model_filename}")
        
        if loaded_count == 0:
            self.logger.warning("No se cargaron modelos. Creando modelo de fallback...")
            self._create_fallback_model()
        else:
            # Normalizar pesos
            self._normalize_weights()
            self.logger.info(f"Ensemble inicializado con {loaded_count} modelos")

    def _create_fallback_model(self):
        """Crea un modelo de fallback simple basado en reglas cuando no hay modelos disponibles."""
        self.logger.info("Creando modelo de fallback basado en reglas...")
        
        # Agregar información del modelo de fallback
        self.model_info.append({
            "filename": "fallback_rule_based",
            "description": "Modelo de fallback basado en reglas",
            "input_shape": (48, 48, 1),
            "weight": 1.0,
            "loaded_at": datetime.now().isoformat(),
            "type": "rule_based"
        })
        
        self.fallback_mode = True

    def _normalize_weights(self):
        """Normaliza los pesos del ensemble para que sumen 1."""
        if self.model_weights:
            total_weight = sum(self.model_weights)
            self.model_weights = [w / total_weight for w in self.model_weights]
            
            self.logger.info("Pesos normalizados del ensemble:")
            for i, (info, weight) in enumerate(zip(self.model_info, self.model_weights)):
                self.logger.info(f"  {info['description']}: {weight:.3f}")

    def preprocess_face(self, face_img: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Preprocesa imagen facial para un modelo específico.
        
        Args:
            face_img (np.ndarray): Imagen del rostro
            target_shape (Tuple[int, int, int]): Forma objetivo (height, width, channels)
            
        Returns:
            np.ndarray: Imagen preprocesada lista para predicción
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            elif len(face_img.shape) == 3 and face_img.shape[2] == 1:
                gray = face_img[:, :, 0]
            else:
                gray = face_img
            
            # Redimensionar a la forma objetivo
            height, width = target_shape[0], target_shape[1]
            resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            
            # Normalización y ecualización adaptativa
            # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized.astype(np.uint8))
            
            # Normalizar a rango [0, 1]
            face_normalized = enhanced.astype('float32') / 255.0
            
            # Agregar dimensión de canal si es necesario
            if len(target_shape) == 3 and target_shape[2] == 1:
                face_normalized = np.expand_dims(face_normalized, axis=-1)
            
            # Agregar dimensión de batch
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            return face_batch
            
        except Exception as e:
            self.logger.error(f"Error preprocesando rostro: {e}")
            # Retornar imagen básica en caso de error
            fallback = np.zeros((1, target_shape[0], target_shape[1], target_shape[2]))
            return fallback

    def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Predice emoción usando el ensemble de modelos.
        
        Args:
            face_img (np.ndarray): Imagen del rostro
            
        Returns:
            Tuple[str, float]: (emoción_predicha, confianza)
        """
        start_time = datetime.now()
        
        try:
            # Si estamos en modo fallback
            if hasattr(self, 'fallback_mode') and self.fallback_mode:
                return self._fallback_prediction(face_img)
            
            if not self.models:
                self.logger.warning("No hay modelos cargados, usando fallback")
                return self._fallback_prediction(face_img)
            
            # Recopilar predicciones de todos los modelos
            predictions = []
            valid_predictions = 0
            
            for i, (model, input_shape, weight) in enumerate(zip(self.models, self.input_shapes, self.model_weights)):
                try:
                    # Preprocesar para este modelo específico
                    processed_face = self.preprocess_face(face_img, input_shape)
                    
                    # Realizar predicción
                    pred = model.predict(processed_face, verbose=0)[0]
                    
                    # Aplicar peso del modelo
                    weighted_pred = pred * weight
                    predictions.append(weighted_pred)
                    valid_predictions += 1
                    
                except Exception as e:
                    self.logger.error(f"Error en modelo {i}: {e}")
                    self.performance_metrics['model_errors'] += 1
                    continue
            
            if valid_predictions == 0:
                self.logger.warning("Ningún modelo pudo procesar la imagen, usando fallback")
                return self._fallback_prediction(face_img)
            
            # Combinar predicciones (promedio ponderado)
            if valid_predictions == 1:
                avg_prediction = predictions[0]
            else:
                avg_prediction = np.mean(predictions, axis=0)
            
            # Aplicar suavizado para evitar sobreconfianza
            avg_prediction = self._apply_confidence_smoothing(avg_prediction)
            
            # Obtener emoción con mayor probabilidad
            emotion_idx = np.argmax(avg_prediction)
            emotion = self.emotions[emotion_idx]
            confidence = float(avg_prediction[emotion_idx])
            
            # Actualizar métricas de rendimiento
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['processing_times'].append(processing_time)
            self.performance_metrics['predictions_made'] += 1
            
            if confidence > 0.7:
                self.performance_metrics['high_confidence_predictions'] += 1
            
            return emotion, confidence
            
        except Exception as e:
            self.logger.error(f"Error en predicción ensemble: {e}")
            self.performance_metrics['model_errors'] += 1
            return self._fallback_prediction(face_img)

    def _apply_confidence_smoothing(self, predictions: np.ndarray, 
                                   smoothing_factor: float = 0.1) -> np.ndarray:
        """
        Aplica suavizado a las predicciones para evitar sobreconfianza.
        
        Args:
            predictions (np.ndarray): Predicciones originales
            smoothing_factor (float): Factor de suavizado
            
        Returns:
            np.ndarray: Predicciones suavizadas
        """
        # Aplicar suavizado de Laplace
        uniform_dist = np.ones(len(predictions)) / len(predictions)
        smoothed = (1 - smoothing_factor) * predictions + smoothing_factor * uniform_dist
        
        # Renormalizar para asegurar que sumen 1
        smoothed = smoothed / np.sum(smoothed)
        
        return smoothed

    def _fallback_prediction(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Predicción de fallback basada en análisis simple de la imagen.
        
        Args:
            face_img (np.ndarray): Imagen del rostro
            
        Returns:
            Tuple[str, float]: (emoción_predicha, confianza_baja)
        """
        try:
            # Convertir a escala de grises
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            # Análisis básico de características
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Reglas heurísticas simples
            if mean_intensity < 80:  # Imagen muy oscura
                emotion = "Sad"
                confidence = 0.3
            elif std_intensity < 20:  # Poca variación (rostro plano)
                emotion = "Neutral" 
                confidence = 0.4
            elif mean_intensity > 180:  # Imagen muy brillante
                emotion = "Happy"
                confidence = 0.35
            else:
                # Análisis de gradientes para detectar expresiones
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                avg_gradient = np.mean(gradient_magnitude)
                
                if avg_gradient > 50:  # Muchos cambios (posible expresión intensa)
                    emotion = "Surprise"
                    confidence = 0.4
                else:
                    emotion = "Neutral"
                    confidence = 0.3
            
            self.logger.info(f"Predicción fallback: {emotion} (confianza: {confidence:.2f})")
            return emotion, confidence
            
        except Exception as e:
            self.logger.error(f"Error en predicción fallback: {e}")
            return "Neutral", 0.1

    def predict_emotions_batch(self, face_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Predice emociones para un lote de imágenes.
        
        Args:
            face_images (List[np.ndarray]): Lista de imágenes de rostros
            
        Returns:
            List[Tuple[str, float]]: Lista de (emoción, confianza) para cada imagen
        """
        results = []
        
        for i, face_img in enumerate(face_images):
            try:
                emotion, confidence = self.predict_emotion(face_img)
                results.append((emotion, confidence))
            except Exception as e:
                self.logger.error(f"Error procesando imagen {i}: {e}")
                results.append(("Unknown", 0.0))
        
        return results

    def get_emotion_distribution(self, face_img: np.ndarray) -> Dict[str, float]:
        """
        Obtiene la distribución completa de probabilidades emocionales.
        
        Args:
            face_img (np.ndarray): Imagen del rostro
            
        Returns:
            Dict[str, float]: Diccionario con probabilidades por emoción
        """
        try:
            if hasattr(self, 'fallback_mode') and self.fallback_mode:
                # En modo fallback, retornar distribución simple
                emotion, confidence = self._fallback_prediction(face_img)
                distribution = {emotion: confidence}
                remaining = (1.0 - confidence) / (len(self.emotions) - 1)
                for emo in self.emotions:
                    if emo not in distribution:
                        distribution[emo] = remaining
                return distribution
            
            if not self.models:
                return self.get_fallback_distribution()
            
            # Obtener predicciones de todos los modelos
            all_predictions = []
            
            for model, input_shape, weight in zip(self.models, self.input_shapes, self.model_weights):
                try:
                    processed_face = self.preprocess_face(face_img, input_shape)
                    pred = model.predict(processed_face, verbose=0)[0]
                    weighted_pred = pred * weight
                    all_predictions.append(weighted_pred)
                except Exception as e:
                    self.logger.error(f"Error obteniendo distribución: {e}")
                    continue
            
            if not all_predictions:
                return self.get_fallback_distribution()
            
            # Promedio de predicciones
            avg_predictions = np.mean(all_predictions, axis=0)
            avg_predictions = self._apply_confidence_smoothing(avg_predictions)
            
            # Crear diccionario de distribución
            distribution = {}
            for emotion, prob in zip(self.emotions, avg_predictions):
                distribution[emotion] = float(prob)
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error calculando distribución emocional: {e}")
            return self.get_fallback_distribution()

    def get_fallback_distribution(self) -> Dict[str, float]:
        """Retorna distribución emocional uniforme como fallback."""
        uniform_prob = 1.0 / len(self.emotions)
        return {emotion: uniform_prob for emotion in self.emotions}

    def get_model_info(self) -> List[Dict]:
        """
        Obtiene información detallada de los modelos cargados.
        
        Returns:
            List[Dict]: Lista con información de cada modelo
        """
        return self.model_info.copy()

    def get_performance_metrics(self) -> Dict:
        """
        Obtiene métricas de rendimiento del ensemble.
        
        Returns:
            Dict: Métricas de rendimiento
        """
        metrics = self.performance_metrics.copy()
        
        # Calcular estadísticas de tiempo de procesamiento
        if metrics['processing_times']:
            metrics['avg_processing_time'] = np.mean(metrics['processing_times'])
            metrics['max_processing_time'] = np.max(metrics['processing_times'])
            metrics['min_processing_time'] = np.min(metrics['processing_times'])
        
        # Calcular tasa de éxito
        total_attempts = metrics['predictions_made'] + metrics['model_errors']
        if total_attempts > 0:
            metrics['success_rate'] = metrics['predictions_made'] / total_attempts
            metrics['high_confidence_rate'] = metrics['high_confidence_predictions'] / metrics['predictions_made'] if metrics['predictions_made'] > 0 else 0
        
        # Duración de sesión
        metrics['session_duration'] = str(datetime.now() - metrics['session_start'])
        
        return metrics

    def reset_performance_metrics(self):
        """Reinicia las métricas de rendimiento."""
        self.performance_metrics = {
            'predictions_made': 0,
            'high_confidence_predictions': 0,
            'processing_times': [],
            'model_errors': 0,
            'session_start': datetime.now()
        }
        self.logger.info("Métricas de rendimiento reiniciadas")

    def validate_models(self) -> Dict[str, bool]:
        """
        Valida que todos los modelos funcionen correctamente.
        
        Returns:
            Dict[str, bool]: Estado de validación por modelo
        """
        validation_results = {}
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        
        for i, (model, input_shape) in enumerate(zip(self.models, self.input_shapes)):
            model_name = self.model_info[i]['filename']
            
            try:
                processed_test = self.preprocess_face(test_image, input_shape)
                prediction = model.predict(processed_test, verbose=0)
                
                # Verificar que la predicción tenga la forma correcta
                if prediction.shape[1] == len(self.emotions):
                    validation_results[model_name] = True
                    self.logger.info(f"✓ Modelo {model_name} validado correctamente")
                else:
                    validation_results[model_name] = False
                    self.logger.error(f"✗ Modelo {model_name} produce salida incorrecta")
                    
            except Exception as e:
                validation_results[model_name] = False
                self.logger.error(f"✗ Error validando modelo {model_name}: {e}")
        
        return validation_results

    def get_emotion_spanish(self, emotion_english: str) -> str:
        """
        Convierte emoción de inglés a español.
        
        Args:
            emotion_english (str): Emoción en inglés
            
        Returns:
            str: Emoción en español
        """
        return self.emotions_spanish.get(emotion_english, emotion_english)

    def analyze_emotion_confidence(self, face_img: np.ndarray) -> Dict:
        """
        Análisis detallado de confianza emocional.
        
        Args:
            face_img (np.ndarray): Imagen del rostro
            
        Returns:
            Dict: Análisis detallado de confianza
        """
        distribution = self.get_emotion_distribution(face_img)
        
        # Encontrar las top 3 emociones
        sorted_emotions = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        # Calcular métricas de confianza
        top_confidence = sorted_emotions[0][1]
        second_confidence = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0
        confidence_gap = top_confidence - second_confidence
        
        # Determinar nivel de certeza
        if top_confidence > 0.8 and confidence_gap > 0.3:
            certainty_level = "Muy Alta"
        elif top_confidence > 0.6 and confidence_gap > 0.2:
            certainty_level = "Alta"
        elif top_confidence > 0.4 and confidence_gap > 0.1:
            certainty_level = "Media"
        else:
            certainty_level = "Baja"
        
        return {
            "top_emotion": sorted_emotions[0][0],
            "top_confidence": top_confidence,
            "second_emotion": sorted_emotions[1][0] if len(sorted_emotions) > 1 else None,
            "second_confidence": second_confidence,
            "confidence_gap": confidence_gap,
            "certainty_level": certainty_level,
            "full_distribution": distribution,
            "top_3_emotions": sorted_emotions[:3]
        }