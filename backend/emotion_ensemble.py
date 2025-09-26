import os
import numpy as np
import cv2
from keras.models import load_model
from keras.optimizers import Adam

class EmotionEnsemble:
    def __init__(self, models_dir="./models"):
        self.models_dir = models_dir
        self.model_files = ["FER_model.h5", "fer2013_mini_XCEPTION.99-0.65.hdf5"]
        self.models = []
        self.input_shapes = []
        for f in self.model_files:
            model_path = os.path.join(self.models_dir, f)
            model = load_model(model_path, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            self.models.append(model)
            # Guardar tamaño esperado de entrada sin batch size
            input_shape = model.input_shape[1:]  # ej: (64,64,1)
            self.input_shapes.append(input_shape)

        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def preprocess_face(self, face_img, target_shape):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (target_shape[1], target_shape[0]))  # ancho,alto
        face = resized.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1) if len(target_shape) == 3 else face
        face = np.expand_dims(face, axis=0)  # añadir batch dimension
        return face

    def predict_emotion(self, face_img):
        preds = []
        for model, input_shape in zip(self.models, self.input_shapes):
            face = self.preprocess_face(face_img, input_shape)
            pred = model.predict(face)[0]
            preds.append(pred)
        avg_pred = np.mean(preds, axis=0)
        idx = np.argmax(avg_pred)
        emotion = self.emotions[idx]
        return emotion, float(avg_pred[idx])




