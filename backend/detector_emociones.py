import os
import cv2
from .emotion_ensemble import EmotionEnsemble

class DetectorEmociones:
    def __init__(self, cascade_path="./haarcascade_frontalface_default.xml", save_frames_path="./frames"):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.emotion_model = EmotionEnsemble()
        self.save_frames_path = save_frames_path
        os.makedirs(save_frames_path, exist_ok=True)

    def detectar_rostros(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 5)

    def analizar_emocion(self, rostro_img):
        emocion, confianza = self.emotion_model.predict_emotion(rostro_img)
        return {"emotion": emocion, "confidence": confianza}

    def analizar_video(self, video_path, intervalo_ms=1000):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        skip_frames = max(1, int((intervalo_ms/1000)*fps))
        frame_id = 0
        resultados = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % skip_frames == 0:
                rostros = self.detectar_rostros(frame)
                emociones = []
                for idx, (x, y, w, h) in enumerate(rostros):
                    rostro_img = frame[y:y+h, x:x+w]
                    emo = self.analizar_emocion(rostro_img)
                    emociones.append(emo)
                    cv2.imwrite(f"{self.save_frames_path}/frame{frame_id}_face{idx}_{emo['emotion']}.png", rostro_img)
                resultados.append({'frame': frame_id, 'num_faces': len(rostros), 'emociones': emociones})
            frame_id += 1
        cap.release()
        return resultados



