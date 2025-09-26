import matplotlib.pyplot as plt
import os
from utils.logger import log_error

class GeneradorInformes:
    def __init__(self, carpeta_resultados="./resultados"):
        self.carpeta_resultados = carpeta_resultados
        os.makedirs(self.carpeta_resultados, exist_ok=True)

    def generar_histograma_emociones(self, datos_emociones, nombre_archivo="histograma_emociones.png"):
        try:
            emociones = list(datos_emociones.keys())
            conteos = list(datos_emociones.values())

            plt.figure(figsize=(8,5))
            plt.bar(emociones, conteos, color='skyblue')
            plt.title("Distribución de emociones")
            plt.xlabel("Emociones")
            plt.ylabel("Cantidad detectada")
            plt.tight_layout()

            ruta_guardado = os.path.join(self.carpeta_resultados, nombre_archivo)
            plt.savefig(ruta_guardado)
            plt.close()
            return ruta_guardado
        except Exception as e:
            log_error(f"Error generando histograma: {e}")
            return None

