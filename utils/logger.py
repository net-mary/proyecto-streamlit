import os
import datetime

LOG_FILE_PATH = "./logs/error_log.txt"

def log_error(mensaje):
    try:
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{fecha}] ERROR: {mensaje}\n")
    except Exception as e:
        print(f"No se pudo escribir el log: {e}")

