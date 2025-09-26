# main.py

from frontend.interface import main_interface
from utils.logger import log_error

def iniciar_aplicacion():
    try:
        # Ejecuta la interfaz Streamlit principal
        main_interface()
    except Exception as e:
        log_error(f"Error en la aplicación principal: {e}")
        print(f"Error en la aplicación principal: {e}")

if __name__ == "__main__":
    iniciar_aplicacion()

