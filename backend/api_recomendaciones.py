import requests
from utils.logger import log_error

class ApiRecomendaciones:
    def __init__(self, base_url, token=None):
        self.base_url = base_url
        self.token = token

    def obtener_recomendaciones(self, diagnostico, contexto_usuario):
        url = f"{self.base_url}/recomendaciones"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        payload = {
            "diagnostico": diagnostico,
            "contexto": contexto_usuario
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            log_error("Timeout en llamada a API")
            return {"error": "Solicitud API expiró"}
        except requests.exceptions.RequestException as e:
            log_error(f"Error API: {e}")
            return {"error": str(e)}
        except ValueError:
            log_error("Respuesta inválida API")
            return {"error": "Respuesta inválida de la API"}


