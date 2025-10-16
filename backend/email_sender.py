import smtplib
import logging
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

class EmailSender:
    """Envía reportes de análisis emocional por email."""
    
    def __init__(self, smtp_server: str = "smtp.gmail.com", 
                 smtp_port: int = 587,
                 sender_email: str = None,
                 sender_password: str = None):
        """
        Inicializa el cliente de email.
        
        Args:
            smtp_server: Servidor SMTP
            smtp_port: Puerto SMTP
            sender_email: Email del remitente
            sender_password: Contraseña o token de aplicación
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.environ.get('EMAIL_USER')
        self.sender_password = sender_password or os.environ.get('EMAIL_PASSWORD')
        
        if not self.sender_email or not self.sender_password:
            self.logger.warning(
                "Email o contraseña no configurados. "
                "Configura EMAIL_USER y EMAIL_PASSWORD en variables de entorno."
            )
    
    def generar_cuerpo_html(self, info_personal: Dict = None, 
                           datos_analisis: Dict = None) -> str:
        """Genera cuerpo HTML profesional del email."""
        
        emociones_total = {}
        confianzas = []
        
        if datos_analisis:
            for frame in datos_analisis.get('emociones', []):
                for emocion in frame.get('emociones', []):
                    key = str(emocion.get('emotion', 'Unknown')).lower()
                    emociones_total[key] = emociones_total.get(key, 0) + 1
                    confianzas.append(emocion.get('confidence', 0.0))
        
        conf_promedio = np.mean(confianzas) if confianzas else 0
        
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reporte Análisis Emocional</title>
        </head>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #f5f5f5; margin: 0; padding: 20px;">
            <div style="max-width: 700px; margin: 0 auto; background-color: white; border-radius: 8px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden;">
                
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 30px; text-align: center;">
                    <h1 style="margin: 0; font-size: 24px;">📊 Análisis Emocional</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 14px;">
                        Sistema de Análisis Multimodal para Niños con Discapacidad
                    </p>
                </div>
                
                <div style="padding: 30px;">
                    
                    <h2 style="color: #2C3E50; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
                        👤 Información del Participante
                    </h2>
        """
        
        if info_personal:
            html += '<table style="width: 100%; margin: 15px 0;">'
            for key, value in info_personal.items():
                if value:
                    html += f"""
                        <tr>
                            <td style="padding: 8px; font-weight: bold; width: 40%; color: #667eea;">
                                {key.replace('_', ' ').title()}:
                            </td>
                            <td style="padding: 8px; color: #333;">
                                {value}
                            </td>
                        </tr>
                    """
            html += '</table>'
        
        total_frames = len(datos_analisis.get('emociones', [])) if datos_analisis else 0
        total_emociones = sum(emociones_total.values())
        
        html += f"""
                    <h2 style="color: #2C3E50; border-bottom: 2px solid #667eea; padding-bottom: 10px; 
                               margin-top: 25px;">
                        📈 Resumen de Resultados
                    </h2>
                    
                    <table style="width: 100%; margin: 15px 0;">
                        <tr>
                            <td style="padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                                <div style="font-size: 24px; font-weight: bold; color: #667eea;">
                                    {total_frames}
                                </div>
                                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                    Frames Analizados
                                </div>
                            </td>
                            <td style="padding: 10px; background-color: #f8f9fa; border-radius: 4px; margin-left: 10px;">
                                <div style="font-size: 24px; font-weight: bold; color: #764ba2;">
                                    {total_emociones}
                                </div>
                                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                    Emociones Detectadas
                                </div>
                            </td>
                            <td style="padding: 10px; background-color: #f8f9fa; border-radius: 4px; margin-left: 10px;">
                                <div style="font-size: 24px; font-weight: bold; color: #2ECC71;">
                                    {conf_promedio:.0%}
                                </div>
                                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                    Confianza Promedio
                                </div>
                            </td>
                        </tr>
                    </table>
                    
                    <h2 style="color: #2C3E50; border-bottom: 2px solid #667eea; padding-bottom: 10px; 
                               margin-top: 25px;">
                        😊 Distribución de Emociones
                    </h2>
        """
        
        if emociones_total:
            emotion_colors = {
                'happy': '#2ECC71',
                'sad': '#3498DB',
                'angry': '#E74C3C',
                'fear': '#9B59B6',
                'surprise': '#F39C12',
                'disgust': '#795548',
                'neutral': '#95A5A6'
            }
            
            total_emo = sum(emociones_total.values())
            for emocion in sorted(emociones_total.keys(), key=lambda x: emociones_total[x], reverse=True):
                count = emociones_total[emocion]
                percentage = (count / total_emo) * 100
                color = emotion_colors.get(emocion, '#95A5A6')
                
                html += f"""
                    <div style="margin: 12px 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-weight: bold; color: #333;">{emocion.title()}</span>
                            <span style="color: #666;">{percentage:.1f}% ({count})</span>
                        </div>
                        <div style="background-color: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden;">
                            <div style="background-color: {color}; height: 100%; width: {percentage}%; 
                                        transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                """
        
        html += """
                    <h2 style="color: #2C3E50; border-bottom: 2px solid #667eea; padding-bottom: 10px; 
                               margin-top: 25px;">
                        📎 Archivos Adjuntos
                    </h2>
                    <p style="color: #666; font-size: 13px; line-height: 1.8;">
                        ✓ Histograma de emociones (PNG)<br>
                        ✓ Análisis temporal (Heatmap - PNG)<br>
                        ✓ Análisis de confianza (PNG)<br>
                        ✓ Datos completos (CSV)<br>
                        ✓ Reporte detallado (JSON)<br>
                        ✓ Resumen ejecutivo (TXT)
                    </p>
                    
                </div>
                
                <div style="background-color: #f8f9fa; padding: 20px; text-align: center; 
                            border-top: 1px solid #e0e0e0; font-size: 12px; color: #666;">
                    <p style="margin: 5px 0;">
                        Reporte generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
                    </p>
                    <p style="margin: 5px 0; opacity: 0.8;">
                        Sistema de Análisis Emocional v2.0
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def enviar_reporte(self, destinatarios: List[str], 
                      archivos: Dict[str, str],
                      info_personal: Dict = None,
                      datos_analisis: Dict = None,
                      asunto: str = None) -> bool:
        """Envía reporte por email con todos los adjuntos."""
        
        if not self.sender_email or not self.sender_password:
            self.logger.error("Email o contraseña no configurados")
            return False
        
        try:
            mensaje = MIMEMultipart('related')
            
            nombre_participante = info_personal.get('nombre', 'Participante') if info_personal else 'Participante'
            if asunto is None:
                asunto = f"Reporte de Análisis Emocional - {nombre_participante}"
            
            mensaje['Subject'] = asunto
            mensaje['From'] = self.sender_email
            mensaje['To'] = ', '.join(destinatarios)
            
            cuerpo_html = self.generar_cuerpo_html(info_personal, datos_analisis)
            mensaje.attach(MIMEText(cuerpo_html, 'html', 'utf-8'))
            
            self.logger.info("Adjuntando archivos...")
            
            archivos_a_adjuntar = [
                ('histograma', 'image/png'),
                ('heatmap', 'image/png'),
                ('confianza', 'image/png'),
                ('html', 'text/html'),
                ('resumen', 'text/plain'),
                ('csv', 'text/csv'),
                ('json', 'application/json')
            ]
            
            for tipo, mime_type in archivos_a_adjuntar:
                ruta = archivos.get(tipo)
                if ruta and os.path.exists(ruta):
                    try:
                        with open(ruta, 'rb') as adjunto:
                            if 'image' in mime_type:
                                parte = MIMEImage(adjunto.read(), _subtype=mime_type.split('/')[1])
                            elif 'text' in mime_type:
                                parte = MIMEText(adjunto.read().decode('utf-8'), _subtype=mime_type.split('/')[1])
                            else:
                                parte = MIMEBase(*mime_type.split('/'))
                                parte.set_payload(adjunto.read())
                                encoders.encode_base64(parte)
                            
                            nombre_archivo = os.path.basename(ruta)
                            parte.add_header('Content-Disposition', 'attachment', filename=nombre_archivo)
                            mensaje.attach(parte)
                            
                            self.logger.info(f"  ✓ {nombre_archivo}")
                    except Exception as e:
                        self.logger.warning(f"  Error adjuntando {tipo}: {e}")
            
            self.logger.info(f"Conectando a {self.smtp_server}:{self.smtp_port}...")
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as servidor:
                servidor.starttls()
                self.logger.info("TLS iniciado")
                
                servidor.login(self.sender_email, self.sender_password)
                self.logger.info(f"Autenticado como {self.sender_email}")
                
                servidor.send_message(mensaje)
                self.logger.info(f"Email enviado a: {', '.join(destinatarios)}")
            
            return True
            
        except smtplib.SMTPAuthenticationError:
            self.logger.error("Error de autenticación")
            return False
        except smtplib.SMTPException as e:
            self.logger.error(f"Error SMTP: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return False
    
    def verificar_conexion(self) -> bool:
        """Verifica conexión al servidor SMTP."""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as servidor:
                servidor.starttls()
                self.logger.info(f"Conexión exitosa a {self.smtp_server}")
                return True
        except Exception as e:
            self.logger.error(f"No se pudo conectar: {e}")
            return False
    
    @staticmethod
    def generar_instrucciones_gmail():
        """Imprime instrucciones para Gmail."""
        print("""
╔════════════════════════════════════════════════════════════════╗
║         CONFIGURAR ENVÍO DE EMAIL CON GMAIL                    ║
╚════════════════════════════════════════════════════════════════╝

1. Ve a myaccount.google.com
2. Selecciona "Seguridad" 
3. Habilita "Verificación en dos pasos"
4. Busca "Contraseñas de aplicación"
5. Selecciona "Correo" y "Windows"
6. Google generará una contraseña de 16 caracteres
7. Usa esa contraseña en la aplicación
        """)