import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import base64

class GeneradorInformes:
    """
    Generador avanzado de informes y visualizaciones para análisis emocional.
    Crea gráficos interactivos, reportes detallados y visualizaciones profesionales.
    """
    
    def __init__(self, carpeta_resultados: str = "./resultados"):
        """
        Inicializa el generador de informes.
        
        Args:
            carpeta_resultados (str): Carpeta donde guardar los resultados
        """
        # ⚡ CRÍTICO: Inicializar logger PRIMERO, antes de TODO
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Crear handler si no existe
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Ahora sí, el resto de la inicialización
        self.carpeta_resultados = carpeta_resultados
        self.setup_directories()
        
        # Configurar estilo de matplotlib y seaborn
        self.setup_plot_style()
        
        # Colores profesionales para emociones
        self.emotion_colors = {
            'Happy': '#2E8B57',
            'Sad': '#4682B4',
            'Angry': '#DC143C',
            'Fear': '#800080',
            'Surprise': '#FF8C00',
            'Disgust': '#8B4513',
            'Neutral': '#708090'
        }
        
        self.color_palette = ['#2E8B57', '#4682B4', '#DC143C', '#800080', 
                             '#FF8C00', '#8B4513', '#708090', '#20B2AA']
        
        self.logger.info(f"✅ GeneradorInformes inicializado correctamente")

    def setup_directories(self):
        """Configura directorios necesarios con estructura por fecha."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            
            self.daily_dir = os.path.join(self.carpeta_resultados, f"informes_{timestamp}")
            self.charts_dir = os.path.join(self.daily_dir, "graficos")
            self.reports_dir = os.path.join(self.daily_dir, "reportes")
            self.exports_dir = os.path.join(self.daily_dir, "exportaciones")
            
            for directory in [self.daily_dir, self.charts_dir, self.reports_dir, self.exports_dir]:
                os.makedirs(directory, exist_ok=True)
            
            self.logger.info(f"Directorios configurados en: {self.daily_dir}")
        except Exception as e:
            self.logger.error(f"Error configurando directorios: {e}")
            # Usar directorio por defecto
            self.charts_dir = self.carpeta_resultados
            self.reports_dir = self.carpeta_resultados
            self.exports_dir = self.carpeta_resultados

    def setup_plot_style(self):
        """Configura el estilo profesional para los gráficos."""
        try:
            plt.style.use('default')
            plt.rcParams.update({
                'figure.figsize': (12, 8),
                'font.size': 11,
                'font.family': 'sans-serif',
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.spines.top': False,
                'axes.spines.right': False
            })
            
            sns.set_palette("husl")
        except Exception as e:
            self.logger.warning(f"No se pudo configurar estilo de plots: {e}")

    def generar_histograma_emociones(self, datos_emociones: Dict, 
                                   nombre_archivo: str = None,
                                   incluir_estadisticas: bool = True) -> str:
        """
        Genera histograma avanzado de distribución de emociones.
        """
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"histograma_emociones_{timestamp}.png"
            
            if not datos_emociones or sum(datos_emociones.values()) == 0:
                self.logger.warning("No hay datos de emociones para graficar")
                return self._generar_grafico_vacio("Sin datos de emociones", nombre_archivo)
            
            emociones = list(datos_emociones.keys())
            conteos = list(datos_emociones.values())
            total = sum(conteos)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = [self.emotion_colors.get(emo, '#708090') for emo in emociones]
            bars = ax.bar(emociones, conteos, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            for bar, count in zip(bars, conteos):
                height = bar.get_height()
                percentage = (count / total) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + max(conteos) * 0.01,
                        f'{count}\n({percentage:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_title("Distribución de Emociones Detectadas", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Emociones", fontsize=12)
            ax.set_ylabel("Frecuencia", fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            promedio = total / len(emociones) if emociones else 0
            ax.axhline(y=promedio, color='red', linestyle='--', alpha=0.7, 
                       label=f'Promedio: {promedio:.1f}')
            ax.legend()
            
            plt.tight_layout()
            
            ruta_guardado = os.path.join(self.charts_dir, nombre_archivo)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            self.logger.info(f"Histograma generado: {ruta_guardado}")
            return ruta_guardado
            
        except Exception as e:
            self.logger.error(f"Error generando histograma: {e}")
            return self._generar_grafico_error(str(e), nombre_archivo if nombre_archivo else "error.png")

    def generar_timeline_emocional(self, resultados_emociones: List[Dict], 
                                  nombre_archivo: str = None) -> str:
        """Genera timeline detallado de emociones."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"timeline_emocional_{timestamp}.png"
            
            if not resultados_emociones:
                return self._generar_grafico_vacio("Sin datos de timeline", nombre_archivo)
            
            timeline_data = []
            
            for frame_result in resultados_emociones:
                frame_id = frame_result.get('frame_id', 0)
                tiempo_video = frame_result.get('tiempo_video', frame_id)
                
                for emocion_data in frame_result.get('emociones', []):
                    timeline_data.append({
                        'frame': frame_id,
                        'tiempo': tiempo_video,
                        'emocion': emocion_data.get('emotion', 'Unknown'),
                        'confianza': emocion_data.get('confidence', 0.0),
                        'face_id': emocion_data.get('face_id', 0)
                    })
            
            if not timeline_data:
                return self._generar_grafico_vacio("Sin datos de emociones", nombre_archivo)
            
            df = pd.DataFrame(timeline_data)
            
            fig, ax = plt.subplots(figsize=(16, 8))
            
            for emocion in df['emocion'].unique():
                datos_emocion = df[df['emocion'] == emocion]
                color = self.emotion_colors.get(emocion, '#708090')
                ax.scatter(datos_emocion['tiempo'], datos_emocion['confianza'], 
                           label=emocion, color=color, alpha=0.7, s=60)
            
            ax.set_xlabel('Tiempo (segundos)')
            ax.set_ylabel('Nivel de Confianza')
            ax.set_title('Evolución Temporal de Emociones', fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            ruta_guardado = os.path.join(self.charts_dir, nombre_archivo)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            self.logger.info(f"Timeline emocional generado: {ruta_guardado}")
            return ruta_guardado
            
        except Exception as e:
            self.logger.error(f"Error generando timeline: {e}")
            return self._generar_grafico_error(str(e), nombre_archivo)

    def generar_reporte_completo(self, datos_analisis: Dict, 
                               info_personal: Dict = None,
                               nombre_archivo: str = None) -> str:
        """Genera reporte completo en formato texto."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"reporte_completo_{timestamp}.txt"
            
            ruta_archivo = os.path.join(self.reports_dir, nombre_archivo)
            
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("REPORTE DE ANÁLISIS EMOCIONAL MULTIMODAL\n")
                f.write("Sistema de Análisis para Niños con Discapacidad\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("INFORMACIÓN DE LA SESIÓN\n")
                f.write("-" * 40 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Versión del sistema: 2.0\n\n")
                
                if info_personal:
                    f.write("INFORMACIÓN DEL PARTICIPANTE\n")
                    f.write("-" * 40 + "\n")
                    for key, value in info_personal.items():
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"Reporte completo generado: {ruta_archivo}")
            return ruta_archivo
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            return ""

    def generar_dashboard_visual(self, datos_analisis: Dict, 
                               nombre_archivo: str = None) -> str:
        """Genera dashboard visual completo."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"dashboard_visual_{timestamp}.png"
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('DASHBOARD DE ANÁLISIS EMOCIONAL', fontsize=20, fontweight='bold')
            
            plt.tight_layout()
            
            ruta_guardado = os.path.join(self.charts_dir, nombre_archivo)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Dashboard visual generado: {ruta_guardado}")
            return ruta_guardado
            
        except Exception as e:
            self.logger.error(f"Error generando dashboard: {e}")
            return self._generar_grafico_error(str(e), nombre_archivo)

    def exportar_datos_csv(self, datos_analisis: Dict, nombre_archivo: str = None) -> str:
        """Exporta datos a formato CSV."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"datos_analisis_{timestamp}.csv"
            
            csv_data = []
            
            for frame_result in datos_analisis.get('emociones', []):
                frame_id = frame_result.get('frame_id', 0)
                tiempo_video = frame_result.get('tiempo_video', 0)
                
                for emocion_data in frame_result.get('emociones', []):
                    csv_data.append({
                        'frame_id': frame_id,
                        'tiempo_segundos': tiempo_video,
                        'emocion': emocion_data.get('emotion', 'Unknown'),
                        'confianza': emocion_data.get('confidence', 0.0),
                        'face_id': emocion_data.get('face_id', 0)
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                ruta_csv = os.path.join(self.exports_dir, nombre_archivo)
                df.to_csv(ruta_csv, index=False, encoding='utf-8')
                
                self.logger.info(f"Datos exportados a CSV: {ruta_csv}")
                return ruta_csv
            else:
                self.logger.warning("No hay datos para exportar a CSV")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error exportando a CSV: {e}")
            return ""

    def exportar_reporte_json(self, datos_analisis: Dict, 
                            info_personal: Dict = None,
                            nombre_archivo: str = None) -> str:
        """Exporta reporte en formato JSON."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"reporte_completo_{timestamp}.json"
            
            reporte_json = {
                "metadata": {
                    "fecha_generacion": datetime.now().isoformat(),
                    "version_sistema": "2.0"
                },
                "informacion_personal": info_personal if info_personal else {},
                "resultados_analisis": datos_analisis
            }
            
            ruta_json = os.path.join(self.exports_dir, nombre_archivo)
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(reporte_json, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Reporte JSON generado: {ruta_json}")
            return ruta_json
            
        except Exception as e:
            self.logger.error(f"Error exportando JSON: {e}")
            return ""

    def _generar_grafico_vacio(self, mensaje: str, nombre_archivo: str) -> str:
        """Genera gráfico con mensaje cuando no hay datos."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, mensaje, ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
            ax.set_title('Sin Datos Disponibles', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            ruta_guardado = os.path.join(self.charts_dir, nombre_archivo)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            plt.close()
            
            return ruta_guardado
        except Exception as e:
            self.logger.error(f"Error generando gráfico vacío: {e}")
            return ""

    def _generar_grafico_error(self, error_msg: str, nombre_archivo: str) -> str:
        """Genera gráfico de error."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error generando gráfico:\n{error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
            ax.set_title('Error en Visualización', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            ruta_guardado = os.path.join(self.charts_dir, f"error_{nombre_archivo}")
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            plt.close()
            
            return ruta_guardado
        except:
            return ""

