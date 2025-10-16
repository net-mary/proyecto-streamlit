import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io

class GeneradorInformes:
    """
    Generador avanzado de informes y visualizaciones profesionales.
    Genera reportes listos para email con gráficos de alta calidad.
    """
    
    def __init__(self, carpeta_resultados: str = "./resultados"):
        """Inicializa el generador de informes."""
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.carpeta_resultados = carpeta_resultados
        self.setup_directories()
        self.setup_plot_style()
        
        # Paletas de colores profesionales
        self.emotion_colors = {
            'Happy': '#2ECC71',
            'Sad': '#3498DB',
            'Angry': '#E74C3C',
            'Fear': '#9B59B6',
            'Surprise': '#F39C12',
            'Disgust': '#795548',
            'Neutral': '#95A5A6',
            'happy': '#2ECC71',
            'sad': '#3498DB',
            'angry': '#E74C3C',
            'fear': '#9B59B6',
            'surprise': '#F39C12',
            'disgust': '#795548',
            'neutral': '#95A5A6'
        }
        
        self.logger.info("GeneradorInformes inicializado correctamente")

    def setup_directories(self):
        """Configura estructura de directorios."""
        timestamp = datetime.now().strftime("%Y%m%d")
        
        self.daily_dir = os.path.join(self.carpeta_resultados, f"informes_{timestamp}")
        self.charts_dir = os.path.join(self.daily_dir, "graficos")
        self.reports_dir = os.path.join(self.daily_dir, "reportes")
        self.exports_dir = os.path.join(self.daily_dir, "exportaciones")
        
        for directory in [self.daily_dir, self.charts_dir, self.reports_dir, self.exports_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Directorios configurados: {self.daily_dir}")

    def setup_plot_style(self):
        """Configura estilo profesional para gráficos."""
        plt.rcParams.update({
            'figure.figsize': (14, 8),
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

    def generar_histograma_emociones_avanzado(self, datos_emociones: Dict, 
                                              nombre_archivo: str = None) -> str:
        """Genera histograma profesional con estadísticas."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"histograma_emociones_{timestamp}.png"
            
            if not datos_emociones or sum(datos_emociones.values()) == 0:
                return self._generar_grafico_vacio("Sin datos de emociones", nombre_archivo)
            
            emociones = list(datos_emociones.keys())
            conteos = list(datos_emociones.values())
            total = sum(conteos)
            
            # Convertir a minúsculas para consultar colores
            colors = [self.emotion_colors.get(str(emo).lower(), '#95A5A6') for emo in emociones]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Gráfico 1: Barras con porcentajes
            bars = ax1.bar(emociones, conteos, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
            
            for bar, count in zip(bars, conteos):
                height = bar.get_height()
                percentage = (count / total) * 100
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(conteos) * 0.02,
                        f'{count}\n({percentage:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax1.set_title("Distribución de Emociones (Frecuencia)", fontsize=13, fontweight='bold')
            ax1.set_xlabel("Emociones", fontsize=11)
            ax1.set_ylabel("Frecuencia", fontsize=11)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            
            promedio = total / len(emociones)
            ax1.axhline(y=promedio, color='red', linestyle='--', alpha=0.7, linewidth=2,
                       label=f'Promedio: {promedio:.1f}')
            ax1.legend()
            
            # Gráfico 2: Pastel
            percentajes = [(count / total) * 100 for count in conteos]
            wedges, texts, autotexts = ax2.pie(conteos, labels=emociones, autopct='%1.1f%%',
                                                colors=colors, startangle=90,
                                                textprops={'fontsize': 10})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            ax2.set_title("Distribución Porcentual de Emociones", fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            
            ruta_guardado = os.path.join(self.charts_dir, nombre_archivo)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"Histograma avanzado generado: {ruta_guardado}")
            return ruta_guardado
            
        except Exception as e:
            self.logger.error(f"Error generando histograma: {e}")
            return ""

    def generar_heatmap_temporal(self, resultados_emociones: List[Dict], 
                                 nombre_archivo: str = None) -> str:
        """Genera heatmap de emociones en el tiempo."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"heatmap_temporal_{timestamp}.png"
            
            if not resultados_emociones:
                return self._generar_grafico_vacio("Sin datos de timeline", nombre_archivo)
            
            # Preparar datos
            timeline_data = []
            for frame_result in resultados_emociones:
                for emocion_data in frame_result.get('emociones', []):
                    timeline_data.append({
                        'frame': frame_result.get('frame_id', 0),
                        'tiempo': frame_result.get('tiempo_video', 0),
                        'emocion': str(emocion_data.get('emotion', 'Unknown')).lower(),
                        'confianza': emocion_data.get('confidence', 0.0)
                    })
            
            if not timeline_data:
                return self._generar_grafico_vacio("Sin emociones detectadas", nombre_archivo)
            
            df = pd.DataFrame(timeline_data)
            
            # Crear tabla pivote
            pivot_table = df.pivot_table(
                values='confianza',
                index='emocion',
                columns='frame',
                aggfunc='mean',
                fill_value=0
            )
            
            fig, ax = plt.subplots(figsize=(16, 6))
            
            sns.heatmap(pivot_table, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Confianza'})
            
            ax.set_title("Heatmap Temporal de Emociones", fontsize=14, fontweight='bold')
            ax.set_xlabel("Frame", fontsize=11)
            ax.set_ylabel("Emoción", fontsize=11)
            
            plt.tight_layout()
            
            ruta_guardado = os.path.join(self.charts_dir, nombre_archivo)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"Heatmap temporal generado: {ruta_guardado}")
            return ruta_guardado
            
        except Exception as e:
            self.logger.error(f"Error generando heatmap: {e}")
            return ""

    def generar_analisis_confianza(self, resultados_emociones: List[Dict],
                                   nombre_archivo: str = None) -> str:
        """Genera gráfico de análisis de confianza."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"analisis_confianza_{timestamp}.png"
            
            timeline_data = []
            for frame_result in resultados_emociones:
                for emocion_data in frame_result.get('emociones', []):
                    timeline_data.append({
                        'emocion': str(emocion_data.get('emotion', 'Unknown')).lower(),
                        'confianza': emocion_data.get('confidence', 0.0)
                    })
            
            if not timeline_data:
                return self._generar_grafico_vacio("Sin datos de confianza", nombre_archivo)
            
            df = pd.DataFrame(timeline_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Análisis Estadístico de Confianza', fontsize=16, fontweight='bold')
            
            # Box plot
            emociones_unicas = df['emocion'].unique()
            colors_list = [self.emotion_colors.get(e, '#95A5A6') for e in emociones_unicas]
            
            bp = axes[0, 0].boxplot([df[df['emocion'] == e]['confianza'].values for e in emociones_unicas],
                                     labels=emociones_unicas, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
            axes[0, 0].set_title('Distribución de Confianza (Box Plot)', fontweight='bold')
            axes[0, 0].set_ylabel('Confianza')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(axis='y', alpha=0.3)
            
            # Violin plot
            parts = axes[0, 1].violinplot([df[df['emocion'] == e]['confianza'].values for e in emociones_unicas])
            axes[0, 1].set_xticks(range(1, len(emociones_unicas) + 1))
            axes[0, 1].set_xticklabels(emociones_unicas, rotation=45)
            axes[0, 1].set_title('Densidad de Confianza (Violin Plot)', fontweight='bold')
            axes[0, 1].set_ylabel('Confianza')
            axes[0, 1].grid(axis='y', alpha=0.3)
            
            # Histograma de confianza general
            axes[1, 0].hist(df['confianza'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Histograma de Confianza General', fontweight='bold')
            axes[1, 0].set_xlabel('Confianza')
            axes[1, 0].set_ylabel('Frecuencia')
            axes[1, 0].axvline(df['confianza'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["confianza"].mean():.2f}')
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # Estadísticas por emoción
            stats_text = "ESTADÍSTICAS POR EMOCIÓN\n" + "="*40 + "\n"
            for emocion in emociones_unicas:
                datos = df[df['emocion'] == emocion]['confianza']
                stats_text += f"\n{emocion.upper()}\n"
                stats_text += f"  Media: {datos.mean():.3f}\n"
                stats_text += f"  Std: {datos.std():.3f}\n"
                stats_text += f"  Min: {datos.min():.3f}\n"
                stats_text += f"  Max: {datos.max():.3f}\n"
                stats_text += f"  Conteo: {len(datos)}\n"
            
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                           fontfamily='monospace', fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            ruta_guardado = os.path.join(self.charts_dir, nombre_archivo)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"Análisis de confianza generado: {ruta_guardado}")
            return ruta_guardado
            
        except Exception as e:
            self.logger.error(f"Error generando análisis: {e}")
            return ""

    def generar_reporte_completo_html(self, datos_analisis: Dict, 
                                      info_personal: Dict = None,
                                      nombre_archivo: str = None) -> str:
        """Genera reporte HTML profesional para email."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"reporte_{timestamp}.html"
            
            # Calcular estadísticas
            emociones_total = {}
            confianza_promedio = []
            
            for frame in datos_analisis.get('emociones', []):
                for emocion in frame.get('emociones', []):
                    emo_key = str(emocion.get('emotion', 'Unknown')).lower()
                    emociones_total[emo_key] = emociones_total.get(emo_key, 0) + 1
                    confianza_promedio.append(emocion.get('confidence', 0.0))
            
            total_frames = len(datos_analisis.get('emociones', []))
            conf_avg = np.mean(confianza_promedio) if confianza_promedio else 0
            
            # Generar HTML
            html_content = f"""
            <!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Reporte de Análisis Emocional</title>
                <style>
                    * {{ margin: 0; padding: 0; }}
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background-color: #f5f5f5;
                        color: #333;
                        line-height: 1.6;
                    }}
                    .container {{
                        max-width: 900px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 40px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .header {{
                        text-align: center;
                        border-bottom: 3px solid #3498DB;
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                    }}
                    .header h1 {{
                        color: #2C3E50;
                        font-size: 28px;
                        margin-bottom: 10px;
                    }}
                    .header p {{
                        color: #7F8C8D;
                        font-size: 14px;
                    }}
                    .section {{
                        margin-bottom: 30px;
                    }}
                    .section h2 {{
                        color: #2C3E50;
                        font-size: 18px;
                        margin-bottom: 15px;
                        border-left: 4px solid #3498DB;
                        padding-left: 10px;
                    }}
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        gap: 15px;
                        margin-bottom: 20px;
                    }}
                    .stat-box {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 6px;
                        text-align: center;
                    }}
                    .stat-box.secondary {{
                        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    }}
                    .stat-value {{
                        font-size: 28px;
                        font-weight: bold;
                        margin-bottom: 5px;
                    }}
                    .stat-label {{
                        font-size: 12px;
                        opacity: 0.9;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 15px;
                    }}
                    th {{
                        background-color: #34495E;
                        color: white;
                        padding: 12px;
                        text-align: left;
                    }}
                    td {{
                        padding: 12px;
                        border-bottom: 1px solid #ECF0F1;
                    }}
                    tr:hover {{
                        background-color: #F8F9FA;
                    }}
                    .emotion-row {{
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }}
                    .emotion-bar {{
                        flex-grow: 1;
                        height: 20px;
                        border-radius: 4px;
                        background-color: #ECF0F1;
                        position: relative;
                    }}
                    .emotion-bar-fill {{
                        height: 100%;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-size: 11px;
                        font-weight: bold;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 30px;
                        padding-top: 20px;
                        border-top: 1px solid #ECF0F1;
                        color: #7F8C8D;
                        font-size: 12px;
                    }}
                    .warning {{
                        background-color: #FFF3CD;
                        border-left: 4px solid #FFC107;
                        padding: 12px;
                        margin-bottom: 15px;
                        border-radius: 4px;
                    }}
                    .success {{
                        background-color: #D4EDDA;
                        border-left: 4px solid #28A745;
                        padding: 12px;
                        margin-bottom: 15px;
                        border-radius: 4px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 Reporte de Análisis Emocional</h1>
                        <p>Sistema de Análisis Multimodal para Niños con Discapacidad</p>
                        <p style="margin-top: 10px; font-size: 13px;">{datetime.now().strftime('%d de %B de %Y - %H:%M:%S')}</p>
                    </div>
            """
            
            # Información personal
            if info_personal:
                html_content += """
                    <div class="section">
                        <h2>👤 Información del Participante</h2>
                        <table>
                            <tr>
                """
                for key, value in info_personal.items():
                    html_content += f"<th>{key.replace('_', ' ').title()}</th>"
                html_content += "</tr><tr>"
                for value in info_personal.values():
                    html_content += f"<td>{value}</td>"
                html_content += "</tr></table></div>"
            
            # Estadísticas generales
            html_content += f"""
                    <div class="section">
                        <h2>📈 Estadísticas Generales</h2>
                        <div class="stats-grid">
                            <div class="stat-box">
                                <div class="stat-value">{total_frames}</div>
                                <div class="stat-label">Frames Analizados</div>
                            </div>
                            <div class="stat-box secondary">
                                <div class="stat-value">{sum(emociones_total.values())}</div>
                                <div class="stat-label">Emociones Detectadas</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value">{conf_avg:.1%}</div>
                                <div class="stat-label">Confianza Promedio</div>
                            </div>
                            <div class="stat-box secondary">
                                <div class="stat-value">{len(emociones_total)}</div>
                                <div class="stat-label">Tipos de Emociones</div>
                            </div>
                        </div>
                    </div>
            """
            
            # Distribución de emociones
            if emociones_total:
                html_content += """
                    <div class="section">
                        <h2>😊 Distribución de Emociones</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Emoción</th>
                                    <th>Frecuencia</th>
                                    <th>Porcentaje</th>
                                </tr>
                            </thead>
                            <tbody>
                """
                
                total_emo = sum(emociones_total.values())
                for emocion in sorted(emociones_total.keys(), key=lambda x: emociones_total[x], reverse=True):
                    count = emociones_total[emocion]
                    percentage = (count / total_emo) * 100
                    color = self.emotion_colors.get(emocion, '#95A5A6')
                    
                    html_content += f"""
                        <tr>
                            <td><strong>{emocion.title()}</strong></td>
                            <td>{count}</td>
                            <td>
                                <div class="emotion-bar">
                                    <div class="emotion-bar-fill" style="width: {percentage}%; background-color: {color};">
                                        {percentage:.1f}%
                                    </div>
                                </div>
                            </td>
                        </tr>
                    """
                
                html_content += """
                            </tbody>
                        </table>
                    </div>
                """
            
            html_content += """
                    <div class="footer">
                        <p>Este reporte fue generado automáticamente por el Sistema de Análisis Emocional.</p>
                        <p>Para más información, consulte la documentación del sistema.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            ruta_guardado = os.path.join(self.reports_dir, nombre_archivo)
            with open(ruta_guardado, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Reporte HTML generado: {ruta_guardado}")
            return ruta_guardado
            
        except Exception as e:
            self.logger.error(f"Error generando reporte HTML: {e}")
            return ""

    def exportar_datos_csv(self, datos_analisis: Dict, nombre_archivo: str = None) -> str:
        """Exporta datos a CSV."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"datos_{timestamp}.csv"
            
            csv_data = []
            for frame_result in datos_analisis.get('emociones', []):
                frame_id = frame_result.get('frame_id', 0)
                tiempo = frame_result.get('tiempo_video', 0)
                
                for emocion_data in frame_result.get('emociones', []):
                    csv_data.append({
                        'frame_id': frame_id,
                        'tiempo_segundos': tiempo,
                        'emocion': emocion_data.get('emotion', 'Unknown'),
                        'confianza': emocion_data.get('confidence', 0.0),
                        'face_id': emocion_data.get('face_id', 0)
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                ruta_csv = os.path.join(self.exports_dir, nombre_archivo)
                df.to_csv(ruta_csv, index=False, encoding='utf-8')
                self.logger.info(f"CSV exportado: {ruta_csv}")
                return ruta_csv
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error exportando CSV: {e}")
            return ""

    def exportar_reporte_json(self, datos_analisis: Dict, 
                             info_personal: Dict = None,
                             nombre_archivo: str = None) -> str:
        """Exporta reporte completo en JSON."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"reporte_{timestamp}.json"
            
            reporte_json = {
                "metadata": {
                    "fecha": datetime.now().isoformat(),
                    "version": "2.0"
                },
                "informacion_personal": info_personal or {},
                "resultados": datos_analisis
            }
            
            ruta_json = os.path.join(self.exports_dir, nombre_archivo)
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(reporte_json, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"JSON exportado: {ruta_json}")
            return ruta_json
            
        except Exception as e:
            self.logger.error(f"Error exportando JSON: {e}")
            return ""

    def generar_resumen_ejecutivo(self, datos_analisis: Dict,
                                  info_personal: Dict = None,
                                  nombre_archivo: str = None) -> str:
        """Genera resumen ejecutivo en texto profesional."""
        try:
            if nombre_archivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"resumen_{timestamp}.txt"
            
            # Calcular estadísticas
            emociones_total = {}
            confianzas = []
            
            for frame in datos_analisis.get('emociones', []):
                for emocion in frame.get('emociones', []):
                    key = str(emocion.get('emotion', 'Unknown')).lower()
                    emociones_total[key] = emociones_total.get(key, 0) + 1
                    confianzas.append(emocion.get('confidence', 0.0))
            
            conf_promedio = np.mean(confianzas) if confianzas else 0
            conf_max = np.max(confianzas) if confianzas else 0
            conf_min = np.min(confianzas) if confianzas else 0
            
            # Emoción dominante
            emocion_dominante = max(emociones_total, key=emociones_total.get) if emociones_total else "Ninguna"
            emocion_dominante_pct = (emociones_total.get(emocion_dominante, 0) / sum(emociones_total.values()) * 100) if emociones_total else 0
            
            resumen = f"""
{'='*75}
RESUMEN EJECUTIVO - ANÁLISIS EMOCIONAL MULTIMODAL
{'='*75}

FECHA Y HORA: {datetime.now().strftime('%d de %B de %Y - %H:%M:%S')}
SISTEMA: Análisis Emocional para Niños con Discapacidad v2.0

{'─'*75}
INFORMACIÓN DEL PARTICIPANTE
{'─'*75}
"""
            
            if info_personal:
                for key, value in info_personal.items():
                    resumen += f"\n{key.replace('_', ' ').title()}: {value}"
            else:
                resumen += "\nNo disponible"
            
            total_frames = len(datos_analisis.get('emociones', []))
            total_emociones = sum(emociones_total.values())
            
            resumen += f"""

{'─'*75}
RESULTADOS GENERALES DEL ANÁLISIS
{'─'*75}

Frames analizados: {total_frames}
Emociones detectadas: {total_emociones}
Emoción dominante: {emocion_dominante.title()} ({emocion_dominante_pct:.1f}%)
Confianza promedio: {conf_promedio:.1%}
  - Mínima: {conf_min:.1%}
  - Máxima: {conf_max:.1%}

{'─'*75}
DISTRIBUCIÓN DE EMOCIONES
{'─'*75}
"""
            
            if emociones_total:
                # Ordenar por frecuencia
                emociones_ordenadas = sorted(emociones_total.items(), key=lambda x: x[1], reverse=True)
                
                for emocion, count in emociones_ordenadas:
                    pct = (count / total_emociones) * 100
                    barra = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                    resumen += f"\n{emocion.title():15} | {barra} {pct:5.1f}% ({count:4d})"
            
            resumen += f"""

{'─'*75}
ANÁLISIS ESTADÍSTICO
{'─'*75}

Media de confianza: {conf_promedio:.3f}
Desv. estándar: {np.std(confianzas):.3f} (si hay datos)
Rango: {conf_min:.3f} - {conf_max:.3f}

Total de tipos de emociones detectadas: {len(emociones_total)}

{'─'*75}
RECOMENDACIONES
{'─'*75}

• Los resultados indican una detección {('exitosa' if total_emociones > 0 else 'limitada')} de emociones.
• Nivel de confianza {'alto' if conf_promedio > 0.7 else 'moderado' if conf_promedio > 0.5 else 'bajo'} en las predicciones.
• Se recomienda revisar los videos con {'mayor confianza' if conf_promedio > 0.7 else 'precaución'}.
• Consultar los gráficos generados para análisis más detallados.

{'─'*75}
ARCHIVOS GENERADOS
{'─'*75}

• Histograma de emociones (PNG)
• Heatmap temporal (PNG)
• Análisis de confianza (PNG)
• Reporte HTML (para email)
• Datos CSV (para procesamiento)
• Reporte JSON (completo)

{'─'*75}
NOTAS
{'─'*75}

Este análisis fue realizado automáticamente por el Sistema de Análisis Emocional.
Para validación clínica, consulte con profesionales especializados.
Todos los datos han sido procesados respetando privacidad y confidencialidad.

{'='*75}
FIN DEL REPORTE
{'='*75}
"""
            
            ruta_guardado = os.path.join(self.reports_dir, nombre_archivo)
            with open(ruta_guardado, 'w', encoding='utf-8') as f:
                f.write(resumen)
            
            self.logger.info(f"Resumen ejecutivo generado: {ruta_guardado}")
            return ruta_guardado
            
        except Exception as e:
            self.logger.error(f"Error generando resumen: {e}")
            return ""

    def generar_todos_informes(self, datos_analisis: Dict, 
                               info_personal: Dict = None) -> Dict[str, str]:
        """Genera todos los informes de una vez. Ideal para emails."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            resultado = {
                'histograma': self.generar_histograma_emociones_avanzado(
                    self._extraer_conteo_emociones(datos_analisis.get('emociones', [])),
                    f"histograma_{timestamp}.png"
                ),
                'heatmap': self.generar_heatmap_temporal(
                    datos_analisis.get('emociones', []),
                    f"heatmap_{timestamp}.png"
                ),
                'confianza': self.generar_analisis_confianza(
                    datos_analisis.get('emociones', []),
                    f"confianza_{timestamp}.png"
                ),
                'html': self.generar_reporte_completo_html(
                    datos_analisis,
                    info_personal,
                    f"reporte_{timestamp}.html"
                ),
                'resumen': self.generar_resumen_ejecutivo(
                    datos_analisis,
                    info_personal,
                    f"resumen_{timestamp}.txt"
                ),
                'csv': self.exportar_datos_csv(
                    datos_analisis,
                    f"datos_{timestamp}.csv"
                ),
                'json': self.exportar_reporte_json(
                    datos_analisis,
                    info_personal,
                    f"reporte_{timestamp}.json"
                )
            }
            
            self.logger.info(f"Todos los informes generados correctamente")
            return resultado
            
        except Exception as e:
            self.logger.error(f"Error generando informes: {e}")
            return {}

    def _extraer_conteo_emociones(self, resultados_emociones: List[Dict]) -> Dict:
        """Extrae conteo de emociones."""
        conteo = {}
        for frame in resultados_emociones:
            for emocion in frame.get('emociones', []):
                key = str(emocion.get('emotion', 'Unknown')).lower()
                conteo[key] = conteo.get(key, 0) + 1
        return conteo

    def _generar_grafico_vacio(self, mensaje: str, nombre_archivo: str) -> str:
        """Genera gráfico vacío."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, mensaje, ha='center', va='center',
                   transform=ax.transAxes, fontsize=16,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            ax.axis('off')
            
            ruta = os.path.join(self.charts_dir, nombre_archivo)
            plt.savefig(ruta, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return ruta
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return ""

