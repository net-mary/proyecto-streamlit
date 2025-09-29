import os
import sys
import streamlit as st
import pandas as pd
import json
import base64
from datetime import datetime
from typing import Dict, Optional, List

# Configuración de página DEBE ser lo primero
st.set_page_config(
    page_title="Análisis Emocional Multimodal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Añadir carpeta raíz del proyecto para importar backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from backend.pipeline import PipelineAnalisisEmocional
    from frontend.components.timeline_emotions import TimelineEmotions
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.stop()

# CSS personalizado para diseño profesional
def load_custom_css():
    st.markdown("""
    <style>
    /* Importar fuente profesional */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Estilos globales */
    :root {
        --primary-color: #2e7d8b; /* Teal oscuro */
        --secondary-color: #4a9dba; /* Teal claro */
        --background-color: #f0f2f6; /* Fondo suave */
        --text-color: #1c1e21; /* Texto oscuro */
        --border-color: #cfd8dc; /* Borde gris */
        --shadow-color: rgba(0, 0, 0, 0.05);
    }
    html, body, .main, .stApp {
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
        background-color: var(--background-color);
    }

    /* Header principal */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border-color);
    }
    .main-header h1 {
        color: var(--primary-color);
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #607d8b;
        font-size: 1.1rem;
    }

    /* Cards profesionales */
    .professional-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px var(--shadow-color);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
        transition: transform 0.2s;
    }
    .professional-card:hover {
        transform: translateY(-2px);
    }
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .card-icon {
        font-size: 1.5rem;
        margin-right: 0.75rem;
        color: var(--primary-color);
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
    }

    /* Cards de métricas */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 6px var(--shadow-color);
        flex: 1;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.25rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #607d8b;
        font-weight: 500;
    }

    /* Alertas personalizadas */
    .alert-critical {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #e57373;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .alert-warning {
        background-color: #fffde7;
        color: #ffb300;
        border: 1px solid #fff176;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .alert-success {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #81c784;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Progress bar personalizada */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    }

    /* Selectbox y inputs */
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid var(--border-color);
        padding: 0.75rem;
        font-size: 1rem;
    }

    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(46, 125, 139, 0.1);
    }

    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed var(--primary-color);
        border-radius: 12px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(46, 125, 139, 0.05), rgba(74, 157, 186, 0.05));
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }

        .metric-container {
            flex-direction: column;
            align-items: center;
        }

        .professional-card {
            margin: 0.5rem 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa el estado de la sesión con valores por defecto."""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'session_history' not in st.session_state:
        st.session_state.session_history = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None

def render_header():
    """Renderiza el header principal de la aplicación."""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Análisis Emocional Multimodal</h1>
        <p>Sistema Avanzado de Análisis para Niños con Discapacidad</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Renderiza la barra lateral con configuraciones y formulario de datos."""
    st.sidebar.markdown("### ⚙️ Configuración del Análisis")

    # Sección de upload de video
    st.sidebar.markdown("#### 📹 Video a Analizar")
    video_file = st.sidebar.file_uploader(
        "Selecciona un video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Formatos soportados: MP4, AVI, MOV, MKV"
    )

    # Configuraciones técnicas
    st.sidebar.markdown("#### 🔧 Configuraciones Técnicas")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        models_dir = st.text_input(
            "Directorio de modelos",
            value="./models",
            help="Ruta donde se encuentran los modelos de IA"
        )

    with col2:
        language = st.selectbox(
            "Idioma del audio",
            options=["es-ES", "en-US", "es-MX", "es-AR"],
            help="Idioma para el reconocimiento de voz"
        )

    # Configuraciones avanzadas
    with st.sidebar.expander("⚡ Configuraciones Avanzadas"):
        intervalo_analisis = st.slider(
            "Intervalo de análisis (ms)",
            min_value=500,
            max_value=5000,
            value=1000,
            step=250,
            help="Frecuencia de análisis de frames"
        )

        umbral_confianza = st.slider(
            "Umbral de confianza",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Confianza mínima para detectar emociones"
        )

        guardar_frames = st.checkbox(
            "Guardar fotogramas detectados",
            value=True,
            help="Guarda imágenes de rostros detectados"
        )

    # Información personal del niño
    st.sidebar.markdown("#### 👶 Información del Participante")

    with st.sidebar.form("datos_personales"):
        st.markdown("**Datos del Niño:**")
        nombre = st.text_input("Nombre", placeholder="Nombre del niño")
        edad = st.number_input("Edad", min_value=1, max_value=18, value=5)
        diagnostico = st.selectbox(
            "Diagnóstico",
            options=[
                "",
                "Autismo/TEA",
                "TDAH",
                "Síndrome de Down",
                "Parálisis Cerebral",
                "Discapacidad Intelectual",
                "Trastornos del Lenguaje",
                "Otro"
            ]
        )

        if diagnostico == "Otro":
            diagnostico_otro = st.text_input("Especificar diagnóstico")
            diagnostico = diagnostico_otro if diagnostico_otro else "No especificado"

        st.markdown("**Información del Usuario:**")
        rol_usuario = st.selectbox(
            "Tu rol",
            options=["Padre/Madre", "Educador", "Terapeuta", "Investigador", "Otro"]
        )

        contexto_video = st.text_area(
            "Contexto del video",
            placeholder="Describe la situación en la que se grabó el video...",
            height=100
        )

        submit_info = st.form_submit_button("💾 Guardar Información")

        if submit_info and nombre:
            st.success("✅ Información guardada correctamente")

    # Datos personales para retornar
    datos_personales = {}
    if 'nombre' in locals():
        datos_personales = {
            "nombre": nombre if nombre else "",
            "edad": edad,
            "diagnostico": diagnostico if diagnostico else "",
            "rol_usuario": rol_usuario,
            "contexto_video": contexto_video
        }
    else:
        # Valores por defecto si no se ha renderizado el formulario
        datos_personales = {
            "nombre": "",
            "edad": 5,
            "diagnostico": "",
            "rol_usuario": "Padre/Madre",
            "contexto_video": ""
        }

    # Configuraciones personalizadas para retornar
    configuracion_personalizada = {}
    if 'intervalo_analisis' in locals():
        configuracion_personalizada = {
            "intervalo_analisis_ms": intervalo_analisis,
            "umbral_confianza": umbral_confianza,
            "guardar_frames": guardar_frames
        }
    else:
        # Valores por defecto si no se ha renderizado el expander
        configuracion_personalizada = {
            "intervalo_analisis_ms": 1000,
            "umbral_confianza": 0.5,
            "guardar_frames": True
        }

    return video_file, models_dir, language, datos_personales, configuracion_personalizada

def display_metrics_dashboard(results: Dict):
    """Muestra dashboard con métricas principales."""
    if not results or not results.get('session_info'):
        return

    st.markdown("### 📊 Dashboard de Métricas")

    # Extraer métricas
    session_info = results.get('session_info', {})
    emociones = results.get('emociones', [])
    audio_info = results.get('audio', {})

    # Calcular métricas
    total_frames = len(emociones)
    total_rostros = sum(len(frame.get('emociones', [])) for frame in emociones)
    tiempo_procesamiento = session_info.get('tiempo_procesamiento', 'N/A')
    alertas_count = len(session_info.get('alertas', []))

    # Mostrar métricas en cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_frames}</div>
            <div class="metric-label">Frames Analizados</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_rostros}</div>
            <div class="metric-label">Rostros Detectados</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        palabras = audio_info.get('palabras_totales', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{palabras}</div>
            <div class="metric-label">Palabras Detectadas</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        color_class = "metric-card" if alertas_count == 0 else "metric-card alert-critical-bg" # Se usaría una clase de fondo
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{alertas_count}</div>
            <div class="metric-label">Alertas</div>
        </div>
        """, unsafe_allow_html=True)

def display_alerts(session_info: Dict):
    """Muestra alertas del análisis."""
    alertas = session_info.get('alertas', [])

    if not alertas:
        st.markdown("""
        <div class="alert-success">
            <h4>✅ Sin Alertas Detectadas</h4>
            <p>El análisis no identificó situaciones que requieran atención inmediata.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("### ⚠️ Alertas y Recomendaciones Prioritarias")

    # Agrupar alertas por nivel
    alertas_criticas = [a for a in alertas if a.get('nivel') == 'alto']
    alertas_moderadas = [a for a in alertas if a.get('nivel') == 'medio']
    alertas_info = [a for a in alertas if a.get('nivel') == 'bajo']

    # Mostrar alertas críticas
    for alerta in alertas_criticas:
        st.markdown(f"""
        <div class="alert-critical">
            <h4>🚨 {alerta.get('mensaje', 'Alerta crítica')}</h4>
            <p><strong>Recomendación:</strong> {alerta.get('recomendacion', 'Consultar especialista')}</p>
            <small>Tipo: {alerta.get('tipo', 'General')} | Detectado: {alerta.get('timestamp', '')}</small>
        </div>
        """, unsafe_allow_html=True)

    # Mostrar alertas moderadas
    for alerta in alertas_moderadas:
        st.markdown(f"""
        <div class="alert-warning">
            <h4>⚡ {alerta.get('mensaje', 'Alerta moderada')}</h4>
            <p><strong>Recomendación:</strong> {alerta.get('recomendacion', 'Monitoreo recomendado')}</p>
            <small>Tipo: {alerta.get('tipo', 'General')}</small>
        </div>
        """, unsafe_allow_html=True)

def display_emotions_analysis(emociones: List[Dict]):
    """Muestra análisis detallado de emociones."""
    if not emociones:
        st.warning("No se detectaron emociones en el video.")
        return

    st.markdown("### 😊 Análisis de Emociones Faciales")

    # Crear dos columnas para el análisis
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-header">
            <span class="card-icon">📊</span>
            <h3 class="card-title">Timeline Emocional Interactivo</h3>
        </div>
        """, unsafe_allow_html=True)

        # Timeline emocional interactivo
        try:
            TimelineEmotions(emociones, highlights=[]) # Asumiendo que TimelineEmotions acepta estos parámetros
        except NameError:
            st.error("Error: La componente TimelineEmotions no está definida o importada.")
        except Exception as e:
            st.error(f"Error mostrando timeline: {e}")
            st.info("Mostrando tabla de emociones alternativa...")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-header">
            <span class="card-icon">📈</span>
            <h3 class="card-title">Estadísticas Emocionales</h3>
        </div>
        """, unsafe_allow_html=True)

        # Calcular estadísticas
        conteo_emociones = {}
        confianzas = []

        for frame_result in emociones:
            for emocion_data in frame_result.get('emociones', []):
                emocion = emocion_data.get('emotion', 'Unknown')
                confianza = emocion_data.get('confidence', 0.0)

                conteo_emociones[emocion] = conteo_emociones.get(emocion, 0) + 1
                confianzas.append(confianza)

        if conteo_emociones:
            total = sum(conteo_emociones.values())
            emocion_predominante = max(conteo_emociones, key=conteo_emociones.get)
            porcentaje_pred = (conteo_emociones[emocion_predominante] / total) * 100
            promedio_confianza = sum(confianzas)/len(confianzas) if confianzas else 0.0

            st.metric("Emoción Predominante", emocion_predominante, f"{porcentaje_pred:.1f}%")
            st.metric("Confianza Promedio", f"{promedio_confianza:.2f}", "")
            st.metric("Detecciones Totales", total, "")

            # Mini gráfico de distribución
            st.markdown("**Distribución:**")
            for emocion, count in sorted(conteo_emociones.items(), key=lambda x: x[1], reverse=True)[:5]:
                percentage = (count / total) * 100
                st.write(f"• {emocion}: {percentage:.1f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    # Tabla detallada de emociones
    with st.expander("📋 Ver Tabla Detallada de Emociones", expanded=False):
        data_tabla = []
        for frame_result in emociones:
            frame_id = frame_result.get('frame_id', 0)
            tiempo = frame_result.get('tiempo_video', 0)

            for i, emocion_data in enumerate(frame_result.get('emociones', [])):
                data_tabla.append({
                    "Frame": frame_id,
                    "Tiempo (s)": f"{tiempo:.2f}",
                    "Rostro": i + 1,
                    "Emoción": emocion_data.get('emotion', 'Unknown'),
                    "Confianza": f"{emocion_data.get('confidence', 0.0):.3f}",
                    "Calidad": f"{emocion_data.get('quality_score', 0.0):.3f}"
                })

        if data_tabla:
            df = pd.DataFrame(data_tabla)
            st.dataframe(df, use_container_width=True, height=300)

def display_audio_analysis(audio_data: Dict):
    """Muestra análisis detallado de audio."""
    if not audio_data or audio_data.get('error'):
        st.warning(f"Error en análisis de audio: {audio_data.get('error', 'Error desconocido')}")
        return

    st.markdown("### 🎤 Análisis de Comunicación Verbal")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-header">
            <span class="card-icon">💬</span>
            <h3 class="card-title">Transcripción y Análisis</h3>
        </div>
        """, unsafe_allow_html=True)

        # Transcripción
        transcripcion = audio_data.get('transcription', 'Sin transcripción disponible')
        if transcripcion:
            st.markdown("**Transcripción completa:**")
            st.info(f'"{transcripcion}"')
        else:
            st.warning("No se detectó comunicación verbal clara")

        # Palabras detectadas
        palabras_detectadas = audio_data.get('palabras_detectadas', [])
        if palabras_detectadas:
            st.markdown("**Palabras identificadas:**")
            st.write(" • ".join(palabras_detectadas[:15]))  # Mostrar primeras 15
            if len(palabras_detectadas) > 15:
                st.write(f"... y {len(palabras_detectadas) - 15} más")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-header">
            <span class="card-icon">📊</span>
            <h3 class="card-title">Métricas Comunicativas</h3>
        </div>
        """, unsafe_allow_html=True)

        # Métricas
        palabras_totales = audio_data.get('palabras_totales', 0)
        intentos = audio_data.get('intentos_comunicacion', 0)
        calidad = audio_data.get('calidad_comunicacion', 'No evaluado').replace('_', ' ').title()

        st.metric("Palabras Totales", palabras_totales)
        st.metric("Intentos Comunicativos", intentos)
        st.metric("Calidad Comunicativa", calidad)

        # Palabras infantiles
        palabras_infantiles = audio_data.get('palabras_infantiles', [])
        if palabras_infantiles:
            st.markdown("**Vocabulario infantil:**")
            st.write(f"{len(palabras_infantiles)} palabras apropiadas")

        st.markdown('</div>', unsafe_allow_html=True)

def display_recommendations(recomendaciones: List[str], session_info: Dict):
    """Muestra recomendaciones personalizadas."""
    if not recomendaciones:
        st.info("No se generaron recomendaciones específicas.")
        return

    st.markdown("### 💡 Recomendaciones Personalizadas")

    # Categorizar recomendaciones
    recomendaciones_urgentes = []
    recomendaciones_generales = []

    for rec in recomendaciones:
        if any(palabra in rec.lower() for palabra in ['urgente', '🚨', 'inmediato', 'crítico']):
            recomendaciones_urgentes.append(rec)
        else:
            recomendaciones_generales.append(rec)

    # Mostrar recomendaciones urgentes
    if recomendaciones_urgentes:
        st.markdown("#### 🚨 Recomendaciones Prioritarias")
        for i, rec in enumerate(recomendaciones_urgentes, 1):
            st.markdown(f"""
            <div class="alert-critical">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

    # Mostrar recomendaciones generales
    if recomendaciones_generales:
        st.markdown("#### 📋 Recomendaciones Generales")

        # Crear tabs para organizar recomendaciones
        tabs = st.tabs(["🎯 Emocionales", "🗣️ Comunicativas", "🏠 Familiares", "📚 Todas"])

        with tabs[0]:  # Emocionales
            emocionales = [r for r in recomendaciones_generales if any(word in r.lower() for word in ['emocional', '😢', '😤', '😊', 'regulación'])]
            for i, rec in enumerate(emocionales, 1):
                st.markdown(f"**{i}.** {rec}")

        with tabs[1]:  # Comunicativas
            comunicativas = [r for r in recomendaciones_generales if any(word in r.lower() for word in ['comunicación', '🗣️', 'verbal', 'lenguaje'])]
            for i, rec in enumerate(comunicativas, 1):
                st.markdown(f"**{i}.** {rec}")

        with tabs[2]:  # Familiares
            familiares = [r for r in recomendaciones_generales if any(word in r.lower() for word in ['familia', '👨‍👩‍👧', 'cuidadores', 'hogar'])]
            for i, rec in enumerate(familiares, 1):
                st.markdown(f"**{i}.** {rec}")

        with tabs[3]:  # Todas
            for i, rec in enumerate(recomendaciones_generales, 1):
                st.markdown(f"**{i}.** {rec}")

def display_reports_section(results: Dict):
    """Muestra sección de reportes y descargas."""
    if not results:
        return

    st.markdown("### 📄 Reportes y Exportaciones")

    col1, col2, col3 = st.columns(3)

    # Información de archivos generados
    archivos = results.get('archivos_generados', {})

    with col1:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Dashboard Visual")
        if archivos.get('dashboard'):
            if os.path.exists(archivos['dashboard']):
                # Se asume que datetime, os, y open están disponibles
                with open(archivos['dashboard'], 'rb') as file:
                    st.download_button(
                        label="⬇️ Descargar Dashboard",
                        data=file.read(),
                        file_name=f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                        mime="image/png"
                    )
                st.success("Dashboard generado")
            else:
                st.info("Archivo de Dashboard no encontrado")
        else:
            st.info("Dashboard no disponible")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Timeline Emocional")
        if archivos.get('timeline'):
            if os.path.exists(archivos['timeline']):
                with open(archivos['timeline'], 'rb') as file:
                    st.download_button(
                        label="⬇️ Descargar Timeline",
                        data=file.read(),
                        file_name=f"timeline_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                        mime="image/png"
                    )
                st.success("Timeline generado")
            else:
                st.info("Archivo de Timeline no encontrado")
        else:
            st.info("Timeline no disponible")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Datos Exportados")
        if archivos.get('csv'):
            if os.path.exists(archivos['csv']):
                with open(archivos['csv'], 'rb') as file:
                    st.download_button(
                        label="⬇️ Descargar CSV",
                        data=file.read(),
                        file_name=f"datos_analisis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                st.success("Datos CSV generados")
            else:
                st.info("Archivo CSV no encontrado")
        else:
            st.info("CSV no disponible")
        st.markdown('</div>', unsafe_allow_html=True)

    # Reporte completo
    st.markdown("#### 📑 Reporte Completo")
    if results.get('reporte'):
        try:
            with open(results['reporte'], 'r', encoding='utf-8') as f:
                reporte_contenido = f.read()

            col1, col2 = st.columns([3, 1])
            with col1:
                st.text_area("Reporte Completo", reporte_contenido, height=300)
            with col2:
                st.download_button(
                    label="📄 Descargar Reporte TXT",
                    data=reporte_contenido,
                    file_name=f"reporte_completo_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"Error cargando reporte: {e}")

def main():
    """Función principal de la aplicación."""
    # Cargar CSS personalizado
    load_custom_css()

    # Inicializar estado de sesión
    initialize_session_state()

    # Renderizar header
    render_header()

    # Renderizar sidebar y obtener configuración
    video_file, models_dir, language, datos_personales, configuracion = render_sidebar()

    # Botón de análisis
    if st.sidebar.button("🚀 Ejecutar Análisis Completo", key="analyze_btn", type="primary"):
        if video_file is not None:
            st.session_state.processing = True
            st.session_state.analysis_results = None
        else:
            st.sidebar.error("⚠️ Por favor, sube un video antes de continuar")

    # Procesar análisis si está activado
    if st.session_state.processing and video_file is not None:
        # Crear directorio temporal
        os.makedirs("./temp_uploaded_videos", exist_ok=True)
        temp_path = f"./temp_uploaded_videos/{video_file.name}"

        # Guardar archivo temporal
        with open(temp_path, "wb") as f:
            f.write(video_file.getbuffer())

        # Mostrar progreso
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Procesando Análisis...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Inicializar pipeline
                if st.session_state.pipeline is None:
                    status_text.text("Inicializando sistema...")
                    progress_bar.progress(10)
                    # Se asume que PipelineAnalisisEmocional está disponible
                    st.session_state.pipeline = PipelineAnalisisEmocional(models_dir=models_dir)

                # Ejecutar análisis
                status_text.text("Analizando video y audio...")
                progress_bar.progress(50)

                # Se asume que el método ejecutar_pipeline es accesible
                resultados = st.session_state.pipeline.ejecutar_pipeline(
                    video_path=temp_path,
                    lang=language,
                    datos_personales=datos_personales,
                    configuracion_personalizada=configuracion
                )

                progress_bar.progress(100)
                status_text.text("✅ Análisis completado exitosamente!")

                # Guardar resultados
                st.session_state.analysis_results = resultados
                st.session_state.processing = False

                # Limpiar archivo temporal
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                st.balloons()
                st.success("🎉 ¡Análisis completado con éxito!")

            except Exception as e:
                st.error(f"💥 Error durante el análisis: {str(e)}")
                st.session_state.processing = False

                # Limpiar archivo temporal en caso de error
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # Mostrar resultados si están disponibles
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results

        # Dashboard de métricas
        display_metrics_dashboard(results)

        # Mostrar alertas
        if results.get('session_info', {}).get('alertas'):
            display_alerts(results.get('session_info', {}))

        # Crear tabs principales
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "😊 Análisis Emocional",
            "🎤 Análisis de Audio",
            "💡 Recomendaciones",
            "📊 Reportes",
            "ℹ️ Información Técnica"
        ])

        with tab1:
            display_emotions_analysis(results.get('emociones', []))

        with tab2:
            display_audio_analysis(results.get('audio', {}))

        with tab3:
            display_recommendations(
                results.get('recomendaciones', []),
                results.get('session_info', {})
            )

        with tab4:
            display_reports_section(results)

        with tab5:
            st.markdown("### ⚙️ Información Técnica de la Sesión")
            session_info = results.get('session_info', {})

            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "ID de Sesión": session_info.get('session_id', 'N/A'),
                    "Tiempo de Procesamiento": session_info.get('tiempo_procesamiento', 'N/A'),
                    "Etapas Completadas": len(session_info.get('etapas_completadas', [])),
                    "Errores Encontrados": len(session_info.get('errores', []))
                })

            with col2:
                if session_info.get('errores'):
                    st.markdown("**Errores durante el procesamiento:**")
                    for error in session_info['errores']:
                        st.error(error)
                else:
                    st.success("✅ No se encontraron errores durante el procesamiento")

                st.markdown("**Etapas completadas:**")
                for etapa in session_info.get('etapas_completadas', []):
                    st.write(f"• {etapa.replace('_', ' ').title()}")

    # Mostrar información de ayuda si no hay resultados
    elif not st.session_state.processing:
        st.markdown("### 🎯 Cómo usar el sistema")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="professional-card">
                <div class="card-header">
                    <span class="card-icon">📹</span>
                    <h3 class="card-title">1. Sube tu Video</h3>
                </div>
                <p>Selecciona un video del niño en formato MP4, AVI, MOV o MKV.
                Asegúrate de que tenga buena iluminación y el rostro sea visible.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="professional-card">
                <div class="card-header">
                    <span class="card-icon">📝</span>
                    <h3 class="card-title">2. Completa la Información</h3>
                </div>
                <p>Proporciona datos del niño como edad, diagnóstico y contexto.
                Esta información ayuda a generar recomendaciones más precisas.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="professional-card">
                <div class="card-header">
                    <span class="card-icon">🚀</span>
                    <h3 class="card-title">3. Ejecuta el Análisis</h3>
                </div>
                <p>Haz clic en "Ejecutar Análisis" y espera mientras el sistema
                procesa el video y genera recomendaciones personalizadas.</p>
            </div>
            """, unsafe_allow_html=True)

        # Información adicional
        st.markdown("### 📚 Características del Sistema")

        features_col1, features_col2 = st.columns(2)

        with features_col1:
            st.markdown("""
            **🧠 Análisis Emocional Avanzado:**
            - Detección de 7 emociones básicas
            - Análisis temporal y de patrones
            - Sistema ensemble de múltiples modelos
            - Evaluación de calidad de detección

            **🎤 Análisis de Comunicación:**
            - Transcripción automática de audio
            - Detección de palabras y vocabulario
            - Evaluación de calidad comunicativa
            - Análisis por segmentos temporales
            """)

        with features_col2:
            st.markdown("""
            **💡 Recomendaciones Personalizadas:**
            - Específicas por diagnóstico
            - Basadas en patrones detectados
            - Integración emoción-comunicación
            - Niveles de prioridad

            **📊 Reportes Completos:**
            - Dashboard visual interactivo
            - Timeline emocional detallado
            - Exportación en múltiples formatos
            - Análisis estadístico completo
            """)

        # Consejos para mejores resultados
        st.markdown("### 💡 Consejos para Mejores Resultados")

        tips_col1, tips_col2 = st.columns(2)

        with tips_col1:
            st.info("""
            **📹 Calidad del Video:**
            - Iluminación clara y uniforme
            - Rostro del niño visible en la mayoría de frames
            - Evitar movimientos bruscos de cámara
            - Duración recomendada: 30 segundos - 5 minutos
            """)

        with tips_col2:
            st.info("""
            **🔊 Calidad del Audio:**
            - Minimizar ruido de fondo
            - Volumen apropiado (ni muy bajo ni saturado)
            - Habla clara del niño
            - Evitar múltiples voces simultáneas
            """)

if __name__ == "__main__":
    main()