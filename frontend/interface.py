import os
import sys
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List

st.set_page_config(
    page_title="Análisis Emocional Multimodal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from backend.pipeline import PipelineAnalisisEmocional
    from frontend.components.timeline_emotions import TimelineEmotions
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.stop()

def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #2E7D8B;
        --secondary-color: #4A9DBA;
        --accent-color: #F4A261;
        --success-color: #2A9D8F;
        --warning-color: #E76F51;
    }
    
    .main > div {
        padding-top: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 600;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--success-color), var(--secondary-color));
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Análisis Emocional Multimodal</h1>
        <p>Sistema Avanzado de Análisis para Niños con Discapacidad</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    st.sidebar.markdown("### ⚙️ Configuración del Análisis")
    
    st.sidebar.markdown("#### 📹 Video a Analizar")
    video_file = st.sidebar.file_uploader(
        "Selecciona un video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Formatos soportados: MP4, AVI, MOV, MKV"
    )
    
    st.sidebar.markdown("#### 🔧 Configuraciones Técnicas")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        models_dir = st.text_input("Directorio de modelos", value="./models")
    
    with col2:
        language = st.selectbox("Idioma del audio", ["es-ES", "en-US", "es-MX", "es-AR"])
    
    with st.sidebar.expander("⚡ Configuraciones Avanzadas"):
        intervalo_analisis = st.slider("Intervalo de análisis (ms)", 500, 5000, 1000, 250)
        umbral_confianza = st.slider(
            "Umbral de confianza", 
            0.01, 0.9, 0.05, 0.01,
            help="Valor bajo (0.05-0.15) detecta más emociones."
        )
        guardar_frames = st.checkbox("Guardar fotogramas detectados", value=True)
        
        if umbral_confianza > 0.3:
            st.warning(f"Umbral {umbral_confianza:.2f} es alto. Puede descartar detecciones.")
        elif umbral_confianza < 0.10:
            st.info(f"Umbral {umbral_confianza:.2f} es óptimo.")
    
    st.sidebar.markdown("#### 👶 Información del Participante")
    
    with st.sidebar.form("datos_personales"):
        nombre = st.text_input("Nombre", placeholder="Nombre del niño")
        edad = st.number_input("Edad", min_value=1, max_value=18, value=5)
        diagnostico = st.selectbox("Diagnóstico", [
            "", "Autismo/TEA", "TDAH", "Síndrome de Down", "Parálisis Cerebral",
            "Discapacidad Intelectual", "Trastornos del Lenguaje", "Otro"
        ])
        
        if diagnostico == "Otro":
            diagnostico = st.text_input("Especificar diagnóstico")
        
        rol_usuario = st.selectbox("Tu rol", ["Padre/Madre", "Educador", "Terapeuta", "Investigador", "Otro"])
        contexto_video = st.text_area("Contexto del video", placeholder="Describe la situación...", height=100)
        
        submit_info = st.form_submit_button("Guardar Información")
        if submit_info and nombre:
            st.success("Información guardada")
    
    datos_personales = {
        "nombre": nombre if 'nombre' in locals() and nombre else "",
        "edad": edad if 'edad' in locals() else 5,
        "diagnostico": diagnostico if 'diagnostico' in locals() and diagnostico else "",
        "rol_usuario": rol_usuario if 'rol_usuario' in locals() else "Padre/Madre",
        "contexto_video": contexto_video if 'contexto_video' in locals() else ""
    } if 'nombre' in locals() else {}
    
    configuracion = {
        "intervalo_analisis_ms": intervalo_analisis if 'intervalo_analisis' in locals() else 1000,
        "umbral_confianza": umbral_confianza if 'umbral_confianza' in locals() else 0.05,
        "guardar_frames": guardar_frames if 'guardar_frames' in locals() else True
    } if 'intervalo_analisis' in locals() else {}
    
    return video_file, models_dir, language, datos_personales, configuracion

def display_metrics_dashboard(results: Dict):
    if not results or not results.get('session_info'):
        return
    
    st.markdown("### 📊 Dashboard de Métricas")
    
    emociones = results.get('emociones', [])
    audio_info = results.get('audio', {})
    
    total_frames = len(emociones)
    total_rostros = sum(len(f.get('emociones', [])) for f in emociones)
    palabras = audio_info.get('palabras_totales', 0)
    alertas_count = len(results.get('session_info', {}).get('alertas', []))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_frames}</div>
            <div class="metric-label">Marcos analizados</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_rostros}</div>
            <div class="metric-label">Rostros detectados</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{palabras}</div>
            <div class="metric-label">Palabras detectadas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{alertas_count}</div>
            <div class="metric-label">Alertas</div>
        </div>
        """, unsafe_allow_html=True)
    
    if total_rostros > 0:
        st.info(f"Total de emociones detectadas: {total_rostros}")

def display_emotions_analysis(emociones: List[Dict]):
    if not emociones:
        st.warning("No se detectaron emociones en el video.")
        return
    
    st.markdown("### Análisis de Emociones Faciales")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("#### Estadísticas")
        
        conteo_emociones = {}
        for frame_result in emociones:
            for emocion_data in frame_result.get('emociones', []):
                emocion = emocion_data.get('emotion', 'Unknown')
                conteo_emociones[emocion] = conteo_emociones.get(emocion, 0) + 1
        
        if conteo_emociones:
            total = sum(conteo_emociones.values())
            emocion_predominante = max(conteo_emociones, key=conteo_emociones.get)
            porcentaje = (conteo_emociones[emocion_predominante] / total) * 100
            
            st.metric("Emoción Predominante", emocion_predominante, f"{porcentaje:.1f}%")
            st.metric("Detecciones Totales", total)
            
            st.markdown("**Distribución:**")
            for emocion, count in sorted(conteo_emociones.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                st.write(f"• {emocion}: {percentage:.1f}% ({count})")
    
    with col1:
        with st.expander("Ver Tabla Detallada"):
            data_tabla = []
            for frame_result in emociones:
                for i, emocion_data in enumerate(frame_result.get('emociones', [])):
                    data_tabla.append({
                        "Frame": frame_result.get('frame_id', 0),
                        "Tiempo (s)": f"{frame_result.get('tiempo_video', 0):.2f}",
                        "Emoción": emocion_data.get('emotion', 'Unknown'),
                        "Confianza": f"{emocion_data.get('confidence', 0.0):.3f}"
                    })
            
            if data_tabla:
                df = pd.DataFrame(data_tabla)
                st.dataframe(df, use_container_width=True, height=300)

def display_audio_analysis(audio_data: Dict):
    if not audio_data or audio_data.get('error'):
        st.warning(f"Error en análisis de audio: {audio_data.get('error', 'Error desconocido')}")
        return
    
    st.markdown("### Análisis de Comunicación Verbal")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### Transcripción")
        transcripcion = audio_data.get('transcription', 'Sin transcripción disponible')
        if transcripcion:
            st.info(f'"{transcripcion}"')
        else:
            st.warning("No se detectó comunicación verbal")
    
    with col2:
        st.markdown("#### Métricas")
        st.metric("Palabras Totales", audio_data.get('palabras_totales', 0))
        st.metric("Intentos Comunicativos", audio_data.get('intentos_comunicacion', 0))

def display_recommendations(recomendaciones: List[str]):
    st.markdown("### Recomendaciones Personalizadas")
    
    if not recomendaciones:
        st.info("No hay recomendaciones disponibles.")
    else:
        for i, rec in enumerate(recomendaciones, 1):
            st.info(f"**{i}.** {rec}")

def display_reports_section(results: Dict):
    st.markdown("### Reportes y Exportaciones")
    
    archivos = results.get('archivos_generados', {})
    histograma = results.get('histograma', '')
    reporte = results.get('reporte', '')
    
    if histograma and os.path.exists(histograma):
        st.markdown("#### Histograma de Emociones")
        st.image(histograma, use_column_width=True)
        
        with open(histograma, 'rb') as file:
            st.download_button("Descargar Histograma", file.read(),
                f"histograma_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                "image/png", use_container_width=True)

def main():
    load_custom_css()
    initialize_session_state()
    render_header()
    
    video_file, models_dir, language, datos_personales, configuracion = render_sidebar()
    
    if st.sidebar.button("Ejecutar Análisis Completo", type="primary"):
        if video_file:
            st.session_state.processing = True
            st.session_state.analysis_results = None
        else:
            st.sidebar.error("Por favor, sube un video antes de continuar")
    
    if st.session_state.processing and video_file:
        # Crear directorio temporal
        temp_dir = os.path.abspath("./videos_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Guardar con ruta absoluta
        temp_path = os.path.join(temp_dir, video_file.name)
        
        st.write(f"DEBUG: Guardando en: {temp_path}")
        
        try:
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            st.write(f"DEBUG: Archivo guardado. Tamaño: {os.path.getsize(temp_path)} bytes")
            
            # Verificar que existe
            if not os.path.exists(temp_path):
                st.error("ERROR: Archivo no se guardó correctamente")
                st.stop()
            
        except Exception as e:
            st.error(f"Error guardando archivo: {e}")
            st.stop()
        
        with st.container():
            st.markdown("### Procesando Análisis...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if not st.session_state.pipeline:
                    status_text.text("Inicializando sistema...")
                    progress_bar.progress(10)
                    st.session_state.pipeline = PipelineAnalisisEmocional(models_dir=models_dir)
                
                status_text.text("Analizando video y audio...")
                progress_bar.progress(50)
                
                resultados = st.session_state.pipeline.ejecutar_pipeline(
                    video_path=temp_path,
                    lang=language,
                    datos_personales=datos_personales,
                    configuracion_personalizada=configuracion
                )
                
                progress_bar.progress(100)
                status_text.text("Análisis completado!")
                
                st.session_state.analysis_results = resultados
                st.session_state.processing = False
                
                st.balloons()
                st.success("Análisis completado con éxito!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.processing = False
            
            finally:
                # Limpiar archivo
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        st.write("DEBUG: Archivo eliminado")
                except:
                    pass
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        if "error" in results:
            st.error(f"Error en análisis: {results['error']}")
        else:
            display_metrics_dashboard(results)
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Análisis Emocional",
                "Análisis de Audio",
                "Recomendaciones",
                "Reportes"
            ])
            
            with tab1:
                display_emotions_analysis(results.get('emociones', []))
            
            with tab2:
                display_audio_analysis(results.get('audio', {}))
            
            with tab3:
                display_recommendations(results.get('recomendaciones', []))
            
            with tab4:
                display_reports_section(results)

if __name__ == "__main__":
    main()