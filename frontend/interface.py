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
    from backend.generador_informes import GeneradorInformes
    from backend.email_sender import EmailSender
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
    if 'informes_generados' not in st.session_state:
        st.session_state.informes_generados = None

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
    
    datos_personales = {}
    configuracion = {}
    
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
    
    datos_personales = {
        "nombre": nombre,
        "edad": edad,
        "diagnostico": diagnostico,
        "rol_usuario": rol_usuario,
        "contexto_video": contexto_video
    }
    
    configuracion = {
        "intervalo_analisis_ms": intervalo_analisis,
        "umbral_confianza": umbral_confianza,
        "guardar_frames": guardar_frames
    }
    
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
        st.metric("Marcos analizados", total_frames)
    
    with col2:
        st.metric("Rostros detectados", total_rostros)
    
    with col3:
        st.metric("Palabras detectadas", palabras)
    
    with col4:
        st.metric("Alertas", alertas_count)
    
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

def display_recommendations(recomendaciones: List):
    st.markdown("### Recomendaciones Personalizadas")
    
    if not recomendaciones:
        st.info("No hay recomendaciones disponibles.")
    else:
        for i, rec in enumerate(recomendaciones, 1):
            if isinstance(rec, dict):
                st.info(f"**{i}. {rec.get('title', 'Recomendación')}**\n\n{rec.get('text', '')}")
            else:
                st.info(f"**{i}.** {rec}")

def display_reports_section(results: Dict, datos_personales: Dict):
    st.markdown("### Generar y Descargar Reportes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Generar Todos los Informes", type="primary"):
            with st.spinner("Generando informes..."):
                try:
                    gen_informes = GeneradorInformes()
                    archivos = gen_informes.generar_todos_informes(
                        datos_analisis=results,
                        info_personal=datos_personales
                    )
                    st.session_state.informes_generados = archivos
                    st.success("Informes generados correctamente")
                except Exception as e:
                    st.error(f"Error generando informes: {e}")
    
    if st.session_state.informes_generados:
        st.markdown("#### Descargar Archivos")
        
        archivos = st.session_state.informes_generados
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if archivos.get('histograma') and os.path.exists(archivos['histograma']):
                with open(archivos['histograma'], 'rb') as f:
                    st.download_button("📊 Histograma", f, "histograma.png", "image/png")
        
        with col2:
            if archivos.get('heatmap') and os.path.exists(archivos['heatmap']):
                with open(archivos['heatmap'], 'rb') as f:
                    st.download_button("🔥 Heatmap", f, "heatmap.png", "image/png")
        
        with col3:
            if archivos.get('confianza') and os.path.exists(archivos['confianza']):
                with open(archivos['confianza'], 'rb') as f:
                    st.download_button("📈 Confianza", f, "confianza.png", "image/png")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if archivos.get('html') and os.path.exists(archivos['html']):
                with open(archivos['html'], 'rb') as f:
                    st.download_button("📄 HTML", f, "reporte.html", "text/html")
        
        with col2:
            if archivos.get('resumen') and os.path.exists(archivos['resumen']):
                with open(archivos['resumen'], 'rb') as f:
                    st.download_button("📋 Resumen", f, "resumen.txt", "text/plain")
        
        with col3:
            if archivos.get('csv') and os.path.exists(archivos['csv']):
                with open(archivos['csv'], 'rb') as f:
                    st.download_button("📊 CSV", f, "datos.csv", "text/csv")
        
        with col4:
            if archivos.get('json') and os.path.exists(archivos['json']):
                with open(archivos['json'], 'rb') as f:
                    st.download_button("🔗 JSON", f, "reporte.json", "application/json")

def display_email_section(results: Dict, datos_personales: Dict):
    st.markdown("### Enviar Reporte por Email")
    
    st.info("Genera los informes primero antes de enviarlos por email")
    
    col1, col2 = st.columns(2)
    with col1:
        email_remitente = st.text_input(
            "Tu email (Gmail)",
            placeholder="ejemplo@gmail.com",
            type="password"
        )
    
    with col2:
        email_password = st.text_input(
            "Contraseña de aplicación",
            placeholder="xxxxxx xxxxxx xxxxxx xxxxxx",
            type="password",
            help="Genera en myaccount.google.com > Seguridad > Contraseñas de aplicación"
        )
    
    emails_destinatarios = st.text_area(
        "Emails destinatarios (separados por comas)",
        placeholder="padre@example.com, educador@example.com",
        height=80
    )
    
    asunto_custom = st.text_input(
        "Asunto (opcional)",
        placeholder=f"Análisis Emocional - {datos_personales.get('nombre', 'Participante')}"
    )
    
    if st.button("Enviar por Email", type="primary"):
        if not email_remitente or not email_password:
            st.error("Completa email y contraseña")
        elif not emails_destinatarios:
            st.error("Ingresa al menos un email destinatario")
        elif not st.session_state.informes_generados:
            st.error("Genera los informes primero")
        else:
            with st.spinner("Enviando email..."):
                try:
                    sender = EmailSender(
                        smtp_server="smtp.gmail.com",
                        smtp_port=587,
                        sender_email=email_remitente,
                        sender_password=email_password
                    )
                    
                    if not sender.verificar_conexion():
                        st.error("No se pudo conectar al servidor")
                        with st.expander("Ver instrucciones"):
                            EmailSender.generar_instrucciones_gmail()
                    else:
                        destinatarios = [e.strip() for e in emails_destinatarios.split(',')]
                        
                        exito = sender.enviar_reporte(
                            destinatarios=destinatarios,
                            archivos=st.session_state.informes_generados,
                            info_personal=datos_personales,
                            datos_analisis=results,
                            asunto=asunto_custom or f"Análisis Emocional - {datos_personales.get('nombre')}"
                        )
                        
                        if exito:
                            st.success(f"Email enviado a {len(destinatarios)} destinatarios")
                            st.balloons()
                        else:
                            st.error("Error al enviar")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def display_session_info(results: Dict, datos_personales: Dict):
    st.markdown("### Información de la Sesión")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Datos del Participante:**")
        for key, value in datos_personales.items():
            if value:
                st.write(f"- **{key.replace('_', ' ').title()}**: {value}")
    
    with col2:
        st.markdown("**Información Técnica:**")
        session_info = results.get('session_info', {})
        st.write(f"- **Session ID**: {session_info.get('session_id', 'N/A')}")
        st.write(f"- **Etapas**: {len(session_info.get('etapas_completadas', []))}")
        st.write(f"- **Errores**: {len(session_info.get('errores', []))}")
        st.write(f"- **Tiempo**: {results.get('session_info', {}).get('tiempo_procesamiento', 'N/A')}")

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
        temp_dir = os.path.abspath("./videos_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, video_file.name)
        
        try:
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            
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
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        if "error" in results:
            st.error(f"Error en análisis: {results['error']}")
        else:
            display_metrics_dashboard(results)
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Análisis Emocional",
                "Análisis de Audio",
                "Recomendaciones",
                "Reportes",
                "Email",
                "Información"
            ])
            
            with tab1:
                display_emotions_analysis(results.get('emociones', []))
            
            with tab2:
                display_audio_analysis(results.get('audio', {}))
            
            with tab3:
                display_recommendations(results.get('recomendaciones', []))
            
            with tab4:
                display_reports_section(results, datos_personales)
            
            with tab5:
                display_email_section(results, datos_personales)
            
            with tab6:
                display_session_info(results, datos_personales)

if __name__ == "__main__":
    main()