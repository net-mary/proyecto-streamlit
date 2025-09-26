import os
import sys
import streamlit as st
import pandas as pd

# Añadir carpeta raíz del proyecto para importar backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.pipeline import ejecutar_pipeline
from frontend.components.timeline_emotions import TimelineEmotions

def cargar_reporte():
    if os.path.exists("reporte.txt"):
        with open("reporte.txt", "r", encoding="utf-8") as f:
            return f.read()
    return "No hay reporte generado aún."

def cargar_histograma():
    hist_path = "histograma.png"
    if os.path.exists(hist_path):
        st.image(hist_path, caption="Distribución de emociones detectadas")

def main():
    st.set_page_config(page_title="Análisis Emocional Multimodal", layout="wide")
    st.title("Dashboard avanzado para análisis emocional infantil")

    st.sidebar.header("Configuración")
    video_file = st.sidebar.file_uploader("Sube tu video (.mp4, .avi)", type=["mp4", "avi"])
    models_dir = st.sidebar.text_input("Carpeta de modelos:", "./models")
    language = st.sidebar.selectbox("Idioma del audio:", ["es-ES", "en-US"])
    modo_experto = st.sidebar.checkbox("Modo experto", value=False)

    resultados = None
    if st.sidebar.button("Ejecutar análisis") and video_file is not None:
        temp_path = f"./temp_uploaded_videos/{video_file.name}"
        with open(temp_path, "wb") as f:
            f.write(video_file.getbuffer())
        with st.spinner("Procesando video y audio..."):
            resultados = ejecutar_pipeline(temp_path, models_dir=models_dir, lang=language)
        st.success("Análisis completado!")

    if resultados:
        col1, col2 = st.columns([2, 1])
        with col1:
            cargar_histograma()
            st.header("Timeline emocional interactivo")
            if "emociones" in resultados and resultados["emociones"] is not None:
                TimelineEmotions(resultados["emociones"], highlights=[])
            else:
                st.warning("No se generaron resultados emocionales.")

            st.header("Tabla de emociones")
            if "emociones" in resultados and resultados["emociones"] is not None:
                data_tabla = []
                for frame_data in resultados["emociones"]:
                    for e in frame_data["emociones"]:
                        data_tabla.append({
                            "Frame": frame_data["frame"],
                            "Emoción": e["emotion"],
                            "Confianza": round(e["confidence"], 2)
                        })
                df = pd.DataFrame(data_tabla)
                st.dataframe(df)

        with col2:
            st.header("Recomendaciones inteligentes")
            if "recomendaciones" in resultados and resultados["recomendaciones"]:
                for rec in resultados["recomendaciones"]:
                    st.info(rec)
            else:
                st.write("No hay recomendaciones disponibles.")

            st.header("Reporte generado")
            reporte_texto = cargar_reporte()
            st.text_area("Informe emocional", value=reporte_texto, height=280)

        st.sidebar.download_button("Descargar reporte completo", resultados["reporte"], file_name="reporte.txt", mime="text/plain")

if __name__ == "__main__":
    main()

