import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Optional

def TimelineEmotions(emociones: List[Dict], highlights: Optional[List[str]] = None):
    """
    Componente avanzado para visualizar timeline emocional interactivo.
    
    Args:
        emociones (List[Dict]): Lista de resultados emocionales por frame
        highlights (List[str]): Lista de momentos destacados (opcional)
    """
    
    if highlights is None:
        highlights = []
    
    # Verificar si hay datos
    if not emociones:
        st.warning("📊 No hay datos emocionales para mostrar en el timeline.")
        return
    
    try:
        # Preparar datos para visualización
        timeline_data = []
        
        for frame_result in emociones:
            frame_id = frame_result.get('frame_id', 0)
            tiempo_video = frame_result.get('tiempo_video', frame_id / 30.0)  # Asume 30 FPS
            
            for face_idx, emocion_data in enumerate(frame_result.get('emociones', [])):
                timeline_data.append({
                    'frame': frame_id,
                    'tiempo': tiempo_video,
                    'emocion': emocion_data.get('emotion', 'Unknown'),
                    'confianza': emocion_data.get('confidence', 0.0),
                    'face_id': face_idx + 1,
                    'calidad': emocion_data.get('quality_score', 0.0),
                    'area': emocion_data.get('area', 0)
                })
        
        if not timeline_data:
            st.warning("📊 No se encontraron emociones válidas en los datos.")
            return
        
        # Crear DataFrame
        df = pd.DataFrame(timeline_data)
        
        # Definir paleta de colores profesional para emociones
        emotion_colors = {
            'Happy': '#2E8B57',      # Verde mar
            'Sad': '#4682B4',        # Azul acero  
            'Angry': '#DC143C',      # Rojo carmesí
            'Fear': '#800080',       # Púrpura
            'Surprise': '#FF8C00',   # Naranja oscuro
            'Disgust': '#8B4513',    # Marrón
            'Neutral': '#708090'     # Gris pizarra
        }
        
        # Crear subplots para múltiples visualizaciones
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '📈 Timeline Principal de Emociones',
                '🎯 Distribución de Confianza',
                '⏱️ Evolución Temporal Detallada', 
                '📊 Heatmap de Intensidad',
                '🔄 Patrones de Transición',
                '📋 Resumen Estadístico'
            ],
            specs=[
                [{"colspan": 2}, None],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Timeline principal (scatter plot)
        for emocion in df['emocion'].unique():
            datos_emocion = df[df['emocion'] == emocion]
            color = emotion_colors.get(emocion, '#708090')
            
            fig.add_trace(
                go.Scatter(
                    x=datos_emocion['tiempo'],
                    y=[emocion] * len(datos_emocion),
                    mode='markers+lines',
                    marker=dict(
                        color=color,
                        size=datos_emocion['confianza'] * 15 + 5,  # Tamaño proporcional a confianza
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    line=dict(color=color, width=2, dash='dot'),
                    name=f'{emocion}',
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Tiempo: %{x:.2f}s<br>' +
                                 'Confianza: %{marker.size}<br>' +
                                 '<extra></extra>',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. Distribución de confianza por emoción (box plot)
        for emocion in df['emocion'].unique():
            datos_emocion = df[df['emocion'] == emocion]
            color = emotion_colors.get(emocion, '#708090')
            
            fig.add_trace(
                go.Box(
                    y=datos_emocion['confianza'],
                    name=emocion,
                    marker_color=color,
                    boxmean=True,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 3. Evolución temporal con líneas suaves
        # Calcular emoción predominante por ventana de tiempo
        time_windows = np.arange(0, df['tiempo'].max() + 1, 0.5)  # Ventanas de 0.5s
        emotion_evolution = []
        
        for i in range(len(time_windows) - 1):
            start_time = time_windows[i]
            end_time = time_windows[i + 1]
            
            # Filtrar datos en esta ventana
            window_data = df[(df['tiempo'] >= start_time) & (df['tiempo'] < end_time)]
            
            if not window_data.empty:
                # Encontrar emoción con mayor confianza promedio
                emotion_means = window_data.groupby('emocion')['confianza'].mean()
                predominant_emotion = emotion_means.idxmax()
                avg_confidence = emotion_means.max()
                
                emotion_evolution.append({
                    'tiempo': (start_time + end_time) / 2,
                    'emocion_predominante': predominant_emotion,
                    'confianza_promedio': avg_confidence
                })
        
        if emotion_evolution:
            evolution_df = pd.DataFrame(emotion_evolution)
            
            for emocion in evolution_df['emocion_predominante'].unique():
                emo_data = evolution_df[evolution_df['emocion_predominante'] == emocion]
                color = emotion_colors.get(emocion, '#708090')
                
                fig.add_trace(
                    go.Scatter(
                        x=emo_data['tiempo'],
                        y=emo_data['confianza_promedio'],
                        mode='lines+markers',
                        name=f'{emocion} (Evolución)',
                        line=dict(color=color, width=3),
                        marker=dict(size=8, color=color),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # 4. Heatmap de intensidad emocional
        # Crear matriz de intensidad tiempo vs emoción
        pivot_data = df.pivot_table(
            values='confianza', 
            index='emocion', 
            columns=pd.cut(df['tiempo'], bins=10),  # 10 bins temporales
            fill_value=0, 
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=[f"{interval.left:.1f}-{interval.right:.1f}s" for interval in pivot_data.columns],
                    y=pivot_data.index,
                    colorscale='Viridis',
                    showscale=True,
                    hoverongaps=False,
                    colorbar=dict(title="Intensidad"),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # 5. Gráfico de barras con estadísticas
        emotion_stats = df.groupby('emocion').agg({
            'confianza': ['count', 'mean', 'std']
        }).round(3)
        
        emotion_stats.columns = ['Frecuencia', 'Confianza_Media', 'Desv_Std']
        emotion_stats = emotion_stats.reset_index()
        
        colors_list = [emotion_colors.get(emo, '#708090') for emo in emotion_stats['emocion']]
        
        fig.add_trace(
            go.Bar(
                x=emotion_stats['emocion'],
                y=emotion_stats['Frecuencia'],
                marker_color=colors_list,
                name='Frecuencia',
                text=emotion_stats['Confianza_Media'].round(2),
                textposition='outside',
                showlegend=False
            ),
            row=3, col=2
        )
        
        # Configurar layout
        fig.update_layout(
            height=900,
            title=dict(
                text="📊 Análisis Temporal Completo de Emociones",
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#264653')
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Actualizar ejes
        fig.update_xaxes(title_text="Tiempo (segundos)", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Emociones", row=1, col=1, gridcolor='lightgray')
        
        fig.update_xaxes(title_text="Emociones", row=2, col=1)
        fig.update_yaxes(title_text="Nivel de Confianza", row=2, col=1)
        
        fig.update_xaxes(title_text="Tiempo (segundos)", row=2, col=2)
        fig.update_yaxes(title_text="Confianza Promedio", row=2, col=2)
        
        fig.update_xaxes(title_text="Períodos Temporales", row=3, col=1)
        fig.update_yaxes(title_text="Emociones", row=3, col=1)
        
        fig.update_xaxes(title_text="Emociones", row=3, col=2)
        fig.update_yaxes(title_text="Frecuencia de Detección", row=3, col=2)
        
        # Mostrar gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Mostrar estadísticas adicionales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "⏱️ Duración Total",
                f"{df['tiempo'].max():.2f}s",
                f"{len(df)} detecciones"
            )
        
        with col2:
            emocion_predominante = df['emocion'].mode().iloc[0] if not df.empty else "N/A"
            frecuencia = df[df['emocion'] == emocion_predominante].shape[0] if emocion_predominante != "N/A" else 0
            st.metric(
                "🎯 Emoción Predominante", 
                emocion_predominante,
                f"{frecuencia} detecciones"
            )
        
        with col3:
            confianza_promedio = df['confianza'].mean()
            st.metric(
                "📊 Confianza Promedio",
                f"{confianza_promedio:.3f}",
                f"±{df['confianza'].std():.3f}"
            )
        
        # Mostrar momentos destacados si se proporcionan
        if highlights:
            st.subheader("🌟 Momentos Destacados")
            for i, highlight in enumerate(highlights, 1):
                st.info(f"**{i}.** {highlight}")
        
        # Análisis de patrones
        with st.expander("🔍 Análisis Avanzado de Patrones", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Tendencias Temporales:**")
                
                # Dividir timeline en segmentos y analizar tendencias
                n_segments = 5
                segment_size = len(df) // n_segments
                
                for i in range(n_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(df)
                    segment_data = df.iloc[start_idx:end_idx]
                    
                    if not segment_data.empty:
                        segment_emotion = segment_data['emocion'].mode().iloc[0]
                        segment_confidence = segment_data['confianza'].mean()
                        
                        start_time = segment_data['tiempo'].min()
                        end_time = segment_data['tiempo'].max()
                        
                        st.write(f"• **Segmento {i+1}** ({start_time:.1f}s - {end_time:.1f}s): "
                               f"{segment_emotion} (conf: {segment_confidence:.2f})")
            
            with col2:
                st.markdown("**🔄 Transiciones Emocionales:**")
                
                # Analizar transiciones entre emociones
                transitions = []
                for i in range(len(df) - 1):
                    current_emotion = df.iloc[i]['emocion']
                    next_emotion = df.iloc[i + 1]['emocion']
                    
                    if current_emotion != next_emotion:
                        transitions.append(f"{current_emotion} → {next_emotion}")
                
                # Contar transiciones más frecuentes
                if transitions:
                    from collections import Counter
                    transition_counts = Counter(transitions)
                    
                    st.write("**Transiciones más frecuentes:**")
                    for transition, count in transition_counts.most_common(5):
                        st.write(f"• {transition}: {count} veces")
                else:
                    st.write("• No se detectaron transiciones significativas")
        
        # Tabla resumen por tipo de dato
        with st.expander("📋 Tabla de Datos Detallada", expanded=False):
            st.markdown("**Datos procesados para el timeline:**")
            
            # Preparar tabla más legible
            display_df = df.copy()
            display_df['tiempo'] = display_df['tiempo'].round(2)
            display_df['confianza'] = display_df['confianza'].round(3)
            display_df['calidad'] = display_df['calidad'].round(3)
            
            # Renombrar columnas para mejor presentación
            display_df = display_df.rename(columns={
                'frame': 'Frame',
                'tiempo': 'Tiempo (s)',
                'emocion': 'Emoción',
                'confianza': 'Confianza',
                'face_id': 'ID Rostro',
                'calidad': 'Calidad',
                'area': 'Área Rostro'
            })
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300
            )
    
    except Exception as e:
        st.error(f"❌ Error generando timeline emocional: {str(e)}")
        st.info("💡 Asegúrate de que los datos de entrada tengan el formato correcto.")
        
        # Mostrar información de debug si hay error
        if st.checkbox("🔧 Mostrar información de debug"):
            st.write("**Estructura de datos recibida:**")
            st.json(emociones[:2] if len(emociones) > 2 else emociones)

def create_simple_timeline(emociones: List[Dict]) -> go.Figure:
    """
    Crea un timeline simplificado para uso en otros componentes.
    
    Args:
        emociones (List[Dict]): Datos de emociones
        
    Returns:
        go.Figure: Figura de Plotly simple
    """
    
    if not emociones:
        fig = go.Figure()
        fig.add_annotation(
            text="No hay datos disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Preparar datos básicos
    timeline_data = []
    for frame_result in emociones:
        tiempo = frame_result.get('tiempo_video', frame_result.get('frame_id', 0) / 30.0)
        
        for emocion_data in frame_result.get('emociones', []):
            timeline_data.append({
                'tiempo': tiempo,
                'emocion': emocion_data.get('emotion', 'Unknown'),
                'confianza': emocion_data.get('confidence', 0.0)
            })
    
    if not timeline_data:
        return go.Figure()
    
    df = pd.DataFrame(timeline_data)
    
    # Colores para emociones
    colors = {
        'Happy': '#2E8B57', 'Sad': '#4682B4', 'Angry': '#DC143C',
        'Fear': '#800080', 'Surprise': '#FF8C00', 'Disgust': '#8B4513',
        'Neutral': '#708090'
    }
    
    fig = go.Figure()
    
    for emocion in df['emocion'].unique():
        data = df[df['emocion'] == emocion]
        
        fig.add_trace(go.Scatter(
            x=data['tiempo'],
            y=[emocion] * len(data),
            mode='markers',
            marker=dict(
                size=data['confianza'] * 10 + 5,
                color=colors.get(emocion, '#708090'),
                opacity=0.7
            ),
            name=emocion
        ))
    
    fig.update_layout(
        title="Timeline Emocional",
        xaxis_title="Tiempo (segundos)",
        yaxis_title="Emociones",
        height=400,
        showlegend=True
    )
    
    return fig
