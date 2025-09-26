import plotly.express as px
import pandas as pd
import streamlit as st

def TimelineEmotions(emociones, highlights=None):
    data = []
    for r in emociones:
        frame = r['frame']
        for e in r['emociones']:
            data.append({
                'frame': frame,
                'emotion': e['emotion'],
                'confidence': round(e['confidence'], 2)
            })
    if not data:
        st.warning("No hay datos para mostrar.")
        return

    df = pd.DataFrame(data)
    df['time'] = df['frame'] / 30  # si son 30 fps

    fig = px.scatter(df, x='time', y='emotion', color='emotion',
                     size='confidence', hover_data=['frame', 'confidence'],
                     title='Timeline emocional')

    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if highlights:
        st.subheader("Momentos destacados")
        for h in highlights:
            st.info(f"Highlight: {h}")
