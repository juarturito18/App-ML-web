import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pages.config import section, set_page

set_page("Datos")
st.title("Conociendo los Datos")
st.markdown('''
            Antes de construir cualquier modelo predictivo, un científico de datos dedica el 80% de su tiempo a lo más importante: 
            **entender y explorar los datos**. En esta sección, te convertirás en un **detective inmobiliario**.''')
st.markdown(''' 
            :blue[Tu misión será examinar el dataset de 
            viviendas de California para descubrir pistas ocultas, patrones interesantes y relaciones que puedan explicar por qué algunas casas valen más que otras. 
            **¡Manos a la obra!**]
            ''')

section("✔ Objetivos de Aprendizaje",'''
         <ul>
        <li>Identificar las características principales del dataset</li>
        <li>Visualizar las distribuciones de las variables clave</li>
        <li>Descubrir relaciones entre diferentes características</li>
        <li>Formular hipótesis iniciales sobre qué afecta el precio</li>
        </ul>
        ''',"#C7C8FF","#252DF5")

with st.expander("Historia del dataset"):
    st.markdown('''
                Elemento Interactivo 1: "La Ficha Técnica"
                Diseño: Una tarjeta expandible con iconos
                Funcionamiento: Al hacer clic en cada ítem, se expande una breve explicación
                Contenido:
                📅 Año de recolección: 1990 (explicar contexto histórico)
                
                📍 Ubicación: California, USA
                
                🏘️ Unidad de análisis: Distritos/bloques censales
                
                🔢 Número de registros: 20,640
                
                📊 Número de características: 10
                
                🎯 Variable objetivo: median_house_value (valor mediano de la vivienda)
                ''')

with st.expander("Conoce a los Personajes: Las Variables"):
    st.markdown('''
        Elemento Interactivo 2: "La Galería de Variables"
        
        Diseño: Tarjetas con pestañas o acordeón
        
        Organización: Divididas en 3 categorías con colores diferentes      
                ''')

st.markdown('''
            <h4 style="font-weight: bold;">Primer Contacto: Vista Previa de los Datos</h4>
            ''',unsafe_allow_html=True)
st.markdown('''
            Los científicos de datos siempre comienzan mirando **una muestra de los datos en formato tabla**. 
            Esto nos da una primera impresión de cómo está organizada la información, qué valores tienen las variables, y si encontramos datos extraños o faltantes.
        ''')

data=pd.read_csv("data_regresion_lineal\housing.csv")
st.dataframe(data)

#COSAS POR HACER
st.button("resaltado condicional", type="primary")
st.button("Filtrar por variable y cantidad", type="primary")
#COSAS POR HACER


st.markdown('''
            <h4 style="font-weight: bold;">El Detective Estadístico: Distribuciones y Formas</h4>
            ''',unsafe_allow_html=True)
st.markdown('''
    Cada variable tiene su propia :blue[personalidad estadística]. Algunas son **simétricas**, otras están **sesgadas**, 
    algunas tienen **valores concentrados** y otras **dispersas**. Entender estas distribuciones es crucial para saber qué técnicas aplicar después.        
''')

col1,col2=st.columns(2)
with col1:
    valores_histo=st.selectbox("¿Que Variable deseas graficar?", data.columns)
with col2:
    numero_bins=st.slider("Numero de bins",min_value=1,value=20, help='Los bins (también llamados **intervalos, clases o grupos**) son los bloques fundamentales de un histograma. Representan **rangos de valores en los que se divide el conjunto total** de datos numéricos para poder graficarlos')

#FUNCION GRAFICOS INTERACTIVOS DE BARRA
fig = px.histogram(
    data,
    x=valores_histo,
    nbins=numero_bins,
    labels={valores_histo: valores_histo},
)
fig.update_layout(
    bargap=0.05,
    height=450
)
st.plotly_chart(fig, use_container_width=True)

#COSAS POR HACER
st.text('''
        Botón "Curva de densidad": Superpone una curva suavizada sobre el histograma

        Marcadores de estadísticos: Líneas verticales que muestran media (azul) y mediana (roja)
        ''')
#COSAS POR HACER

st.markdown('''
            <h4 style="font-weight: bold;">En Busca de Relaciones: El Mapa de Conexiones</h4>
            ''',unsafe_allow_html=True)
