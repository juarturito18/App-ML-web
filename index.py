
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

pages = {
    "📊 Análisis de datos": [
        st.Page("pages/1_Analisis_de_datos.py", title="📊 Análisis de Datos"),
        st.Page("pages/2_preprocesamiento.py", title="⚙️ Preprocesamiento de datos"),
    ],
    "🤖 Modelado y explicación": [
        st.Page("pages/3_Modelo_regresion.py", title="🤖 Modelo"),
        st.Page("pages/4_Explicabilidad.py", title="🔍 Explicabilidad"),
    ],
    "🎯 Práctica": [
        st.Page("pages/5_Ejercicios.py", title="🎯 Ejercicios"),
    ],
}

pg=st.navigation(pages)
pg.run()