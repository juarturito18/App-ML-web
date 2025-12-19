# archivo: app.py
import streamlit as st
import streamlit_shadcn_ui as ui
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

#PS C:\Users\Usuario\Documents\App-ML-web\code_regresion_lineal> streamlit run test.py
#Comando para inicia el codigo 
#.\venv\Scripts\Activate.ps1 - Para activar el entorno virtual

def selector_data_training(): #Funcion para seleccionar el dataset a usar

    st.sidebar.header("Seleccionar el dataset a usar")
    datasets ={"Elige un dataset": None,
            "Precio de casa":r"C:\Users\Usuario\Documents\App-ML-web\data_regresion_lineal\housing.csv",
           "Salarios":r"C:\Users\Usuario\Documents\App-ML-web\data_regresion_lineal\Salary_dataset.csv"}
    select = st.sidebar.selectbox(
        "Elige el dataset a utilizar",
        list(datasets.keys())
    )
    upload_dataset = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv", "json"])
    if datasets[select] is  None and upload_dataset is not None:
        datasets[select] = upload_dataset
    return datasets[select]

def clean_data(df): #Funcion para limpiar los datos
    
    pass


def split_data_training(): #Función para seleccionar el porcentaje de datos a usar para training
    porcentaje = st.sidebar.slider(
        "Porcentaje de datos para entrenamiento (%)", 
        min_value=10, 
        max_value=90, 
        value=80, 
    )
    return porcentaje / 100  # Convertir a decimal (0.1 a 0.9)

def calculate_metrics(y_true, y_pred): #Funcion para calcular las metricas de evaluacion del modelo
    """Calcula todas las métricas de evaluación del modelo"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error) - Evitar división por cero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

def show_model_metrics(y_train, y_pred_train, y_test, y_pred_test, model, target, features):
    """Función completa para mostrar todas las métricas y scores del modelo"""
    
    # Calcular métricas para entrenamiento y prueba
    metrics_train = calculate_metrics(y_train, y_pred_train)
    metrics_test = calculate_metrics(y_test, y_pred_test)
    
    st.subheader("📈 Métricas y Scores del Modelo")
    
    # Mostrar ecuación del modelo
    st.markdown("### Ecuación del Modelo")
    equation = f"**{target} = {model.intercept_:.2f}**"
    for coef, feat in zip(model.coef_, features):
        sign = "+" if coef >= 0 else ""
        equation += f" {sign} {coef:.4f}·{feat}"
    st.write(equation)
    
    # Mostrar métricas en columnas
    st.markdown("### 📊 Métricas de Evaluación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Datos de Entrenamiento")
        st.metric("R² Score", f"{metrics_train['R²']:.4f}", 
                 help="Coeficiente de determinación. Más cercano a 1 es mejor.")
        st.metric("RMSE", f"{metrics_train['RMSE']:.4f}",
                 help="Raíz del Error Cuadrático Medio. Más cercano a 0 es mejor.")
        st.metric("MSE", f"{metrics_train['MSE']:.4f}",
                 help="Error Cuadrático Medio. Más cercano a 0 es mejor.")
        st.metric("MAE", f"{metrics_train['MAE']:.4f}",
                 help="Error Absoluto Medio. Más cercano a 0 es mejor.")
        if not np.isnan(metrics_train['MAPE']):
            st.metric("MAPE", f"{metrics_train['MAPE']:.2f}%",
                     help="Error Porcentual Absoluto Medio. Más cercano a 0% es mejor.")
        else:
            st.metric("MAPE", "N/A", help="No se puede calcular (valores cero en datos)")
    
    with col2:
        st.markdown("#### 🧪 Datos de Prueba")
        st.metric("R² Score", f"{metrics_test['R²']:.4f}",
                 help="Coeficiente de determinación. Más cercano a 1 es mejor.")
        st.metric("RMSE", f"{metrics_test['RMSE']:.4f}",
                 help="Raíz del Error Cuadrático Medio. Más cercano a 0 es mejor.")
        st.metric("MSE", f"{metrics_test['MSE']:.4f}",
                 help="Error Cuadrático Medio. Más cercano a 0 es mejor.")
        st.metric("MAE", f"{metrics_test['MAE']:.4f}",
                 help="Error Absoluto Medio. Más cercano a 0 es mejor.")
        if not np.isnan(metrics_test['MAPE']):
            st.metric("MAPE", f"{metrics_test['MAPE']:.2f}%",
                     help="Error Porcentual Absoluto Medio. Más cercano a 0% es mejor.")
        else:
            st.metric("MAPE", "N/A", help="No se puede calcular (valores cero en datos)")
    
    # Comparación de rendimiento
    st.markdown("### 🔍 Análisis de Rendimiento")
    
    # Diferencia entre entrenamiento y prueba
    diff_r2 = metrics_train['R²'] - metrics_test['R²']
    diff_rmse = metrics_test['RMSE'] - metrics_train['RMSE']
    
    col3, col4 = st.columns(2)
    
    with col3:
        if diff_r2 < 0.05:
            st.success(f"✅ **Buen ajuste:** La diferencia en R² entre entrenamiento y prueba es {diff_r2:.4f}. El modelo generaliza bien.")
        elif diff_r2 < 0.15:
            st.warning(f"⚠️ **Ajuste moderado:** La diferencia en R² es {diff_r2:.4f}. El modelo puede estar sobreajustándose ligeramente.")
        else:
            st.error(f"❌ **Sobreajuste:** La diferencia en R² es {diff_r2:.4f}. El modelo está memorizando los datos de entrenamiento.")
    
    with col4:
        if metrics_test['R²'] > 0.8:
            st.success(f"✅ **Excelente modelo:** R² de prueba = {metrics_test['R²']:.4f}")
        elif metrics_test['R²'] > 0.6:
            st.info(f"ℹ️ **Modelo aceptable:** R² de prueba = {metrics_test['R²']:.4f}")
        else:
            st.warning(f"⚠️ **Modelo mejorable:** R² de prueba = {metrics_test['R²']:.4f}. Considera revisar las variables o el modelo.")
    
    # Tabla comparativa
    st.markdown("### 📋 Tabla Comparativa de Métricas")
    comparison_df = pd.DataFrame({
        'Métrica': ['R² Score', 'RMSE', 'MSE', 'MAE', 'MAPE (%)'],
        'Entrenamiento': [
            f"{metrics_train['R²']:.4f}",
            f"{metrics_train['RMSE']:.4f}",
            f"{metrics_train['MSE']:.4f}",
            f"{metrics_train['MAE']:.4f}",
            f"{metrics_train['MAPE']:.2f}%" if not np.isnan(metrics_train['MAPE']) else "N/A"
        ],
        'Prueba': [
            f"{metrics_test['R²']:.4f}",
            f"{metrics_test['RMSE']:.4f}",
            f"{metrics_test['MSE']:.4f}",
            f"{metrics_test['MAE']:.4f}",
            f"{metrics_test['MAPE']:.2f}%" if not np.isnan(metrics_test['MAPE']) else "N/A"
        ],
        'Diferencia': [
            f"{diff_r2:.4f}",
            f"{diff_rmse:.4f}",
            f"{metrics_test['MSE'] - metrics_train['MSE']:.4f}",
            f"{metrics_test['MAE'] - metrics_train['MAE']:.4f}",
            f"{metrics_test['MAPE'] - metrics_train['MAPE']:.2f}%" if not (np.isnan(metrics_test['MAPE']) or np.isnan(metrics_train['MAPE'])) else "N/A"
        ]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Explicación de métricas
    with st.expander("📚 ¿Qué significan estas métricas?"):
        st.markdown("""
        - **R² Score (Coeficiente de Determinación):** 
          - Rango: 0 a 1 (puede ser negativo si el modelo es muy malo)
          - Indica qué tan bien el modelo explica la variabilidad de los datos
          - **1.0 = Perfecto** | **>0.8 = Excelente** | **>0.6 = Bueno** | **<0.5 = Mejorable**
        
        - **RMSE (Raíz del Error Cuadrático Medio):**
          - Mide el error promedio en las mismas unidades que la variable objetivo
          - Penaliza más los errores grandes
          - **Más cercano a 0 = Mejor**
        
        - **MSE (Error Cuadrático Medio):**
          - Similar al RMSE pero en unidades cuadradas
          - Penaliza mucho los errores grandes
          - **Más cercano a 0 = Mejor**
        
        - **MAE (Error Absoluto Medio):**
          - Error promedio en valor absoluto
          - No penaliza tanto los errores grandes como MSE/RMSE
          - **Más cercano a 0 = Mejor**
        
        - **MAPE (Error Porcentual Absoluto Medio):**
          - Error expresado como porcentaje
          - Útil para comparar modelos con diferentes escalas
          - **Más cercano a 0% = Mejor**
        """)

def show_results_model(X_train, y_train, X_test, y_test, model, target, features, tipo="ambos"): 
    """Función para calcular predicciones y retornarlas (mantiene compatibilidad)"""
    if tipo == "entrenamiento" or tipo == "ambos":
        y_pred_train = model.predict(X_train)
    
    if tipo == "prueba" or tipo == "ambos":
        y_pred_test = model.predict(X_test)
    
    if tipo == "ambos":
        return y_pred_train, y_pred_test
    elif tipo == "entrenamiento":
        return y_pred_train
    else:
        return y_pred_test

def show_graph_variables(X_train, y_train, X_test, y_test, y_pred_train, y_pred_test, target, features):
    if len(features) == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de entrenamiento
        ax1.scatter(X_train, y_train, color="blue", alpha=0.6, label="Datos reales")
        ax1.plot(X_train, y_pred_train, color="red", linewidth=2, label="Predicción")
        ax1.set_xlabel(features[0])
        ax1.set_ylabel(target)
        ax1.set_title("📊 Datos de Entrenamiento")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de prueba
        ax2.scatter(X_test, y_test, color="green", alpha=0.6, label="Datos reales")
        ax2.plot(X_test, y_pred_test, color="orange", linewidth=2, label="Predicción")
        ax2.set_xlabel(features[0])
        ax2.set_ylabel(target)
        ax2.set_title("🧪 Datos de Prueba")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Explicación de los gráficos:**
        - **Gráfico izquierdo (Entrenamiento):** Muestra cómo el modelo se ajusta a los datos con los que fue entrenado
        - **Gráfico derecho (Prueba):** Muestra cómo el modelo predice datos que nunca ha visto. 
          Si el modelo generaliza bien, ambos gráficos deberían verse similares.
        """)

st.title("📊 Regresión Lineal Interactiva")

st.markdown("La **regresión lineal** es uno de los algoritmos más simples y fundamentales en el aprendizaje automático. " \
"Su objetivo es encontrar la relación entre una o más variables independientes y una variable dependiente, ajustando una línea recta que mejor represente los datos." \
" Gracias a su facilidad de interpretación y aplicación, la regresión lineal es ideal para comprender los conceptos básicos de modelado predictivo y sentar las bases para algoritmos más complejos.")

# 1. Cargar datos

datasets = selector_data_training()

uploaded_file = datasets

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Sube un archivo CSV para comenzar")
    st.stop()

st.write("### Vista previa de los datos")

df_edit = st.data_editor(df)

st.markdown("Lo primero a analizar de nuestro conjunto de datos o dataset, es saber cual es nuestra ***variable objetivo*** " \
"y nuestra ***variable predictora***, esto porque nuestro objetivo es predecir el comportamiento de nuestra variable " \
"en base a las condiciones que tenga la variable predictora y a si tener una linea de predicción.")
st.info("Selecciona las variables que deseas analizar y el porcentaje de datos para entrenamiento")

# 2. Seleccionar variables
target = st.sidebar.selectbox("Variable objetivo (Y)", df_edit.columns)
features = st.sidebar.multiselect("Variables predictoras (X)", df_edit.columns, default=[col for col in df_edit.columns if col != target])

# 2.1. Configuración de división de datos
st.sidebar.header("⚙️ Configuración del Modelo")
porcentaje_train = split_data_training()

if not features:
    st.warning("Selecciona al menos una variable predictora")
    st.stop()

if len(features) == 1:
    plt.scatter(df_edit[features], df_edit[target], color="blue")
    plt.xlabel(features[0])
    plt.ylabel(target)
    st.pyplot(plt)

st.markdown(f"""A primera vista como observamos en este grafico con solo los datos, 
podemos ver que existe una **relación** entre nuestras variables donde son directamente proporcionas donde podemos
que nuestra variable objetivo: **'{target}'**, tiene una cierta relación con nuestra variable predictora **'{features[0]}**'
""")
# 3. Preparar datos y dividir en entrenamiento y prueba
X = df_edit[features]
y = df_edit[target]

# Dividir datos usando train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=1 - porcentaje_train,  # El complemento del porcentaje de entrenamiento
    random_state=42  # Semilla para reproducibilidad
)

# Mostrar información de la división
st.info(f"📊 **División de datos:** {len(X_train)} registros para entrenamiento ({porcentaje_train*100:.0f}%) | {len(X_test)} registros para prueba ({(1-porcentaje_train)*100:.0f}%)")

# 4. Entrenar modelo SOLO con datos de entrenamiento
model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("Entrenar al modelo")
model_train = f"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size={1 - porcentaje_train:.2f},  # {(1-porcentaje_train)*100:.0f}% para prueba
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)  # Entrenar SOLO con datos de entrenamiento
"""
st.code(model_train)

st.markdown(f""" Luego de definir cual es nuestra intención por la que vamos a analizar las variables
            tenemos que **dividir nuestros datos** entre entrenamiento y prueba. Esto es importante porque
            necesitamos datos que el modelo nunca haya visto para evaluar su rendimiento real.
            
            Después, entrenamos el modelo con **'model.fit(X_train, y_train)'** usando SOLO los datos de entrenamiento.
            Esto permite que el modelo aprenda los patrones sin "memorizar" los datos.
""")

# 5. Calcular predicciones
y_pred_train, y_pred_test = show_results_model(X_train, y_train, X_test, y_test, model, target, features, tipo="ambos")

# 6. Mostrar métricas completas del modelo
show_model_metrics(y_train, y_pred_train, y_test, y_pred_test, model, target, features)

# 7. Visualización si es una sola variable
show_graph_variables(X_train, y_train, X_test, y_test, y_pred_train, y_pred_test, target, features)
