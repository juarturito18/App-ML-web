# archivo: test_con_pipeline.py
# VERSIÓN CON PIPELINE - Ejemplo mejorado
import streamlit as st
import streamlit_shadcn_ui as ui
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

#PS C:\Users\Usuario\Documents\App-ML-web\code_regresion_lineal> streamlit run test_con_pipeline.py

"""
====================================================
¿QUÉ ES UN PIPELINE Y POR QUÉ USARLO?
====================================================

Un Pipeline en scikit-learn es una secuencia de transformaciones 
seguidas de un estimador (modelo). 

VENTAJAS:
1. ✅ Evita "data leakage": Las transformaciones solo aprenden con datos de entrenamiento
2. ✅ Código más limpio: Todo el proceso en un solo objeto
3. ✅ Más seguro: No puedes olvidar aplicar transformaciones a nuevos datos
4. ✅ Fácil de guardar: Todo el proceso en un solo archivo
5. ✅ Mejor para validación cruzada

ESTRUCTURA BÁSICA:
Pipeline = [Transformación] → [Modelo]
Ejemplo: Estandarizar → Entrenar Modelo
"""

def selector_data_training():
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

def clean_data(df):
    """Función para limpiar los datos (puedes agregar lógica aquí)"""
    # Ejemplo: eliminar duplicados, manejar valores faltantes, etc.
    return df

def split_data_training():
    porcentaje = st.sidebar.slider(
        "Porcentaje de datos para entrenamiento (%)", 
        min_value=10, 
        max_value=90, 
        value=80, 
    )
    return porcentaje / 100

def calculate_metrics(y_true, y_pred):
    """Calcula todas las métricas de evaluación del modelo"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
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

def show_model_metrics(y_train, y_pred_train, y_test, y_pred_test, pipeline, target, features):
    """Función completa para mostrar todas las métricas y scores del modelo"""
    
    # Obtener el modelo del pipeline
    model = pipeline.named_steps['model']
    
    # Calcular métricas para entrenamiento y prueba
    metrics_train = calculate_metrics(y_train, y_pred_train)
    metrics_test = calculate_metrics(y_test, y_pred_test)
    
    st.subheader("📈 Métricas y Scores del Modelo")
    
    # Mostrar información del pipeline
    with st.expander("🔧 Ver información del Pipeline"):
        st.markdown("### Componentes del Pipeline:")
        for i, (nombre, componente) in enumerate(pipeline.steps, 1):
            st.write(f"{i}. **{nombre}**: {type(componente).__name__}")
        st.info("💡 El pipeline aplica las transformaciones automáticamente antes de predecir")
    
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
        st.metric("R² Score", f"{metrics_train['R²']:.4f}")
        st.metric("RMSE", f"{metrics_train['RMSE']:.4f}")
        st.metric("MSE", f"{metrics_train['MSE']:.4f}")
        st.metric("MAE", f"{metrics_train['MAE']:.4f}")
        if not np.isnan(metrics_train['MAPE']):
            st.metric("MAPE", f"{metrics_train['MAPE']:.2f}%")
    
    with col2:
        st.markdown("#### 🧪 Datos de Prueba")
        st.metric("R² Score", f"{metrics_test['R²']:.4f}")
        st.metric("RMSE", f"{metrics_test['RMSE']:.4f}")
        st.metric("MSE", f"{metrics_test['MSE']:.4f}")
        st.metric("MAE", f"{metrics_test['MAE']:.4f}")
        if not np.isnan(metrics_test['MAPE']):
            st.metric("MAPE", f"{metrics_test['MAPE']:.2f}%")
    
    # Comparación de rendimiento
    st.markdown("### 🔍 Análisis de Rendimiento")
    diff_r2 = metrics_train['R²'] - metrics_test['R²']
    
    if diff_r2 < 0.05:
        st.success(f"✅ **Buen ajuste:** Diferencia en R² = {diff_r2:.4f}. El modelo generaliza bien.")
    elif diff_r2 < 0.15:
        st.warning(f"⚠️ **Ajuste moderado:** Diferencia en R² = {diff_r2:.4f}")
    else:
        st.error(f"❌ **Sobreajuste:** Diferencia en R² = {diff_r2:.4f}")

def show_results_model(pipeline, X_train, X_test):
    """Función para calcular predicciones usando el pipeline"""
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    return y_pred_train, y_pred_test

def show_graph_variables(X_train, y_train, X_test, y_test, y_pred_train, y_pred_test, target, features):
    if len(features) == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.scatter(X_train, y_train, color="blue", alpha=0.6, label="Datos reales")
        ax1.plot(X_train, y_pred_train, color="red", linewidth=2, label="Predicción")
        ax1.set_xlabel(features[0])
        ax1.set_ylabel(target)
        ax1.set_title("📊 Datos de Entrenamiento")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(X_test, y_test, color="green", alpha=0.6, label="Datos reales")
        ax2.plot(X_test, y_pred_test, color="orange", linewidth=2, label="Predicción")
        ax2.set_xlabel(features[0])
        ax2.set_ylabel(target)
        ax2.set_title("🧪 Datos de Prueba")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

st.title("📊 Regresión Lineal con Pipeline")

st.markdown("""
Esta versión muestra cómo usar **Pipelines de scikit-learn** para mejorar tu código.

### 🔄 ¿Qué cambió?

**ANTES (sin pipeline):**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

**AHORA (con pipeline):**
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)  # ¡Todo junto!
y_pred = pipeline.predict(X_test)  # Aplica transformaciones automáticamente
```

### ✅ Ventajas:
- **Más seguro**: No puedes olvidar aplicar transformaciones
- **Más limpio**: Todo en un solo objeto
- **Más fácil de mantener**: Cambios en un solo lugar
""")

# 1. Cargar datos
datasets = selector_data_training()
uploaded_file = datasets

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)
else:
    st.info("Sube un archivo CSV para comenzar")
    st.stop()

st.write("### Vista previa de los datos")
df_edit = st.data_editor(df)

# 2. Seleccionar variables
target = st.sidebar.selectbox("Variable objetivo (Y)", df_edit.columns)
features = st.sidebar.multiselect("Variables predictoras (X)", df_edit.columns, 
                                   default=[col for col in df_edit.columns if col != target])

# Opción para usar o no estandarización
st.sidebar.header("⚙️ Configuración del Modelo")
usar_escalado = st.sidebar.checkbox("Usar estandarización (StandardScaler)", value=False,
                                    help="Recomendado cuando las variables tienen escalas muy diferentes")
porcentaje_train = split_data_training()

if not features:
    st.warning("Selecciona al menos una variable predictora")
    st.stop()

# 3. Preparar datos y dividir
X = df_edit[features]
y = df_edit[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=1 - porcentaje_train,
    random_state=42
)

st.info(f"📊 **División de datos:** {len(X_train)} registros para entrenamiento ({porcentaje_train*100:.0f}%) | {len(X_test)} registros para prueba ({(1-porcentaje_train)*100:.0f}%)")

# 4. CREAR Y ENTRENAR PIPELINE
st.subheader("🔧 Crear y Entrenar Pipeline")

if usar_escalado:
    # Pipeline con estandarización
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    codigo_pipeline = """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    
    # Crear pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),      # Paso 1: Estandarizar
        ('model', LinearRegression())      # Paso 2: Modelo
    ])
    
    # Entrenar TODO el pipeline
    pipeline.fit(X_train, y_train)
    """
else:
    # Pipeline sin estandarización (solo modelo)
    pipeline = Pipeline([
        ('model', LinearRegression())
    ])
    codigo_pipeline = """
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    
    # Crear pipeline (solo modelo)
    pipeline = Pipeline([
        ('model', LinearRegression())
    ])
    
    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)
    """

st.code(codigo_pipeline)

# Entrenar el pipeline
pipeline.fit(X_train, y_train)

st.success("✅ Pipeline entrenado exitosamente!")

st.markdown("""
### 💡 ¿Cómo funciona el Pipeline?

1. **fit()**: 
   - Si hay scaler: aprende la media y desviación de X_train
   - Entrena el modelo con los datos transformados

2. **predict()**:
   - Aplica las mismas transformaciones aprendidas a los nuevos datos
   - Usa el modelo entrenado para predecir

**Ventaja clave**: Las transformaciones solo se aprenden con datos de entrenamiento,
evitando "data leakage" (filtración de información).
""")

# 5. Calcular predicciones
y_pred_train, y_pred_test = show_results_model(pipeline, X_train, X_test)

# 6. Mostrar métricas
show_model_metrics(y_train, y_pred_train, y_test, y_pred_test, pipeline, target, features)

# 7. Visualización
if len(features) == 1:
    show_graph_variables(X_train[features[0]], y_train, 
                        X_test[features[0]], y_test, 
                        y_pred_train, y_pred_test, target, features)

# 8. Guardar pipeline (opcional)
st.subheader("💾 Guardar Pipeline (para producción)")

with st.expander("Ver código para guardar y cargar pipeline"):
    st.code("""
    # Guardar pipeline completo
    import joblib
    joblib.dump(pipeline, 'mi_modelo_regresion.pkl')
    
    # Cargar pipeline completo
    pipeline_cargado = joblib.load('mi_modelo_regresion.pkl')
    
    # Usar con nuevos datos (aplica transformaciones automáticamente)
    nuevas_predicciones = pipeline_cargado.predict(nuevos_datos)
    """, language='python')
    
    st.info("""
    💡 **Ventaja**: Guardas TODO (transformaciones + modelo) en un solo archivo.
    Cuando cargas y usas el pipeline, automáticamente aplica todas las transformaciones
    aprendidas durante el entrenamiento. ¡Perfecto para producción!
    """)

