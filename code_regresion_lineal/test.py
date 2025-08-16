# archivo: app.py
import streamlit as st
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("📊 Regresión Lineal Interactiva")

st.markdown("La **regresión lineal** es uno de los algoritmos más simples y fundamentales en el aprendizaje automático. " \
"Su objetivo es encontrar la relación entre una o más variables independientes y una variable dependiente, ajustando una línea recta que mejor represente los datos." \
" Gracias a su facilidad de interpretación y aplicación, la regresión lineal es ideal para comprender los conceptos básicos de modelado predictivo y sentar las bases para algoritmos más complejos.")

# 1. Cargar datos
st.sidebar.header("Seleccionar el dataset a usar")

datasets ={"Elige un dataset": None,
            "Precio de casa":"data_regresion_lineal\housing.csv",
           "Salarios":"data_regresion_lineal\Salary_dataset.csv"}

select = st.sidebar.selectbox(
    "Elige el dataset a utilizar",
    list(datasets.keys())
)

uploaded_file = datasets[select]

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

# 2. Seleccionar variables
target = st.sidebar.selectbox("Variable objetivo (Y)", df_edit.columns)
features = st.sidebar.multiselect("Variables predictoras (X)", df_edit.columns, default=[col for col in df_edit.columns if col != target])

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

# 3. Entrenar modelo
X = df_edit[features]
y = df_edit[target]
model = LinearRegression()
model.fit(X, y)

st.subheader("Entrenar al modelo")
model_train = """
X = df_edit[features]
y = df_edit[target]
model = LinearRegression()
model.fit(X, y)
"""
st.code(model_train)

st.markdown(f""" Luego de definir cual es nuestra intención por la que vamos a analizar las variables
            tenemos que entrenar el modelo que, donde vamos a crear y a su vez entrenar el modelo
            esto por medio del **'model = LinearRegression()'**, donde definimos nuestro **modelo de regresión lineal**
            y por medío del **'model.fit(X,y)'** tomar estos datos de variables predictoria (X) y objetiva (y).
""")

# 4. Mostrar resultados
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.subheader("Resultados del modelo")
st.write("Ecuación: ", f"{target} = {model.intercept_:.2f} + " + " + ".join(f"{coef:.2f}·{feat}" for coef, feat in zip(model.coef_, features)))
st.write("MSE:", mse)
st.write("R²:", r2)

# 5. Visualización si es una sola variable
if len(features) == 1:
    plt.scatter(X, y, color="blue")
    plt.plot(X, y_pred, color="red")
    plt.xlabel(features[0])
    plt.ylabel(target)
    st.pyplot(plt)
