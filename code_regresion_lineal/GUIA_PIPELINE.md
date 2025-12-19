# 🔄 Guía de Pipelines en scikit-learn

## ¿Qué es un Pipeline?

Un **Pipeline** (tubería) en scikit-learn es un objeto que encadena múltiples pasos de transformación de datos seguidos de un estimador (modelo). Es como una línea de producción donde los datos pasan por diferentes etapas antes de llegar al modelo final.

## 📊 Estructura Visual

```
Datos Crudos
    ↓
[Transformación 1: Estandarizar]
    ↓
[Transformación 2: Imputar valores faltantes]
    ↓
[Modelo: Regresión Lineal]
    ↓
Predicciones
```

## 🎯 Ventajas de usar Pipelines

### 1. **Evita Data Leakage (Filtración de Datos)**
   - Las transformaciones solo se "aprenden" con datos de entrenamiento
   - Los mismos parámetros aprendidos se aplican a datos de prueba
   - Esto es **CRÍTICO** para tener evaluaciones confiables

### 2. **Código más Limpio y Organizado**
   ```python
   # ❌ Sin Pipeline (múltiples pasos separados)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   model = LinearRegression()
   model.fit(X_train_scaled, y_train)
   y_pred = model.predict(X_test_scaled)
   
   # ✅ Con Pipeline (todo en uno)
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('model', LinearRegression())
   ])
   pipeline.fit(X_train, y_train)
   y_pred = pipeline.predict(X_test)
   ```

### 3. **Más Seguro**
   - No puedes olvidar aplicar transformaciones a nuevos datos
   - Si usas `pipeline.predict()`, automáticamente aplica todos los pasos

### 4. **Fácil de Guardar y Cargar**
   ```python
   # Guardar TODO el proceso (transformaciones + modelo)
   import joblib
   joblib.dump(pipeline, 'modelo.pkl')
   
   # Cargar TODO el proceso
   pipeline_cargado = joblib.load('modelo.pkl')
   pipeline_cargado.predict(nuevos_datos)  # Funciona automáticamente
   ```

### 5. **Mejor para Validación Cruzada**
   - scikit-learn puede hacer cross-validation con pipelines
   - Evita errores comunes de aplicar transformaciones incorrectamente

## 💻 Ejemplo Básico

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Crear pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Nombre: 'scaler', Transformador: StandardScaler()
    ('model', LinearRegression())      # Nombre: 'model', Estimador: LinearRegression()
])

# Entrenar (automáticamente aplica StandardScaler antes de entrenar)
pipeline.fit(X_train, y_train)

# Predecir (automáticamente aplica StandardScaler antes de predecir)
y_pred = pipeline.predict(X_test)
```

## 🔍 Cómo Funciona Internamente

Cuando llamas a `pipeline.fit(X_train, y_train)`:

1. **Paso 1 - Scaler:**
   - `scaler.fit(X_train)` → Aprende la media y desviación estándar de X_train
   - `X_train_scaled = scaler.transform(X_train)` → Estandariza X_train

2. **Paso 2 - Model:**
   - `model.fit(X_train_scaled, y_train)` → Entrena el modelo con datos estandarizados

Cuando llamas a `pipeline.predict(X_test)`:

1. **Paso 1 - Scaler:**
   - `X_test_scaled = scaler.transform(X_test)` → Aplica la MISMA transformación (usando parámetros aprendidos)

2. **Paso 2 - Model:**
   - `y_pred = model.predict(X_test_scaled)` → Predice con el modelo entrenado

## 🛠️ Pipeline con Múltiples Pasos

Puedes agregar tantos pasos como necesites:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression

pipeline_completo = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Paso 1: Llenar valores faltantes
    ('scaler', RobustScaler()),                    # Paso 2: Estandarizar (robusto a outliers)
    ('model', LinearRegression())                  # Paso 3: Modelo
])

pipeline_completo.fit(X_train, y_train)
y_pred = pipeline_completo.predict(X_test)
```

## 🔑 Acceder a Componentes Individuales

```python
# Acceder al modelo entrenado
modelo = pipeline.named_steps['model']
print(f"Coeficientes: {modelo.coef_}")
print(f"Intercepto: {modelo.intercept_}")

# Acceder al scaler
scaler = pipeline.named_steps['scaler']
print(f"Media aprendida: {scaler.mean_}")

# Ver todos los pasos
for nombre, componente in pipeline.steps:
    print(f"{nombre}: {type(componente).__name__}")
```

## 📝 Aplicación para tu código test.py

### Versión Actual (sin Pipeline):
```python
# Líneas 307-309 de test.py
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
```

### Versión Mejorada (con Pipeline):
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Crear pipeline (opcional: agregar StandardScaler si lo necesitas)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Opcional
    ('model', LinearRegression())
])

# Entrenar TODO
pipeline.fit(X_train, y_train)

# Predecir (aplica transformaciones automáticamente)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Acceder al modelo para mostrar coeficientes
model = pipeline.named_steps['model']
```

## ⚠️ Cuándo NO usar Pipeline

- Si solo usas el modelo sin transformaciones
- Si necesitas control manual de cada paso
- Para casos muy simples (puede ser excesivo)

## ✅ Cuándo SÍ usar Pipeline

- ✅ Cuando tienes transformaciones de datos (escalado, imputación, etc.)
- ✅ Cuando quieres código más profesional y mantenible
- ✅ Cuando planeas usar validación cruzada
- ✅ Cuando necesitas guardar/cargar el modelo completo
- ✅ Cuando trabajas en proyectos de producción

## 📚 Transformadores Comunes para Pipelines

```python
from sklearn.preprocessing import (
    StandardScaler,      # Estandarización (media=0, std=1)
    RobustScaler,        # Estandarización robusta a outliers
    MinMaxScaler,        # Escalado a rango [0, 1]
    Normalizer,          # Normalización por fila
)

from sklearn.impute import (
    SimpleImputer,       # Llenar valores faltantes
    KNNImputer,          # Imputación con k-vecinos más cercanos
)

from sklearn.preprocessing import (
    PolynomialFeatures,  # Crear características polinomiales
    OneHotEncoder,       # Codificar variables categóricas
)
```

## 🎓 Ejemplo Completo para Regresión

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Cargar datos
df = pd.read_csv('datos.csv')
X = df[['feature1', 'feature2']]
y = df['target']

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Crear y entrenar pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

# 4. Evaluar
y_pred = pipeline.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 5. Guardar (opcional)
import joblib
joblib.dump(pipeline, 'modelo_completo.pkl')
```

## 🚀 Próximos Pasos

1. **Ejecuta** `pipeline_example.py` para ver ejemplos interactivos
2. **Revisa** `test_con_pipeline.py` para ver cómo integrarlo en tu código
3. **Experimenta** agregando diferentes transformadores a tu pipeline
4. **Prueba** guardar y cargar tu pipeline con `joblib`

---

**¿Preguntas?** Los pipelines son una herramienta poderosa que hace tu código más profesional y seguro. ¡Vale la pena aprenderlos!

