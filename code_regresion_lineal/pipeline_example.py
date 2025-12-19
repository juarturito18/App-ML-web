"""
EJEMPLO DE PIPELINE PARA REGRESIÓN LINEAL
==========================================

¿QUÉ ES UN PIPELINE?
--------------------
Un Pipeline en scikit-learn es una cadena de transformaciones seguidas por un estimador (modelo).
Permite aplicar múltiples pasos de preprocesamiento y entrenamiento de forma secuencial y ordenada.

VENTAJAS:
1. Evita data leakage: Las transformaciones se aprenden solo con datos de entrenamiento
2. Código más limpio y organizado
3. Reutilizable: Puedes guardar y cargar el pipeline completo
4. Consistencia: Los mismos pasos se aplican a entrenamiento y prueba
5. Cross-validation: Facilita la validación cruzada sin errores

ESTRUCTURA:
Pipeline = [Transformación 1] → [Transformación 2] → ... → [Modelo Final]

EJEMPLO PRÁCTICO:
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# ===========================================
# EJEMPLO 1: Pipeline Simple
# ===========================================
print("=" * 60)
print("EJEMPLO 1: Pipeline Simple (una sola transformación)")
print("=" * 60)

# Crear datos de ejemplo
np.random.seed(42)
X = np.random.randn(100, 3)
y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.5

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ❌ FORMA ANTIGUA (sin pipeline) - RIESGO DE DATA LEAKAGE
print("\n❌ FORMA ANTIGUA (riesgosa):")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Aprende parámetros
X_test_scaled = scaler.transform(X_test)        # Usa los mismos parámetros
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# ✅ FORMA CON PIPELINE (recomendada)
print("\n✅ FORMA CON PIPELINE (recomendada):")
pipeline = Pipeline([
    ('scaler', StandardScaler()),        # Paso 1: Estandarizar datos
    ('model', LinearRegression())        # Paso 2: Entrenar modelo
])

# El pipeline se entrena TODO junto
pipeline.fit(X_train, y_train)
y_pred_pipeline = pipeline.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred_pipeline):.4f}")
print("\nNota: Ambas formas dan el mismo resultado, pero el pipeline es más seguro y organizado")

# ===========================================
# EJEMPLO 2: Pipeline con Múltiples Pasos
# ===========================================
print("\n" + "=" * 60)
print("EJEMPLO 2: Pipeline con Múltiples Transformaciones")
print("=" * 60)

# Crear datos con valores faltantes
X_with_missing = X.copy()
X_with_missing[10:15, 0] = np.nan  # Agregar valores faltantes

X_train, X_test, y_train, y_test = train_test_split(
    X_with_missing, y, test_size=0.2, random_state=42
)

# Pipeline con múltiples pasos
pipeline_completo = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Paso 1: Llenar valores faltantes
    ('scaler', RobustScaler()),                    # Paso 2: Estandarizar (robusto a outliers)
    ('model', LinearRegression())                  # Paso 3: Entrenar modelo
])

pipeline_completo.fit(X_train, y_train)
y_pred = pipeline_completo.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print("\nPipeline aplicado:")
print("1. Imputación de valores faltantes (solo aprende con train)")
print("2. Estandarización robusta (solo aprende con train)")
print("3. Entrenamiento del modelo")

# ===========================================
# EJEMPLO 3: Cómo funciona internamente
# ===========================================
print("\n" + "=" * 60)
print("EJEMPLO 3: Cómo funciona internamente")
print("=" * 60)

# Ver los pasos del pipeline
print("\nPasos del pipeline:")
for i, (nombre, transformador) in enumerate(pipeline_completo.steps, 1):
    print(f"{i}. {nombre}: {type(transformador).__name__}")

# Acceder a componentes individuales
print(f"\nPuedes acceder a componentes individuales:")
print(f"  - Scaler: {pipeline_completo.named_steps['scaler']}")
print(f"  - Modelo: {pipeline_completo.named_steps['model']}")
print(f"  - Coeficientes del modelo: {pipeline_completo.named_steps['model'].coef_}")

# ===========================================
# EJEMPLO 4: Para tu código test.py
# ===========================================
print("\n" + "=" * 60)
print("EJEMPLO 4: Cómo aplicarlo en test.py")
print("=" * 60)

print("""
En tu archivo test.py, podrías reemplazar:

    # ❌ Código actual (líneas 307-309):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

Por:

    # ✅ Con Pipeline (recomendado):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Opcional: si quieres estandarizar
        ('model', LinearRegression())
    ])
    
    # Entrenar TODO el pipeline
    pipeline.fit(X_train, y_train)
    
    # Predecir (automáticamente aplica las transformaciones)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Para acceder al modelo directamente:
    model = pipeline.named_steps['model']
    print(f"Coeficientes: {model.coef_}")

BENEFICIOS ESPECÍFICOS PARA TU CÓDIGO:
- Si agregas limpieza de datos en clean_data(), puedes integrarla
- Más fácil agregar preprocesamiento (normalización, imputación, etc.)
- El código es más mantenible y profesional
- Si guardas el pipeline, puedes reutilizarlo fácilmente
""")

# ===========================================
# EJEMPLO 5: Guardar y Cargar Pipeline
# ===========================================
print("\n" + "=" * 60)
print("EJEMPLO 5: Guardar y Cargar Pipeline (para producción)")
print("=" * 60)

import joblib

# Guardar pipeline completo
# joblib.dump(pipeline_completo, 'modelo_regresion.pkl')

# Cargar pipeline completo
# pipeline_cargado = joblib.load('modelo_regresion.pkl')

print("""
Con joblib puedes guardar TODO el pipeline:
    - Las transformaciones aprendidas
    - El modelo entrenado
    - Todo en un solo archivo .pkl

Esto es MUY útil para producción, porque solo necesitas:
    pipeline_cargado.predict(nuevos_datos)
    
Y automáticamente aplica TODAS las transformaciones + predicción.
""")

print("\n" + "=" * 60)
print("FIN DE EJEMPLOS")
print("=" * 60)

