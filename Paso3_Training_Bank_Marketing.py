import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os

# 1) Cargar datos procesados desde Paso 2
X_train = pd.read_csv("output/X_train.csv")
X_test = pd.read_csv("output/X_test.csv")
y_train = pd.read_csv("output/y_train.csv").squeeze()  # Convertimos a Serie 1D
y_test = pd.read_csv("output/y_test.csv").squeeze()    # Convertimos a Serie 1D

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test:", y_test.shape)

# 2) Definir modelos a entrenar
models = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42))
]

# Crear carpeta output si no existe
os.makedirs("output", exist_ok=True)

# 3) Entrenamiento, evaluación y guardado de modelos
for name, model in models:
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy :", accuracy_score(y_test, y_pred_test))
    print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_test))
    
    # Guardar modelo
    dump(model, f"output/{name}.joblib")

print("Modelos guardados en la carpeta output/")

""" Logistic Regression: Accuracy 91–92%, buen rendimiento general, pero recall del target positivo 
(clientes que sí se suscriben) bajo (43%).

Random Forest: Perfecto en entrenamiento (sobreajuste), test similar a LR, recall un poco mejor (50%).

Gradient Boosting: Mejor balance entre precisión y recall (55%), test accuracy más alto (92.17%), 
menos sobreajuste que Random Forest.
"""
