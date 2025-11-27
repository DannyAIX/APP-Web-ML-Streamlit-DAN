import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint, uniform
from joblib import dump

# Cargar datos

X_train = pd.read_csv("output/X_train.csv")
X_test = pd.read_csv("output/X_test.csv")
y_train = pd.read_csv("output/y_train.csv").squeeze()
y_test = pd.read_csv("output/y_test.csv").squeeze()

# Paso 1: RandomizedSearch rápido

gb = GradientBoostingClassifier(random_state=42)

param_dist = {
'n_estimators': randint(100, 400),
'learning_rate': uniform(0.05, 0.2),
'max_depth': randint(3, 6),
'min_samples_split': randint(2, 11),
'subsample': uniform(0.8, 0.2)
}

random_search = RandomizedSearchCV(
estimator=gb,
param_distributions=param_dist,
n_iter=30,
scoring='f1',
cv=3,
n_jobs=-1,
verbose=2,
random_state=42
)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print("RandomizedSearch mejores parámetros:", best_params)

# Paso 2: GridSearch reducido alrededor de los mejores parámetros

param_grid = {
'n_estimators': [best_params['n_estimators']-50, best_params['n_estimators'], best_params['n_estimators']+50],
'learning_rate': [best_params['learning_rate']*0.8, best_params['learning_rate'], best_params['learning_rate']*1.2],
'max_depth': [best_params['max_depth']-1, best_params['max_depth'], best_params['max_depth']+1],
'min_samples_split': [best_params['min_samples_split']-1, best_params['min_samples_split'], best_params['min_samples_split']+1],
'subsample': [max(0.7, best_params['subsample']-0.05), best_params['subsample'], min(1.0, best_params['subsample']+0.05)]
}

grid_search = GridSearchCV(
estimator=gb,
param_grid=param_grid,
scoring='f1',
cv=3,
n_jobs=-1,
verbose=2
)
grid_search.fit(X_train, y_train)
best_gb = grid_search.best_estimator_
print("GridSearch mejores parámetros:", grid_search.best_params_)

# Evaluación final

y_pred = best_gb.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Guardar modelo optimizado

dump(best_gb, "output/GradientBoosting_optimized_hybrid.joblib")
print("Modelo híbrido optimizado guardado como output/GradientBoosting_optimized_hybrid.joblib")
