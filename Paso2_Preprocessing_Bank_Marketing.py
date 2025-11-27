import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump

# 1) Cargar dataset crudo
df = pd.read_csv("output/bank_additional_full_raw.csv")

# 2) Separar features y target
X = df.drop(columns=['y', 'y_num'])
y = df['y_num']

# 3) Columnas categóricas y numéricas
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 4) Escalar numéricas
scaler = StandardScaler()
X_num = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols)

# 5) One-hot encoding de categóricas (CORREGIDO: agregado handle_unknown='ignore')
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_cat = pd.DataFrame(encoder.fit_transform(X[cat_cols]),
                     columns=encoder.get_feature_names_out(cat_cols))

# 6) Concatenar numéricas y categóricas
X_processed = pd.concat([X_num, X_cat], axis=1)

# 7) Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y)

# 8) Guardar datasets preprocesados y transformadores
os.makedirs("output", exist_ok=True)
X_train.to_csv("output/X_train.csv", index=False)
X_test.to_csv("output/X_test.csv", index=False)
y_train.to_csv("output/y_train.csv", index=False)
y_test.to_csv("output/y_test.csv", index=False)

# IMPORTANTE: Guardar scaler, encoder y las columnas
dump(scaler, "output/scaler.joblib")
dump(encoder, "output/encoder.joblib")
dump({'cat_cols': cat_cols, 'num_cols': num_cols}, "output/column_info.joblib")

print("✅ Preprocesamiento completado.")
print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test:", y_test.shape)
print("✅ Scaler, encoder y columnas guardados.")