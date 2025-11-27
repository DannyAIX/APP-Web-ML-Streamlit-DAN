# Paso1_EDA_bank_marketing.py
import os
import zipfile
import requests
import io
import pandas as pd

# 1) Descargar y extraer dataset desde UCI
uci_zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
r = requests.get(uci_zip_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("data")

# Ruta del CSV principal
csv_path = os.path.join("data", "bank-additional", "bank-additional-full.csv")

# 2) Cargar dataset
df = pd.read_csv(csv_path, sep=';')

# 3) Resumen general
print("FILAS, COLUMNAS:", df.shape)
print("\nTIPOS DE DATOS:\n", df.dtypes)
print("\nNulos por columna:\n", df.isnull().sum())
print("\nDescripción numérica:\n", df.describe().T)

# 4) Distribución del target
print("\nDistribución del target 'y':\n", df['y'].value_counts(dropna=False))
print("\nProporciones:\n", df['y'].value_counts(normalize=True))

# 5) Columnas categóricas y sus valores más comunes
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("\nColumnas categóricas:", categorical_cols)

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts().head(10))

# 6) Correlación numérica con el target
df['y_num'] = df['y'].map({'no': 0, 'yes': 1})

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

corr_matrix = df[num_cols + ['y_num']].corr()

# 🔥 FIX INFALIBLE
corr_target = corr_matrix['y_num']
if isinstance(corr_target, pd.DataFrame):
    corr_target = corr_target.iloc[:, 0]

print("\nCorrelaciones (numéricas con y_num):\n",
      corr_target.sort_values(ascending=False))

# 7) Guardar copia del dataset crudo
os.makedirs("output", exist_ok=True)
df.to_csv("output/bank_additional_full_raw.csv", index=False)
print("\nArchivo guardado: output/bank_additional_full_raw.csv")
