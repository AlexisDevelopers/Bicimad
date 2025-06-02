import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar y combinar archivos JSON
json_files = ["202111_movements.json", "202210_movements.json", "202212_movements.json"]
dfs = []

for file in json_files:
    with open(file, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
        lines = [json.loads(line.strip()) for line in lines]
        df_temp = pd.DataFrame(lines)
        dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

# Explode y normalizar columna 'stations'
df_exploded = df.explode('stations')
df_expanded = pd.json_normalize(df_exploded['stations'])

# Variables a usar en el modelo

features = ['number', 'total_bases', 'dock_bikes', 'free_bases', 'no_available', 'reservations_count']
df_model = df_expanded[features].dropna()

# Convertir a numérico y manejar errores
for col in df_model.columns:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

df_model.dropna(inplace=True)

# Definir variables predictoras y variable objetivo
X = df_model.drop('dock_bikes', axis=1)
y = df_model['dock_bikes']

# División y escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo de regresión
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluación del modelo
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# Visualización
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual dock_bikes")
plt.ylabel("Predicted dock_bikes")
plt.title("Actual vs Predicted dock_bikes")
plt.grid(True)
plt.tight_layout()
plt.show()
