# Importar bibliotecas
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Subir archivos JSON manualmente desde la barra lateral
# (No se necesita código: haga clic en el icono de carpeta a la izquierda > subir 3 archivos JSON)

# Autenticarse con Google Drive
auth.authenticate_user()
drive_service = build('drive', 'v3')

# Cargar JSON, calcular distancias
json_files = ["202111_movements.json", "202210_movements.json", "202212_movements.json"]
distancias = [] 

# Fórmula de Haversine para calcular la distancia entre dos puntos de latitud y longitud (en metros)
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance between two geographical points using the Haversine formula."""
    R = 6371000 
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c 

# Recorra cada archivo para extraer latitudes y longitudes de la columna 'estaciones' y calcular distancias
for file in json_files:
    if not os.path.exists(file):
        print(f"⚠️ Warning: {file} not found. Skipping.")
        continue
    df = pd.read_json(file, encoding="ISO8859-1", lines=True)
    print(f"✅ Loaded {file} with {len(df)} rows")
    df.dropna(inplace=True) 

   # Extrae latitudes y longitudes de la columna 'estaciones' (que es una lista de diccionarios)
    origin_latitudes = []
    origin_longitudes = []

    # Iterar a través de la lista de 'estaciones' de cada fila
    for row in df['stations']:
        for station in row:  
            if 'latitude' in station and 'longitude' in station:
               # Añadir latitudes y longitudes de las estaciones
                origin_latitudes.append(float(station['latitude'])) 
                origin_longitudes.append(float(station['longitude'])) 

    # Verificar si hay suficientes estaciones (al menos 2) para el cálculo de la distancia
    if len(origin_latitudes) >= 2 and len(origin_longitudes) >= 2:
       # Calcular distancias entre estaciones consecutivas
        for i in range(len(origin_latitudes)-1):  
            try:
                dist = haversine_distance(origin_latitudes[i], origin_longitudes[i], origin_latitudes[i+1], origin_longitudes[i+1])
                distancias.append(dist) 
            except:
                continue

print(f"📊 Distancias totales extraídas: {len(distancias)}")

# Imprima los datos de las estaciones para inspeccionar su estructura
for file in json_files:
    df = pd.read_json(file, encoding="ISO8859-1", lines=True)
    print(f"✅ Loaded {file} with {len(df)} rows")

    # Imprima la primera fila de la columna 'estaciones' para inspeccionar la estructura
    print(df['stations'].head())
    break 

# Cargar JSON, calcular distancias
# Guardar las distancias calculadas en un DataFrame
df_viajes = pd.DataFrame({"distancias": distancias})
df_viajes['distancia_media'] = df_viajes['distancias'].mean() 
df_viajes.to_csv("viajes_final.csv", index=False) 

# Sube el CSV final a Google Drive
media = MediaFileUpload('viajes_final.csv', mimetype='text/plain', resumable=True)
upload_final = drive_service.files().create(
    body={'name': 'viajes_final.csv', 'mimeType': 'text/plain'},
    media_body=media,
    fields='id'
).execute()
print(f'ID de archivo final: {upload_final.get("id")}')

# Entrenar modelos de regresión
# Preparar los datos para el entrenamiento: características (X) y objetivo (Y)
X = df_viajes.drop(['distancias'], axis=1) 
y = df_viajes['distancias'] 

# Reduce the test size to speed up training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) 

# Estandarizar los datos (escalar funciones para un mejor rendimiento del modelo)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir los modelos para entrenamiento (eliminando MLP y KNN para simplificar)
modelos = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'SGD': SGDRegressor(),
}

# Entrene cada modelo e imprima el puntaje R², MSE y MAE
for nombre, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    score = modelo.score(X_test_scaled, y_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{nombre}: R²={score:.4f} | MSE={mse:.2f} | MAE={mae:.2f}")

# Visualización de la distribución de distancias

plt.figure(figsize=(8,5))
sns.histplot(distancias, kde=True, bins=40, color='skyblue')
plt.title('Distribución de Distancias Bicimad')
plt.xlabel('Distancia (m)')
plt.ylabel('Frecuencia de los Viajes')
plt.grid(True)
plt.show()


# Análisis de patrones útiles para recomendaciones operativas

# 1. Estadísticas descriptivas de distancias
print("📈 Estadísticas de las distancias:")
print(df_viajes['distancias'].describe())

# 2. Segmentación de distancias en rangos operativos
df_viajes['rango'] = pd.cut(df_viajes['distancias'],
                             bins=[0, 500, 1000, 2000, 5000, 10000],
                             labels=['0-500m', '501-1000m', '1001-2000m', '2001-5000m', '5001-10000m'])

# 3. Conteo por rango
rango_counts = df_viajes['rango'].value_counts().sort_index()
print("\n📊 Distribución por rangos de distancia:")
print(rango_counts)

# 4. Visualización de la segmentación
plt.figure(figsize=(7,5))
rango_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Frecuencia de Viajes por Rango de Distancia')
plt.xlabel('Rango de Distancia')
plt.ylabel('Cantidad de Viajes')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
