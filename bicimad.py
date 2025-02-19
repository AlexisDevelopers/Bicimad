# Importar librerías necesarias
import requests as req # type: ignore
from bs4 import BeautifulSoup as soup # type: ignore
from zipfile import ZipFile
import io

# Definir URLs de datos y hacer la solicitud
base_url = "https://opendata.emtmadrid.es"
response = req.get(f'{base_url}/Datos-estaticos/Datos-generales-(1)')
page_content = soup(response.text, 'lxml')

# Extraer archivos de datos
links = page_content.find_all('a')
for link in links:
    title_tag = link.get('title')
    if str(title_tag).startswith('Datos'):
        file_url = base_url + link.get('href')
        file_data = req.get(file_url)
        zip_file = ZipFile(io.BytesIO(file_data.content))
        file_list = zip_file.namelist()
        for file_name in file_list:
            extracted_path = zip_file.extract(file_name)
            print(f"Archivo extraído: {extracted_path}")


# Autenticación y subida de archivos a Google Drive usando Colab
from google.colab import auth # type: ignore
auth.authenticate_user()
from googleapiclient.discovery import build # type: ignore
drive_service = build('drive', 'v3')

# Subir archivos procesados a Google Drive
json_files = ['201704.json', '201705.json', '201903.json']
local_files = ['/content/201704_Usage_Bicimad.json', '/content/201705_Usage_Bicimad.json', '/content/201903_Usage_Bicimad.json']
indices = list(range(24))

# Subir múltiples archivos
from googleapiclient.http import MediaFileUpload # type: ignore
for idx in indices:
    metadata = {'name': json_files[idx % len(json_files)], 'mimeType': 'text/plain'}
    media = MediaFileUpload(local_files[idx % len(local_files)], mimetype='text/plain', resumable=True)
    uploaded = drive_service.files().create(body=metadata, media_body=media, fields='id').execute()
    print(f'File ID: {uploaded.get("id")}')



# Preprocesamiento de datos de Bicimad
import pandas as pd # type: ignore

data_dir = '/content/drive/MyDrive/Bicimad/20'
json_files = ["1704.json", "1705.json", "1903.json"]
csv_names = ["201704.csv", "201705.csv", "201903.csv"]

for i, json_file in enumerate(json_files):
    file_path = f"{data_dir}/{json_file}"
    data = pd.read_json(file_path, encoding="ISO8859-1", lines=True)
    data.dropna(inplace=True)
    csv_data = data.to_csv()
    with open(csv_names[i], "w") as file:
        file.write(csv_data)



# Procesar archivos de coordenadas de estaciones y viajes
coords_path = '/content/drive/MyDrive/stations_data.csv'
stations_df = pd.read_csv(coords_path).drop(['Unnamed: 0'], axis=1)

# Calcular distancia entre estaciones
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Radio de la Tierra en metros
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Calcular distancias de todas las estaciones a cada viaje
distances = []
for i in range(len(stations_df)):
    lat1, lon1 = stations_df.loc[i, ['lat', 'lon']]
    for j in range(len(stations_df)):
        lat2, lon2 = stations_df.loc[j, ['lat', 'lon']]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        distances.append(dist)



# Cargar datos y añadir columnas de distancias a DataFrame de viajes
df_viajes = pd.DataFrame(distances, columns=['distancias'])
df_viajes['distancia_media'] = df_viajes['distancias'].mean()

# Subir CSVs finales procesados a Google Drive
csv_final = df_viajes.to_csv()
with open("viajes_final.csv", "w") as file:
    file.write(csv_final)

# Subir archivo CSV a Google Drive
media = MediaFileUpload('/content/viajes_final.csv', mimetype='text/plain', resumable=True)
upload_final = drive_service.files().create(body={'name': 'viajes_final.csv', 'mimeType': 'text/plain'}, media_body=media, fields='id').execute()
print(f'File ID Final: {upload_final.get("id")}')



# Machine Learning: Regresión con Scikit-Learn
from sklearn.model_selection import GridSearchCV, train_test_split # type: ignore
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor # type: ignore
from sklearn.neighbors import KNeighborsRegressor # type: ignore
from sklearn.neural_network import MLPRegressor # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Preprocesamiento de datos
X = df_viajes.drop(['distancias'], axis=1)
y = df_viajes['distancias']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir y ajustar modelos
modelos = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'SGD': SGDRegressor(),
    'KNN': KNeighborsRegressor(),
    'MLP': MLPRegressor(hidden_layer_sizes=(100,50), max_iter=300, activation='relu', solver='adam')
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    score = modelo.score(X_test_scaled, y_test)
    print(f"{nombre}: R²={score:.4f}")

