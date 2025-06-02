# 📊 Predicción de Bicicletas Ancladas en BiciMad

Este proyecto forma parte de un Trabajo Fin de Máster (TFM) cuyo objetivo es construir un modelo de predicción supervisada para estimar la cantidad de bicicletas ancladas (`dock_bikes`) en las estaciones del sistema público BiciMad de Madrid.

## 📁 Estructura del proyecto

- `202111_movements.json`, `202210_movements.json`, `202212_movements.json`: Archivos JSON con los datos históricos de estaciones y movimientos.
- `modelo_bicimad.py` o notebook equivalente: Contiene todo el código para procesar los datos, entrenar el modelo y visualizar los resultados.
- Este `README.md`

## 🚀 ¿Qué hace el modelo?

El modelo predice cuántas bicicletas hay ancladas en una estación en un momento determinado, usando como entrada:

- `number`: ID de la estación
- `total_bases`: Número total de anclajes
- `free_bases`: Anclajes libres disponibles
- `no_available`: Anclajes fuera de servicio
- `reservations_count`: Número de reservas activas

Este enfoque es útil para completar datos faltantes, detectar estaciones desbalanceadas y apoyar la planificación operativa del sistema.

## 🧠 Técnicas utilizadas

- **Regresión Lineal** como modelo base
- **Escalado de datos** con `StandardScaler`
- **Evaluación con métricas**:
  - R² Score
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)

## ⚙️ Resultado del modelo

- **R² Score**: 0.8840 → El modelo explica el 88.4% de la variabilidad de los datos.
- **MAE**: 1.27 → En promedio, se equivoca por 1.27 bicicletas por estación.
- **MSE**: 3.78 → Error cuadrático medio aceptable.

> La diferencia entre el valor real y el predicho es, en general, muy baja, lo que indica una predicción precisa y útil para propósitos operativos.

## 📈 Visualización

El modelo genera una gráfica comparando los valores reales con los valores predichos:

![Scatter plot]([ruta/a/tu/imagen.png](https://drive.google.com/file/d/12U_Qk-G1h6e5etA15PpqSfLCQdKyFz-Q/view?usp=sharing))

## 🛠️ Requisitos

- Python 3.x
- Bibliotecas:
  - `pandas`
  - `json`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Instalación rápida:

```bash
pip install pandas scikit-learn matplotlib seaborn
