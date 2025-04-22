# 🚲 Análisis Predictivo de BiciMAD - Optimización de Operaciones

## 📋 Descripción

Este proyecto corresponde al Trabajo Fin de Máster en Ciencia de Datos. Su objetivo es analizar patrones de uso del sistema de bicicletas compartidas BiciMAD en Madrid, calcular distancias entre estaciones y aplicar modelos de regresión para estimar comportamientos de uso, como parte de una propuesta de mejora operativa.

## ⚙️ Funcionalidades Principales

- Carga y procesamiento de archivos `.json` con datos históricos anonimizados de BiciMAD.
- Cálculo de distancias entre estaciones utilizando la fórmula de Haversine.
- Generación automática de dataset final con métricas útiles para el análisis.
- Entrenamiento y evaluación de modelos de regresión.
- Visualización de la distribución de distancias de viaje.
- Almacenamiento de resultados en Google Drive.

## 🛠️ Tecnologías Utilizadas

- Python 3.x
- `pandas` para manipulación de datos
- `scikit-learn` para modelos de machine learning
- `matplotlib` y `seaborn` para visualización
- Google Colab + API de Google Drive para ejecución y almacenamiento

## 📊 Modelos de Machine Learning

- Regresión Lineal
- Ridge Regression
- SGD Regressor

## 📦 Estructura del Proyecto

```bash
.
├── bicimad_modelo.ipynb     # Notebook principal con el pipeline de análisis
├── datos/                   # Archivos .json con datos de estaciones
├── viajes_final.csv         # Archivo final generado con resultados
└── README.md
