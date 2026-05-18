# 🩺 Diabetes Detection

> **Sistema automatizado de Machine Learning para la detección de diabetes utilizando múltiples datasets, validación cruzada y optimización de hiperparámetros.**

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

> ⚠️ Proyecto con fines educativos y de investigación.
---

## 📌 Descripción General

**Diabetes Detection** es un proyecto de aprendizaje automático desarrollado en Python que permite entrenar, evaluar y comparar múltiples modelos de clasificación para predecir si una persona tiene diabetes (`0 = No`, `1 = Sí`).

El sistema está diseñado para:

- Cargar automáticamente varios datasets en formato CSV.
- Normalizar la variable objetivo a una columna estándar llamada `target`.
- Detectar variables numéricas y categóricas.
- Construir pipelines de preprocesamiento.
- Optimizar hiperparámetros con `GridSearchCV`.
- Aplicar validación cruzada estratificada.
- Evaluar los modelos con métricas estándar.
- Generar reportes, gráficas y artefactos reproducibles.

---

## 🎯 Objetivo del Proyecto

El objetivo es responder preguntas como:

- ¿Qué modelo predice mejor la diabetes?
- ¿Qué dataset ofrece mejores resultados?
- ¿Qué variables tienen mayor influencia?
- ¿Qué tan confiables son las predicciones?

Este proyecto es ideal para:

- Proyectos universitarios.
- Portafolio profesional en GitHub.
- Experimentación en ciencia de datos.
- Base para una API REST o dashboard interactivo.

---

## 🧠 Modelos Implementados

### Modelos principales

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Gaussian Naive Bayes

### Modelos opcionales

6. Support Vector Machine (SVM)
7. MLPClassifier
8. XGBoost

> Los modelos opcionales pueden activarse o desactivarse desde `src/configuracion.py`.

---

## 📊 Métricas de Evaluación

Para cada combinación de dataset y modelo se calculan:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC (opcional)
- Matriz de confusión
- Confianza (%) = `F1-score × 100`

---

## 🗂️ Datasets Compatibles

Coloca tus archivos CSV en:

```text
data/raw/
```

Datasets sugeridos:

- Diabetes Prediction Dataset
- Diabetes Health Indicators Dataset
- Pima Indians Diabetes Dataset
- Otros datasets con una variable objetivo binaria

Algunos datasets que te pueden ayudar son:

- https://www.kaggle.com/datasets/mrsimple07/diabetes-prediction
- https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
- https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset
- https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

El sistema detecta automáticamente la columna objetivo y la renombra como target.

## ⚙️ Flujo de Trabajo

1. Detectar datasets CSV
2. Cargar dataset
3. Normalizar variable objetivo
4. Identificar variables numéricas y categóricas
5. Dividir datos en entrenamiento y prueba
6. Construir pipeline de preprocesamiento
7. Ejecutar GridSearchCV
8. Evaluar modelo optimizado
9. Guardar modelo entrenado
10. Generar métricas, gráficas y reportes

## 🏗️ Arquitectura del Proyecto

```text
diabetes_detection/
│
├── data/
│   ├── raw/                  # Datasets originales
│   └── processed/            # Datasets normalizados
│
├── models/                   # Modelos entrenados (.pkl)
│
├── reports/
│   ├── 01/
│   ├── 02/
│   └── ...
│
├── src/
│   ├── configuracion.py
│   ├── carga_datos.py
│   ├── preprocesamiento.py
│   ├── entrenamiento.py
│   ├── evaluacion.py
│   ├── visualizacion.py
│   ├── reportes.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 📁 Descripción de Módulos

configuracion.py
Centraliza:

- Rutas del proyecto.
- Hiperparámetros.
- Modelos habilitados.
- Configuración general.

carga_datos.py

- Detecta datasets automáticamente.
- Carga archivos CSV.
- Normaliza la variable objetivo.

preprocesamiento.py

- Identifica columnas numéricas y categóricas.
- Construye el ColumnTransformer.
- Realiza train_test_split.

entrenamiento.py

- Instancia modelos.
- Ejecuta GridSearchCV.
- Devuelve el mejor modelo.

evaluacion.py

- Calcula métricas de clasificación.

visualizacion.py

- Genera matrices de confusión y gráficas comparativas.

reportes.py

- Exporta resultados a CSV y Markdown.

main.py

- Orquesta toda la ejecución del proyecto.

## 🧪 Preprocesamiento Automático

Variables numéricas

- Imputación de valores faltantes con la mediana.
- Escalado con StandardScaler.

Variables categóricas

- Imputación con la moda.
- Codificación con OneHotEncoder.

Esto garantiza que el pipeline sea reproducible y evita fugas de información.

## 🔍 Optimización con Grid Search

Cada modelo se optimiza mediante:

- GridSearchCV
- StratifiedKFold (cv=5)
- scoring='f1'

Esto permite seleccionar automáticamente la mejor combinación de hiperparámetros.

## 💻 Requisitos del Sistema

- Python 3.13 o superior
- 8 GB de RAM recomendados
- Windows, Linux o macOS

## 🚀 Instalación

1. Clonar el repositorio

```text
git clone https://github.com/tu-usuario/diabetes_detection.git
cd diabetes_detection
```

2. Crear un entorno virtual
   Windows

```text
python -m venv venv
venv\Scripts\activate
```

Linux/macOS

```text
python3 -m venv venv
source venv/bin/activate
```

3. Instalar dependencias

```text
pip install -r requirements.txt
```

4. Colocar los datasets

Copia tus archivos .csv dentro de:

```text
data/raw/
```

5. Ejecutar el proyecto

```text
python -m src.main
```

## 📈 Salidas Generadas

Cada ejecución crea una carpeta incremental dentro de reports/:

```text
reports/01/
reports/02/
reports/03/
```

Archivos generados:

- metrics.csv
- reporte_final.md
- dashboard_resumen.png
- figures/\*.png

Además, los modelos se guardan en:

```text
models/
```

## 📄 Ejemplo de metrics.csv

| dataset      | model                  | accuracy | precision | recall |    f1 |
| ------------ | ---------------------- | -------: | --------: | -----: | ----: |
| diabetes.csv | RandomForestClassifier |    0.982 |     0.975 |  0.971 | 0.973 |

## 📊 Interpretación de Métricas

Accuracy

Porcentaje total de predicciones correctas.

Precision

De todas las predicciones positivas, cuántas fueron correctas.

Recall

De todos los casos positivos reales, cuántos fueron detectados.

F1-score

Media armónica entre precision y recall.

## 🔬 Justificación Metodológica

Se utiliza esta metodología porque:

- Los datasets contienen variables heterogéneas.
- El preprocesamiento debe integrarse dentro del pipeline.
- GridSearchCV automatiza la optimización.
- La validación cruzada reduce el riesgo de sobreajuste.
- El F1-score es ideal para clases desbalanceadas.

## 🛠️ Personalización

Puedes modificar en src/configuracion.py:

- Modelos habilitados.
- Hiperparámetros.
- Número de folds.
- Métrica principal.
- Rutas de salida.

## 🧭 Próximas Mejoras

- API REST con FastAPI.
- Dashboard con Streamlit.
- Docker.
- SHAP para interpretabilidad.
- Pruebas automatizadas.

## 🤝 Contribuciones

Las contribuciones son bienvenidas.

```text
fork → branch → commit → pull request
```

## 📜 Licencia

Este proyecto se distribuye bajo la licencia MIT.

## 👨‍💻 Autor

Salvador Vargas Pelayo

Proyecto desarrollado con fines académicos y profesionales para el análisis de modelos de clasificación aplicados a la detección de diabetes.

## ⭐ Apóyalo

Si este proyecto te resulta útil:

- Dale una estrella al repositorio.
- Comparte el proyecto.
- Contribuye con mejoras.
