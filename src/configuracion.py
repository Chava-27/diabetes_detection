"""
Módulo de Configuración Global.
Define los hiperparámetros de los modelos, rutas de carpetas,
mapeo de variables objetivo y el soporte para aceleración por GPU.
"""

import os

# --- RUTAS DEL PROYECTO ---
RUTA_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DATOS_RAW = os.path.join(RUTA_BASE, "data", "raw")
RUTA_MODELOS = os.path.join(RUTA_BASE, "models")
RUTA_DATOS_PROCESSED = os.path.join("data", "processed")
RUTA_REPORTES_BASE = os.path.join(RUTA_BASE, "reports")  # Carpeta raíz de reportes

# Asegurar que las carpetas base existan
for carpeta in [RUTA_DATOS_RAW, RUTA_MODELOS, RUTA_REPORTES_BASE]:
    os.makedirs(carpeta, exist_ok=True)


def obtener_siguiente_carpeta_iteracion() -> str:
    """
    Busca en reports/ las carpetas numéricas existentes y determina
    la ruta de la siguiente iteración (01, 02, etc.).
    """
    iteracion = 1
    while True:
        nombre_carpeta = f"{iteracion:02d}"  # Formato de dos dígitos: 01, 02...
        ruta_candidata = os.path.join(RUTA_REPORTES_BASE, nombre_carpeta)
        if not os.path.exists(ruta_candidata):
            os.makedirs(ruta_candidata, exist_ok=True)
            # También creamos una carpeta interna para las figuras individuales de esa iteración
            os.makedirs(os.path.join(ruta_candidata, "figures"), exist_ok=True)
            return ruta_candidata
        iteracion += 1


# --- CONFIGURACIÓN DE HARDWARE (GPU) ---
# Cambiar a False si se desea forzar el uso exclusivo de CPU
USAR_GPU = True


def obtener_configuracion_xgboost() -> dict:
    """Devuelve los parámetros de inicialización de XGBoost según el hardware."""
    if USAR_GPU:
        # Configuración para XGBoost con soporte CUDA en versiones recientes
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "hist", "device": "cpu"}


# --- MAPEO DINÁMICO DE COLUMNAS OBJETIVO ---
# 1. Ampliamos el mapeo de targets para capturar más variantes de Kaggle
MAPEO_TARGET = {
    "Outcome": "target",
    "Diabetes_binary": "target",
    "Diabetes_012": "target",
    "diabetes": "target",
    "Diabetes": "target",
    "class": "target",
    "Diabetes_status": "target",
    "Prediction": "target", # veremos
    "Diagnosis": "target",
}

# --- CONFIGURACIÓN DE MODELOS E HIPERPARAMETROS ---
# 2. Modificamos el solver de LogisticRegression a 'lbfgs' para soportar datasets multiclase (0, 1, 2)
GRID_HIPERPARAMETROS = {
    "LogisticRegression": {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__solver": [
            "lbfgs"
        ],  # 'lbfgs' maneja tanto binaria como multiclase de forma nativa
        "classifier__max_iter": [
            1000
        ],  # Añadimos iteraciones para asegurar convergencia en datos grandes
    },
    "DecisionTreeClassifier": {
        "classifier__max_depth": [3, 5, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
    },
    "RandomForestClassifier": {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [None, 10, 20],
    },
    "KNeighborsClassifier": {
        "classifier__n_neighbors": [3, 5, 7, 9],
        "classifier__weights": ["uniform", "distance"],
    },
    "GaussianNB": {},
}

# Modelos Opcionales (Habilitar cambiando el valor a True si se desea)
MODELOS_OPCIONALES = {
    "SVC": False,  # Support Vector Machine (Costoso en datasets grandes)
    "XGBClassifier": True,  # Habilitado por defecto para probar la optimización por GPU
    "MLPClassifier": False,  # Multi-Layer Perceptron (Red Neuronal)
}

# Hiperparámetros de modelos opcionales
GRID_HIPERPARAMETROS_OPCIONALES = {
    "SVC": {"classifier__C": [0.1, 1, 10], "classifier__kernel": ["linear", "rbf"]},
    "XGBClassifier": {
        "classifier__n_estimators": [50, 100],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
    },
    "MLPClassifier": {
        "classifier__hidden_layer_sizes": [(50,), (50, 25)],
        "classifier__activation": ["relu", "tanh"],
        "classifier__max_iter": [200],
    },
}
