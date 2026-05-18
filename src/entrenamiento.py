"""
Módulo de Entrenamiento y Optimización.
Orquesta las ejecuciones de GridSearchCV aplicando StratifiedKFold.
"""
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.configuracion import (
    GRID_HIPERPARAMETROS, MODELOS_OPCIONALES, 
    GRID_HIPERPARAMETROS_OPCIONALES, obtener_configuracion_xgboost
)

def instanciar_modelo_base(nombre_modelo: str) -> Any:
    """Instancia el clasificador de Scikit-Learn o XGBoost correspondiente."""
    modelos = {
        "LogisticRegression": LogisticRegression(random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "SVC": SVC(probability=True, random_state=42),
        "MLPClassifier": MLPClassifier(random_state=42),
        "XGBClassifier": XGBClassifier(random_state=42, **obtener_configuracion_xgboost())
    }
    return modelos[nombre_modelo]

def obtener_mapa_modelos_y_grids() -> Tuple[Dict[str, Any], Dict[str, dict]]:
    """Genera un diccionario consolidado de modelos y grillas activos para el bucle."""
    modelos_activos = {}
    grids_activos = {}
    
    # Cargar obligatorios
    for nombre in GRID_HIPERPARAMETROS.keys():
        modelos_activos[nombre] = instanciar_modelo_base(nombre)
        grids_activos[nombre] = GRID_HIPERPARAMETROS[nombre]
        
    # Cargar opcionales si están habilitados
    for nombre, habilitado in MODELOS_OPCIONALES.items():
        if habilitado:
            modelos_activos[nombre] = instanciar_modelo_base(nombre)
            grids_activos[nombre] = GRID_HIPERPARAMETROS_OPCIONALES[nombre]
            
    return modelos_activos, grids_activos

def ejecutar_grid_search(preprocesador, modelo, param_grid, X_entrenar, y_entrenar) -> GridSearchCV:
    """
    Crea un pipeline unificado (Preprocesamiento + Clasificador) y ejecuta
    la búsqueda de hiperparámetros optimizada mediante validación cruzada estratificada.
    """
    pipeline_completo = Pipeline(steps=[
        ('preprocesador', preprocesador),
        ('classifier', modelo)
    ])
    
    # Validación cruzada de 5 pliegues estratificada
    cv_estratificado = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=pipeline_completo,
        param_grid=param_grid,
        cv=cv_estratificado,
        scoring='f1', # Optimizado para F1-score por desbalances médicos típicos
        n_jobs=-1 if not (hasattr(modelo, "device") and modelo.get_params().get("device") == "cuda") else 1,
        verbose=1
    )
    
    grid_search.fit(X_entrenar, y_entrenar)
    return grid_search