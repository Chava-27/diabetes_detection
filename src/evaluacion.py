"""
Módulo de Evaluación de Modelos.
Calcula métricas de clasificación estándar sobre el conjunto de test aislado.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluar_modelo_test(modelo_optimizado, X_prueba, y_prueba) -> Dict[str, Any]:
    """
    Calcula predicciones y retorna un diccionario con las métricas de rendimiento evaluadas.
    """
    predicciones = modelo_optimizado.predict(X_prueba)
    
    metricas = {
        "accuracy": accuracy_score(y_prueba, predicciones),
        "precision": precision_score(y_prueba, predicciones, zero_division=0),
        "recall": recall_score(y_prueba, predicciones, zero_division=0),
        "f1": f1_score(y_prueba, predicciones, zero_division=0),
        "matriz_confusion": confusion_matrix(y_prueba, predicciones)
    }
    return metricas