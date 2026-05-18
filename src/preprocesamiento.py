"""
Módulo de Preprocesamiento Automático.
Identifica tipos de variables y construye los pipelines de Scikit-Learn.
"""

import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def detectar_tipos_columnas(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifica de forma automática cuáles columnas son numéricas y cuáles categóricas,
    excluyendo la columna objetivo 'target'.
    """
    X = df.drop(columns=['target'])
    columnas_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()
    return columnas_numericas, columnas_categoricas

def preparar_datos_entrenamiento(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el DataFrame en conjuntos de entrenamiento y prueba usando una partición estratificada.
    """
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Stratified split para conservar proporciones de la variable objetivo
    X_entrenar, X_prueba, y_entrenar, y_prueba = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_entrenar, X_prueba, y_entrenar, y_prueba

def crear_pipeline_preprocesamiento(columnas_num: List[str], columnas_cat: List[str]) -> ColumnTransformer:
    """
    Construye un ColumnTransformer automatizado con pipelines independientes para
    variables numéricas y categóricas evitando fugas de información.
    """
    # Pipeline para variables numéricas: Imputación por mediana + Escalado estándar
    pipeline_numerico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para variables categóricas: Imputación por el más frecuente + OneHot Encoding
    pipeline_categorico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combinar transformadores
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', pipeline_numerico, columnas_num),
            ('cat', pipeline_categorico, columnas_cat)
        ],
        remainder='drop'
    )
    return preprocesador