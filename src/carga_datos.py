# src/carga_datos.py

import os
import pandas as pd
from typing import List, Tuple
# Importamos la ruta de procesados desde la configuración
from src.configuracion import RUTA_DATOS_RAW, MAPEO_TARGET

# NOTA: Si no tienes definida RUTA_DATOS_PROCESSED en configuracion.py, 
# puedes definirla aquí directamente de forma segura:
RUTA_DATOS_PROCESSED = os.path.join("data", "processed")

def detectar_datasets() -> List[str]:
    archivos = [f for f in os.listdir(RUTA_DATOS_RAW) if f.endswith('.csv')]
    return archivos

def cargar_y_normalizar_dataset(nombre_archivo: str) -> Tuple[pd.DataFrame, str]:
    """
    Carga un archivo CSV, renombra la variable objetivo a 'target',
    fuerza el formato binario (0/1) y GUARDA el resultado en data/processed/.
    """
    ruta_completa = os.path.join(RUTA_DATOS_RAW, nombre_archivo)
    df = pd.read_csv(ruta_completa)
    
    # Renombrar columnas según el diccionario de mapeo
    columnas_actuales = df.columns
    for col_original, col_nueva in MAPEO_TARGET.items():
        if col_original in columnas_actuales:
            df = df.rename(columns={col_original: col_nueva})
            break
            
    if 'target' not in df.columns:
        raise KeyError(f"No se pudo identificar la columna objetivo en {nombre_archivo}. Columnas: {list(df.columns)}")
    
    # Homogeneización a clasificación binaria (0 = No, 1 = Sí)
    valores_unicos = df['target'].unique()
    if len(valores_unicos) > 2:
        print(f"    [INFO] Dataset multiclase detectado ({valores_unicos}). Convirtiendo a formato binario (0/1)...")
        df['target'] = (df['target'] > 0).astype(int)
        
    # --- NUEVO: GUARDADO FÍSICO EN LA CARPETA PROCESSED ---
    # Nos aseguramos de que la carpeta exista en el disco
    os.makedirs(RUTA_DATOS_PROCESSED, exist_ok=True)
    
    # Creamos la ruta de destino (ej. data/processed/clean_diabetes.csv)
    ruta_guardado = os.path.join(RUTA_DATOS_PROCESSED, f"clean_{nombre_archivo}")
    
    # Guardamos el archivo limpio sin el índice aburrido de pandas
    df.to_csv(ruta_guardado, index=False, encoding='utf-8')
    print(f"    [PROCESSED] Archivo depurado guardado con éxito en: {ruta_guardado}")
    # ------------------------------------------------------
        
    return df, nombre_archivo