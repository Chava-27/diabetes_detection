"""
Módulo de Visualización.
Genera y almacena mapas de calor de las matrices de confusión dentro del directorio de la iteración.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def guardar_matriz_confusion(matriz: np.ndarray, nombre_dataset: str, nombre_modelo: str, ruta_carpeta_iteracion: str):
    """Genera un mapa de calor para la matriz de confusión y lo exporta a la subcarpeta figures/."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f"Matriz de Confusión\n{nombre_modelo} - {nombre_dataset}")
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    # Se guarda dentro de la subcarpeta 'figures' de la iteración actual
    nombre_limpio = f"matriz_{nombre_dataset.replace('.csv', '')}_{nombre_modelo}.png"
    ruta_destino = os.path.join(ruta_carpeta_iteracion, "figures", nombre_limpio)
    
    plt.savefig(ruta_destino, dpi=150)
    plt.close()