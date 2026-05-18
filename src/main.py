"""
Script Principal (Orquestador).
Ejecuta secuencialmente la canalización de Machine Learning completa.
Organiza automáticamente los modelos entrenados en subcarpetas por iteración.
"""

import os
import joblib
from src.configuracion import RUTA_MODELOS, USAR_GPU, obtener_siguiente_carpeta_iteracion
from src.carga_datos import detectar_datasets, cargar_y_normalizar_dataset
from src.preprocesamiento import detectar_tipos_columnas, preparar_datos_entrenamiento, crear_pipeline_preprocesamiento
from src.entrenamiento import obtener_mapa_modelos_y_grids, ejecutar_grid_search
from src.evaluacion import evaluar_modelo_test
from src.visualizacion import guardar_matriz_confusion
from src.reportes import exportar_metricas_csv, generar_reporte_markdown, generar_reporte_imagen

def ejecutar_pipeline():
    print("="*60)
    print(" INICIANDO PIPELINE DE DETECCIÓN DE DIABETES ")
    print(f" Aceleración por GPU activa: {USAR_GPU}")
    print("="*60)
    
    datasets_disponibles = detectar_datasets()
    if not datasets_disponibles:
        print("[ALERTA] No se encontraron archivos CSV en 'data/raw/'.")
        print("Por favor, descarga los datasets y colócalos en la ruta correspondiente antes de continuar.")
        return

    # 1. Determinar y crear de forma automática la carpeta de la iteración (01, 02, etc.)
    ruta_iteracion_actual = obtener_siguiente_carpeta_iteracion()
    nombre_carpeta_corta = os.path.basename(ruta_iteracion_actual)
    print(f"[REPOSITORIO] Los resultados se almacenarán en la carpeta de iteración: reports/{nombre_carpeta_corta}/\n")

    modelos_activos, grids_activos = obtener_mapa_modelos_y_grids()
    historico_resultados = []
    
    for archivo_csv in datasets_disponibles:
        print(f"\n[DATASET] Procesando origen de datos: {archivo_csv}")
        try:
            df, nombre_ds = cargar_y_normalizar_dataset(archivo_csv)
            columnas_num, columnas_cat = detectar_tipos_columnas(df)
            
            X_entrenar, X_prueba, y_entrenar, y_prueba = preparar_datos_entrenamiento(df)
            preprocesador = crear_pipeline_preprocesamiento(columnas_num, columnas_cat)
            
            for nombre_modelo, objeto_modelo in modelos_activos.items():
                print(f"  -> Entrenando {nombre_modelo} mediante GridSearchCV (5-Fold CV)...")
                param_grid = grids_activos[nombre_modelo]
                
                # Ejecutar búsqueda en malla con validación cruzada
                grid_res = ejecutar_grid_search(preprocesador, objeto_modelo, param_grid, X_entrenar, y_entrenar)
                
                # Evaluar el estimador óptimo en Test
                mejores_metricas = evaluar_modelo_test(grid_res.best_estimator_, X_prueba, y_prueba)
                
                # Almacenar en histórico
                resultado_nodo = {
                    "dataset": nombre_ds,
                    "model": nombre_modelo,
                    "accuracy": mejores_metricas["accuracy"],
                    "precision": mejores_metricas["precision"],
                    "recall": mejores_metricas["recall"],
                    "f1": mejores_metricas["f1"],
                    "best_params": str(grid_res.best_params_),
                    "matriz_confusion": mejores_metricas["matriz_confusion"]
                }
                historico_resultados.append(resultado_nodo)
                
                # Graficar matrices individuales pasándole la carpeta de la iteración
                guardar_matriz_confusion(mejores_metricas["matriz_confusion"], nombre_ds, nombre_modelo, ruta_iteracion_actual)
                
                # --- LOGICA CORREGIDA: ENRUTAMIENTO SEGURO DE BINARIOS ---
                # Creamos la ruta limpia para la carpeta contenedora: models/mejor_iter01
                carpeta_destino_modelos = os.path.join(RUTA_MODELOS, f"mejor_iter{nombre_carpeta_corta}")
                os.makedirs(carpeta_destino_modelos, exist_ok=True)
                
                # Limpiamos el nombre eliminando la extensión .csv
                nombre_ds_limpio = nombre_ds.replace('.csv', '')
                nombre_pkl = f"{nombre_ds_limpio}_{nombre_modelo}.pkl"
                
                # Concatenamos la ruta definitiva sin espacios fantasmas
                ruta_final_modelo = os.path.join(carpeta_destino_modelos, nombre_pkl)
                
                # Guardamos el archivo binario encapsulado
                joblib.dump(grid_res.best_estimator_, ruta_final_modelo)
                print(f"     [MODELO GUARDADO] Guardado correctamente en: {ruta_final_modelo}")
                
        except Exception as e:
            print(f"  [ERROR] Falla crítica procesando el dataset {archivo_csv}: {e}")
            continue

    # Generación de la capa de reportes consolidados dentro de la carpeta autoincremental
    if historico_resultados:
        print(f"\n[REPORTES] Generando entregables finales en 'reports/{nombre_carpeta_corta}/'...")
        exportar_metricas_csv(historico_resultados, ruta_iteracion_actual)
        generar_reporte_markdown(historico_resultados, ruta_iteracion_actual)
        generar_reporte_imagen(historico_resultados, ruta_iteracion_actual)
        print(f"¡Éxito! Iteración {nombre_carpeta_corta} completada de forma correcta.")
    else:
        print("\n[FIN] No se procesaron resultados con éxito.")

if __name__ == "__main__":
    ejecutar_pipeline()