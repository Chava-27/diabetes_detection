"""
Módulo de Reportes Finales y Dashboards Avanzados.
Optimizado para alta legibilidad con rotaciones de etiquetas y espaciados corregidos.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def exportar_metricas_csv(historico_resultados: List[Dict[str, Any]], ruta_carpeta: str):
    """Convierte los resultados en un DataFrame añadiendo el porcentaje de confianza y escribe metrics.csv."""
    df_resultados = pd.DataFrame(historico_resultados)
    if "matriz_confusion" in df_resultados.columns:
        df_resultados = df_resultados.drop(columns=["matriz_confusion"])
    
    df_resultados['confianza_porcentaje'] = df_resultados['f1'] * 100
    ruta_csv = os.path.join(ruta_carpeta, "metrics.csv")
    df_resultados.to_csv(ruta_csv, index=False, encoding='utf-8')

def generar_reporte_markdown(historico_resultados: List[Dict[str, Any]], ruta_carpeta: str):
    """Crea un documento resumido tipo bitácora en formato Markdown."""
    df = pd.DataFrame(historico_resultados)
    ruta_md = os.path.join(ruta_carpeta, "reporte_final.md")
    
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Reporte de Resultados Finales - Detección de Diabetes\n\n")
        f.write("## Tabla Comparativa General\n\n")
        f.write("| Dataset | Modelo | Accuracy | Precision | Recall | F1-Score | Confianza (%) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for _, fila in df.iterrows():
            confianza = fila['f1'] * 100
            f.write(f"| {fila['dataset']} | {fila['model']} | {fila['accuracy']:.4f} | {fila['precision']:.4f} | {fila['recall']:.4f} | {fila['f1']:.4f} | {confianza:.2f}% |\n")

def generar_reporte_imagen(historico_resultados: List[Dict[str, Any]], ruta_carpeta: str):
    """
    Genera una infografía ejecutiva de alta resolución corrigiendo problemas de solapamiento:
    - Etiquetas de datos en las barras en posición VERTICAL (90°).
    - Nombres de los ejes X en posición DIAGONAL (45°).
    - Distribución de espaciado extendida para mitigar colisiones de texto.
    """
    df = pd.DataFrame(historico_resultados)
    df['confianza_pct'] = df['f1'] * 100
    
    # --- 1. CREACIÓN DE ALIAS / LEYENDAS ---
    datasets_unicos = sorted(list(df['dataset'].unique()))
    modelos_unicos = sorted(list(df['model'].unique()))
    
    mapa_ds = {ds: f"datst_{i+1}" for i, ds in enumerate(datasets_unicos)}
    mapa_mdl = {mdl: f"mdl_{i+1}" for i, mdl in enumerate(modelos_unicos)}
    
    df['ds_alias'] = df['dataset'].map(mapa_ds)
    df['mdl_alias'] = df['model'].map(mapa_mdl)
    df['eje_x_alias'] = df['mdl_alias'] + " (" + df['ds_alias'] + ")"
    
    # --- 2. DISEÑO DEL LIENZO EXPANSO ---
    # Incrementamos el ancho a 20 y reajustamos los altos para que entren las 24 filas cómodamente
    fig = plt.figure(figsize=(20, 32))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.8, 6.5, 2.0, 5.5, 5.5, 0.8])
    fig.suptitle("Dashboard Ejecutivo: Control de Rendimiento de Modelos", fontsize=22, fontweight='bold', color='#2C3E50', y=0.99)
    
    # --- COMPONENTE 1: CUADRO DE LEYENDAS (SUPERIOR) ---
    ax_leyenda = fig.add_subplot(gs[0, :])
    ax_leyenda.axis('off')
    
    texto_leyenda_ds = " Leyendas de Datasets:\n" + "\n".join([f" • {alias}: {orig}" for orig, alias in mapa_ds.items()])
    texto_leyenda_mdl = " Leyendas de Modelos:\n" + "\n".join([f" • {alias}: {orig}" for orig, alias in mapa_mdl.items()])
    
    ax_leyenda.text(0.01, 0.1, texto_leyenda_ds, fontsize=11, bbox=dict(facecolor='#F8F9F9', edgecolor='#BDC3C7', boxstyle='round,pad=0.8'), va='bottom')
    ax_leyenda.text(0.51, 0.1, texto_leyenda_mdl, fontsize=11, bbox=dict(facecolor='#F8F9F9', edgecolor='#BDC3C7', boxstyle='round,pad=0.8'), va='bottom')
    ax_leyenda.set_title("Mapeo Analítico de Variables (Leyendas de Configuración)", fontsize=14, fontweight='bold', color='#34495E', loc='left', pad=10)

    # --- COMPONENTE 2: TABLA COMPARATIVA GENERAL ---
    ax_tabla1 = fig.add_subplot(gs[1, :])
    ax_tabla1.axis('off')
    ax_tabla1.set_title("Tabla 1: Resumen General de Experimentos (Nomenclatura Filtrada)", fontsize=14, fontweight='bold', color='#34495E', loc='left', pad=10)
    
    columnas_t1 = ['Dataset', 'Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Confianza (%)']
    datos_t1 = []
    for _, fila in df.iterrows():
        datos_t1.append([
            fila['ds_alias'], fila['mdl_alias'],
            f"{fila['accuracy']:.4f}", f"{fila['precision']:.4f}",
            f"{fila['recall']:.4f}", f"{fila['f1']:.4f}", f"{fila['confianza_pct']:.2f}%"
        ])
        
    tabla1 = ax_tabla1.table(cellText=datos_t1, colLabels=columnas_t1, loc='upper center', cellLoc='center')
    tabla1.auto_set_font_size(False)
    tabla1.set_fontsize(10.5)
    tabla1.scale(1, 1.5)  # Escalado vertical óptimo para las celdas
    
    for (f_idx, c_idx), celda in tabla1._cells.items():
        if f_idx == 0:
            celda.set_text_props(weight='bold', color='white')
            celda.set_facecolor('#2C3E50')

    # --- COMPONENTE 3: TABLA DE MEJORES RESULTADOS POR DATASET ---
    ax_tabla2 = fig.add_subplot(gs[2, :])
    ax_tabla2.axis('off')
    ax_tabla2.set_title("Tabla 2: Cuadro de Honor (Máxima Confianza Predictiva Alcanzada por Dataset)", fontsize=14, fontweight='bold', color='#34495E', loc='left', pad=10)
    
    df_mejores = df.loc[df.groupby('ds_alias')['confianza_pct'].idxmax()].sort_values(by='confianza_pct', ascending=False)
    
    columnas_t2 = ['Dataset (Alias)', 'Modelo (Alias)', 'Estructura o Algoritmo Real', 'Porcentaje de Confianza']
    datos_t2 = []
    for _, fila in df_mejores.iterrows():
        datos_t2.append([
            fila['ds_alias'], fila['mdl_alias'], fila['model'], f"{fila['confianza_pct']:.2f}%"
        ])
        
    tabla2 = ax_tabla2.table(cellText=datos_t2, colLabels=columnas_t2, loc='center', cellLoc='center')
    tabla2.auto_set_font_size(False)
    tabla2.set_fontsize(11)
    tabla2.scale(1, 1.6)
    
    for (f_idx, c_idx), celda in tabla2._cells.items():
        if f_idx == 0:
            celda.set_text_props(weight='bold', color='white')
            celda.set_facecolor('#16A085')

    # --- COMPONENTE 4: CUADRANTE DE GRÁFICAS DE MÉTRICAS (2X2) ---
    metricas_graficas = [
        {"clave": "accuracy", "titulo": "Métrica: Accuracy (Exactitud Global)", "color": "#34495E", "pos": gs[3, 0]},
        {"clave": "precision", "titulo": "Métrica: Precision (Precisión de Diagnóstico)", "color": "#2980B9", "pos": gs[3, 1]},
        {"clave": "recall", "titulo": "Métrica: Recall (Sensibilidad / Tasa de Captura)", "color": "#27AE60", "pos": gs[4, 0]},
        {"clave": "f1", "titulo": "Métrica: F1-Score (Equilibrio General armónico)", "color": "#D35400", "pos": gs[4, 1]}
    ]
    
    for m in metricas_graficas:
        ax_g = fig.add_subplot(m["pos"])
        barras = ax_g.bar(df['eje_x_alias'], df[m["clave"]], color=m["color"], edgecolor='black', alpha=0.85, width=0.5)
        
        ax_g.set_title(m["titulo"], fontsize=13, fontweight='bold', color='#2C3E50', pad=12)
        
        # CORRECCIÓN DE ESPACIADO VERTICAL: Subimos el límite superior a 1.35 para que las etiquetas verticales no se corten
        ax_g.set_ylim(0, 1.35)
        ax_g.grid(axis='y', linestyle='--', alpha=0.4)
        
        # CORRECCIÓN 1: Eje X en Diagonal (45 grados, alineación a la derecha)
        ax_g.set_xticks(range(len(df)))
        ax_g.set_xticklabels(df['eje_x_alias'], rotation=45, ha='right', fontsize=9.5)
        
        # CORRECCIÓN 2: Etiquetas numéricas sobre las barras en posición VERTICAL (90 grados)
        for b in barras:
            h = b.get_height()
            ax_g.annotate(f'{h:.3f}',
                          xy=(b.get_x() + b.get_width() / 2, h),
                          xytext=(0, 6), textcoords="offset points", # Separado 6 puntos del tope de la barra
                          ha='center', va='bottom', 
                          fontsize=8.5, fontweight='bold', 
                          rotation=90, color='#2C3E50') # <--- Rotación vertical aplicada aquí

    # --- COMPONENTE 5: BANNER DE CONCLUSIÓN ABSOLUTA ---
    ax_texto = fig.add_subplot(gs[5, :])
    ax_texto.axis('off')
    
    ganador_absoluto = df.loc[df['confianza_pct'].idxmax()]
    
    texto_final = (
        f"CONCLUSIÓN CRÍTICA DEL SISTEMA: Basado en el rendimiento armónico ponderado, el modelo real '{ganador_absoluto['model']}' "
        f"({ganador_absoluto['mdl_alias']}) ejecutado sobre el dataset '{ganador_absoluto['dataset']}' ({ganador_absoluto['ds_alias']}) "
        f"registró el índice de confianza más alto del ecosistema con un valor de: {ganador_absoluto['confianza_pct']:.2f}%."
    )
    
    ax_texto.text(0.5, 0.5, texto_final, fontsize=12, fontweight='bold', color='white',
                  bbox=dict(facecolor='#27AE60', edgecolor='none', boxstyle='round,pad=0.8'),
                  ha='center', va='center')

    # --- AJUSTE MANUAL DE MÁRGENES (Evita colisiones de subplots de manera estricta) ---
    fig.subplots_adjust(top=0.96, bottom=0.04, left=0.06, right=0.94, hspace=0.55, wspace=0.2)
    
    ruta_png = os.path.join(ruta_carpeta, "reporte_imagen.png")
    plt.savefig(ruta_png, dpi=220, bbox_inches='tight')
    plt.close()