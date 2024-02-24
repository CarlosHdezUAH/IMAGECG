import os
import numpy as np

# Rutas de las carpetas
carpeta_muestras = "muestras_registros_totales_norm"
carpeta_anomalias = "anomalias"
carpeta_anomalias_masks = "anomalias_masks_1280"

# Crear la carpeta "anomalias_masks" si no existe
if not os.path.exists(carpeta_anomalias_masks):
    os.mkdir(carpeta_anomalias_masks)

# Listar los archivos de señales en "muestras_registros_totales_norm"
archivos_muestras = [f for f in os.listdir(carpeta_muestras) if f.endswith('.npy')]

valores_a_b = {
    "14046": (12, 40),  
    "14134": (10, 55),  
    "14157": (10, 55),  
    "14172": (15, 75), 
    "14184": (15, 50)
}

num_column = 1280
# Procesar cada archivo de señal
for archivo_muestra in archivos_muestras:
    # Obtener el número de señal
    numero_muestra = archivo_muestra.split("_")[0]
    
    if numero_muestra in valores_a_b:
        a, b = valores_a_b[numero_muestra]
        
    # Cargar la señal desde el archivo .npy
    señal = np.load(os.path.join(carpeta_muestras, archivo_muestra))

    # Cargar los índices de anomalías correspondientes
    archivo_indices = f"{numero_muestra}_indices.npy"
    if archivo_indices in os.listdir(carpeta_anomalias):
        indices_anomalías = np.load(os.path.join(carpeta_anomalias, archivo_indices))
    else:
        indices_anomalías = []

    # Crear la matriz de máscaras
    longitud_señal = len(señal)
    máscara = np.zeros(longitud_señal)

    for índice_anomalía in indices_anomalías:
        inicio = max(0, índice_anomalía - a)
        fin = min(longitud_señal, índice_anomalía + b + 1)
        máscara[inicio:fin] = 1
        
        indice_anterior = índice_anomalía - num_column
        
        inicio = max(0, indice_anterior - a)
        fin = min(longitud_señal, indice_anterior + b + 1)
        if fin >= 0:
            máscara[inicio:fin] = 1
        
        
        indice_posterior = índice_anomalía + num_column
        
        inicio = max(0, indice_posterior - a)
        fin = min(longitud_señal, indice_posterior + b + 1)
        if fin >= 0:
            máscara[inicio:fin] = 1

    # Guardar la matriz de máscaras en un archivo .npy
    nombre_archivo_máscara = f"{numero_muestra}_máscara.npy"
    np.save(os.path.join(carpeta_anomalias_masks, nombre_archivo_máscara), máscara)

