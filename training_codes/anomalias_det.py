import cv2
import os
import numpy as np
from scipy import ndimage

# Directorio de entrada (refined_masks) y salida (mascaras_recuperadas)
directorio_entrada = './metricas/refined_masks'
directorio_salida = './metricas/mascaras_recuperadas'

# Asegurémonos de que el directorio de salida exista
if not os.path.exists(directorio_salida):
    os.makedirs(directorio_salida)

# Listar archivos en el directorio de entrada
archivos = os.listdir(directorio_entrada)

for archivo in archivos:
    # Cargar la imagen desde el directorio de entrada
    ruta_entrada = os.path.join(directorio_entrada, archivo)
    imagen = cv2.imread(ruta_entrada)

    # Reducir la imagen a 1280x1280 y quedarse con un valor cada 4 casillas
    imagen_reducida = imagen[::2, ::2]

    # Guardar la imagen reducida en el directorio de salida
    ruta_salida = os.path.join(directorio_salida, archivo)
    cv2.imwrite(ruta_salida, imagen_reducida)

# Directorio que contiene las imágenes
directorio = directorio_salida

# Obtener el nombre de la carpeta actual (LOO_XXXXX)
nombre_carpeta_actual = os.path.basename(os.getcwd())[4:]  # Elimina "LOO_" del nombre de la carpeta

# Cargar el archivo "XXXXX_indices.npy"
indices = np.load(f"{nombre_carpeta_actual}_indices.npy")

# Número de archivos en el directorio de entrada
num_archivos_entrada = len(archivos)

# Nombres de archivo de las imágenes que quieres leer
nombres_archivos = [f'{nombre_carpeta_actual}_{i}.png' for i in range(num_archivos_entrada)] 

# Inicializar un vector vacío para concatenar los valores
vector_concatenado = np.array([])

# Leer las imágenes en orden y concatenar los valores
for nombre_archivo in nombres_archivos:
    ruta_imagen = os.path.join(directorio, nombre_archivo)
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)  # Lee la imagen en escala de grises

    # Aplanar la matriz de la imagen y agregar sus valores al vector
    vector_concatenado = np.concatenate((vector_concatenado, imagen.ravel()), axis=None)

# Normalizar el vector concatenado
vector_concatenado = vector_concatenado / 255

# Obtener el nombre del archivo de máscara correspondiente (XXXXX_máscara.npy)
nombre_mascara = f"{nombre_carpeta_actual}_máscara.npy"

# Cargar el archivo de máscara
mascara = np.load(nombre_mascara)

# Obtener los índices donde el valor de la máscara es igual a 1
indices_unos = np.where(mascara == 1)[0]

# Inicializar variables para el inicio y fin de los tramos de "1"
inicio_tramo = None
fin_tramo = None

# Inicializar una lista para almacenar los tramos de "1"
tramos_unos = []

# Recorrer los índices de "1"
for indice in indices_unos:
    if inicio_tramo is None:
        # Si es el primer "1" encontrado, establecer el inicio del tramo
        inicio_tramo = indice
    elif indice == fin_tramo + 1:
        # Si el índice es consecutivo al tramo actual, extender el tramo
        fin_tramo = indice
    else:
        # Si el índice no es consecutivo, finalizar el tramo actual y comenzar uno nuevo
        tramos_unos.append((inicio_tramo, fin_tramo))
        inicio_tramo = indice
    fin_tramo = indice

# Agregar el último tramo a la lista si es necesario
if inicio_tramo is not None:
    tramos_unos.append((inicio_tramo, fin_tramo))

# Calcular el porcentaje de tramos que superan el umbral de 50%
total_tramos = len(tramos_unos)
tramos_al_menos_50_porcentaje = 0
umbral_porcentaje = 50  # Cambia este valor si deseas un umbral diferente

for inicio, fin in tramos_unos:
    subvector = vector_concatenado[inicio:fin + 1]
    porcentaje_unos = (np.sum(subvector == 1) / len(subvector)) * 100
    if porcentaje_unos >= umbral_porcentaje:
        tramos_al_menos_50_porcentaje += 1

porcentaje_al_menos_50 = (tramos_al_menos_50_porcentaje / total_tramos) * 100

print(f"Porcentaje de anomalías detectadas ({umbral_porcentaje}% de la anomalía marcada): {porcentaje_al_menos_50:.2f}%")

