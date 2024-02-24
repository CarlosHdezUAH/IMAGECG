import os
import cv2
import numpy as np
from rich.progress import Progress
from rich import print
from rich.table import Table

# Directorio de entrada y salida
directorio_entrada = "./metricas/reconstructed_masks/"
directorio_salida = "./metricas/refined_masks/"

# Crear el directorio de salida si no existe
if not os.path.exists(directorio_salida):
    os.makedirs(directorio_salida)
    
# Obtener la lista de archivos en el directorio de entrada
archivos = os.listdir(directorio_entrada)
num_archivos = len(archivos)

with Progress() as progress:
    task = progress.add_task("[cyan]Refining masks...", total=num_archivos)	
    # Iterar sobre cada archivo en el directorio de entrada
    for archivo in archivos:
        # Comprobar si es un archivo de imagen
        if archivo.endswith('.png') or archivo.endswith('.jpg') or archivo.endswith('.jpeg'):
            # Obtener la ruta completa del archivo de entrada
            ruta_entrada = os.path.join(directorio_entrada, archivo)
    
            # Cargar la imagen
            imagen = cv2.imread(ruta_entrada, 0)
    
            # Binarizar la imagen (asegurarse de que las manchas sean blancas y el fondo negro)
            _, imagen_binaria = cv2.threshold(imagen, 128, 255, cv2.THRESH_BINARY)
    
            # Obtener dimensiones de la imagen
            alto, ancho = imagen_binaria.shape
    
            # Tamaño de fila de procesamiento
            fila_alto = 2
    
            # Lista para almacenar los rectángulos finales
            rectangulos_finales = []
    
            # Iterar por filas de la imagen
            for fila_inicio in range(0, alto, fila_alto):
                fila_fin = fila_inicio + fila_alto
                # Asegurarse de no exceder los límites de la imagen
                if fila_fin > alto:
                    fila_fin = alto
    
                # Obtener la fila actual
                fila_actual = imagen_binaria[fila_inicio:fila_fin+1, :]
    
                # Encontrar los contornos en la fila actual
                contornos, _ = cv2.findContours(fila_actual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                # Iterar sobre los contornos encontrados en la fila
                for contorno in contornos:
                    # Obtener el rectángulo delimitador ajustado al contorno
                    (x, y, w, h) = cv2.boundingRect(contorno)
    
                    # Crear el rectángulo que ocupe todo el alto de la fila y el ancho correspondiente al contorno detectado
                    rectangulo = ((x, fila_inicio), (x + w, fila_fin))
    
                    # Calcular el área del rectángulo
                    area_rectangulo = w * fila_alto
    
                    # Calcular el área relativa del contorno respecto al rectángulo
                    area_relativa = cv2.contourArea(contorno) / area_rectangulo
    
                    # Verificar si el área relativa cumple la condición del 50% de área del contorno respecto al rectángulo
                    if area_relativa >= 0.0001:
                        rectangulos_finales.append(rectangulo)
    
            # Crear una imagen completamente negra del mismo tamaño que la imagen original
            nueva_imagen = np.zeros_like(imagen)
    
            # Dibujar los rectángulos finales en la nueva imagen
            for rectangulo in rectangulos_finales:
                (x, y), (x2, y2) = rectangulo
                cv2.rectangle(nueva_imagen, (x, y), (x2, y2), (255, 255, 255), -1)
                
            # Obtener el nombre del archivo sin extensión
            nombre_archivo = os.path.splitext(archivo)[0]
    
            # Obtener la ruta completa del archivo de salida
            ruta_salida = os.path.join(directorio_salida, nombre_archivo + '.png')
    
            # Guardar la nueva imagen
            cv2.imwrite(ruta_salida, nueva_imagen)
            progress.update(task, advance=1)
