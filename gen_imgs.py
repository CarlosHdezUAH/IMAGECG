import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pylab import cm

def colorize(nombre_senal, image, colormap, num_imagenes):
    for j in range(num_imagenes):
        im = image[1280*j:1280*(j+1),:]
        im = (im+1)/2
        im = np.repeat(np.repeat(im, 2, axis=0), 2, axis=1)
        colorized = colormap(im)
        colorized = cv2.cvtColor(np.uint8(255*colorized), cv2.COLOR_RGB2BGR)
        nombre_carpeta = "full_images/images/"
        nombre_senal = nombre_senal.split("_")[0]
        nombre_salida_senal = f"{nombre_senal}_{j}.png"
        ruta_salida = nombre_carpeta + nombre_salida_senal
        cv2.imwrite(ruta_salida, colorized)

def colorize_mask(nombre_mascara, image, colormap, num_imagenes):
    for j in range(num_imagenes):
        im = image[1280*j:1280*(j+1),:]
        im = np.repeat(np.repeat(im, 2, axis=0), 2, axis=1)
        colorized = colormap(im)
        nombre_carpeta = "full_images/masks/"
        nombre_mascara = nombre_mascara.split("_")[0]
        nombre_salida_senal = f"{nombre_mascara}_{j}.png"
        ruta_salida = nombre_carpeta + nombre_salida_senal
        cv2.imwrite(ruta_salida, 255*colorized)
    
def generar_imagenes():
    # Directorio base
    base_dir = "./full_images"
    
    # Rutas completas de las carpetas
    full_images_dir = os.path.join(base_dir, "images")
    full_masks_dir = os.path.join(base_dir, "masks")
    
    # Crear directorio base
    try:
        os.makedirs(base_dir)
        print(f"Directorio creado: {base_dir}")
    except FileExistsError:
        print(f"El directorio {base_dir} ya existe.")
    
    # Crear subdirectorios
    try:
        os.makedirs(full_images_dir)
        print(f"Directorio creado: {full_images_dir}")
    except FileExistsError:
        print(f"El directorio {full_images_dir} ya existe.")
    
    try:
        os.makedirs(full_masks_dir)
        print(f"Directorio creado: {full_masks_dir}")
    except FileExistsError:
        print(f"El directorio {full_masks_dir} ya existe.")
        
    # Lista de nombres de archivos de señales y máscaras
    nombres_senales = ["14046_transformada", "14134_transformada", "14157_transformada", "14172_transformada", "14184_transformada"]
    nombres_mascaras = ["14046_máscara", "14134_máscara", "14157_máscara", "14172_máscara", "14184_máscara"]

    # Directorio de entrada y salida
    carpeta_entrada_senales = "muestras_registros_totales_norm"
    carpeta_entrada_mascaras = "anomalias_masks_1280"

    # Tamaño de recorte
    tamano_recorte = 1280

    for nombre_senal, nombre_mascara in zip(nombres_senales, nombres_mascaras):
        # Cargar señal y máscara
        
        senal = np.load(os.path.join(carpeta_entrada_senales, f"{nombre_senal}.npy"))
        mascara = np.load(os.path.join(carpeta_entrada_mascaras, f"{nombre_mascara}.npy"))
        print(f"La longitud original de la señal {nombre_senal} es de {len(senal)}")
        # Calcula el tamaño objetivo basado en el múltiplo más cercano de tamano_recorte^2
        tamano_objetivo = int(np.ceil(len(senal) / (tamano_recorte * tamano_recorte)) * tamano_recorte * tamano_recorte)
        media_a_anadir = np.mean(senal)
        valores_a_anadir = tamano_objetivo - len(senal)
        senal = senal[:,0]
        
        array_a_agregar = np.full(valores_a_anadir, media_a_anadir)
        array_a_agregar_mascara = np.full(valores_a_anadir, 0)
        mascara = np.concatenate((mascara, array_a_agregar_mascara))
        senal = np.concatenate((senal, array_a_agregar))
        num_filas = int(len(senal) / tamano_recorte)
        # Reshape de la señal y la máscara
        senal = senal.reshape(num_filas, tamano_recorte)
        
        # Obtener el número de filas y columnas de la matriz redimensionada
        num_filas, num_columnas = senal.shape
        
        # Imprimir el número de filas y columnas
        print(f"La matriz tiene {num_filas} filas y {num_columnas} columnas.")
        num_imagenes = int(num_filas/tamano_recorte)
        print(f"Se van a generar {num_imagenes} imágenes para la señal {nombre_senal}")
        mascara = (mascara).reshape(num_filas, tamano_recorte)
        
        colorize(nombre_senal, senal, plt.get_cmap('viridis'), num_imagenes)
        colorize_mask(nombre_mascara, mascara, cm.gray, num_imagenes)
            

if __name__ == "__main__":
    generar_imagenes()
