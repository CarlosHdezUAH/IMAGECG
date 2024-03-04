import os
from PIL import Image
import re
import cv2
import csv
import torch
import shutil
import imageio
import torchvision
import numpy as np
import yaml
from skimage.morphology import binary_opening, disk
from torchvision.transforms import ToTensor, Resize
import torchvision.transforms as tf
from rich.progress import Progress
from rich import print
from rich.table import Table
import warnings

warnings.filterwarnings("ignore")

def segment_image(image_name):
    ruta_madre = "./metricas/"
    nombre_archivo = os.path.splitext(os.path.basename(image_name))[0]
    nueva_carpeta = os.path.join(ruta_madre, nombre_archivo)  # Ruta completa de la nueva carpeta
    # Comprobar si la carpeta ya existe
    if not os.path.exists(nueva_carpeta):
        os.makedirs(nueva_carpeta)

    # Cargar la imagen original
    original_image = Image.open(image_name)

    # Obtener las dimensiones de la imagen original
    width, height = original_image.size

    # Definir el tamaño de los parches
    patch_size = 256
    
    total_patches = (width // patch_size) * (height // patch_size)
    
    # Generar y guardar los parches
    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            		# Calcular las coordenadas del parche
            		left = i
            		top = j
            		right = i + patch_size
            		bottom = j + patch_size

            		# Recortar el parche
            		patch = original_image.crop((left, top, right, bottom))

            		# Guardar el parche con un nombre único
            		patch_name = f"patch_{i}_{j}.png"
            		patch_path = os.path.join(nueva_carpeta, patch_name)
            		patch.save(patch_path)
    return nueva_carpeta
    
def process_images(carpeta, image_name):
    width=height=256 # image width and height
    
    # Obtener la ruta al archivo config.yml
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')

    # Cargar el archivo de configuración
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    epochs = config['epochs']
    
    model_path = f"./{epochs}.torch"
    directory = carpeta
    ruta_madre = "./metricas/"
    nombre_archivo = os.path.splitext(os.path.basename(image_name))[0] + "_predicted_masks"
    output_directory = os.path.join(ruta_madre, nombre_archivo)  # Ruta completa de la nueva carpeta
    # Comprobar si la carpeta ya existe
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        #print("Se ha creado la carpeta:", output_directory)
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    
    Net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)
    Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    Net = Net.to(device)  # Set net to GPU or CPU
    
    # Cargar los pesos del modelo guardado
    state_dict = torch.load(model_path)
    
    # Remover las claves incompatibles
    state_dict.pop("aux_classifier.0.weight", None)
    state_dict.pop("aux_classifier.1.weight", None)
    state_dict.pop("aux_classifier.1.bias", None)
    state_dict.pop("aux_classifier.1.running_mean", None)
    state_dict.pop("aux_classifier.1.running_var", None)
    state_dict.pop("aux_classifier.1.num_batches_tracked", None)
    state_dict.pop("aux_classifier.4.weight", None)
    state_dict.pop("aux_classifier.4.bias", None)
    
    # Cargar los pesos en el modelo
    Net.load_state_dict(state_dict, strict=False)
    Net.eval()  # Cambiar a modo de evaluación
    # Ruta de la carpeta que contiene las imágenes
    carpeta_imagenes = './data/dataset/images/'

    # Obtener la lista de nombres de archivo de imágenes en la carpeta
    nombres_imagenes = os.listdir(carpeta_imagenes)

    # Inicializar listas para almacenar los valores de píxeles de todas las imágenes
    valores_pixeles = []

    # Leer todas las imágenes y almacenar sus valores de píxeles normalizados
    for nombre_imagen in nombres_imagenes:
        ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
        imagen = cv2.imread(ruta_imagen)
        imagen_normalizada = imagen / 255.0  # Normalizar los valores de píxeles
        valores_pixeles.append(imagen_normalizada)

    # Calcular la media y la desviación estándar de cada canal de color
    valores_pixeles = np.array(valores_pixeles)
    mean = np.mean(valores_pixeles, axis=(0, 1, 2))
    std = np.std(valores_pixeles, axis=(0, 1, 2))
    
    transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize(mean, std)])
    
    # Obtener la lista de imágenes en el directorio
    image_files = os.listdir(os.path.join(directory))
    total_images=len(image_files)
    for image_file in image_files:
        	# Obtener el nombre de la imagen de prueba
        	image_name = image_file
        	# Cargar la imagen
        	image_path = os.path.join(directory, image_name)
       		image = cv2.imread(image_path)
        	# Preprocesar la imagen
        	image_cv = image.astype(np.uint8)
        	image_tensor = transformImg(image_cv) 
        	image_tensor = torch.autograd.Variable(image_tensor, requires_grad=False).to(device).unsqueeze(0)
        
        	# Obtener la máscara predicha por la red neuronal
        	with torch.no_grad():
            		pred = Net(image_tensor)['out']
		# Redimensionar a tamaño original
        	pred = tf.Resize((256, 256))(pred[0])
        	# Convertir la máscara predicha a una representación binaria
        	seg = torch.argmax(pred, 0).cpu().detach().numpy()
        	#print("Valores únicos de los píxeles de la máscara predicha:", np.unique(seg))
        	# Aplicar binary_opening a la máscara predicha
        	seg_bin = seg > 0
        	selem = disk(radius=1.7)  # Modifica el tamaño del elemento estructurante según tus necesidades
        	seg_bin_opening = binary_opening(seg_bin, selem=selem)
        	seg_bin_opening_2 = binary_opening(seg_bin_opening, selem=selem)
        	# Guardar la máscara predicha en el directorio de salida con el mismo nombre de la imagen
        	mask_pred_path = os.path.join(output_directory, image_name)
        	cv2.imwrite(mask_pred_path, seg_bin_opening_2.astype(np.uint8) * 255)
        
        	#print("Máscara predicha guardada:", mask_pred_path)
    return output_directory
	
def reconstruct_image(folder_path, image_name):
    # Obtener la lista de archivos en la carpeta
    patch_files = os.listdir(folder_path)

    # Ordenar los archivos por nombre
    patch_files.sort()

    # Obtener el tamaño de los parches
    patch_size = 256
    rows = 10  # Número de filas
    cols = 10 # Número de columnas

    # Crear una imagen vacía para la reconstrucción
    image_width = cols * patch_size
    image_height = rows * patch_size
    reconstructed_image = Image.new("RGB", (image_width, image_height))
    total_patches = len(patch_files)
    for i, patch_file in enumerate(patch_files):
            patron = r"patch_(\d+)_(\d+)\.png"
            coincidencias = re.search(patron, patch_file)
            numero1 = int(coincidencias.group(1))
            numero2 = int(coincidencias.group(2))
            row = numero2 // patch_size
            col = numero1 // patch_size
            left = col * patch_size
            top = row * patch_size
            right = left + patch_size
            bottom = top + patch_size
            patch_path = os.path.join(folder_path, patch_file)
            patch = Image.open(patch_path)
            reconstructed_image.paste(patch, (left, top, right, bottom))
		
            ruta_madre = "./metricas/"
            nombre_archivo = os.path.splitext(os.path.basename(image_name))[0] + "_reconstructed_image"
            output_directory = os.path.join(ruta_madre, "reconstructed_masks") 
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Guardar la imagen reconstruida en un archivo
            reconstructed_image.save(os.path.join(output_directory, os.path.splitext(os.path.basename(image_name))[0]) + ".png")
            
    return output_directory
    
def copiar_imagenes():
    # Directorio de las carpetas reconstructed_image
    directorio_reconstructed = "./metricas/"

    # Directorio de la carpeta true_masks
    directorio_true_masks = "./metricas/true_masks"

    # Obtener la lista de carpetas reconstructed_image
    carpetas_reconstructed = [carpeta for carpeta in os.listdir(directorio_reconstructed)
                              if os.path.isdir(os.path.join(directorio_reconstructed, carpeta)) and carpeta.endswith("_reconstructed_image")]

    # Recorrer cada carpeta reconstructed_image
    for carpeta in carpetas_reconstructed:
        # Obtener el nombre de la carpeta sin la extensión y separar en partes
        partes = carpeta.split("_")

        # Obtener el número de imagen
        numero_imagen = partes[0] + "_" + partes[1]

        # Construir el nombre de la imagen correspondiente
        nombre_imagen = numero_imagen + ".png"

        # Construir la ruta completa de la imagen en la carpeta true_masks
        ruta_imagen_origen = os.path.join(directorio_true_masks, nombre_imagen)

        # Construir la ruta completa de la carpeta reconstructed_image
        ruta_carpeta_destino = os.path.join(directorio_reconstructed, carpeta)

        # Verificar si la imagen existe en la carpeta true_masks
        if os.path.exists(ruta_imagen_origen):
            # Copiar la imagen a la carpeta reconstructed_image
            shutil.copy(ruta_imagen_origen, ruta_carpeta_destino)

def modificar_imagenes():
    # Directorio de las carpetas reconstructed_image
    directorio_reconstructed = "./metricas/"

    # Obtener la lista de carpetas reconstructed_image
    carpetas_reconstructed = [carpeta for carpeta in os.listdir(directorio_reconstructed)
                              if os.path.isdir(os.path.join(directorio_reconstructed, carpeta)) and carpeta.endswith("_reconstructed_image")]

    # Recorrer cada carpeta reconstructed_image
    for carpeta in carpetas_reconstructed:
        # Obtener la ruta completa de la carpeta reconstructed_image
        ruta_carpeta = os.path.join(directorio_reconstructed, carpeta)

        # Obtener la lista de archivos en la carpeta
        archivos = os.listdir(ruta_carpeta)

        # Filtrar las imágenes que no son "imagen_reconstruida.png"
        imagenes = [archivo for archivo in archivos if archivo.lower().endswith(".png") and archivo != "imagen_reconstruida.png"]

        # Recorrer las imágenes y modificarlas
        for imagen in imagenes:
            # Construir la ruta completa de la imagen
            ruta_imagen = os.path.join(ruta_carpeta, imagen)

            # Abrir la imagen utilizando la biblioteca PIL
            try:
                imagen_pil = Image.open(ruta_imagen)

                # Modificar la imagen para que sea legible
                imagen_modificada = imagen_pil.point(lambda p: 0 if p == 0 else 255)

                # Guardar la imagen sobreescribiendo la original
                imagen_modificada.save(ruta_imagen)
            except Exception as e:
                print(f"Error al abrir o modificar la imagen {imagen} en la carpeta {carpeta}: {e}")

def convert_to_binary(image):
    # Convierte la imagen a escala de grises
    image = image.convert('L')
    
    # Define el umbral para la binarización
    threshold = 128
    
    # Aplica la binarización
    binary_image = image.point(lambda p: p > threshold and 255)
    
    return binary_image  
def calculate_pixel_match(image1, image2):
    
    # Obtiene los datos de píxeles de las imágenes
    pixels1 = list(image1.getdata())
    pixels2 = list(image2.getdata())
    
    # Calcula el número total de píxeles
    total_pixels = len(pixels1)
    
    # Calcula el número de píxeles coincidentes
    matching_pixels = sum(p1 == p2 for p1, p2 in zip(pixels1, pixels2))
    
    # Calcula el porcentaje de píxeles coincidentes
    match_percentage = (matching_pixels / total_pixels) * 100
    
    return match_percentage
 
# Ruta de la carpeta de imágenes
ruta_carpeta = "./metricas/images/"

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(ruta_carpeta)

# Crear una lista para almacenar las imágenes
imagenes = []

# Recorrer los archivos en la carpeta
for archivo in archivos:
    # Comprobar si el archivo es una imagen (puedes agregar más extensiones si lo deseas)
    if archivo.endswith(".jpg") or archivo.endswith(".jpeg") or archivo.endswith(".png"):
        # Agregar la ruta completa del archivo a la lista de imágenes
        imagenes.append(os.path.join(ruta_carpeta, archivo))

total_imgs = len(imagenes)

with Progress() as progress:
    task = progress.add_task("[cyan]Processing images...", total=total_imgs)	
    # Procesar las imágenes en un bucle
    for imagen in imagenes:
        carpeta = segment_image(imagen)
        carpeta_pred_masks = process_images(carpeta, imagen)
        carpeta_reconstructed_img = reconstruct_image(carpeta_pred_masks, imagen)
        nombre_imagen = os.path.splitext(os.path.basename(imagen))[0]
        progress.update(task, advance=1)
    copiar_imagenes()
    modificar_imagenes()

