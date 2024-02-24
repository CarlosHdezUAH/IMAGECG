import os
import cv2
import yaml

# Directorio base donde se encuentran las carpetas LOO_XXXXXX
base_dir = "./LOO/"

# Obtener la ruta al archivo config.yml
config_path = os.path.join(os.path.dirname(__file__), '../config.yml')

# Cargar el archivo de configuración
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    
# Tamaño del recorte
crop_size = 256

# Número de parches que se guardarán por imagen
num_patches = config['num_patches']

def patch_images_in_folder(folder_path):
    # Ruta de las carpetas de imágenes y máscaras
    images_folder = os.path.join(folder_path, "data", "imagenesparch", "images")
    masks_folder = os.path.join(folder_path, "data", "imagenesparch", "masks")

    # Ruta de las carpetas para guardar los recortes de imágenes y máscaras
    output_images_folder = os.path.join(folder_path, "data", "dataset", "images")
    output_masks_folder = os.path.join(folder_path, "data", "dataset", "masks")

    # Obtener la lista de archivos en las carpetas de imágenes y máscaras
    image_files = os.listdir(images_folder)
    mask_files = os.listdir(masks_folder)

    # Asegurarse de que las listas de archivos estén ordenadas
    image_files.sort()
    mask_files.sort()

    # Verificar que haya el mismo número de imágenes y máscaras
    if len(image_files) != len(mask_files):
        print("Error: El número de imágenes y máscaras no coincide.")
        exit()

    # Crear las carpetas de destino si no existen
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_masks_folder, exist_ok=True)

    # Recortar las imágenes y máscaras
    for i in range(len(image_files)):
        # Cargar la imagen y la máscara correspondiente
        image_path = os.path.join(images_folder, image_files[i])
        mask_path = os.path.join(masks_folder, mask_files[i])
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Obtener el tamaño de la imagen
        height, width, _ = image.shape

        # Contador de parches
        patches_count = 0

        # Recortar la imagen y la máscara sin solapamiento
        for y in range(0, height, crop_size):
            for x in range(0, width, crop_size):
                # Calcular las coordenadas del recorte
                x_end = min(x + crop_size, width)
                y_end = min(y + crop_size, height)

                # Extraer el recorte de la imagen y la máscara
                crop_image = image[y:y_end, x:x_end]
                crop_mask = mask[y:y_end, x:x_end]

                # Generar el nombre del archivo
                filename = f"{i}_{x}_{y}.png"

                # Guardar el recorte de la imagen y la máscara
                output_image_path = os.path.join(output_images_folder, filename)
                output_mask_path = os.path.join(output_masks_folder, filename)
                cv2.imwrite(output_image_path, crop_image)
                cv2.imwrite(output_mask_path, crop_mask)

                # Incrementar el contador de parches
                patches_count += 1

                # Detener el bucle si se ha alcanzado el número deseado de parches
                if patches_count >= num_patches:
                    break

            if patches_count >= num_patches:
                break

# Recorrer todas las carpetas LOO_XXXXXX en base_dir
for subdir in os.listdir(base_dir):
    if subdir.startswith("LOO_"):
        folder_path = os.path.join(base_dir, subdir)
        patch_images_in_folder(folder_path)

print("Proceso de parcheo completado en todas las carpetas LOO_XXXXXX.")

