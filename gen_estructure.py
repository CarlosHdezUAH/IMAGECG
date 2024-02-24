import os
import shutil
import json

# Obtener nombres de archivos .npy en /muestras_registros_totales
npy_files_dir = "./muestras_registros_totales"
file_names = [file.split('.')[0] for file in os.listdir(npy_files_dir) if file.endswith('.npy')]

# Crear carpeta LOO en el directorio del script
loo_dir = os.path.join(os.path.dirname(__file__), "LOO")
os.makedirs(loo_dir, exist_ok=True)

# Crear carpetas LOO_XXXXX y su estructura dentro de la carpeta LOO
for name in file_names:
    loo_subdir = os.path.join(loo_dir, f"LOO_{name}")
    os.makedirs(loo_subdir, exist_ok=True)
    
    # Crear estructura de carpetas dentro de LOO_XXXXX
    data_dir = os.path.join(loo_subdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_dir = os.path.join(data_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    imagenes_parch = os.path.join(data_dir, "imagenesparch")
    os.makedirs(imagenes_parch, exist_ok=True)
    
    imagenes_parch_images = os.path.join(imagenes_parch, "images")
    os.makedirs(imagenes_parch_images, exist_ok=True)
    
    imagenes_parch_masks = os.path.join(imagenes_parch, "masks")
    os.makedirs(imagenes_parch_masks, exist_ok=True)
    
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    masks_dir = os.path.join(dataset_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    metricas_dir = os.path.join(loo_subdir, "metricas")
    os.makedirs(metricas_dir, exist_ok=True)
    
    # Crear archivo data.json en blanco
    data_json_path = os.path.join(loo_subdir, "data.json")
    with open(data_json_path, "w") as json_file:
        json.dump({}, json_file)

    # Crear archivo resultados.txt en blanco
    resultados_txt_path = os.path.join(loo_subdir, "resultados.txt")
    with open(resultados_txt_path, "w") as resultados_file:
        resultados_file.write("")

    metricas_subdirs = ["images", "mascaras_recuperadas", "reconstructed_masks", "refined_masks", "true_masks"]
    for subdir in metricas_subdirs:
        subdir_path = os.path.join(metricas_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

print("Estructura de carpetas creada exitosamente en el directorio del script.")

# Obtener nombres de archivos XXXXX_0.png en /full_images/images y /full_images/masks
images_dir = "./full_images/images"
masks_dir = "./full_images/masks"

# Obtener los nombres de archivos XXXXX_0.png
image_files = [file.split('_')[0] for file in os.listdir(images_dir) if file.endswith('_0.png')]
mask_files = [file.split('_')[0] for file in os.listdir(masks_dir) if file.endswith('_0.png')]

# Obtener nombres únicos
unique_files = list(set(image_files + mask_files))

# Copiar archivos a las carpetas LOO_XXXX/data/imagenesparch/images y masks
loo_dir = os.path.join(os.path.dirname(__file__), "LOO")

for name in unique_files:
    loo_subdirs = [os.path.join(loo_dir, subdir) for subdir in os.listdir(loo_dir) if os.path.isdir(os.path.join(loo_dir, subdir)) and subdir.startswith("LOO_")]
    for loo_subdir in loo_subdirs:
        loo_name = loo_subdir.split('_')[-1]
        if loo_name != name:
            # Copiar imagen
            dest_image_dir = os.path.join(loo_subdir, "data", "imagenesparch", "images")
            if not os.path.exists(dest_image_dir):
                os.makedirs(dest_image_dir, exist_ok=True)
            shutil.copy(os.path.join(images_dir, f"{name}_0.png"), dest_image_dir)
            
            # Copiar máscara
            dest_mask_dir = os.path.join(loo_subdir, "data", "imagenesparch", "masks")
            if not os.path.exists(dest_mask_dir):
                os.makedirs(dest_mask_dir, exist_ok=True)
            shutil.copy(os.path.join(masks_dir, f"{name}_0.png"), dest_mask_dir)

print("Archivos copiados exitosamente según el criterio de leave one out.")

# Directorio de la carpeta con los códigos de entrenamiento
training_codes_dir = "./training_codes"

# Directorio base donde se encuentran las carpetas LOO_XXXXXX
base_dir = loo_dir

# Recorrer todas las carpetas LOO_XXXXXX en base_dir
for subdir in os.listdir(base_dir):
    if subdir.startswith("LOO_"):
        loo_folder = os.path.join(base_dir, subdir)
        # Copiar todos los archivos de la carpeta training_codes a la carpeta LOO_XXXXX
        for file_name in os.listdir(training_codes_dir):
            src_file_path = os.path.join(training_codes_dir, file_name)
            dest_file_path = os.path.join(loo_folder, file_name)
            shutil.copy(src_file_path, dest_file_path)

print("Se han copiado todos los archivos de la carpeta training_codes a cada una de las carpetas LOO_XXXXX.")


# Directorio donde se encuentran las imágenes y máscaras completas
full_images_dir = "./full_images/"

# Recorrer todas las carpetas LOO_XXXXXX en base_dir
for subdir in os.listdir(base_dir):
    if subdir.startswith("LOO_"):
        loo_folder = os.path.join(base_dir, subdir)
        loo_number = subdir.split("_")[-1]

        # Directorio de métricas dentro de la carpeta LOO_XXXX
        metrics_dir = os.path.join(loo_folder, "metricas")

        # Directorio de imágenes y máscaras en métricas
        images_dir = os.path.join(metrics_dir, "images")
        true_masks_dir = os.path.join(metrics_dir, "true_masks")

        # Crear directorios de imágenes y máscaras si no existen
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(true_masks_dir, exist_ok=True)

        # Copiar imágenes correspondientes
        for image_file in os.listdir(os.path.join(full_images_dir, "images")):
            if image_file.startswith(loo_number):
                src_image_path = os.path.join(full_images_dir, "images", image_file)
                dest_image_path = os.path.join(images_dir, image_file)
                shutil.copy(src_image_path, dest_image_path)

        # Copiar máscaras correspondientes
        for mask_file in os.listdir(os.path.join(full_images_dir, "masks")):
            if mask_file.startswith(loo_number):
                src_mask_path = os.path.join(full_images_dir, "masks", mask_file)
                dest_mask_path = os.path.join(true_masks_dir, mask_file)
                shutil.copy(src_mask_path, dest_mask_path)

print("Proceso completado: Se han copiado las imágenes y máscaras correspondientes a cada carpeta LOO_XXXX/metricas.")

# Directorio donde se encuentran los archivos de índices de anomalías y máscaras de anomalías
anomalias_indices_dir = "./anomalias"
anomalias_masks_dir = "./anomalias_masks_1280"

# Recorrer todas las carpetas LOO_XXXXXX en base_dir
for subdir in os.listdir(base_dir):
    if subdir.startswith("LOO_"):
        loo_folder = os.path.join(base_dir, subdir)
        loo_number = subdir.split("_")[-1]

        # Copiar archivos de índices de anomalías correspondientes
        src_indices_file = os.path.join(anomalias_indices_dir, f"{loo_number}_indices.npy")
        dest_indices_file = os.path.join(loo_folder, f"{loo_number}_indices.npy")
        shutil.copy(src_indices_file, dest_indices_file)

        # Copiar archivos de máscaras de anomalías correspondientes
        src_masks_file = os.path.join(anomalias_masks_dir, f"{loo_number}_máscara.npy")
        dest_masks_file = os.path.join(loo_folder, f"{loo_number}_máscara.npy")
        shutil.copy(src_masks_file, dest_masks_file)

print("Archivos de anomalías copiados a cada carpeta LOO_XXXXX según corresponde.")


