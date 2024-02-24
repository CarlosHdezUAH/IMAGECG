#!/bin/bash

# Ejecutar get_muestras.py
python get_muestras.py

# Ejecutar get_ann.py
python get_ann.py

# Ejecutar normalization.py
python normalization.py

# Ejecutar anomalias_masks_1280.py
python gen_ann_matrix.py

# Ejecutar gen_imgs.py
python gen_imgs.py

# Ejecutar la generación de la estructura de carpetas
python gen_estructure.py

# aquí se puede variar el número de parches por imagen
python parch.py

# Bucle para ejecutar run.sh en cada carpeta LOO_XXXXX
for loo_folder in /home/carloshf/imagecg/generacion_imagenes/LOO/LOO_*; do
    if [ -d "$loo_folder" ] && [ -f "$loo_folder/run.sh" ]; then
        echo "Ejecutando run.sh en $loo_folder"
        (cd "$loo_folder" && bash run.sh)
    fi
done

echo "Proceso completado: Se han ejecutado los archivos run.sh en todas las carpetas LOO_XXXXX."

