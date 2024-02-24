# IMAGECG

IMAGECG es un implementación basada en Deep Learning que pretende abordar el problema del procesado de señales de electrocardiograma de larga duración a través su conversión a imágenes y la generación de máscaras binarias para entrenar una red de segmentación semántica.

## Instalación

Siga estos pasos para configurar el entorno Conda utilizando el archivo `environment.yml` proporcionado en este repositorio:

1. Clone el repositorio a su máquina local utilizando Git:

```bash
git clone https://github.com/CarlosHdezUAH/IMAGECG.git
```

2. Navegue al directorio del repositorio clonado:

```bash
cd tu-repositorio
```

3. Cree y active un nuevo entorno Conda a partir del archivo `environment.yml`:

```bash
conda env create -f environment.yml
```

4. Active el entorno Conda recién creado:

```bash
conda activate nombre-del-entorno
```

Ahora, el entorno Conda está configurado con todas las dependencias necesarias especificadas en el archivo `environment.yml`.

## Descripción de los Archivos


### Directorio `scripts`

- **get_muestras.py**: Este script se utiliza para obtener muestras de datos.

- **get_ann.py**: Utilizado para obtener anotaciones de datos.

- **normalization.py**: Script para normalizar los datos de entrada.

- **gen_ann_matrix.py**: Contiene código para generar matrices de anotaciones.

- **gen_imgs.py**: Se utiliza para generar imágenes a partir de los datos.

- **gen_estructure.py**: Código para generar la estructura de carpetas necesaria para el proyecto.

- **parch.py**: Contiene código relacionado con la generación de parches de imágenes para entrenar la red.


### Directorio `training_codes`

- **FPTP.py**: Este archivo contiene el código para el preprocesamiento de datos antes del entrenamiento.

- **results.py**: Aquí se encuentra el código para la generación de resultados y métricas después del entrenamiento del modelo.

- **run.sh**: Un script de shell que automatiza varias tareas relacionadas con el entrenamiento del modelo.

- **train.py**: Contiene el código principal para el entrenamiento del modelo.

- **refinemasks.py**: Este archivo contiene el código para refinar las máscaras generadas por el modelo.

- **anomalias_det.py**: Código para detectar anomalías en los datos de entrada.

- **plots.py**: Archivo con el código para generar diferentes tipos de gráficos y visualizaciones.

### Archivo `run.sh`

Este archivo es un script de shell que automatiza varias tareas del proyecto. Se encarga de ejecutar los scripts necesarios y coordinar el flujo de trabajo del proyecto.

### Archivo `config.yml`

Este archivo de configuración permite cambiar algunos parámetros del entrenamiento como el learning rate, el batch size, el número de parches extraidos por cada registro ECG, la red empleada, el número de steps y los pesos asociados a cada clase.


