import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import json
import yaml
from PIL import Image
from rich.progress import Progress
from rich import print
from rich.table import Table


# Fijar las semillas para la reproducibilidad
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Obtener la ruta al archivo config.yml
config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')

# Cargar el archivo de configuración
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    
with Progress() as progress:

    #HIPERPARÁMETROS
    Learning_Rate= config['learning_rate']
    width=height=256 # image width and height
    batchSize= config['batch_size']
    steps = config['epochs']
    
    # Nombre del archivo de texto de salida
    output_file = "resultados.txt"

    # Comprueba si el archivo ya existe
    file_exists = os.path.exists(output_file)

    # Guardar la tabla en un archivo de texto
    with open(output_file, "a" if file_exists else "w") as file:
        file.write("Learning rate: " + str(Learning_Rate) + " " + "Batch size: " + str(batchSize) + "\n")
        
    print("[b]*************************************[/b]")
    print(f" [b u]Learning Rate:[/b u] [u]{Learning_Rate}[/u]  [b u]Batch Size:[/b u] [u]{batchSize}[/u]")
    print("[b]*************************************[/b]")
    # Directorio de la carpeta dataset
    task = progress.add_task(f"[cyan]Preprocessing...", total=5)  
    dataset_dir = "./data/dataset/"
    ListImages = os.listdir(os.path.join(dataset_dir, "images"))  # Create list of images
    ListMasks = os.listdir(os.path.join(dataset_dir, "masks"))  # Create list of masks
    
    # SPLIT INTO TRAIN AND TEST
    progress.update(task, completed=1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(ListImages, ListMasks, test_size = 0.25, random_state = seed)
    
    #----------------------------------------------Transform image-------------------------------------------------------------------
    progress.update(task, completed=2)
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
    transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()])
    #---------------------Read image ---------------------------------------------------------
    def ReadRandomImage():
            idx = np.random.randint(0, len(X_train))
            Img = cv2.imread(os.path.join(dataset_dir, "images", X_train[idx]))[:, :, 0:3]
            Mask = cv2.imread(os.path.join(dataset_dir, "masks", y_train[idx]), 0)
            AnnMap = np.zeros(Img.shape[0:2], np.float32)
            AnnMap[Mask == 255] = 1
            Img = transformImg(Img.astype(np.uint8))
            AnnMap = transformAnn(Mask.astype(np.uint8))
            return Img, AnnMap
    #--------------Load batch of images-----------------------------------------------------
    def LoadBatch():
            images = torch.zeros([batchSize, 3, height, width])
            ann = torch.zeros([batchSize, height, width])
            for i in range(batchSize):
                images[i], ann[i] = ReadRandomImage()
            return images, ann
    #--------------Load and set net and optimizer-------------------------------------
    progress.update(task, completed=3)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights = config['weights']
    net = config['net']

    if 'deeplabv3_resnet50' in net:
        Net = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
    elif 'deeplabv3_resnet101' in net:
        Net = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
    elif 'fcn_resnet50' in net:
        Net = torchvision.models.segmentation.fcn_resnet50(weights=weights)
    elif 'fcn_resnet101' in net:
        Net = torchvision.models.segmentation.fcn_resnet101(weights=weights)
    else:
        raise ValueError("Invalid network configuration provided in config file.")

    Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes
    Net=Net.to(device)
    optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer
    
    progress.update(task, completed=4)
    data_file = "data.json"  # Nombre del archivo para almacenar los datos

    # Cargar los datos existentes si el archivo existe
    if os.path.exists(data_file):
        with open(data_file, 'r') as file:
            data = json.load(file)
    else:
        data = {}
    progress.update(task, completed=5)
    #----------------Train--------------------------------------------------------------------------
    task = progress.add_task("[cyan]Training...", total=steps+1)  
    for itr in range(steps+1):  # Training loop
        images, ann = LoadBatch()  # Load training batch
        images = torch.autograd.Variable(images, requires_grad=False).to(device)  # Load image
        ann = torch.autograd.Variable(ann, requires_grad=False).to(device)  # Load annotation
        Pred = Net(images)['out']  # Make prediction
        Net.zero_grad()
        # Obtener los valores de 'class_weights' del archivo de configuración
        class_weights = config['class_weights']

        # Crear un tensor de PyTorch con los valores de 'class_weights'
        weights = torch.FloatTensor(class_weights)

        # Mover el tensor al dispositivo deseado (por ejemplo, GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Verifica el dispositivo del tensor de predicción (Pred)
        device = Pred.device
        weights = weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)  # Set loss function
        Loss = criterion(Pred, ann.long())  # Calculate cross entropy loss
        Loss.backward()  # Backpropagate loss
        optimizer.step()  # Apply gradient descent change to weight
        seg = torch.argmax(Pred, 1).cpu().detach().numpy()  # Get prediction classes
    
        # Calculate IoU
        iou = []
        for pred, mask in zip(seg, ann.cpu().numpy()):
            intersection = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou_value = np.sum(intersection) / np.sum(union)
            if not np.isnan(iou_value):  # Verificar si el valor no es NaN
                iou.append(iou_value)
    
        key = (Learning_Rate, batchSize)
        key_str = str(key)  # Convertir la tupla en una cadena de texto
        if key_str not in data:
            data[key_str] = {'loss': [], 'iou': []}
        data[key_str]['loss'].append(Loss.data.cpu().numpy())
        data[key_str]['iou'].append(np.mean([val for val in iou if not np.isnan(val)]))
        #print(itr, ") Loss=", Loss.data.cpu().numpy(), " IoU=", np.mean(iou))
        
        progress.update(task, completed=itr+1, description="[cyan]Training...")
    
        if itr % steps == 0 and itr != 0:
            torch.save(Net.state_dict(), str(itr) + ".torch")
            #torch.save(Net, str(itr)  + "completo.torch")     
    
    def convert_ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(key): convert_ndarray_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_ndarray_to_list(item) for item in obj]
        else:
            return obj
    
    # Guardar los datos en el archivo
    with open(data_file, 'w') as file:
        converted_data = convert_ndarray_to_list(data)
        json.dump(converted_data, file)
    
    for key_str, values in data.items():
        key = eval(key_str)  # Convertir la cadena de texto en una tupla nuevamente
        learning_rate, batch_size = key
        loss = values['loss']
        iou = values['iou']


