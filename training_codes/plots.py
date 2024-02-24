import json
import matplotlib.pyplot as plt

# Cargar el JSON desde un archivo externo
with open('data.json') as json_file:
    data = json.load(json_file)

# Configurar la resolución y la calidad de las imágenes
dpi = 300  # Puntos por pulgada (dots per inch)

# Crear las gráficas
plt.figure(figsize=(19.2, 10.8), dpi=dpi)  # Tamaño de la figura en pulgadas

# Gráfica de Loss
for key, values in data.items():
    learning_rate, batch_size = eval(key)
    loss = values['loss']
    plt.plot(loss, label=f"LR: {learning_rate}, BS: {batch_size}")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')
plt.savefig('loss_plot.png', dpi=dpi)  # Guardar la gráfica de Loss como imagen

# Gráfica de IoU
plt.figure(figsize=(19.2, 10.8), dpi=dpi)  # Tamaño de la figura en pulgadas
for key, values in data.items():
    learning_rate, batch_size = eval(key)
    iou = values['iou']
    plt.plot(iou, label=f"LR: {learning_rate}, BS: {batch_size}")

plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.title('IoU')
plt.savefig('iou_plot.png', dpi=dpi)  # Guardar la gráfica de IoU como imagen

