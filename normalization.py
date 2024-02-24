import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import matplotlib.pyplot as plt
import os

# Cargar las señales 14046, 14134, 14157 y 14172 (asegúrate de tener los archivos correspondientes)
señal_14046 = np.load('muestras_registros_totales/14046.npy')
señal_14134 = np.load('muestras_registros_totales/14134.npy')
señal_14157 = np.load('muestras_registros_totales/14157.npy')
señal_14172 = np.load('muestras_registros_totales/14172.npy')
señal_14184 = np.load('muestras_registros_totales/14184.npy')

# Definir el número total de muestras deseadas
total_muestras = 30_000_000

# Crear una lista de señales
señales = [señal_14046, señal_14134, señal_14157, señal_14172]

# Inicializar la señal total
señal_total = []

# Tomar 1 millón de muestras de cada señal de manera intercalada en puntos aleatorios
muestras_por_señal = 1_000_000

while len(señal_total) < total_muestras:
    señal_actual = np.random.choice(señales)  # Seleccionar una señal aleatoriamente
    inicio = np.random.randint(0, len(señal_actual) - muestras_por_señal)  # Punto aleatorio
    muestras_seleccionadas = señal_actual[inicio:inicio + muestras_por_señal]
    señal_total.extend(muestras_seleccionadas)

# Convertir la señal total a un arreglo numpy
señal_total = np.array(señal_total)

# Crear el transformador de potencia y ajustarlo a la señal total
power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
señal_total_transformada = power_transformer.fit_transform(señal_total.reshape(-1, 1))

# Ahora, normalizar la señal 14184 utilizando el transformador ajustado previamente
señal_14184 = np.load('muestras_registros_totales/14184.npy')
señal_14184_transformada = power_transformer.transform(señal_14184.reshape(-1, 1))

# GUARDAR SEÑALES NORMALIZADAS

# Ruta de la carpeta para guardar las señales normalizadas
carpeta_salida = 'muestras_registros_totales_norm'

# Crear directorio base
os.makedirs(carpeta_salida)

        
# Crear una lista de señales y sus nombres de archivo correspondientes
señales_y_nombres = [(señal_14046, '14046.npy'), (señal_14134, '14134.npy'), (señal_14157, '14157.npy'), (señal_14172, '14172.npy'), (señal_14184, '14184.npy')]

scaler = MinMaxScaler(feature_range=(-1, 1))

for señal, nombre_archivo in señales_y_nombres:
    señal_transformada = power_transformer.transform(señal.reshape(-1, 1))
    señal_transformada = scaler.fit_transform(señal_transformada)
    np.save(os.path.join(carpeta_salida, nombre_archivo.replace('.npy', '_transformada.npy')), señal_transformada)

