import wfdb
import os
import numpy as np

# Base de datos y directorio local
database_name = "ltdb/1.0.0"

# Registros que deseas procesar
record_names = ["14046", "14134", "14157", "14172", "14184"]

# Directorio para guardar los archivos .npy
output_dir = "anomalias"

# Crea el directorio si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterar sobre los registros y guardar los índices de las anomalías en archivos .npy
for record_name in record_names:
    print(f"Procesando el registro {record_name}...")
    record = wfdb.rdrecord(record_name, pn_dir=database_name)
    annotation = wfdb.rdann(record_name, 'atr', pn_dir=database_name)

    # Obtener todos los índices de las anomalías V o F
    anomaly_indices = []

    for i, annot in enumerate(annotation.symbol):
        if annot in ['V', 'F']:
            samp_start = annotation.sample[i]
            anomaly_indices.append(samp_start)

    # Convertir la lista de índices a un arreglo numpy
    anomaly_indices = np.array(anomaly_indices)

    # Guardar los índices en un archivo .npy
    output_filename = os.path.join(output_dir, f"{record_name}_indices.npy")
    np.save(output_filename, anomaly_indices)
    print(f"Se han guardado los índices en {output_filename}")

print("Proceso completado.")
