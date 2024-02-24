import wfdb
import os
import numpy as np

# Base de datos y directorio local
database_name = "ltdb/1.0.0"
# Registros que deseas guardar
record_names = ["14046", "14134", "14157", "14172", "14184"]

# Directorio para guardar los archivos
output_dir = "muestras_registros_totales"

# Crea el directorio si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterar sobre los registros y guardar todas las muestras en un solo archivo
for record_name in record_names:
    print(f"Obteniendo señal para el registro {record_name}...")
    record = wfdb.rdrecord(record_name, pn_dir=database_name)
    signals, fields = wfdb.rdsamp(record_name,channels=[0], pn_dir=database_name)

    # Guardar todas las muestras en un archivo npy con el mismo nombre del registro
    output_filename = os.path.join(output_dir, f"{record_name}.npy")
    np.save(output_filename, signals)
    print(f"Se han guardado todas las muestras en {output_filename}")

print("Proceso completado.")

