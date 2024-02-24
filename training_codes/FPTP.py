import os
import numpy as np
from PIL import Image
from rich.progress import Progress
from rich import print
from rich.table import Table
from io import StringIO
from contextlib import redirect_stdout

def calculate_metrics(predicted_mask, true_mask):
    # Convertir las imágenes a escala de grises
    predicted_mask = Image.fromarray(predicted_mask.astype(np.uint8)).convert("L")
    true_mask = Image.fromarray(true_mask.astype(np.uint8)).convert("L")
    
    # Convertir a arrays numpy binarios
    predicted_mask = np.array(predicted_mask, dtype=bool)
    true_mask = np.array(true_mask, dtype=bool)
    
    #predicted_mask_unique_values = np.unique(predicted_mask)
    #print(predicted_mask_unique_values)
    #true_mask_unique_values = np.unique(true_mask)
    #print(true_mask_unique_values)
    
    # True Positives (TP): píxeles correctamente clasificados como positivos
    tp = np.sum(np.logical_and(predicted_mask, true_mask))

    # False Positives (FP): píxeles incorrectamente clasificados como positivos
    fp = np.sum(np.logical_and(predicted_mask, np.logical_not(true_mask)))

    # True Negatives (TN): píxeles correctamente clasificados como negativos
    tn = np.sum(np.logical_and(np.logical_not(predicted_mask), np.logical_not(true_mask)))

    # False Negatives (FN): píxeles incorrectamente clasificados como negativos
    fn = np.sum(np.logical_and(np.logical_not(predicted_mask), true_mask))

    return tp, fp, tn, fn

def process_images(refined_masks_folder, true_masks_folder):
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    total_pixels = 0
    archivos = os.listdir(refined_masks_folder)
    num_archivos = len(archivos)
    with Progress() as progress:
        task = progress.add_task("[cyan]Obtaning results...", total=num_archivos)	
        
        for filename in os.listdir(refined_masks_folder):
            if filename.endswith(".png"):
                predicted_mask_path = os.path.join(refined_masks_folder, filename)
                true_mask_path = os.path.join(true_masks_folder, filename)
    
                # Cargar las máscaras como imágenes en escala de grises (numpy arrays)
                predicted_mask = np.array(Image.open(predicted_mask_path).convert("L"), dtype=bool)
                true_mask = np.array(Image.open(true_mask_path).convert("L"), dtype=bool)
    
                # Calcular las métricas para esta imagen
                tp, fp, tn, fn = calculate_metrics(predicted_mask, true_mask)
    
                # Actualizar los totales
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn
                total_pixels += predicted_mask.size
                progress.update(task, advance=1)
                
    return total_tp, total_fp, total_tn, total_fn, total_pixels

if __name__ == "__main__":
    refined_masks_folder = "./metricas/refined_masks"
    true_masks_folder = "./metricas/true_masks"

    total_tp, total_fp, total_tn, total_fn, total_pixels = process_images(refined_masks_folder, true_masks_folder)
    
    TP = total_tp / total_pixels
    FP = total_fp / total_pixels
    TN = total_tn / total_pixels
    FN = total_fn / total_pixels
    
    if (TP+FP) != 0:
        	precision = TP / (TP + FP)
    else:
        	precision = 0
    if (TP + FN) != 0:
        	recall = TP / (TP + FN)
    else:
        	recall = 0
    
    if (precision + recall) != 0:
        	F1 = 2 * (precision * recall) / (precision + recall)
    else:
        	F1 = 0

    # Crear una tabla
    table = Table(title="Resultados de la Segmentación")
    table.add_column("Métrica", style="bold", justify="center")
    table.add_column("Valor", style="bold", justify="center")

    # Función para formatear los valores con dos decimales
    def format_value(value, decimal_places=2):
        if decimal_places == 0:
            return str(int(value))
        return f"{value:.{decimal_places}f}"

    # Agregar filas formateando los valores según corresponda
    table.add_row("Verdaderos Positivos (TP):", format_value(TP))
    table.add_row("Falsos Positivos (FP):", format_value(FP))
    table.add_row("Verdaderos Negativos (TN):", format_value(TN))
    table.add_row("Falsos Negativos (FN):", format_value(FN))
    table.add_row("Número total de píxeles procesados:", format_value(total_pixels, decimal_places=0))
    table.add_row("Precision:", format_value(precision))
    table.add_row("Recall:", format_value(recall))
    table.add_row("F1:", format_value(F1))
    
    # Nombre del archivo de texto de salida
    output_file = "resultados.txt"

    # Comprueba si el archivo ya existe
    file_exists = os.path.exists(output_file)

    # Captura la salida de la tabla en una cadena de texto
    with StringIO() as buf:
        with redirect_stdout(buf):
            print(table)
    
        # Guarda la tabla en el archivo de texto
        with open(output_file, "a") as file:
            file.write(buf.getvalue())
            file.write("\n")
    # Imprimir la tabla
    print(table)
