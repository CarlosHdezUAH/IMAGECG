#!/bin/bash

# Definir los valores para Learning_Rate
learning_rates=("1e-4")

# Definir los valores para batchSize
batch_sizes=(32)

# Bucle para probar diferentes combinaciones de parámetros
for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do

            # Modificar los valores de las variables en el código Python
            sed -i "s/Learning_Rate=.*/Learning_Rate=$lr/" train.py
            sed -i "s/batchSize=.*/batchSize=$bs/" train.py	  
	    
            # Ejecutar el código Python
            python train.py
            python results.py
            python refinemasks.py
            python FPTP.py
    done
done

# Después de finalizar las iteraciones
python plots.py

python anomalias_det.py
