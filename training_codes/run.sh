#!/bin/bash


# Ejecutar el código Python

python train.py
python results.py
python refinemasks.py
python FPTP.py

# Después de finalizar las iteraciones
python plots.py

python anomalias_det.py
