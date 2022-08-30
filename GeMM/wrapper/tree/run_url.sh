#!/bin/bash

#SBATCH -J KNN_URL
#SBATCH -o logs/log_URL_9
#SBATCH -n 1 
#SBATCH -N 1 
#SBATCH -t 10:00:00
#SBATCH -p v100


source activate gen

LEVELS=12
K=64
MAX_ITER=300
python url.py -use_gpu 1 -levels $LEVELS -k $K -iter $MAX_ITER

conda deactivate

