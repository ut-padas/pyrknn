#!/bin/bash


#SBATCH -J fiknn
#SBATCH -o log_avazu
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:0:00



source activate fiknn

DATASET=avazu
N=$((2**22))
K=4
D=16
ITER=151
L=16
LOG=../results/sparse/log_${DATASET}_L${L}K${K}T${ITER}.txt

python -u run_sparse.py -n $N -d $D -iter $ITER -levels $L -k $K -dataset $DATASET > $LOG 



conda deactivate








