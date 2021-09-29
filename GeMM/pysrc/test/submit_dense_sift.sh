#!/bin/bash


#SBATCH -J fiknn
#SBATCH -o log_s
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:0:00



source activate fiknn

DATASET=sift
N=$((2**22))
D=128
K=64
ITER=70
L=10

LOG=../results/dense/log_${DATASET}_L${L}K${K}T${ITER}.txt

python -u run_dense.py -n $N -d $D -iter $ITER -levels $L -k $K -dataset $DATASET > $LOG



conda deactivate








