#!/bin/bash


#SBATCH -J fiknn
#SBATCH -o log_g
#SBATCH -p v100-lm
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:0:00



source activate fiknn

DATASET=gaussian
N=$((2**22))
D=15
K=64
ITER=35
L=13
LOG=../results/dense/log_${DATASET}_L${L}K${K}T${ITER}.txt

python -u run_dense.py -n $N -d $D -iter $ITER -levels $L -k $K -dataset $DATASET > $LOG 



conda deactivate








