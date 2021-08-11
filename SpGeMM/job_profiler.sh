#!/bin/bash

#SBATCH -J SpGeMM
#SBATCH -N 1 	# Total number of nodes requested
#SBATCH -n 1 	# Total number of mpi tasks requested
#SBATCH -t 2:00:00 	# Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p rtx



file=build/GPU_FIKNN_dense
res=results/GPU_FIKNN_dense
write_tag=8M


$CUDA_HOME/nsight-compute-2019.4.0/nv-nsight-cu-cli -f --details-all  --nvtx --export ${res}_${write_tag}_report ${file}
