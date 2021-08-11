#!/bin/bash

#SBATCH -J SpGeMM
#SBATCH -N 1 	# Total number of nodes requested
#SBATCH -n 1 	# Total number of mpi tasks requested
#SBATCH -t 2:00:00 	# Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p rtx


#g++ -I $TACC_PYTHON_INC/python3.7m wrapper.cpp -o wrapper
#source activate py36

file=SpGeMM_TriPart_v1
#file=SpGeMM_Iter_stripes
write_tag=4M_32
#file=SpGeMM_1D_TriPart
#file=SpGeMM_Iter
#python SpGeMM_run.py
$CUDA_HOME/nsight-compute-2019.4.0/nv-nsight-cu-cli -f --details-all  --nvtx --export ${file}_${write_tag}_report ${file}
#./${file}_${write_tag}
#conda deactivate  
