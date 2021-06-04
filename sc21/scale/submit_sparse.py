#!/usr/bin/env python

import os
import argparse

parser = argparse.ArgumentParser(description="Submit TACC jobs for sparse knn")
parser.add_argument('-run', type=int, default=0)
args = parser.parse_args()


def submit(ranks, threads, dataset, iter, blocksize, blockleaf, levels, k, leafsize, ltrees, gpu_flag=False, merge=1, overlap=1):
  
  print(ranks)
  OMP="OMP_NUM_THREADS="+str(threads)

  if gpu_flag:
      CMD=f"mpirun -n {ranks} python run_sparse.py -iter {iter} -dataset {dataset} -bs {blocksize} -bl {blockleaf} -levels {levels} -cores {threads} -ltrees {ltrees} -use_gpu {gpu_flag} -leafsize {leafsize} -overlap {overlap} -merge {merge}"
  else:
      CMD=f"mpirun -n {ranks} python run_sparse.py -iter {iter} -dataset {dataset} -bs {blocksize} -bl {blockleaf} -levels {levels} -cores {threads} -ltrees {ltrees} -leafsize {leafsize} -overlap {overlap} -merge {merge}"

  filename = f"_{dataset}_ranks_{ranks}_bs_{blocksize}_bl_{blockleaf}_gpu_{gpu_flag}_threads_{threads}_lt_{ltrees}_levels_{levels}_leaf_{leafsize}_overlap_{overlap}_merge_{merge}_run_{args.run}"
  submit_file = "job"+filename+".slm"
  output_file = "out"+filename
  print(filename)

  if gpu_flag:
    print("Using GPU")
    machine = "rtx"
    nodes = (ranks-1)//4+1
    tpn = 4
    tasks = ranks
    env_file = "set_env.sh"
  else:
    machine = "normal"
    nodes = ranks
    tasks = ranks
    tpn = 1
    env_file = "set_env_cpu.sh"
  
  with open(submit_file, 'w') as f:
    f.writelines("#!/bin/bash\n")
    f.writelines("#SBATCH -J %s-%d\n" % (dataset, levels))
    f.writelines(f"#SBATCH -e {output_file}.err \n")
    f.writelines(f"#SBATCH -o {output_file}.txt \n")
    f.writelines(f"#SBATCH -p {machine} \n")
    f.writelines(f"#SBATCH -N {nodes}\n")
    f.writelines("#SBATCH -A CCR20001\n")
    f.writelines(f"#SBATCH --tasks-per-node {tpn} \n")
    f.writelines("#SBATCH -t 01:00:00\n")
    f.writelines("#SBATCH --mail-user=will.ruys@gmail.com\n")
    f.writelines("#SBATCH --mail-type=end\n")
    f.writelines("hostname \n")
    f.writelines("export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH \n")
    f.writelines(f"source {env_file} \n")
    #f.writelines("module load intel/19.0.5  impi/19.0.5 \n")
    #f.writelines("module load remora \n")
    f.writelines("ulimit -c unlimited \n")  
    f.writelines("%s %s\n" % (OMP, CMD))

  os.system("sbatch %s" %submit_file)
  #os.system("rm -f %s" %submit_file)


if __name__ == '__main__':

  #if not os.path.exists(folder):
  #  os.makedirs(folder)

  cpu_rank_list = [4, 8, 16, 32, 64, 128, 256]
  gpu_rank_list = [16]
  local_iterations = [1] #[2, 5, 10]

  #NOTE: Submit arguments
  #ranks, threads, dataset, iter, blocksize, blockleaf, levels, k, leafsize, ltrees
  levels = 30
  """
  levels = 20
  #Run URL GPU
  for rank in gpu_rank_list:
    for ltrees in local_iterations:
      submit(rank, 4, "kdd", 4, 64, 6, levels, 32, 9500, ltrees, True, overlap=1, merge=1)
  """ 
  
  #Run URL CPU
  for ranks in cpu_rank_list:
    for ltrees in local_iterations:
      submit(ranks, 56, "kdd", 2, 512, 256, levels, 32, 16384, ltrees, False, overlap=0, merge=1)
 
  """ 
  levels = 21
  #Run AVAZU GPU
  gpu_rank_list = [1, 2, 3, 4]
  for ranks in gpu_rank_list:
    for ltrees in local_iterations:
      submit(ranks, 4, "avazu", 5, 64, 256, levels, 32, 1600, ltrees, True, overlap=0, merge=0)

  levels = 21
  #Run AVAZU GPU
  gpu_rank_list = [1, 2, 3, 4]
  for ranks in gpu_rank_list:
    for ltrees in local_iterations:
      submit(ranks, 4, "avazu", 5, 64, 256, levels, 32, 1600, ltrees, True, overlap=1, merge=0)
  """ 

  """
  #Run AVAZU CPU
  for ranks in cpu_rank_list:
    for ltrees in local_iterations:
      submit(ranks, 56, "avazu", 5, 64, 256, levels, 32, 1600, ltrees, False, overlap=False, merge=True)
  """

  """
  #Run kdd12 CPU
  for ranks in cpu_rank_list:
    for ltrees in local_iterations:
      submit(ranks, 56, "avazu", 5, 64, 256, levels, 32, 1600, ltrees, False)


  #Run kdd12 GPU
  for ranks in gpu_rank_list:
    for ltrees in local_iterations:
      submit(ranks, 56, "avazu", 5, 64, 256, levels, 32, 1600, ltrees, True)
   """

  os.system("squeue -u wlruys")

