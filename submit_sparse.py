#!/usr/bin/env python

import os


def submit(ranks, threads, dataset, iter, blocksize, blockleaf, levels, k, leafsize, ltrees, gpu_flag=False):
  
  print(ranks)
  OMP="OMP_NUM_THREADS="+str(threads)
  CMD=f"mpirun -n {ranks} python run_sparse.py -iter {iter} -dataset {dataset} -bs {blocksize} -bl {blockleaf} -levels {levels} -cores {threads} -ltrees {ltrees} -use_gpu {gpu_flag}"

  filename = f"_{dataset}_ranks_{ranks}_bs_{blocksize}_bl_{blockleaf}_gpu_{gpu_flag}_threads_{threads}"
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
    tasks = 1
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
    f.writelines(f"#SBATCH -n {tasks}\n")
    f.writelines(f"#SBATCH --tasks-per-node {tpn} \n")
    f.writelines("#SBATCH -t 02:00:00\n")
    f.writelines("#SBATCH --mail-user=will.ruys@gmail.com\n")
    f.writelines("#SBATCH --mail-type=end\n")
    f.writelines("hostname \n")
    f.writelines("export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH \n")
    f.writelines(f"source {env_file}")
    #f.writelines("module load intel/19.0.5  impi/19.0.5 \n")
    #f.writelines("module load remora \n")
    f.writelines("ulimit -c unlimited \n")  
    f.writelines("%s %s\n" % (OMP, CMD))

  os.system("sbatch %s" %submit_file)
  #os.system("rm -f %s" %submit_file)


if __name__ == '__main__':

  #if not os.path.exists(folder):
  #  os.makedirs(folder)

  cpu_rank_list = [1, 2] #4, 8, 16, 32]
  gpu_rank_list = [1, 2, 4, 8, 16]
  local_iterations = [2] #[2, 5, 10]

  #NOTE: Submit arguments
  #ranks, threads, dataset, iter, blocksize, blockleaf, levels, k, leafsize, ltrees

  
  levels = 11
  #Run URL GPU
  for rank in gpu_rank_list:
    for ltrees in local_iterations:
      submit(rank, 4, "url", 90, 64, 256, levels, 32, 512, ltrees, True)
  

  """
  #Run URL CPU
  for ranks in cpu_rank_list:
    for ltrees in local_iterations:
      submit(ranks, 56, "url", 90, 64, 256, levels, 32, 512, ltrees)
  """

  levels = 13
  #Run AVAZU GPU
  for ranks in gpu_rank_list:
    for ltrees in local_iterations:
      submit(ranks, 4, "avazu", 90, 64, 256, levels, 32, 512, ltrees, True)
  
  """
  #Run AVAZU CPU
  for ranks in cpu_rank_list:
    for ltrees in local_iterations:
      submit(ranks, 56, "avazu", 90, 64, 256, levels, 32, 512, ltrees)
  """

  os.system("squeue -u wlruys")
