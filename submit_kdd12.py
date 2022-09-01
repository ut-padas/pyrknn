#!/usr/bin/env python

import os
import argparse

parser = argparse.ArgumentParser(description="Submit TACC jobs for sparse knn")
parser.add_argument('-run', type=int, default=0)
args = parser.parse_args()


def submit(ranks, threads, iter, leafsize, k, ltrees, gpu_flag=False, size=2**23):
  dataset = "kdd12"
  print(ranks)
  OMP="OMP_NUM_THREADS="+str(threads)
  CMD=f"ibrun -n {ranks} python run_kdd12.py -iter {iter} -leafsize {leafsize} -cores {threads} -ltrees {ltrees} -overlap 1 -use_gpu {gpu_flag} -n {size}"

  filename = f"kdd12_ranks_{ranks}_gpu_{gpu_flag}_threads_{threads}_lt_{ltrees}_leafsize_{leafsize}_run_{args.run}"
  submit_file = "job"+filename+".slm"
  output_file = "out"+filename
  print(filename)

  if gpu_flag:
    print("Using GPU")
    machine = "v100"
    nodes = (ranks-1)//4+1
    tpn = 4
    tasks = ranks
    env_file = "set_env_longhorn.sh"
  else:
    machine = "normal"
    nodes = ranks
    tasks = 1
    tpn = 1
    env_file = "set_env_cpu.sh"
  
  with open(submit_file, 'w') as f:
    f.writelines("#!/bin/bash\n")
    f.writelines("#SBATCH -J %s-%d\n" % (dataset, leafsize))
    f.writelines(f"#SBATCH -e {output_file}.err \n")
    f.writelines(f"#SBATCH -o {output_file}.txt \n")
    f.writelines(f"#SBATCH -p {machine} \n")
    f.writelines(f"#SBATCH -N {nodes}\n")
    f.writelines("#SBATCH -A ASC21002 \n")
    f.writelines(f"#SBATCH -n {tasks}\n")
    f.writelines(f"#SBATCH --tasks-per-node {tpn} \n")
    f.writelines("#SBATCH -t 01:00:00\n")
    f.writelines("#SBATCH --mail-user=will.ruys@gmail.com\n")
    f.writelines("#SBATCH --mail-type=end\n")
    f.writelines("hostname \n")
    f.writelines("source /scratch/06081/wlruys/miniconda3/etc/profile.d/conda.sh \n")
    f.writelines("conda activate /scratch/06081/wlruys/env/knn_mpi \n")
    f.writelines("export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH \n")
    f.writelines(f"source {env_file} \n")
    #f.writelines("module load remora \n")
    f.writelines("ulimit -c unlimited \n")  
    f.writelines("%s %s\n" % (OMP, CMD))

  os.system("sbatch %s" %submit_file)
  #os.system("rm -f %s" %submit_file)


if __name__ == '__main__':

  #Arguments
  #submit(ranks, threads, iter, leafsize, k, ltrees, gpu_flag=False, size=2**23)

  global_iterations = 10
  local_iterations = [1]
  use_gpu = True
  local_size = 2**23
  leafsize = 1024
  k = 8

  for r in [1]:
    for ltrees in local_iterations:
        submit(r, 10, global_iterations, leafsize, k, ltrees, use_gpu, local_size)

  os.system("squeue -u wlruys")

