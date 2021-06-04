#!/usr/bin/env python

import os


def submit(level, nThread, itr, data, trial):
    
  OMP="OMP_NUM_THREADS="+str(nThread)
  CMD="remora ./driver -dataset "+data+" -k 64 -l "+str(level)+" -bt 1 -t "+str(itr)

  filename = "job_"+data+"_real_lvl_"+str(level)+"_nt_"+str(nThread)+"_"+str(trial)+".slm"
  print(filename)

  if data == 'kdd':
    machine = 'nvdimm'
  else:
    machine = 'small'

  with open(filename, 'w') as f:
    f.writelines("#!/bin/bash\n")
    f.writelines("#SBATCH -J %s-%d-%d\n" % (data, level, trial))
    f.writelines("#SBATCH -e %s_real_lvl_%d_nt_%d_%d.err \n" % (data, level, nThread, trial))
    f.writelines("#SBATCH -o %s_real_lvl_%d_nt_%d_%d.txt \n" % (data, level, nThread, trial))
    f.writelines("#SBATCH -p %s\n" % machine)
    f.writelines("#SBATCH -N 1\n")
    f.writelines("#SBATCH -A ASC21002 \n")
    f.writelines("#SBATCH -n 1\n")
    f.writelines("#SBATCH -t 4:00:00\n")
    f.writelines("#SBATCH --mail-user=will.ruys@gmail.com\n")
    f.writelines("#SBATCH --mail-type=end\n")
    f.writelines("hostname \n")
    f.writelines("export LD_LIBRARY_PATH=$SCRATCH/pyrknn/chao/readSVM:$LD_LIBRARY_PATH \n")
    f.writelines("export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH \n")
    f.writelines("module load intel/19.0.5  impi/19.0.5 \n")
    f.writelines("module load remora \n") 
    f.writelines("%s %s\n" % (OMP, CMD))
    #f.writelines("OMP_NUM_THREADS=%d ./driver -dataset url -l 64 -l %d -bt 1 -t 400\n" % nThread, level)
    #f.writelines("OMP_NUM_THREADS=%d ./driver -dataset %s -bp 64 -l %d -bt 1 -t %d \n" % (nThread, level, itr))

  os.system("sbatch %s" %filename)
  #os.system("rm -f %s" %filename)


if __name__ == '__main__':
  for trial in range(13, 14):
      for level in range(13, 15):
        submit(level, 56, 100, 'avazu_full', trial);
  
  #for level in range(9, 15):
  #  submit(level, 56, 150, 'url');
  
  #for level in range(12, 17):
   # submit(level, 56, 10000, 'criteo');
  
  #for level in range(14,15):
  #  submit(14, 112, 200, 'kdd');
  
  #for level in range(15, 16):
  #  submit(level, 56, 6, 'kdd');
  
  #for nt in [56, 32, 16, 8, 4, 2, 1]:
   # submit(14, nt, 5, 'avazu')
    #submit(16, nt, 5, 'avazu')
  
  #for nt in [56, 32, 16, 8, 4, 2, 1]:
   # submit(11, nt, 5, 'url')
   # submit(10, nt, 5, 'url')
    
  os.system("squeue -u wlruys")

