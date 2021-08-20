#!/usr/bin/env python

import os


def submit(level, nThread, itr, data):
    
  OMP="OMP_NUM_THREADS="+str(nThread)
  CMD="./driver -dataset "+data+" -k 64 -l "+str(level)+" -bt 1 -t "+str(itr)

  filename = "job_"+data+"_lvl_"+str(level)+"_nt_"+str(nThread)+".slm"
  print(filename)

  if data == 'kdd':
    machine = 'nvdimm'
  else:
    machine = 'normal'

  with open(filename, 'w') as f:
    f.writelines("#!/bin/bash\n")
    f.writelines("#SBATCH -J %s-%d\n" % (data, level))
    f.writelines("#SBATCH -e error.txt\n")
    f.writelines("#SBATCH -o %s_lvl_%d_nt_%d.txt\n" % (data, level, nThread))
    f.writelines("#SBATCH -p %s\n" % machine)
    f.writelines("#SBATCH -N 1\n")
    f.writelines("#SBATCH -n 1\n")
    f.writelines("#SBATCH -t 24:00:00\n")
    f.writelines("#SBATCH --mail-user=chenchao.nk@gmail.com\n")
    f.writelines("#SBATCH --mail-type=end\n")
    f.writelines("hostname\n")
    f.writelines("%s %s\n" % (OMP, CMD))
    
    #f.writelines("OMP_NUM_THREADS=%d ./driver -dataset url -l 64 -l %d -bt 1 -t 400\n" % nThread, level)
    #f.writelines("\n")

  os.system("sbatch %s" %filename)
  os.system("rm -f %s" %filename)


if __name__ == '__main__':

  #for level in range(13, 17):
   # submit(level, 56, 500, 'avazu');
  
  #for level in range(9, 14):
   # submit(level, 56, 500, 'url');
  
  for level in range(12, 17):
    submit(level, 56, 10000, 'criteo');
  
  #for level in range(12, 13):
   # submit(level, 112, 1000, 'kdd');
  
  #for level in range(17, 18):
   # submit(level, 56, 2000, 'kdd');
  
  #for nt in [56, 32, 16, 8, 4, 2, 1]:
   # submit(14, nt, 5, 'avazu')
    #submit(16, nt, 5, 'avazu')
  
  #for nt in [56, 32, 16, 8, 4, 2, 1]:
   # submit(11, nt, 5, 'url')
   # submit(10, nt, 5, 'url')
    
  #for nt in [56, 32, 16, 8, 4, 2, 1]:
   # submit(13, nt, 5, 'criteo')
   # submit(14, nt, 5, 'criteo')
    
    
  os.system("squeue -u chaochen")

