import os

foldername = "single_url"

#Strong Scaling Set
size_list = [1]

i = 0
for p in size_list:
    lines = []
    i += 1

    filename = "submit_"+str(foldername)+"_run_"+str(i)+".sh"
    name = 's_gpu_'+str(p)

    lines.append('#!/bin/sh \n')
    lines.append('#SBATCH -J '+name+'\n')
    lines.append('#SBATCH -o '+foldername+"/"+name+'.o%j \n')
    lines.append('#SBATCH -e '+foldername+"/"+name+'.e%j \n')
    lines.append('#SBATCH -p skx-dev \n')
    lines.append('#SBATCH -N '+str(p)+' \n')
    lines.append("#SBATCH -n "+str(p)+"\n")
    lines.append('#SBATCH --tasks-per-node 48 \n')
    lines.append('#SBATCH -t 00:20:00 \n')
    lines.append('source set_env_cpu.sh\n')
    lines.append('ulimit -c unlimited \n')
    lines.append('ibrun -n '+str(p)+' python run_url.py \n')

    f=open(foldername+'/'+filename, "w")
    f.writelines(lines)
    f.close()

    os.system('sbatch '+str(foldername)+"/"+str(filename))
