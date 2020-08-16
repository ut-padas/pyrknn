import os

foldername = "multinode_url"

#Strong Scaling Set
size_list = [1, 2, 4, 8]

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
    lines.append('#SBATCH -p rtx \n')
    lines.append('#SBATCH -N '+str(p)+' \n')
    lines.append("#SBATCH -n "+str(p)+"\n")
    lines.append('#SBATCH --tasks-per-node 56 \n')
    lines.append('#SBATCH -t 00:10:00 \n')

    lines.append('source set_env.sh \n')
    lines.append('ulimit -c unlimited \n')
    lines.append('ibrun -n '+str(p)+' python run_url.py \n')

    f=open(foldername+'/'+filename, "w")
    f.writelines(lines)
    f.close()

    os.system('sbatch '+str(foldername)+"/"+str(filename))
