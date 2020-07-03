import os

foldername = "6_25_dense_cpp"

#Begin strong scaling test
repeat_list = [1]*40
i = 0
for p in repeat_list:
    lines = []
    i += 1
    filename = "submit_"+str(foldername)+"_run_"+str(i)+".sh"
    name = 's_gpu_'+str(p)

    lines.append('#!/bin/sh \n')
    lines.append('#SBATCH -J '+name+'\n')
    lines.append('#SBATCH -o '+foldername+"/"+name+'.o%j \n')
    lines.append('#SBATCH -e '+foldername+"/"+name+'.e%j \n')
    lines.append('#SBATCH -p rtx \n')
    lines.append('#SBATCH -N '+str((p+3)//4)+' \n')
    lines.append("#SBATCH -n "+str(p)+"\n")
    lines.append('#SBATCH --tasks-per-node 4 \n')
    lines.append('#SBATCH -t 00:12:00 \n')

    lines.append('source set_env.sh \n')
    lines.append('ulimit -c unlimited \n')
    #lines.append('export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \n')
    lines.append('export CUDA_LAUNCH_BLOCKING=1 \n')
    #lines.append('ibrun -n '+str(p)+' strace -f -o dense_'+str(i)+'.trace python test_dense.py \n')
    lines.append('ibrun -n '+str(p)+' python test_dense.py \n')
    f=open(foldername+'/'+filename, "w")
    f.writelines(lines)
    f.close()

    os.system('sbatch '+str(foldername)+"/"+str(filename))
