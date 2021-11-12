
import os , sys
import pandas as pd
import numpy as np


file = '../logs/fused_sparse_url.log'
N = 1000 

with open(file, "r") as f:
  lines = f.readlines()
accs = []
its = []
errs = []
times = []
for l in lines:
  if "Recall accuracy: " in l:
    print(l)
    acc = l.split("Recall accuracy: ")[1].split("mean rel distance error = ")[0]
    accs.append(float(acc))
    #if "it = " in l:
    it = l.split("it = ")[1].split("Recall accuracy: ")[0]
    its.append(int(it))
    #if "distance error = " in l:
    err = l.split("mean rel distance error = ")[1].split("cost = ")[0] 
    errs.append(float(err))
  if "it =" in l and "RKDT :" in l:
    time = l.split("RKDT : ")[1].split(" sec")[0]
    times.append(float(time)) 


fname = file.replace(".log", ".csv")


t = np.asarray(times)
qsec = N/t


dic = {}
dic['its'] = its 
dic['acc'] = accs
dic['err'] = errs
dic['time'] = times
dic['qsec'] = qsec

df = pd.DataFrame(dic)

df.to_csv(fname, index = False)


