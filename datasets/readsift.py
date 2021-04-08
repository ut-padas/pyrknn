filename ='/scratch1/00921/biros/bigann_learn.bvecs'
d=128
vsz = 4+d
start = 0
nc = 2;   #how many to read
v = np.fromfile(filename, dtype=np.uint8,count=nc*vsz,offset= st*vsz)
print(np.reshape(v,(nc,d+4)))

    
