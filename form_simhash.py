import numpy as np 

dataset = "rand15d"

N = 2**15
d = 15

dname = dataset+"_"+str(N//2)+"_data.bin.npy"

data = np.load(dname)

max_hash = 500
ortho = True
hash = np.zeros((N, max_hash))

for i in range(max_hash//d):
    Q = np.random.randn(d, d)
    proj = data@Q.T
    hash[:, i*d:(i+1)*d] = (proj < 0) * 1.0

fname = dataset+"_"+str(N//2)+"_simhash.bin.npy"
np.save(fname, hash)

for i in range(max_hash//d):
    Q = np.random.randn(d, d)
    Q, R = np.linalg.qr(Q)
    proj = data@Q.T
    hash[:, i*d:(i+1)*d] = (proj < 0) * 1.0

fname = dataset+"_"+str(N//2)+"_orthhash.bin.npy"
np.save(fname, hash)