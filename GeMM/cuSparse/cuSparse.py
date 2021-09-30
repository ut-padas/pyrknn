
import numpy as np 

from scipy import sparse 
import time
#import pyculib
import cupy
import math
import time


def gen_SpData(m, d, nnz):
  
  dens = nnz/(d*m)
  S = cupy.sparse.random(m, d, density = dens, format='csr')
  I = list(S.indptr)
  J = list(S.indices)
  V = list(S.data)
  I = np.array(I, dtype=np.int32)
  J = np.array(J, dtype=np.int32)
  V = np.array(V, dtype=np.float32)
  

  return I, J, V





def main_cuSparse():

  er = 0
  d = 10000
  m = 300
  nnz = 60000
  max_nnz = 2000   # max nnz per row
  L = 1
  #cuda.select_device(0) 

  tot_t1 = 0


  er = 0
  for l in range(L):
    #I1, J1, V1 = gen_SpData(m, d, nnz)
    density = (nnz)/(m*d)
    S = cupy.sparse.random(m, d, density = dens, format='csr')
    ST = cupy.transpose(S) 
    #S = pyculib.sparse.CudaCSRMatrix((V1, (U1, J1)), (m,d))
   
    
    D = np.zeros((m, m), dtype = np.float32)
    #D = cuda.to_device(D)
    t0 = time.time()
    d_D = pyculib.sparse.Sparse.csrgemm_ez(S, S, transA='N', transB='T', descrA=None, descrB=None, descrC=None)
    t1 = time.time()
    pyculib.sparse.Sparse.csr2dense(m,m,None, d_D.data, d_D.indptr, d_D.indices, D, (m,m)) 
    
    #_, D_true = rec(I1, J1, V1, m, d)
    #print(C)
    #print(I1)
    #print(J1)
    #print(V1)
    #print(D)
    #print(D_true)
    del d_D
    del I1
    del J1
    del V1
    del D
    delt1 = t1-t0
    #cuda.profile_stop()
    #I2, J2, V2 = gen_SpData(d, m, nnz)
    #I3, J3, V3 = gen_SpData(d, m, nnz)
    #I4, J4, V4 = gen_SpData(d, m, nnz)
    #D1, t1, t2 = cuda_API(I1, J1, V1)
    #D2, t2 = cuda_API(I2, J2, V2)
    #D3, t3 = cuda_API(I3, J3, V3)
    #D4, t4 = cuda_API(I4, J4, V4)
    
    #tot_t += t1 + t2 + t3 + t4
    tot_t1 += delt1
    print(l) 
  output = ''
  print(er)







if __name__ == '__main__':
  main_cuSparse()
















