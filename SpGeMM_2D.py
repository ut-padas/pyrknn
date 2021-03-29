
import numpy as np 

from scipy import sparse 
import time
from numba import cuda, float32, int16, int32, int64, gdb_init, types, typeof
import math
import numba
import time
from scipy import sparse

  








@cuda.jit('void(int32[:], int32[:], float32[:], float32[:,:], int32, int32)') 
def SpGeMM_2D(I, J, V, D, m, max_nnz):
  

  i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
  j = cuda.blockIdx.y
  if i>= m or j>= m: return
   


  ind0_i=I[i]
  ind1_i=I[i+1]

  ind0_j=I[j]
  ind1_j=I[j+1]


  nnz_j = ind1_j - ind0_j
  nnz_i = ind1_i - ind0_i


  if nnz_j==0 or nnz_i == 0 : return

  #sj =J[ind0_j:ind1_j]

  sj = cuda.shared.array(shape=(2000),dtype=int32)
  for l in range(nnz_j):
    sj[l] = J[ind0_j+l]
 
 
  si = J[ind0_i:ind1_i]
  

  
  norm_ij = 0  
  v_i = V[ind0_i:ind1_i]
  v_j = V[ind0_j:ind1_j]
  
  for n_i in range(nnz_i):
    norm_ij += v_i[n_i]**2
  for n_j in range(nnz_j):
    norm_ij += v_j[n_j]**2


  
  c_tmp = 0
  log_n_true = math.floor(math.log(nnz_j)/math.log(2)) 
  #log_n = math.floor(math.log(max_nnz)/math.log(2)) 
  log_n = math.floor(math.log(nnz_j)/math.log(2)) 
  log_n1 = math.ceil(math.log(nnz_j)/math.log(2)) 
  for pos_k in range(0, ind1_i-ind0_i):

    k = si[pos_k]
     
    # Binary_search
    ret = 0
    testInd = 0
    
    for l in range(log_n, 0, -1):
      #tmp = math.ceil(nnz_j/2**l)
      testInd = min(ret + 2**l, nnz_j-1)
      ret = testInd if sj[testInd]<= k else ret 
    testInd = min(ret+1, nnz_j-1)
    ret = testInd if sj[testInd]<=k else ret
    ind_jk = ret if sj[ret] == k else -1

    c = v_i[pos_k]*v_j[ind_jk] if ind_jk != -1 else 0
    #if i == 0 and j==1: print(i, j, k, ind_jk, c)
    c_tmp += c

  #c_tmp = max(-2*c_tmp + norm_ij, 0)
  #print(i,j)
  D[i,j] = c_tmp




def rec(I, J, V, m, d):
  A = np.zeros((m,d))
  for i in range(m):
    if I[i] != I[i+1]:
      nnz = I[i+1] - I[i]
      for j in range(nnz):
        col = J[I[i]+j]
        A[i, col] = V[I[i]+j]
 
  OUT = np.matmul(A, A.transpose())
  return A, OUT








def gen_SpData(m, d, nnz):
  

  dens = nnz/(d*m)
  S = sparse.random(m, d, density = dens, format='csr')
  I = list(S.indptr)
  J = list(S.indices)
  V = list(S.data)
  I = np.array(I, dtype=np.int32)
  J = np.array(J, dtype=np.int32)
  V = np.array(V, dtype=np.float32)
  

  return I, J, V





def main():
  er = 0
  d = 100000
  m = 300
  nnz = 30000
  max_nnz = 1   # max nnz per row
  L = 1
  cuda.select_device(0) 
  tot_t1 = 0
  tot_t2 = 0
  tot_t = 0
  tot_ti = 0
  B = m
  blockpergrid = (m + B-1)//B
  blockpergrid = max(1, blockpergrid)
  blockdim = B, 1
  griddim = blockpergrid, m
  print(blockpergrid)
  print(B)
  er = 0
  print('rows of D ', m)
  for l in range(L):
    I1, J1, V1 = gen_SpData(m, d, nnz)
    D = np.zeros((m, m), dtype = np.float32)
    C, D_true = rec(I1, J1, V1, m, d)
    #print(C)
    #print(I1)
    #print(J1)
    #print(V1)
    d_I = cuda.to_device(I1)
    d_J = cuda.to_device(J1)
    d_V = cuda.to_device(V1)
    d_D = cuda.to_device(D)
    t0 = time.time()
    SpGeMM_2D[griddim, blockdim](d_I,d_J,d_V,d_D, m, max_nnz)
    t1 = time.time()
    D = d_D.copy_to_host()
    #print(D)
    #print(D_true)
    cuda.synchronize()
    er = max(er, np.linalg.norm((D-D_true).flatten()))
    del d_D
    del d_I
    del d_J
    del d_V
    del D
    del I1
    del V1
    del J1
    delt1 = t1-t0
    cuda.profile_stop()
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
  output = "$\\num{%.1e}$ & $\\num{%.1e}$ & $\\num{%.1e}$ & $\\num{%.2e}$ & $\\num{%.3e}$ \\\ "%(nnz, m, d, L,tot_t1)
  

  

  #print("Invalid number of nnz per row") if nnz > 2000 else print("Number of nnz <2000")
  
 

  print(output)

if __name__ == '__main__':
  main()
















