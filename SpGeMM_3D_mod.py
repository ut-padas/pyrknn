
import numpy as np 

from scipy import sparse 
import time
from numba import cuda, float32, int16, int32, int64, gdb_init, types, typeof
import math
import numba
import time
from scipy import sparse
import random
  








@cuda.jit('void(int32[:], int32[:], float32[:], float32[:], int32, int32, int32, int32)') 
def SpGeMM_3D(I, J, V, D, m, nnzperrow, max_nnz, Z):
  

  i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
  j = cuda.blockIdx.y
  z = cuda.blockIdx.z
 
  if i>= m or j>= m or z>= Z: return
  
  nnz = m*nnzperrow
  

  ind0_i=I[i + z*(m+1)] + nnz*z
  ind1_i=I[i+1 + z*(m+1)] + nnz*z

  ind0_j=I[j + z*(m+1)] + nnz*z
  ind1_j=I[j+1 + z*(m+1)] + nnz*z


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
  log_n = math.floor(math.log(max_nnz)/math.log(2)) 
  #log_n = math.floor(math.log(nnz_j)/math.log(2)) 
  #log_n = math.log(nnz_j)/math.log(2)
  for pos_k in range(0, ind1_i-ind0_i):

    k = si[pos_k]
     
    # Binary_search
    ret = 0
    testInd = 0
    
    for l in range(log_n, 0, -1):
      if l>log_n_true: continue
      testInd = min(ret + 2**l, nnz_j-1)
      ret = testInd if sj[testInd]<= k else ret 
    testInd = min(ret+1, nnz_j-1)
    ret = testInd if sj[testInd]<=k else ret
    ind_jk = ret if sj[ret] == k else -1
 
    c = v_i[pos_k]*v_j[ind_jk] if ind_jk != -1 else 0
    #if i == 0 and j==1 and z == 1: 
    #    print(i, j, z, k, pos_k, ind_jk)
    c_tmp += c

  #c_tmp = max(-2*c_tmp + norm_ij, 0)
  #print(i,j)
  ind = m*i + j + z*(m**2)
  D[ind] = c_tmp




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







def gen_SpData_2D(m, d, nnzperrow, Z):
  
  nnz = nnzperrow*m
  I = np.zeros(((m+1)*Z), dtype = np.int32)
  J = np.zeros((nnz*Z), dtype = np.int32)
  V = np.zeros((nnz*Z), dtype = np.float32)


  for z in range(Z):
    I[0 + z*(m+1)] = 0
    for i in range(m):
      I[i+1+z*(m+1)] = I[i+z*(m+1)] + nnzperrow
      for j in range(nnzperrow):
        
        ind = I[i] + j + z*nnz
        val = random.randrange(0, d, 1) 
        while val in J[I[i]+z*nnz:ind]: 
          val = random.randrange(0, d, 1)
        #print(val, J[ind-1])
        J[ind] = val
        V[ind] = j
      ind0 = i*nnzperrow + z*nnz
      ind1 = (i+1)*nnzperrow + z*nnz
      
      J[ind0:ind1] = np.sort(J[ind0:ind1])
      #print(J[ind0:ind1])
       

  #dens = nnz/(d*m)
  #I = np.zeros((m+1,Z), dtype=np.int32)
  #J = np.zeros((nnz, Z), dtype=np.int32)
  #V = np.zeros((nnz, Z), dtype=np.float32)
  
  return I, J, V
  
def out2mat(D, m , nnz, z):

    OUT = np.zeros((m,m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            OUT[i,j] = D[i*m + j + m*m*z]

    return OUT








def main():
  er = 0
  d = 10000
  m = 300
  nnzperrow = 600
  nnz = nnzperrow*m
  max_nnz = 2000   # max nnz per row
  L = 1
  Z = 1	  # num of simoltaneous AA^T

  cuda.select_device(0) 
  tot_t1 = 0
  tot_t2 = 0
  tot_t = 0
  tot_ti = 0
  B = min(256, m)
  blockpergrid = (m + B-1)//B
  blockpergrid = max(1, blockpergrid)
  blockdim = B, 1, 1
  griddim = blockpergrid, m, Z
  print(blockpergrid)
  print(B)
  er = 0
  print('rows of D ', m)
  for l in range(L):
    I1, J1, V1 = gen_SpData_2D(m, d, nnzperrow, Z)

    print('row pointer ')
    #print(I1)
    print('col indices')
    #print(J1)
    print('val indices')
    #print(V1)
		 
    D = np.zeros((m**2*Z), dtype = np.float32)
    d_I = cuda.to_device(I1)
    d_J = cuda.to_device(J1)
    d_V = cuda.to_device(V1)
    d_D = cuda.to_device(D)
    t0 = time.time()
    SpGeMM_3D[griddim, blockdim](d_I,d_J,d_V,d_D, m, nnzperrow, max_nnz, Z)
    t1 = time.time()
    D = d_D.copy_to_host()
    #D_true = np.zeros((m**2, Z), dtype = np.float32)

    D_true = np.zeros((m**2*Z), dtype = np.float32)
    for z in range(Z):
        I_start = (m+1)*z
        I_stop = I_start + m
        J_start = I1[I_start] + nnz*z
        J_stop = I1[I_stop] + nnz*z-1
        
        _, tmp = rec(I1[(m+1)*z:(m+1)*(z+1)], J1[nnz*z:nnz*(z+1)], V1[nnz*z:nnz*(z+1)], m, d)
        '''
        if z == 1:
            print('row ptr for z = 1 :')
            print(I1[I_start:I_stop])
            print('col ind :')
            print(J1[J_start:J_stop])
            print('val ind : ')
            print(V1[J_start:J_stop])
            print('output : ')
            print(tmp)
            out = out2mat(D, m , nnz, 1)
            print(out)
        '''
        D_true[m**2*z:m**2*(z+1)] = tmp.flatten()

    #print(D[:,:,0])
    #print(D_true[:,:,0])
    cuda.synchronize()
    er = max(er, np.linalg.norm((D-D_true).flatten()))
    #norm_rec = np.linalg.norm(D_true.flatten())
    #norm_true = np.linalg.norm(D)

    print('rec D : ')
    #print(D) 

    print('true D : ') 
    #print(D_true)
    del tmp
    del d_D
    del d_I
    del d_J
    del d_V
    del D
    del D_true
    del I1
    del V1
    del J1
    delt1 = t1-t0
    cuda.profile_stop()
    
    #tot_t += t1 + t2 + t3 + t4
    
    tot_t1 += delt1
    print(l) 
  output = ''
  print("Elapased time (s) : %.3e"%tot_t1)
  print('max norm of error : ', er)
  #print('norm of the recostructed : ', norm_rec)
  #print('true norm : ', norm_true)
  print(tot_t1)
  output = "$\\num{%.1e}$ & $\\num{%.1e}$ & $\\num{%d}$ & $\\num{%d}$  &  $\\num{%.2e}$ & $\\num{%.3e}$ \\\ "%(nnz, m, d,Z, L,tot_t1)
  

  

  #print("Invalid number of nnz per row") if nnz > 2000 else print("Number of nnz <2000")
  
 

  print(output)

if __name__ == '__main__':
  main()
















