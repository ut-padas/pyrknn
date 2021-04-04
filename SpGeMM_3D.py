
import numpy as np 

from scipy import sparse 
import time
from numba import cuda, float32, int16, int32, int64, gdb_init, types, typeof
import math
import numba
import time
from scipy import sparse
import random
  








@cuda.jit('void(int32[:], int32[:], float32[:], float32[:], int32, int32[:], int32[:], int32, int32)') 
def SpGeMM_3D(I, J, V, D, m, start, stop, max_nnz, batchsize):
  

  i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
  j = cuda.blockIdx.y
  z = cuda.blockIdx.z
 
  # select threads within the input size 
  if i>= m or j>= m or z>= batchsize or j > i: return
  
  nnz = stop[z] - start[z]
  

  # select indices corresponding to row i , j
  ind0_i=I[i + z*(m+1)] + start[z] 
  ind1_i=I[i+1 + z*(m+1)] + start[z]

  ind0_j=I[j + z*(m+1)] + start[z]
  ind1_j=I[j+1 + z*(m+1)] + start[z]

  # number of nnz within row i , j
  nnz_j = ind1_j - ind0_j
  nnz_i = ind1_i - ind0_i

  # check the violation of max nnz 
  if nnz_i > max_nnz: raise Exception('nnz per row > max nnz')   
  if nnz_j > max_nnz: raise Exception('nnz per row > max nnz')   


  # remove the zero rows
  if nnz_j==0 or nnz_i == 0 : return


  norm_ij = 0 

  # register the data for each row 
  v_i = V[ind0_i:ind1_i]
  v_j = V[ind0_j:ind1_j]
  
  # norm of row i  
  for n_i in range(nnz_i):
    norm_ij += v_i[n_i]**2
  
  # remove rows with zero distance
  
  if i==j: 
    c_tmp = norm_ij
    ind = m*i + i + z*(m**2)
    D[ind] = c_tmp
    return
  
  # norm of row j
  for n_j in range(nnz_j):
    norm_ij += v_j[n_j]**2
 
  
  # assign shared memory for row j (col indices)
  sj = cuda.shared.array(shape=(2048),dtype=int32)
  
  for l in range(nnz_j):
    sj[l] = J[ind0_j + l]
 
  # register col indices for row i
  si = J[ind0_i:ind1_i]

  # search for the same column index among row i,j
  c_tmp = 0
  log_n = math.floor(math.log(max_nnz)/math.log(2)) 

  for pos_k in range(0, ind1_i-ind0_i):

    k = si[pos_k]
    
    # Binary_search
    ret = 0
    testInd = 0
     
    for l in range(log_n, 0, -1):
      
      if nnz_j<pow(2, l): continue
      
      testInd = min(ret + 2**l, nnz_j-1)
      ret = testInd if sj[testInd]<= k else ret 
    
    testInd = min(ret+1, nnz_j-1)
    ret = testInd if sj[testInd]<=k else ret

    # check the result    
    ind_jk = ret if sj[ret] == k else -1
    
    c = v_i[pos_k]*v_j[ind_jk] if ind_jk != -1 else 0
    
    c_tmp += c

  #c_tmp = max(-2*c_tmp + norm_ij, 0)
  
  # write to the global array
  ind_ij = m*i + j + z*(m**2)
  ind_ji = m*j + i + z*(m**2)
  D[ind_ij] = c_tmp
  D[ind_ji] = c_tmp

  
  '''
  if i == 1 and j == 0 and z== 1:
    print('distance matrix')
    for w in range(m**2):
      print(w , D[w + z*m**2])
  '''

  cuda.syncthreads()

  # compute k lowest distances 
  '''

  num_updiag = (m**2 - m)//2
  tid = i*(m-1 - i) + j-1 + z*num_updiag

  

  if i == 1 and j == 0 and z== 0:
    print('dist kernel before sort')
    for w in range(num_updiag):
      print(w , D[tid])
  cuda.syncthreads()

  #sj = cuda.shared.array(shape=(2000),dtype=float32)
  z_ind0 = z*num_updiag + z*m**2 
  z_ind1 = (z+1)*num_updiag + z*m**2
  sj = D[z_ind0:z_ind1]


  cuda.syncthreads()

  log_size = math.log(num_updiag**2)/math.log(2)

  
  if tid > (num_updiag)-1:
    print('error')

  for k_step in range(1, log_size+1, 1):
    k = pow(2, k_step)
    
    for l_step in range(k_step-1, -1, -1):
      l = pow(2, l_step)
      
      ixj = tid ^ l  
      

      if ixj > tid :
        if tid & k == 0: 
          
          #cuda.compare_and_swap()
          if sj[tid] > sj[ixj]:
            sj[tid], sj[ixj] = sj[ixj], sj[tid]
    
        else:
          if sj[tid] < sj[ixj]:
            sj[tid], sj[ixj] = sj[ixj], sj[tid]

      cuda.syncthreads()

  if i == 0 and j == 1 and z== 0:
    print('dist kernel after sort')
    for w in range(num_updiag):
      print(w , sj[tid])

  '''
  return




@cuda.jit('void(float32[:],float32[:], int32[:], int32[:], int32, int32, int32)') 
def compute_knn(D, K ,I_k, J_k, k , m, batchsize):

  i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x 
  j = cuda.blockIdx.y
  
  # bitonic sort for distance matrix 
  

  if m**2 > 2048: raise Exception(' number of distance > 2000')     

  # tid for distance matrix
  col_D = i // m 
  row_D = i - col_D*m 
  
  
  num_updiag = (m**2 - m)//2
  log_size = math.ceil(math.log(num_updiag)/math.log(2))
  if i >= pow(2, log_size+1) : return 
  
  # mapping of 2D to inline indexing 
  #tid_D = row_D*(m-1 - row_D) + col_D-1 + j*m**2 
  tid_D = m*row_D + col_D + j*(m**2)
  
  #if tid_D > batchsize*m**2: return

  # mapping of 2D to updiagonal row-based indexing
  tid =  col_D - row_D - 1 + (m* row_D - (row_D**2 + row_D)//2) #+ j*num_updiag


  #tid for the shared memory 
  sj = cuda.shared.array(shape=(2048),dtype=float32)
  I = cuda.shared.array(shape=(2048),dtype=float32)
  J = cuda.shared.array(shape=(2048),dtype=float32)
  if col_D > row_D:
    sj[tid] = D[tid_D]
    I[tid] = row_D 
    J[tid] = col_D 
  cuda.syncthreads()
  
  
  # sorting
  
  log_size = math.ceil(math.log(num_updiag)/math.log(2))
  '''
  if row_D == 0 and col_D == 1 and j == 1:
    print('before sort')
    for w in range(pow(2, log_size)):
      print(w, sj[w])
  '''

  if i >= pow(2 , log_size) : return
    
  for g_step in range(1, log_size+1, 1):
    g = pow(2, g_step)
    
    for l_step in range(g_step-1, -1, -1):
      l = pow(2, l_step)
      
      ixj = i ^ l  
    
      if ixj > i :
        if i & g == 0: 
          
          #cuda.compare_and_swap()
          if sj[i] > sj[ixj]:
            sj[i], sj[ixj] = sj[ixj], sj[i]
            I[i], I[ixj] = I[ixj], I[i]
            J[i], J[ixj] = J[ixj], J[i]
    
        else:
          if sj[i] < sj[ixj]:
            sj[i], sj[ixj] = sj[ixj], sj[i]
            I[i], I[ixj] = I[ixj], I[i]
            J[i], J[ixj] = J[ixj], J[i]

      cuda.syncthreads()
    

  # write the results back
  diff = int(pow(2, log_size) - num_updiag)

  if i >= diff and i <= k + diff: 
    #ind = i-diff
    K[i-diff + j*k] = sj[i]
    I_k[i-diff + j*k] = I[i]
    J_k[i-diff + j*k] = J[i]
    

  cuda.syncthreads()
  '''
  if row_D == 0 and col_D == 1 and j == 1:
    print('after sort')
    print(k)
    for w in range(k):
      print(w, K[w + j*k], I_k[w + j*k], J_k[w + j*k])
    
  ''' 
  #print(tid, sj[tid])
  
  cuda.syncthreads()

  '''
  if (i == 1 and j == 0):
    print('after sort')
    for n in range(m**2):
      print(n , D[n])
  '''
  return



def gpu_sparse_knn(X, k):


  cuda.select_device(0) 
  # TODO   
  #m, d = X.shape

  # num of rows columns of X 
  m = 300  
  d = 10000
  k = 32
  # test 
  nnzperrow = 600
  # max nonzero per row 
  max_nnz = 1000 
  # batches to calculate in each kernel call 
  batchsize = 2 
  ntrees = 1
  
  threadsperblock_x = 256
  blockpergrid = (threadsperblock_x + m - 1)// threadsperblock_x
  
  blockdim = threadsperblock_x, 1 , 1 
  griddim = blockpergrid, m, batchsize


  for iteration in range(ntrees):

    # TODO 
    #X2 = c_build_tree(X, levels)
   
     
    X2 = gen_SpData_2D(m, d, nnzperrow, batchsize)
    leaves = [1] 
    for batch in leaves:
      #start = batch.start 
      #stop = batch.end

      D = np.zeros((m**2*batchsize), dtype = np.float32)
      K = np.zeros((batchsize*k), dtype = np.float32)
      I_k = np.zeros((batchsize*k), dtype = np.float32)
      J_k = np.zeros((batchsize*k), dtype = np.float32)
      d_I = cuda.to_device(X2['rowptr'])
      d_J = cuda.to_device(X2['colind'])
      d_V = cuda.to_device(X2['data'])
      d_start = cuda.to_device(X2['start'])
      d_stop = cuda.to_device(X2['stop'])
      d_D = cuda.to_device(D)
      d_K = cuda.to_device(K)
      d_I_k = cuda.to_device(I_k)
      d_J_k = cuda.to_device(J_k)
      
      t0 = time.time()
      
      # kernel
      SpGeMM_3D[griddim, blockdim](d_I,d_J,d_V,d_D, m, d_start, d_stop, max_nnz, batchsize)
      cuda.synchronize()
      blockpergrid = (threadsperblock_x + m**2//2 - 1)// threadsperblock_x
      blockdim = threadsperblock_x, 1 
      griddim = blockpergrid, batchsize
      print(blockdim, griddim)
      #log_size = math.log(m**2)/math.log(2)
      
      t1 = time.time()
      compute_knn[griddim, blockdim](d_D, d_K ,d_I_k, d_J_k, k, m, batchsize)

      cuda.synchronize()
      del_t0 = t1 - t0
      del_t1 = time.time() - t1
      D = d_D.copy_to_host()
      

      
      msg = 'iter %d \t batch %d /%d \t kernel time (s) : %.3e \t %.3e'%(iteration, batch, len(leaves), del_t0, del_t1)   
      
      print(msg)
      
      del d_I
      del d_J
      del d_V
      del d_start
      del d_stop
      del d_D



# array for debug (rec of true matrix)
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


# for generation of 2D sparse data for Z as batchsize, 
#m rows and d columns 
def gen_SpData_2D(m, d, nnzperrow, Z):
  
  
  I = np.zeros(((m+1)*Z), dtype = np.int32)
  J = np.array([], dtype = np.int32)
  V = np.array([], dtype = np.float32)
  #X = np.zeros((nnz*Z, 3), dtype = np.int32)
  start = np.zeros((Z), dtype = np.int32)
  stop = np.zeros((Z), dtype = np.int32)
  
  X = {}
 
  nnz_i = 0
  nnz_z = 0
  print('generating random')
  for z in range(Z):
    nnz_z = 0
    I[0 + z*(m+1)] = 0
    start[z] = J.shape[0]
    for i in range(m):
      ind = i + z*(m+1)
      nnz_i = random.randrange(nnzperrow//2, 3*nnzperrow//2, 1)
      nnz_i = min(nnz_i, d)
      nnz_z += nnz_i
      I[ind+1] = I[ind] + nnz_i
      J = np.append(J, np.zeros((nnz_i), dtype = np.int32))
      V = np.append(V, np.zeros((nnz_i), dtype = np.float32))
      for j in range(nnz_i):
        ind = I[i + z*(m+1)] + start[z]
        val = random.randrange(0, d, 1)
        while val in J[ind:ind+j]: 
          val = random.randrange(0, d, 1)
        J[ind+j] = val
        V[ind+j] = j+1
      ind0 = I[i + z*(m+1)] + start[z]
      ind1 = ind0 + nnz_i
      J[ind0:ind1] = np.sort(J[ind0:ind1])
    stop[z] = start[z]+nnz_z  
    
  X['rowptr'] = I
  X['colind'] = J 
  X['data'] = V
  X['start'] = start
  X['stop'] = stop
  return X
 

# for debug 
def inline2mat(D, m , z):

    OUT = np.zeros((m,m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            OUT[i,j] = D[i*m + j + m*m*z]
    return OUT

       

def main():
  er = 0
  d = 10000
  m = 300
  nnzperrow = 100
  max_nnz = 300   # max nnz per row
  L = 1
  Z = 100	  # num of simoltaneous AA^T
  '''
  X = gen_SpData_2D(m, d, nnzperrow, Z)
  print('rowptr')
  print(X['rowptr'])
  print('colind')
  print(X['colind'])
  print('data')
  print(X['data'])
  print('start')
  print(X['start'])
  print('stop')
  print(X['stop'])
  '''
  
  
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
    X = gen_SpData_2D(m, d, nnzperrow, Z)
    #I1, J1, V1 = gen_SpData_2D(m, d, nnzperrow, Z)
    I = X['rowptr']
    J = X['colind']
    V = X['data']
    start = X['start']
    stop = X['stop']
		 
    D = np.zeros((m**2*Z), dtype = np.float32)
    d_I = cuda.to_device(I)
    d_J = cuda.to_device(J)
    d_V = cuda.to_device(V)
    d_D = cuda.to_device(D)
    t0 = time.time()
    SpGeMM_3D[griddim, blockdim](d_I,d_J,d_V,d_D, m, start, stop, max_nnz, Z)
    cuda.synchronize()
    t1 = time.time()
    D = d_D.copy_to_host()
    #D_true = np.zeros((m**2, Z), dtype = np.float32)

    D_true = np.zeros((m**2*Z), dtype = np.float32)
    for z in range(Z):
        I_start = (m+1)*z
        I_stop = (m+1)*(z+1)
        J_start = start[z]
        J_stop = stop[z]
        
        A, tmp = rec(I[I_start:I_stop], J[J_start:J_stop], V[J_start:J_stop], m, d)
         
        if z <-1 :
            print('z is %d'%z)
            print(A)
            print('start z')
            print(start[z])
            print('stop z')
            print(stop[z])
            print('row ptr for z = 0 :')
            print(I[I_start:I_stop])
            print('col ind :')
            print(J[J_start:J_stop])
            print('val ind : ')
            print(V[J_start:J_stop])
            print('true  : ')
            print(tmp)
            out = inline2mat(D, m , z)
            print('rec out :')
            print(out)
        
        D_true[m**2*z:m**2*(z+1)] = tmp.flatten()
    #print(D[:,:,0])
    #print(D_true[:,:,0])
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
    del X
    del start
    del stop
    del I
    del V
    del J
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
  output = "$\\num{%.1e}$ & $\\num{%.1e}$ & $\\num{%d}$ & $\\num{%d}$  &  $\\num{%.2e}$ & $\\num{%.3e}$ \\\ "%(nnzperrow, m, d,Z, L,tot_t1)
  

  

  #print("Invalid number of nnz per row") if nnz > 2000 else print("Number of nnz <2000")
  
 

  #print(output)

if __name__ == '__main__':
  gpu_sparse_knn(0, 0)
















