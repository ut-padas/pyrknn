
import numpy as np 

from scipy import sparse 
import time
from numba import cuda, float32, int16, int32, int64, gdb_init, types, typeof
import math
import numba
import time
from scipy import sparse
import random
  








@cuda.jit('void(int32[:], int32[:], float32[:], int32[:], int32[:], float32[:], float32[:], int32, int32[:], int32[:], int32[:], int32[:], int32, int32)') 
def SpGeMM_3D(R_I, C_I, V_I, R_J, C_J, V_J, D_IJ, m, start_I, stop_I, start_J, stop_J, max_nnz, batchsize_x):
  

  i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x
  j = cuda.blockIdx.y
  z = cuda.blockIdx.z
 
  # select threads within the input size 
  if i>= m or j>= m or z>= batchsize_x: return
  

  # total number of nonzero for X[I, :] and X[J, :]
  nnz_I = stop_I[0] - start_I[0]
  nnz_J = stop_J[z] - start_J[z]


  # select indices corresponding to row i , j
  #ind0_i=I[i + z*(m+1)] + start[z] 
  #ind1_i=I[i+1 + z*(m+1)] + start[z]

  ind0_i = R_I[i] + start_I[0]
  ind1_i = R_I[i+1] + start_I[0]


  #ind0_j=I[j + z*(m+1)] + start[z]
  #ind1_j=I[j+1 + z*(m+1)] + start[z]

  ind0_j=R_J[j + z*(m+1)] + start_J[z]
  ind1_j=R_J[j+1 + z*(m+1)] + start_J[z]


  # number of nnz within row i , j
  nnz_j = ind1_j - ind0_j
  nnz_i = ind1_i - ind0_i

  # check the violation of max nnz 
  if nnz_i > max_nnz: raise Exception('nnz per row > max nnz')   
  if nnz_j > max_nnz: raise Exception('nnz per row > max nnz')   


  # remove the zero rows
  #if nnz_j==0 or nnz_i == 0 : return


  norm_ij = 0 

  # register the data for each row 
  v_i = V_I[ind0_i:ind1_i]
  v_j = V_J[ind0_j:ind1_j]
  
  # norm of row i  
  for n_i in range(nnz_i):
    norm_ij += v_i[n_i]**2
  
  # remove rows with zero distance
  
  
  
  # norm of row j
  for n_j in range(nnz_j):
    norm_ij += v_j[n_j]**2
 
  
  # assign shared memory for row j (col indices)
  sj = cuda.shared.array(shape=(2048),dtype=int32)
  
  for l in range(nnz_j):
    sj[l] = C_J[ind0_j + l]
 
  # register col indices for row i
  si = C_I[ind0_i:ind1_i]

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
  
  D_IJ[ind_ij] = c_tmp
  
  cuda.syncthreads()






  return




@cuda.jit('void(float32[:],float32[:], int32[:], int32, int32, int32[:], int32[:], int32[:], int32[:], int32)') 
def compute_knn(D_IJ, K, ID_K, k, m, start_I, stop_I, start_J, stop_J, max_nnz):

  i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x 
  j = cuda.blockIdx.y
  z = cuda.blockIdx.z

  
  # bitonic sort for distance matrix 
  
  cuda.syncthreads()


  # shared memory specified for row i for each j  
  sj = cuda.shared.array(shape=(2048),dtype=float32)
  #row_k = cuda.shared.array(shape=(2048),dtype=int32)
  id_k = cuda.shared.array(shape=(2048),dtype=int32)
  
  

  sj[i] = D_IJ[m*j + i + z*m**2] 
  
  if z== 1 : print(z , j , i , sj[i],  m*i + j + z*m**2)

  id_k[i] = i 
  cuda.syncthreads()

  
  log_size = math.ceil(math.log(m)/math.log(2))
  
  
  
  # bitonic sort

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
            id_k[i], id_k[ixj] = id_k[ixj], id_k[i]
    
        else:
          if sj[i] < sj[ixj]:
            sj[i], sj[ixj] = sj[ixj], sj[i]
            id_k[i], id_k[ixj] = id_k[ixj], id_k[i]

      cuda.syncthreads()
    

  # write the results back
  diff = int(pow(2, log_size) - m)
  
  

  if i >= diff and i <= k + diff: 
    #ind = i-diff
    K[i-diff + j*k + z*k*m] = sj[i]
    
    ID_K[i-diff + j*k + z*k*m] = id_k[i]

  cuda.syncthreads()
  if z == 1 and j == 3 and i == 0 :
    print('sorted')
    for w in range(k):
      print(w , K[w + z*k], ID_K[w+z*k])
  

  return

@cuda.jit('void(float32[:], int32[:], float32[:], int32[:], int32, int32, int32, int32)') 
def merge_knn(d_knn, d_ID_knn, d_K, d_ID_K, k, m , batchsize, max_nnz):

  i = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x 
  j = cuda.blockIdx.y


  S_dist = cuda.shared.array(shape=(2048),dtype=float32)
  S_id = cuda.shared.array(shape=(2048),dtype=float32)

  S_dist[i] = d_knn[i + j*k] if i < k else d_K[i - k + j*k*m]
  S_id[i] = d_ID_knn[i + j*k] if i < k else d_ID_K[i - k + j*k*m]

  cuda.syncthreads()
  if i == 0 and j ==0 :
    print('appended dist j ', j)
    for w in range(k*(batchsize+1)):
      print(w , S_dist[w])

  cuda.syncthreads()
  # bitonic sort 

  size = (batchsize+1)*k
  log_size = math.ceil(math.log(size)/math.log(2))

  for g_step in range(1, log_size+1, 1):
    g = pow(2, g_step)
    for l_step in range(g_step-1, -1, -1):
      l = pow(2, l_step)
      ixj = i ^ l  
      if ixj > i :
        if i & g == 0: 
          
          #cuda.compare_and_swap()
          if S_dist[i] > S_dist[ixj]:
            S_dist[i], S_dist[ixj] = S_dist[ixj], S_dist[i]
            S_id[i], S_id[ixj] = S_id[ixj], S_id[i]
    
        else:
          if S_dist[i] < S_dist[ixj]:
            S_dist[i], S_dist[ixj] = S_dist[ixj], S_dist[i]
            S_id[i], S_id[ixj] = S_id[ixj], S_id[i]

      cuda.syncthreads()

  diff = int(pow(2, log_size) - size)
  if i >= diff and i <= k + diff: 
    #ind = i-diff
      
    d_knn[i-diff + j*k] = S_dist[i]
      
    d_ID_knn[i-diff + j*k] = S_id[i]

  return

def gpu_sparse_knn(X, k):


  cuda.select_device(0) 
  # TODO   
  #m, d = X.shape

  # num of rows columns of X 
  m = 4  
  d = 4
  k = 4
  dist_max = 100
  # test 
  nnzperrow = 4
  # max nonzero per row 
  max_nnz = 1000 
  # batches to calculate in each kernel call 
  batchsize = 2 
  ntrees = 1
  
  threadsperblock_x = m
  blockpergrid = (threadsperblock_x + m - 1)// threadsperblock_x
  
  blockdim = threadsperblock_x, 1 , 1 
  griddim = blockpergrid, m, batchsize
  print(blockdim)
  print(griddim)
  I_ell = 1
  for ell in range(I_ell):

    # TODO 
    #X2 = c_build_tree(X, levels)
   
     
    X2 = gen_SpData_2D(m, d, nnzperrow, batchsize)
    leaf = [1]
    knn = dist_max*np.ones((len(leaf)*m*k), dtype = np.float32)
    ID_knn = np.zeros((len(leaf)*m*k), dtype = np.int32)
    
    
    for batch in leaf:
      #start = batch.start 
      #stop = batch.end
      X_I = gen_SpData_2D(m, d, nnzperrow, 1)
      X_J = gen_SpData_2D(m, d, nnzperrow, batchsize)
      D_IJ = np.zeros((m**2*batchsize), dtype = np.float32)
      K = np.zeros((batchsize*k*m), dtype = np.float32)
      ID_K = np.zeros((batchsize*k*m), dtype = np.int32)
      
      A, _ = rec(X_I['rowptr'], X_I['colind'], X_I['data'], m, d)
      
      A1, _ = rec(X_J['rowptr'][m+1:], X_J['colind'][X_J['start'][1]:], X_J['data'][X_J['start'][1]:], m, d)
      
      D_true = np.matmul(A, A1.transpose())
      print('true D')
      print(D_true)

      d_R_I = cuda.to_device(X_I['rowptr'])
      d_C_I = cuda.to_device(X_I['colind'])
      d_V_I = cuda.to_device(X_I['data'])
      d_R_J = cuda.to_device(X_J['rowptr'])
      d_C_J = cuda.to_device(X_J['colind'])
      d_V_J = cuda.to_device(X_J['data'])
      d_start_I = cuda.to_device(X_I['start'])
      d_stop_I = cuda.to_device(X_I['stop'])
      d_start_J = cuda.to_device(X_J['start'])
      d_stop_J = cuda.to_device(X_J['stop'])
      d_D_IJ = cuda.to_device(D_IJ)
      d_K = cuda.to_device(K)
      d_ID_K = cuda.to_device(ID_K)
      
      d_knn = cuda.to_device(knn)
      d_ID_knn = cuda.to_device(ID_knn)
      

      t0 = time.time()
      
      # kernel
      SpGeMM_3D[griddim, blockdim](d_R_I, d_C_I, d_V_I, d_R_J, d_C_J, d_V_J, d_D_IJ, m, d_start_I, d_stop_I, d_start_J, d_stop_J, max_nnz, batchsize)
      
      cuda.synchronize()
      
      t1 = time.time()
      compute_knn[griddim, blockdim](d_D_IJ, d_K ,d_ID_K, k, m, d_start_I, d_stop_I, d_start_J, d_stop_J, max_nnz)
      
      cuda.synchronize()


      threadsperblock_x = k*(batchsize+1)
      blockpergrid = threadsperblock_x # (threadsperblock_x + m - 1)// threadsperblock_x
      
      blockdim = threadsperblock_x, 1  
      griddim = blockpergrid, m

      merge_knn[griddim, blockdim](d_knn, d_ID_knn, d_K, d_ID_K, k,m, batchsize, max_nnz)

      cuda.synchronize()


      del_t0 = t1 - t0
      del_t1 = time.time() - t1
      D = d_D_IJ.copy_to_host()
      D = d_D_IJ.copy_to_host()
      print('rec D ')
      for z in range(batchsize):
        print('batch ', z)
        D_tmp = inline2mat(D, m, m, z)
        print(D_tmp)

      K = d_K.copy_to_host()
      ID_K = d_ID_K.copy_to_host()
      
      
      #print(' K ')
      #print(K)
      #print(' id K ')
      #print(ID_K)
      
      for z in range(batchsize):
        print('batch dist ', z)
        
        K_tmp = inline2mat(K, m, k , z)
        print(K_tmp)
      
      ID_K = inline2mat(ID_K, m, k, 1)
      
      knn = d_knn.copy_to_host()
      ID_knn = d_ID_knn.copy_to_host()

      
      knn = inline2mat(knn, m, k , 0)
      
      print('results is ')

      print(knn)
    
      
      


      
      msg = 'iter %d \t batch %d /%d \t kernel time (s) : %.3e \t %.3e'%(I_ell , batch, len(leaf), del_t0, del_t1)   
      
      print(msg)
      





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
        V[ind+j] = random.randrange(0, d, 1)
      ind0 = I[i + z*(m+1)] + start[z]
      ind1 = ind0 + nnz_i
      J[ind0:ind1] = np.sort(J[ind0:ind1])
    stop[z] = start[z]+nnz_z  
  
  print('Z is ' , Z)
  print('row ')
  print(I)
  print('col ')
  print(J)
  print(' data')
  print(V)
  print('start ')
  print(start)
  print('stop ')
  print(stop)

  
  X['rowptr'] = I
  X['colind'] = J 
  X['data'] = V
  X['start'] = start
  X['stop'] = stop
  return X
 

# for debug 
def inline2mat(D, m , d, z):

    OUT = np.zeros((m,d), dtype=np.float32)
    for i in range(m):
        for j in range(d):
            OUT[i,j] = D[i*d + j + m*d*z]
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
















