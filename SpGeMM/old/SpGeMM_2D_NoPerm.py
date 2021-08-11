
import numpy as np 

from scipy import sparse 
import time
from numba import cuda, float32, int16, int32, int64, gdb_init, types, typeof
import math
import numba
import time
from scipy import sparse
import random
import cupy as cp








@cuda.jit('void(int32[:], int32[:], float32[:],  float32[:], int32[:], int32, int32, int32, int32,  int32,  int32, int32, float32, int32, int32, int32, int32)') 
def SpGeMM(R, C, V, K, K_id, m_i, m_j, max_nnz, batchId_I, batchId_J, numbatch_I, numbatch_J, dist_max, k_nn, num_I, num_J, M_I):
  
  print('run1')
  # elements
  colId = cuda.threadIdx.x 
  rowId = cuda.blockIdx.x
  
  blockId_I = cuda.threadIdx.y
  blockId_J = cuda.blockIdx.y

  leafId_local = cuda.threadIdx.z 
  blockId_leaf = cuda.blockIdx.z
  
  leafId = cuda.threadIdx.z + cuda.blockIdx.z*cuda.blockDim.z
  
  g_rowInd_I = leafId*M_I + batchId_I*numbatch_I*m_i + blockId_I*m_i + rowId 
  g_rowInd_J = leafId*M_I + batchId_J*numbatch_J*m_i + blockId_J*m_j + colId 

  
  ind0_i = R[g_rowInd_I]									
  ind1_i = R[g_rowInd_I + 1]					
  
  ind0_j = R[g_rowInd_J]									
  ind1_j = R[g_rowInd_J + 1]					
  
  print('run1')
  
  # number of nnz within row i , j
  nnz_i = ind1_i - ind0_i
  nnz_j = ind1_j - ind0_j

  # check the violation of max nnz 
  if nnz_i > max_nnz: raise Exception('nnz per row > max nnz')   
  if nnz_j > max_nnz: raise Exception('nnz per row > max nnz')   

  norm_ij = 0 

  cuda.syncthreads()
   
  # put the col and val in the register
  si = C[ind0_i:ind1_i]
  sj = C[ind0_j:ind1_j]

  vi = V[ind0_i:ind1_i]
  vj = V[ind0_j:ind1_j]
  
  # norm of row i  
  for n_i in range(nnz_i):
    norm_ij += vi[n_i]**2 
    
  # norm of row j
  for n_j in range(nnz_j):
    norm_ij += vj[n_j]**2
  
  # search for the same column index among row i,j
  c_tmp = 0
  log_n = math.floor(math.log(max_nnz)/math.log(2)) 
  #log_n = math.floor(math.log(nnz_j)/math.log(2))+1

  print('run1')
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
    
    c = vi[pos_k]*vj[ind_jk] if ind_jk != -1 else 0
    
    c_tmp += c

  c_tmp = max(-2*c_tmp + norm_ij, 0)**0.5


  #bitonic sort among rows

  l_K = cuda.shared.array(shape=(2048),dtype=float32)
  l_K_id = cuda.shared.array(shape=(2048),dtype=int32)

  l_K[colId] = c_tmp
  
  l_K_id[colId] = g_rowInd_J

  cuda.syncthreads()
  

  print('run1')
  log_size = math.ceil(math.log(m_j)/math.log(2))

  # bitonic sort
  for g_step in range(1, log_size+1, 1):
    g = pow(2, g_step)
    for l_step in range(g_step-1, -1, -1):
      l = pow(2, l_step)
      ixj = colId ^ l
      #ixj_tmp = ixj + row_ID*m_j  
      if ixj > colId :
        if colId & g == 0: 
          
          #cuda.compare_and_swap()
          if l_K[colId] > l_K[ixj]:
            l_K[colId], l_K[ixj] = l_K[ixj], l_K[colId]
            l_K_id[colId], l_K_id[ixj] = l_K_id[ixj], l_K_id[colId]
    
        else:
          if l_K[colId] < l_K[ixj]:
            l_K[colId], l_K[ixj] = l_K[ixj], l_K[colId]
            l_K_id[colId], l_K_id[ixj] = l_K_id[ixj], l_K_id[colId]

  
      cuda.syncthreads()


  # write the results back
  if colId < k_nn:
    col_write = batchId_J * numbatch_J * k_nn + blockId_J * k_nn + colId
    row_write = g_rowInd_I
    
    ind_write = leafId*M_I*k_nn + row_write * numbatch_I * k_nn + col_write 

    K[ind_write] = l_K[colId]
    K_id[ind_write] = l_K_id[colId]

    
@cuda.jit('void(float32[:], int32[:], float32[:], int32[:], int32, int32, int32, int32, int32, int32, int32, int32, int32, int32, int32)') 
def merge_knn(d_knn, d_ID_knn, d_K, d_ID_K, k, m_i, m_j , num_batch_I, num_batch_J, batchID_I, batchID_J, max_nnz, num_I, num_J, M_I):

  i = cuda.threadIdx.x 
  z = cuda.blockIdx.x
  #leaf_ID_local = cuda.blockIdx.y

  S_dist = cuda.shared.array(shape=(2048),dtype=float32)
  S_id = cuda.shared.array(shape=(2048),dtype=int32)
  
  #subbatch_I = i // (num_batch_J*k)
  #subbatch_J = (i - subbatch_I*num_batch_J*k) // k
  
  leaf_ID = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y

  col_local = batchID_J*num_batch_J*k +  i - k
  row_local = batchID_I*num_batch_I*m_i + z
  ind_ij = row_local*num_batch_J*k + col_local + leaf_ID*num_batch_I*num_batch_J*k*m_i

  #ind_ij = z*k*num_batch_J + i - k + leaf_ID*num_batch_I*num_batch_J*k*m_i
  size = (num_batch_J+1)*k
  log_size = math.ceil(math.log(size)/math.log(2))
  pad_size = int(pow(2, log_size))

  S_dist[i] = d_knn[i + z*k + batchID_I*num_batch_I*m_i*k  + leaf_ID*M_I*k] if i < k else d_K[ind_ij] if i < size else 1e30
  S_id[i] = d_ID_knn[i + z*k + batchID_I*num_batch_I*m_i*k + leaf_ID*M_I*k] if i < k else d_ID_K[ind_ij] if i < size else 0

  cuda.syncthreads()
  
  # bitonic sort 
  cuda.syncthreads()
  size = (num_batch_J+1)*k
  log_size = math.ceil(math.log(size)/math.log(2))
  diff = int(pow(2, log_size) - size)
  '''
  if diff > 0 :
      for w in range(diff):
          S_dist[w+size] = 2e30
          S_id[w+size] = 0
  '''
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

  cuda.syncthreads()
  g_row = batchID_I*num_batch_I*m_i + z
  ind = leaf_ID*M_I*k + g_row*k + i
  #if i >= diff and i < k + diff:
  #ind = i + z*k + batchID_I*num_batch_I*m_i*k  + leaf_ID*M_I*k  
  if i < k: 
    #row = batchID_I*m_i*num_batch_I + z
    #if leaf_ID == 1 and row == 0: print(batchID_I, batchID_J, subbatch_I, subbatch_J, S_dist[i],  S_id[i])
    #d_knn[row*k + i-diff + leaf_ID*M_I*k] = S_dist[i]
    #d_ID_knn[i-diff + row*k + leaf_ID*M_I*k] = S_id[i]
    #if leaf_ID == 13: print(g_row, z , i , ind, S_dist[i], S_id[i])
    d_knn[ind] = S_dist[i]
    d_ID_knn[ind] = S_id[i]
    #if leaf_ID == 12 and batchID_I == 3 and batchID_J == 1: print(z , i, ind, g_row, S_dist[i], S_id[i], d_knn[ind], d_ID_knn[ind])


def gpu_sparse_knn(d_R, d_C, d_V, leaves, M_I, d_knn, d_knn_ID, k, num_batch_I, num_batch_J, m_i, m_j, dist_max, max_nnz):


  # d_R : row pointer with length M+1 . (Multiple leaves are concatenated with length M+1)
  # d_C : column indices for points 
  # d_V : data values 

  # leaves : number of leaves 
  
  # M_I : number of rows for points 
  
  # d_knn : array for k neighbors 
  
  # d_knn_ID : array of IDs for k neighbors 
  
  # k : number of neighbors 
  
  # num_batch_I : number of blocks in y-direction to do in parallel 
  # num_batch_J : number of blocks in x-direction to do in parallel 
  
  # m_i : block size in y-direction 
  # m_j : block size in x-direction 
  
  # dist_max : maximum distance of points 
  # max_nnz : maximum number of zeros per row ( can be ignored )

   
  if m_j > 2048 : print(' Error for batch_size , does not fit in shared memory')

   
  num_I = M_I//(m_i*num_batch_I)
  num_J = M_I//(m_j*num_batch_J)
  
  del_t0 = 0
  del_t1 = 0
  del_t2 = 0

  # batches to calculate in each kernel call 
  
  threadsperblock_x = 1024
  blockdim = threadsperblock_x, num_batch_J
  griddim = m_i, num_batch_I

  print(blockdim)
  print(griddim)

  size = (num_batch_J+1)*k
  log_size = math.ceil(math.log(size)/math.log(2))
  threadsperblock_x_merge = int(pow(2, log_size))
  blockdim_merge = threadsperblock_x_merge
  griddim_merge = m_i*num_batch_I
  
  d_K = cp.zeros((num_batch_I*num_batch_J*k*m_i), dtype = cp.float32)
  d_ID_K = cp.zeros((num_batch_I*num_batch_J*k*m_i), dtype = cp.int32)
  #print(d_knn_ID.shape)  
  for leaf_ID in range(leaves): 
    for batch_I in range(num_I):
      for batch_J in range(num_J):
		   
        t0 = time.time()
  			# kernel distand and knn
        cuda.profile_start()
        SpGeMM[griddim, blockdim](d_R, d_C, d_V, d_K, d_ID_K, m_i, m_j, max_nnz, batch_I, batch_J, num_batch_I, num_batch_J, dist_max, k, num_I, num_J, M_I)
        cuda.synchronize()
        cuda.profile_stop()
        t1 = time.time() 
        #merge_knn[griddim_merge, blockdim_merge](d_knn, d_knn_ID, d_K, d_ID_K, k, m_i, m_j , num_batch_I, num_batch_J, batch_I, batch_J, max_nnz, num_I, num_J, M_I)
        cuda.synchronize()
        t2 = time.time()
        del_t0 += t1 - t0
        del_t1 += t2 - t1
        del_t2 += t2 - t0
    
    msg = 'leaves : %d \n seq_itr : %d, \n batch size : %d, \n parts : %d \n Dist (s) : %.3e \n merge : %.3e \n total : %.3e'%(leaf_ID , num_I*num_J, m_i*m_j , num_batch_I*num_batch_J, del_t0, del_t1, del_t2)   

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

def seq_knn(R, C, V, M, d, k):
    A, _ = rec(R, C, V, M, d)

    D_true = np.matmul(A, A.transpose())
    print('Distance true')
    print(D_true)
    K_true = 1000*np.ones((M, k))

    for w in range(M):
        D_true[w,w] = 1000
        val = 0
        for l in range(k):
            row = D_true[w, :]
            val = min(row[row>val])
            K_true[w, l] = min(row[row>val])


    return K_true



# for generation of 2D sparse data for Z as batchsize, 
#m rows and d columns 
def gen_SpData_2D(m, d, nnzperrow, Z):
  
  
  I = np.zeros(((m+1)*Z), dtype = np.int32)
  J = np.array([], dtype = np.int32)
  V = np.array([], dtype = np.float32)
  #X = np.zeros((nnz*Z, 3), dtype = np.int32)
  
  X = {}
 
  nnz_i = 0
  nnz_z = 0
  print('generating random')
  for z in range(Z):
    nnz_z = 0
    I[0 + z*(m+1)] = 0
    for i in range(m):
      ind = i + z*(m+1)
      nnz_i = random.randrange(nnzperrow//2, 3*nnzperrow//2, 1)
      nnz_i = min(nnz_i, d)
      nnz_z += nnz_i
      I[ind+1] = I[ind] + nnz_i
      J = np.append(J, np.zeros((nnz_i), dtype = np.int32))
      V = np.append(V, np.zeros((nnz_i), dtype = np.float32))
      for j in range(nnz_i):
        ind = I[i + z*(m+1)]
        val = random.randrange(0, d, 1)
        while val in J[ind:ind+j]: 
          val = random.randrange(0, d, 1)
        J[ind+j] = val
        V[ind+j] = random.uniform(0, 1)
        #V[ind+j] = random.randrange(1, 20)
      ind0 = I[i + z*(m+1)]
      ind1 = ind0 + nnz_i
      J[ind0:ind1] = np.sort(J[ind0:ind1])
  

  
  X['rowptr'] = I
  X['colind'] = J 
  X['data'] = V
  return X
 

# for debug 
def inline2mat(D, m , d, z):

    OUT = np.zeros((m,d), dtype=np.float32)
    for i in range(m):
        for j in range(d):
            OUT[i,j] = D[i*d + j + m*d*z]
    return OUT

       

def main():
  
  cuda.select_device(0) 
  M = 1028 #number of rows per leaf
  d = 100
  nnzperrow = 200
  # number of k nearest neighbor, should be less than m_j
  k = 32

  Z = 1 # number of leaves

  # number of blocks to do in parallel
  num_batch_I = 16
  num_batch_J = 16

  # size of the batch
  m_i = 32
  m_j = 64

  # max distance , used for bitonic sort 
  dist_max = 1000
  max_nnz = 1000


  X = gen_SpData_2D(M, d, nnzperrow, Z)
  R = X['rowptr']
  C = X['colind']
  V = X['data']
  #k_true = seq_knn(R, C, V, M, d, k)

  
  
  knn = dist_max * np.ones((M*k), dtype = np.float32)
  knn_ID = np.zeros((M*k), dtype = np.int32)

  d_knn = cuda.to_device(knn)
  d_knn_ID = cuda.to_device(knn_ID)
  d_R = cuda.to_device(R)
  d_C = cuda.to_device(C)
  d_V = cuda.to_device(V)


  # X is concatenated CSR format of Z leaves 
  gpu_sparse_knn(d_R, d_C, d_V, 1, M , d_knn, d_knn_ID, k, num_batch_I, num_batch_J, m_i, m_j, dist_max, max_nnz) 

  
  knn = d_knn.copy_to_host() 
  knn_ID = d_knn_ID.copy_to_host() 
  #diff = knn - k_true.flatten()
  knn_mat = inline2mat(knn, M, k, 0)
  
  #print('k true')
  #print(k_true)
  #print('k rec')
  #print(knn_mat)
  #er = np.linalg.norm(diff)
  #print(er)


if __name__ == '__main__':
  main()















