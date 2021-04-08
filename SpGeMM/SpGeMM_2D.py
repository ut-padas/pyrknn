
import numpy as np 

from scipy import sparse 
import time
from numba import cuda, float32, int16, int32, int64, gdb_init, types, typeof
import math
import numba
import time
from scipy import sparse
import random
  








@cuda.jit('void(int32[:], int32[:], float32[:], int32[:], int32, float32[:], int32[:], int32, int32,  int32, int32, int32, int32,  int32, int32, float32, int32, int32, int32)') 
def SpGeMM(R, C, V, GID, leaf_ID, K, ID_K, m_i, m_j, M , max_nnz, batchID_I, batchID_J, num_batch_I, num_batch_J, dist_max, k_nn, num_I, num_J):
  
  # elements
  i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x

  # batch number
  z = cuda.blockIdx.y    

  col_ID = i

  z_tmp = z // (m_i)
  row_ID = z  - (z_tmp*m_i)
  
  subbatch_I = z_tmp // num_batch_J
  subbatch_J = z_tmp - subbatch_I*num_batch_J

  row_ind_I = batchID_I*num_batch_I*m_i + subbatch_I*m_i + row_ID
  #row_ind_I = subbatch_I*m_i + row_ID
  row_ind_J = batchID_J*num_batch_J*m_j + subbatch_J*m_j + col_ID
  #row_ind_J =  subbatch_J*m_j + col_ID
  #print(i , z, row_ID, col_ID, z_tmp, subbatch_I, subbatch_J)
  # select indices corresponding to row i , j


  tmp = GID[leaf_ID * M + row_ind_I] 

  ind0_i = R[tmp] 
  ind1_i = R[tmp+1] 
  
  #ind0_i = R[row_ind_I]									
  #ind1_i = R[row_ind_I + 1]					
  
  
  #ind0_i = R_I[row_ind_I]
  #ind1_i = R_I[row_ind_I + 1]

  #ind0_j= R[row_ind_J]
  #ind1_j= R[row_ind_J + 1]

  tmp = GID[leaf_ID * M + row_ind_J]
 
  ind0_j = R[tmp] 
  ind1_j = R[tmp+1]
  
  #ind0_j= R_J[row_ind_J]
  #ind1_j= R_J[row_ind_J + 1]

  
  # number of nnz within row i , j
  nnz_i = ind1_i - ind0_i
  nnz_j = ind1_j - ind0_j

  # check the violation of max nnz 
  if nnz_i > max_nnz: raise Exception('nnz per row > max nnz')   
  if nnz_j > max_nnz: raise Exception('nnz per row > max nnz')   

  

  norm_ij = 0 

  cuda.syncthreads()
   
  # norm of row i  
  for n_i in range(nnz_i):
    norm_ij += V[ind0_i + n_i]**2
  
  # remove rows with zero distance
  
  
    
  # norm of row j
  for n_j in range(nnz_j):
    norm_ij += V[ind0_j + n_j]**2
  
  
  # register col indices for row i
  si = C[ind0_i:ind1_i]
  sj = C[ind0_j:ind1_j]


  # search for the same column index among row i,j
  c_tmp = 0
  #log_n = math.floor(math.log(max_nnz)/math.log(2)) 
  log_n = math.floor(math.log(nnz_j)/math.log(2))+1

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
    
    c = V[ind0_i + pos_k]*V[ind0_j + ind_jk] if ind_jk != -1 else 0
    
    c_tmp += c

  #c_tmp = max(-2*c_tmp + norm_ij, 0)

  if batchID_I == batchID_J and subbatch_I == subbatch_J and row_ID == col_ID:
    c_tmp = dist_max  
  # write to the global array

 
  g_col = subbatch_J*m_j + col_ID

   
  #print(z , i , row_ID, col_ID, c_tmp)


  #bitonic sort among rows

  sj = cuda.shared.array(shape=(2048),dtype=float32)
  id_k = cuda.shared.array(shape=(2048),dtype=int32)



  sj[i] = c_tmp

  id_k[i] = GID[g_col + leaf_ID*M]
  #id_k[i] = g_col

  cuda.syncthreads()
  

  log_size = math.ceil(math.log(m_j)/math.log(2))
  # bitonic sort
  for g_step in range(1, log_size+1, 1):
    g = pow(2, g_step)
    for l_step in range(g_step-1, -1, -1):
      l = pow(2, l_step)
      ixj = col_ID ^ l
      #ixj_tmp = ixj + row_ID*m_j  
      if ixj > col_ID :
        if col_ID & g == 0: 
          
          #cuda.compare_and_swap()
          if sj[i] > sj[ixj]:
            #sj[i], sj[ixj_tmp] = sj[ixj_tmp], sj[i]
            #id_k[i], id_k[ixj_tmp] = id_k[ixj_tmp], id_k[i]
            sj[i], sj[ixj] = sj[ixj], sj[i]
            id_k[i], id_k[ixj] = id_k[ixj], id_k[i]
    
        else:
          if sj[i] < sj[ixj]:
            #sj[i], sj[ixj] = sj[ixj_tmp], sj[i]
            sj[i], sj[ixj] = sj[ixj], sj[i]
            #id_k[i], id_k[ixj] = id_k[ixj], id_k[i]
            id_k[i], id_k[ixj] = id_k[ixj], id_k[i]

  
      cuda.syncthreads()
  # write the results back
  diff = int(pow(2, log_size) - m_j)
  if col_ID >= diff and col_ID < k_nn + diff: 
    col_local = subbatch_J*k_nn + col_ID - diff
    row_local = subbatch_I*m_i + row_ID
    ind_ij = row_local*num_batch_J*k_nn + col_local
    
    #if batchID_I == 1 and batchID_J == 1: print(row_local, col_local, sj[i] , id_k[i], ind_ij)
    K[ind_ij] = sj[i]
    ID_K[ind_ij] = id_k[i]

    
@cuda.jit('void(float32[:], int32[:], int32[:], int32, float32[:], int32[:], int32, int32,int32,  int32, int32, int32, int32, int32, int32, int32, int32)') 
def merge_knn(d_knn, d_ID_knn, d_GID, leaf_ID, d_K, d_ID_K, k, m_i, m_j , M, num_batch_I, num_batch_J, batchID_I, batchID_J, max_nnz, num_I, num_J):

  i = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x 
  z = cuda.blockIdx.y


  S_dist = cuda.shared.array(shape=(2048),dtype=float32)
  S_id = cuda.shared.array(shape=(2048),dtype=int32)
  
  subbatch_I = i // (num_batch_J*k)
  subbatch_J = (i - subbatch_I*num_batch_J*k) // k
  

  ind_ij = z*k*num_batch_J + i - k

  #S_dist[i] = d_knn[i + z*k + batchID_I*num_batch_I*m_i*k ] if i < k else d_K[ind_ij]
  #S_id[i] = d_ID_knn[i + z*k + batchID_I*num_batch_I*m_i*k] if i < k else d_ID_K[ind_ij]
  ind_knn = i + z*k + batchID_I*num_batch_I*m_i*k + leaf_ID*M
  S_dist[i] = d_knn[d_GID[ind_knn]] if i < k else d_K[ind_ij]
  S_id[i] = d_ID_knn[ d_GID[ ind_knn ]]  if i < k else d_ID_K[ind_ij]

  cuda.syncthreads()
  
  # bitonic sort 
  cuda.syncthreads()
  size = (num_batch_J+1)*k
  log_size = math.ceil(math.log(size)/math.log(2))
  diff = int(pow(2, log_size) - size)
  if diff > 0 :
      for w in range(diff):
          S_dist[w+size] = 0
          S_id[w+size] = 0
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

  #if batchID_I == 1 and batchID_J == 1: print(z , i , batchID_I, batchID_J, S_dist[i])
  if i >= diff and i < k + diff: 
    row = d_GID[batchID_I*m_i*num_batch_I + z]
    d_knn[row*k + i-diff] = S_dist[i]
    d_ID_knn[i-diff + row*k] = S_id[i]


def gpu_sparse_knn(d_R, d_C, d_V, d_GID, leaves, M_I, d_knn, d_knn_ID, k, num_batch_I, num_batch_J, m_i, m_j, dist_max, max_nnz):


  # FOR CUPY CSR:  d_R: indptr, d_C: indices; d_V: dataa
  #
  # Device pointers

    

  # d_R : row pointer with length M_I +1 . (Multiple leaves are concatenated with length M+1)
  # d_C : column indices for points 
  # d_V : data values 

  # leaves : number of leaves 
  
  # M_I : number of points
  
  # d_knn : K nearest neighbor distances   M_I - by - k matrix, flatten (row major)
  
  # d_knn_ID : K nearest neighbor indices   M_I - by - k matrix, flatten (row major)
  
  # k : number of neighbors  to find
  
  # num_batch_I : number of blocks in y-direction to do in parallel     
  # num_batch_J : number of blocks in x-direction to do in parallel 
  
  # m_i : per-block number of target (cow) points, for which we computer nearest neighbors)
  # m_j : per-block number of source (column) points, for which we computer nearest neighbors)
  #       also number of threads for row and colum
  
  # dist_max : maximum distance of points  (TO REMOVE)
  # max_nnz : maximum number of zeros per row ( Optional )

  # Upon return, d_knn and d_knn_ID are updated to the new values.

  # Everything must power of two


  if m_j > 2048 : print(' Error for batch_size , does not fit in shared memory')

  
  num_I = M_I//(m_i*num_batch_I)
  num_J = M_I//(m_j*num_batch_J)
  
  del_t0 = 0
  del_t1 = 0
  del_t2 = 0

  # batches to calculate in each kernel call 
  
  threadsperblock_x = m_j
  blockpergrid = (threadsperblock_x + m_i - 1)// threadsperblock_x
  blockdim = threadsperblock_x, 1 
  griddim = blockpergrid, num_batch_I*num_batch_J*m_i

  threadsperblock_x_merge = k*(num_batch_J+1)
  blockpergrid_merge = (threadsperblock_x_merge + k - 1)// threadsperblock_x_merge 
  blockdim_merge = threadsperblock_x_merge, 1  
  griddim_merge = blockpergrid_merge, m_i*num_batch_I

  K = np.zeros((num_batch_I*num_batch_J*k*m_i), dtype = np.float32)
  ID_K = np.zeros((num_batch_I*num_batch_J*k*m_i), dtype = np.int32)
  d_K = cuda.to_device(K)
  d_ID_K = cuda.to_device(ID_K)
  
  for leaf_ID in range(leaves):
    for batch_I in range(num_I):
      for batch_J in range(num_J):
        t0 = time.time()
        # kernel distand and knn
        cuda.profile_start()
        
        SpGeMM[griddim, blockdim](d_R, d_C, d_V, d_GID, leaf_ID, d_K, d_ID_K, m_i, m_j, M_I, max_nnz, batch_I, batch_J, num_batch_I, num_batch_J, dist_max, k, num_I, num_J)
        
        cuda.synchronize()
        cuda.profile_stop()
        
        t1 = time.time() 
        
        merge_knn[griddim_merge, blockdim_merge](d_knn, d_knn_ID, d_GID, leaf_ID, d_K, d_ID_K, k, m_i, m_j ,M_I,  num_batch_I, num_batch_J, batch_I, batch_J, max_nnz, num_I, num_J)
        cuda.synchronize()
        
        t2 = time.time()
        #print(batch_I, batch_J)
        #K = d_K.copy_to_host()
        #print('rec K')
        #print(K)
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
def gen_SpData_2D(m, d, nnzperrow, num_leaves):
  
  Z = 1
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
  
  GID = np.zeros((num_leaves*m))
  ref = np.arange(m, dtype = np.int32)
  GID[0:m] = ref
  for i in range(1, num_leaves):
    GID[i*m:(i+1)*m] = np.random.permutation(ref)
    
  
  X['rowptr'] = I
  X['colind'] = J 
  X['data'] = V
  X['GID'] = GID
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

  num_leaves = 10 # number of leaves

  # number of blocks to do in parallel
  num_batch_I = 16
  num_batch_J = 16

  # size of the batch
  m_i = 32
  m_j = 64

  # max distance , used for bitonic sort 
  dist_max = 1000
  max_nnz = 1000


  X = gen_SpData_2D(M, d, nnzperrow, num_leaves)
  R = X['rowptr']
  C = X['colind']
  V = X['data']
  GID = X['GID']
  
  #k_true = seq_knn(R, C, V, M, d, k)

  
  
  knn = dist_max * np.ones((M*k), dtype = np.float32)
  knn_ID = np.zeros((M*k), dtype = np.int32)

  d_knn = cuda.to_device(knn)
  d_knn_ID = cuda.to_device(knn_ID)
  d_R = cuda.to_device(R)
  d_C = cuda.to_device(C)
  d_V = cuda.to_device(V)
  d_GID = cuda.to_device(GID)


  # X is concatenated CSR format of Z leaves 
  gpu_sparse_knn(d_R, d_C, d_V, d_GID, num_leaves, M , d_knn, d_knn_ID, k, num_batch_I, num_batch_J, m_i, m_j, dist_max, max_nnz) 

  
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














