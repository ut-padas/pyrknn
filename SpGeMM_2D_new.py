
import numpy as np 

from scipy import sparse 
import time
from numba import cuda, float32, int16, int32, int64, gdb_init, types, typeof
import math
import numba
import time
from scipy import sparse
import random
  








@cuda.jit('void(int32[:], int32[:], float32[:], int32[:], float32[:], int32[:], int32, int32,int32, int32, int32, float32, int32, int32, int32)') 
def SpGeMM_3D(R_I, C_I, V_I, R_J, K, ID_K, m_i, m_j, max_nnz, batchID_I, batchID_J, num_batch_I, num_batch_J, dist_max, k_nn):
  
  # elements
  i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x

  # batch number
  z = cuda.blockIdx.y    

  row_ID = i // m_j 
  col_ID = i - row_ID*m_j 
 
  # select threads within the input size 
  #if i>= m or j>= m or z>= num_batch_J*num_batch_I: return
  
  subbatch_I = z // num_batch_J
  subbatch_J = z - subbatch_I*num_batch_J


  # total number of nonzero for X[I, :] and X[J, :]
  #nnz_I = R_I[subbatch_I*m_i] - R_I[subbatch_I*(m_i+1)]
  #nnz_J = R_J[subbatch_J*m_j] - R_J[subbatch_J*(m_j+1)]

  # select indices corresponding to row i , j

  ind0_i = R_I[row_ID + subbatch_I*(m_i)]
  ind1_i = R_I[row_ID + 1 + subbatch_I*(m_i)]
  #ind0_i = R_I[i]
  #ind1_i = R_I[i+1]

  ind0_j=R_J[col_ID + subbatch_J*(m_j)]
  ind1_j=R_J[col_ID+1 + subbatch_J*(m_j)]
  #ind0_j=R_J[j]
  #ind1_j=R_J[j+1]

  
  # number of nnz within row i , j
  nnz_j = ind1_j - ind0_j
  nnz_i = ind1_i - ind0_i

  # check the violation of max nnz 
  if nnz_i > max_nnz: raise Exception('nnz per row > max nnz')   
  if nnz_j > max_nnz: raise Exception('nnz per row > max nnz')   

  

  norm_ij = 0 

  cuda.syncthreads()
   
  # norm of row i  
  for n_i in range(nnz_i):
    norm_ij += V_I[ind0_i + n_i]**2
  
  # remove rows with zero distance
  
  
    
  # norm of row j
  for n_j in range(nnz_j):
    norm_ij += V_I[ind0_j + n_j]**2
  
  
  
  #for l in range(nnz_j):
  #  sj[l] = C_J[ind0_j + l]
 
  # register col indices for row i
  si = C_I[ind0_i:ind1_i]
  sj = C_I[ind0_j:ind1_j]

  

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
    
    c = V_I[ind0_i + pos_k]*V_I[ind0_j + ind_jk] if ind_jk != -1 else 0
    
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
  
  id_k[i] = g_col



  cuda.syncthreads()
  

  log_size = math.ceil(math.log(m_j)/math.log(2))
  # bitonic sort
  for g_step in range(1, log_size+1, 1):
    g = pow(2, g_step)
    for l_step in range(g_step-1, -1, -1):
      l = pow(2, l_step)
      ixj = col_ID ^ l
      ixj_tmp = ixj + row_ID*m_j  
      if ixj > col_ID :
        if col_ID & g == 0: 
          
          #cuda.compare_and_swap()
          if sj[i] > sj[ixj_tmp]:
            sj[i], sj[ixj_tmp] = sj[ixj_tmp], sj[i]
            id_k[i], id_k[ixj_tmp] = id_k[ixj_tmp], id_k[i]
    
        else:
          if sj[i] < sj[ixj_tmp]:
            sj[i], sj[ixj_tmp] = sj[ixj_tmp], sj[i]
            id_k[i], id_k[ixj_tmp] = id_k[ixj_tmp], id_k[i]

  
      cuda.syncthreads()
  # write the results back
  diff = int(pow(2, log_size) - m_j)
  if col_ID >= diff and col_ID < k_nn + diff: 
    #col_write = g_col - diff
    #col_write = batchID_J*num_batch_J*k_nn + subbatch_J*k_nn + col_ID - diff
    col_local = subbatch_J*k_nn + col_ID - diff
    row_local = subbatch_I*m_i + row_ID
    ind_ij = row_local*num_batch_J*k_nn + col_local
    
    #ind_ij = g_row*num_batch_J*k_nn + col_write
    #print(g_row, col_write, sj[i] , id_k[i])
    K[ind_ij] = sj[i]
    ID_K[ind_ij] = id_k[i]

    



@cuda.jit('void(float32[:],float32[:], int32[:], int32, int32, int32, int32, int32, int32, int32)') 
def compute_knn(D_IJ, K, ID_K, k, m, max_nnz, batchID_I, batchID_J, num_batch_I, num_batch_J):

  i = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x 
  j = cuda.blockIdx.y
  z = cuda.blockIdx.z

  
  # bitonic sort for distance matrix 
  
  subbatch_I = z // num_batch_I 
  subbatch_J = z - num_batch_J*subbatch_I
  # shared memory specified for row i for each j  
  sj = cuda.shared.array(shape=(2048),dtype=float32)
  #row_k = cuda.shared.array(shape=(2048),dtype=int32)
  id_k = cuda.shared.array(shape=(2048),dtype=int32)
  
  g_row = subbatch_I*m + j 
  g_col = subbatch_J*m + i
  ind_ij = num_batch_I*m*g_row + g_col

  sj[i] = D_IJ[ind_ij] 
  
  id_k[i] = g_col + batchID_J*m*num_batch_J 
  cuda.syncthreads()
 
  log_size = math.ceil(math.log(m)/math.log(2))
  '''
  if z == 0 and j == 0 and i == 0:
    print('before sort')
    for w in range(m):
      print(w , sj[w])
  '''
  # bitonic sort
  cuda.syncthreads()
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
  

  if i >= diff and i < k + diff: 
    g_col = subbatch_J*k + i - diff
    ind_ij = (num_batch_I*m*batchID_I + g_row) + g_col
    K[ind_ij] = sj[i]
    ID_K[ind_ij] = id_k[i]

  cuda.syncthreads()
  '''
  if z == 0 and i == 0 and j == 0 :
    print('after sort')
    for w in range(m):
      print(w , sj[w])
  '''

  return

@cuda.jit('void(float32[:], int32[:], float32[:], int32[:], int32, int32, int32, int32, int32, int32, int32, int32, int32, int32)') 
def merge_knn(d_knn, d_ID_knn, d_K, d_ID_K, k, m_i, m_j , num_batch_I, num_batch_J, batchID_I, batchID_J, max_nnz, num_I, num_J):

  i = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x 
  z = cuda.blockIdx.y


  S_dist = cuda.shared.array(shape=(2048),dtype=float32)
  S_id = cuda.shared.array(shape=(2048),dtype=int32)
  
  subbatch_I = i // (num_batch_J*k)
  subbatch_J = (i - subbatch_I*num_batch_J*k) // k
  
  #col_ID_local = i - subbatch_J*k - subbatch_I*num_batch_J*k

  #g_row = batchID_I*num_batch_I*m_i + subbatch_I*m_i + z
  #g_col = batchID_J*num_batch_J*k + subbatch_J*k + col_ID_local
  
  #col_write = batchID_J*num_batch_J*k + subbatch_J*k_nn + col_ID - diff
  #ind_ij = g_row*num_batch_J**num_J*k + col_ID_local - k
  ind_ij = z*k*num_batch_J + i - k
  #S_dist[i] = d_knn[i + j*k] if i < k else d_K[i - k + j * k * num_batch_J]
  #S_id[i] = d_ID_knn[i + j*k] if i < k else d_ID_K[i - k + j * k * num_batch_J]

  S_dist[i] = d_knn[i + z*k] if i < k else d_K[ind_ij]
  S_id[i] = d_ID_knn[i + z*k] if i < k else d_ID_K[ind_ij]

  #if z == 0:
  #    print(i , S_dist[i])
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
  if i >= diff and i < k + diff: 
     
    #d_knn[i-diff + z*k] = S_dist[i]
    d_knn[z*k + i-diff] = S_dist[i]
    #if z == 0:
    #  print(i , S_dist[i])
    #if z == 0 :
    #  print(z , i , k , S_dist[i], diff)
    d_ID_knn[i-diff + z*k] = S_id[i]
    #d_ID_knn[i-diff + z*k] = S_id[i]
    #print(S_id[i], S_dist[i])
  '''
  if j == 3 and i == 0 :
    print('end')
    for w in range(k):
      print(w , d_knn[w + j*k])
  '''
def gpu_sparse_knn(X, k):


  cuda.select_device(0) 
  # TODO   
  #m, d = X.shape

  # num of points of X 
  M_I = 1028
  d = 100
  nnzperrow = 200
  num_batch_I = 1028 
  num_batch_J = 8
  m_i = 1
  m_j = 128
  k = 32
  dist_max = 10000
  
  if m_i*m_j > 2048 : print(' Error for batch_size , does not fit in shared memory')

  #if (k>m) : print('k should be less than m')

  X = gen_SpData_2D(M_I, d, nnzperrow, 1)
  
  R = X['rowptr']
  C = X['colind']
  V = X['data']
  '''
  print(R) 
  print(C) 
  print(V) 
  '''
  # partiioning in distance
  
  
  num_I = M_I//(m_i*num_batch_I)
  num_J = M_I//(m_j*num_batch_J)
  del_t0 = 0
  del_t1 = 0
  del_t2 = 0

  # number of nearest neighbor
  # test 
  # max nonzero per row 
  max_nnz = 30 
  # batches to calculate in each kernel call 
  
  threadsperblock_x = m_i*m_j
  blockpergrid = (threadsperblock_x + m_i - 1)// threadsperblock_x
  blockdim = threadsperblock_x, 1 
  griddim = blockpergrid, num_batch_I*num_batch_J

  threadsperblock_x_merge = k*(num_batch_J+1)
  blockpergrid_merge = (threadsperblock_x_merge + k - 1)// threadsperblock_x_merge 
  blockdim_merge = threadsperblock_x_merge, 1  
  griddim_merge = blockpergrid_merge, m_i*num_batch_I

  I_ell = 1000
  for ell in range(I_ell):

    
    #X2 = gen_SpData_2D(m, d, nnzperrow, batchsize)
    knn = dist_max * np.ones((M_I*k), dtype = np.float32)
    ID_knn = np.zeros((M_I*k), dtype = np.int32)
        
    for batch_I in range(num_I):
      knn_local = knn[batch_I * num_batch_I * m_i * k : (batch_I+1) * num_batch_I * m_i * k]
      ID_knn_local = ID_knn[batch_I * num_batch_I * m_i * k : (batch_I+1) * num_batch_I * m_i * k]
      for batch_J in range(num_J):
      
        #start = batch.start 
        #stop = batch.end
        print(batch_I, batch_J)
        if batch_I == num_I - 1:
          R_I = R[batch_I*num_batch_I*m_i:] #- R[batch_I*num_batch_I*m_i]
          #C_I = C[R_I[0]:R_I[-1]]
          C_I = C
          #V_I = V[R_I[0]:R_I[-1]]
          V_I = V
        else:
          R_I = R[batch_I*num_batch_I*m_i:(batch_I+1)*num_batch_I*m_i+1] #- R[batch_I*num_batch_I*m_i]
          #C_I = C[R_I[0]:R_I[-1]]
          #V_I = V[R_I[0]:R_I[-1]]
          C_I = C
          V_I = V
        if batch_J == num_J - 1:
          R_J = R[batch_J*num_batch_J*m_j:] #- R[batch_J*num_batch_J*m_j]
          #C_J = C[R_J[0]:R_J[-1]]
          #V_J = V[R_J[0]:R_J[-1]]
          C_J = C
          V_J = V
        else:
          R_J = R[batch_J*num_batch_J*m_j:(batch_J+1)*num_batch_J*m_j+1] #- R[batch_J*num_batch_J*m_j]
          C_J = C
          V_J = V
          #C_J = C[R_J[0]:R_J[-1]]
        
          #V_J = V[R_J[0]:R_J[-1]]
        '''
        print(R_I)
        print(C_I)
        print(V_I)
        print(R_J)
        print(C_J)
        print(V_J)
        '''
        #D_IJ = np.zeros((m**2*num_batch_I*num_batch_J), dtype = np.float32)

        #K = np.zeros((num_batch_I*num_batch_J*k*), dtype = np.float32)
        '''
        #ID_K = np.zeros((num_batch_I*num_batch_J*k*m), dtype = np.int32)
        A_I, _ = rec(R_I, C_I, V_I, m_i*num_batch_I, d)
        A_J, _ = rec(R_J, C_J, V_J, m_j*num_batch_J, d)
           
        #print('A_I for i = ', batch_I , ', j = ',batch_J)
        #print(A_I)  
        #print('A_J for i = ', batch_I , ', j = ',batch_J)
        #print(A_J)
        D_true = np.matmul(A_I, A_J.transpose())
        K_true = np.zeros((D_true.shape[0], k))
        for tmp in range(M_I):
            D_true[tmp, tmp] = dist_max
            min_val = 0
            row = D_true[tmp, :]
            for w in range(k):
                min_val = min(row[row>min_val])
                K_true[tmp, w] = min_val
        print(D_true)
        print('k_true')
        print(K_true)      
        #print('true D for i = ', batch_I , ', j = ', batch_J)
        #print(D_true)
        '''
        K = np.zeros((num_batch_I*num_batch_J*k*m_i), dtype = np.float32)
        ID_K = np.zeros((num_batch_I*num_batch_J*k*m_i), dtype = np.int32)
        d_R_I = cuda.to_device(R_I)
        d_C_I = cuda.to_device(C_I)
        d_V_I = cuda.to_device(V_I)
        d_R_J = cuda.to_device(R_J)
        d_C_J = cuda.to_device(C_J)
        d_V_J = cuda.to_device(V_J)
        d_K = cuda.to_device(K)
        d_ID_K = cuda.to_device(ID_K)
      
        d_knn_local = cuda.to_device(knn_local)
        d_ID_knn_local = cuda.to_device(ID_knn_local)
      
        
        t0 = time.time()
        # kernel
        SpGeMM_3D[griddim, blockdim](d_R_I, d_C_I, d_V_I, d_R_J, d_K, d_ID_K, m_i, m_j, max_nnz, batch_I, batch_J, num_batch_I, num_batch_J, dist_max, k)
        cuda.synchronize()
        
        t1 = time.time()
        del_t0 += t1 - t0
        cuda.profile_start()
        merge_knn[griddim_merge, blockdim_merge](d_knn_local, d_ID_knn_local, d_K, d_ID_K, k, m_i, m_j , num_batch_I, num_batch_J, batch_I, batch_J, max_nnz, num_I, num_J)
        cuda.profile_stop()
        cuda.synchronize()
        t2 = time.time()
        del_t1 += t2 - t1
        K = d_K.copy_to_host()
        ID_K = d_ID_K.copy_to_host()
        cuda.synchronize()
        del_t2 += t2 - t0
        ''' 
        print('inline  K ')
        #print(K)
        print(' id K ')
        #print(ID_K)
        print(' k nearest ') 
        K_tmp = inline2mat(K, m_i*num_batch_I*num_I, k , 0)
        ID_K_tmp = inline2mat(ID_K, m_i*num_batch_I*num_I, k , 0)
        print(K_tmp)
        print(ID_K_tmp)
        knn_local = d_knn_local.copy_to_host()
        ID_knn_local = d_ID_knn_local.copy_to_host()
        knn_local_tmp = inline2mat(knn_local, m_i*num_batch_I, k , 0) 
        print('results is ')
        print(knn_local_tmp)
        print(knn_local)
        print(knn_local_tmp)
        compute_knn[griddim, blockdim](d_D_IJ, d_K ,d_ID_K, min(k,m), m, max_nnz,batch_I, batch_J,  num_batch_J, num_batch_J)
      
        cuda.synchronize()
         
        t2 = time.time()

        merge_knn[griddim_merge, blockdim_merge](d_knn_local, d_ID_knn_local, d_K, d_ID_K, k, m , num_batch_I, num_batch_J, batch_I, batch_J, max_nnz)

        cuda.synchronize()
        t3 = time.time()

        del_t0 += t1 - t0
        del_t1 += t2 - t1
        del_t2 += t3 - t2
        #D = d_D_IJ.copy_to_host()
        #print(D)
        #print('rec D ')
        
        #D_tmp = inline2mat(D, m*num_batch_I, m*num_batch_J, 0)
        #print(D_tmp)
        #K = d_K.copy_to_host()
        #ID_K = d_ID_K.copy_to_host()
      
        print('inline  K ')
        print(K)
        #print(' id K ')
        #print(ID_K)
        print(' k nearest ') 
        #K_tmp = inline2mat(K, m*num_batch_I, k*num_batch_J , 0)
        #print(K_tmp)
      
        #ID_K_tmp = inline2mat(ID_K, m*num_batch_I, k*num_batch_J,0)
        #print('id s ')
        #print(ID_K_tmp)
      
        #ID_knn_local = d_ID_knn_local.copy_to_host()
        #knn_local_tmp = inline2mat(knn_local, m*num_batch_I, k , 0)
      
        #print('results is ')

        #print(knn_local)
  
  
  
     '''

      knn_local = d_knn_local.copy_to_host()
      #print(knn_local)
      #print(K_true.flatten() - knn_local) 
      ID_knn_local = d_ID_knn_local.copy_to_host()
      knn[batch_I * num_batch_I * m_i * k:(batch_I+1) * num_batch_I * k * m_i] = knn_local
      ID_knn[batch_I * num_batch_I * m_i * k : (batch_I+1) * num_batch_I * k * m_i ] = ID_knn_local
      
    
      
      

      
    msg = 'leaves : %d \n seq_itr : %d, \n batch size : %d, \n parts : %d \n Dist (s) : %.3e \n merge : %.3e \n total : %.3e'%(ell , num_I*num_J, m_i*m_j , num_batch_I*num_batch_J, del_t0, del_t1, del_t2)   
      
    print(msg)

    #knn_mat = inline2mat(knn, M_I, k, 0)
    #ID_knn_mat = inline2mat(ID_knn, M_I, k, 0)
    #print('knn tot')
    #print(knn_mat)
    #print(ID_knn_mat)

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
  start = np.zeros((Z+1), dtype = np.int32)
  stop = np.zeros((Z+1), dtype = np.int32)
  
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
        #V[ind+j] = random.uniform(0, 1)
        V[ind+j] = random.randrange(1, 20)
      ind0 = I[i + z*(m+1)] + start[z]
      ind1 = ind0 + nnz_i
      J[ind0:ind1] = np.sort(J[ind0:ind1])
    stop[z] = start[z]+nnz_z  
  
  start[Z] = stop[Z-1]

  
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
    #cuda.profile_stop()
    
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















