from libcpp.vector cimport vector
from libcpp.string cimport string
import cython
import cupy as cp
import time


cdef extern from "queryleafknn.h" nogil:
  void query_leafknn(float *V_ref, float *V_q, int *QId, int ppl, int leaves, int k, float *d_knn, int *d_knn_Id, int deviceId, int verbose, int nq,  int *glob_pointIds, int num_search_leaves, int *local_leafIds, int dim);

cdef wrapper_dense_queryknn(size_t V_ref, size_t V_q, size_t QId, int ppl, int leaves, int k, size_t NDistPtr, size_t NIdPtr, int deviceId, int verbose, int nq, size_t glob_pointIds, int num_search_leaves, size_t local_leafIds, int dim):
  query_leafknn(<float*> V_ref, <float*> V_q, <int*> QId, ppl, leaves, k, <float*> NDistPtr, <int*> NIdPtr, deviceId, verbose, nq,  <int*> glob_pointIds, num_search_leaves, <int*> local_leafIds, dim);



@cython.boundscheck(False)
@cython.wraparound(False)
def py_queryleafknn(data_ref, data_q, leaves, ppl, k, knndis, knnidx, deviceId, verbose, morId_q, gids_leaves, n_iter):
 
 
  assert(n_iter == 1, 'more than one iteration per kernel invocation')
   
  nq = cp.int32(data_q.shape[0])
  dim = cp.int32(data_ref.shape[1])
  
  qId = cp.asarray(cp.argsort(morId_q[0, :]), dtype=cp.int32)
  unique = morId_q[0, qId];
  inverse = cp.arange(0, nq, dtype = cp.int32)
  
  tmp = cp.arange(0, ppl, dtype=cp.int32)
  indices = cp.tile(tmp, (unique.shape[0], 1))
  indices += cp.tile(cp.expand_dims(unique, 1), (1, indices.shape[1]))*ppl
  indices = indices.ravel()
  indices = gids_leaves[0, indices]
  data_ref_cop = data_ref[indices, :]
   
  data_q_c = data_q[qId, :]

  v_ref = data_ref_cop.data.ptr
 
  unique = cp.asarray(unique, dtype=cp.int32)
  inverse = cp.asarray(inverse, dtype=cp.int32)

  num_search_leaves = unique.shape[0]

  v_q = data_q_c.data.ptr

  hId = qId.data.ptr
  glob_pointIds = indices.data.ptr
  local_leafIds = inverse.data.ptr
  
  nDist = knndis.ravel().data.ptr
  nId = knnidx.ravel().data.ptr
  
  wrapper_dense_queryknn(v_ref, v_q, hId, ppl, leaves, k, nDist, nId, deviceId, verbose, nq, glob_pointIds, num_search_leaves, local_leafIds, dim)

  return knnidx, knndis, qId[0]



 
