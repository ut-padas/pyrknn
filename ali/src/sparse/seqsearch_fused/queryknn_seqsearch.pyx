from libcpp.vector cimport vector
from libcpp.string cimport string
import cython
import cupy as cp
import time


cdef extern from "queryknn_seqsearch.h" nogil:
  void query_leafknn_seqsearch(int *R_ref, int *C_ref, float *V_ref, int *R_q,  int *C_q, float *V_q, int *QId, int ppl, int leaves, int k, float *d_knn, int *d_knn_Id, int deviceId, int verbose, int nq, int dim, int avgnnz, int *glob_leafIds, int num_search_leaves, int *local_leafIds);

cdef wrapper_sparse_queryknn_seqsearch(size_t R_ref, size_t C_ref, size_t V_ref, size_t R_q, size_t C_q, size_t V_q, size_t QId, int ppl, int leaves, int k, size_t NDistPtr, size_t NIdPtr, int deviceId, int verbose, int nq, int dim, int avgnnz, size_t glob_leafIds, int num_search_leaves, size_t local_leafIds):
  query_leafknn_seqsearch(<int*> R_ref, <int*> C_ref, <float*> V_ref, <int*> R_q, <int*> C_q, <float*> V_q, <int*> QId, ppl, leaves, k, <float*> NDistPtr, <int*> NIdPtr, deviceId, verbose, nq,  dim, avgnnz, <int*> glob_leafIds, num_search_leaves, <int*> local_leafIds);




@cython.boundscheck(False)
@cython.wraparound(False)
def py_queryknn_seqsearch_fused(X_ref, X_q, leaves, ppl, k, knndis, knnidx, deviceId, verbose, leafIds):
 
  

  qId = cp.asarray(cp.argsort(leafIds), dtype=cp.int32)
  unique, inverse = cp.unique(leafIds[qId], return_inverse=True)
  tmp = cp.arange(0, ppl, dtype=cp.int32)
  indices = cp.tile(tmp, (unique.shape[0], 1))
  indices += cp.tile(cp.expand_dims(unique, 1), (1, indices.shape[1]))*ppl
  indices = indices.ravel()
  X_ref_cop = X_ref[indices, :]
  
    
  X_q_c = X_q[qId, :]

  v_ref = X_ref_cop.data.data.ptr
  idx_ref = X_ref_cop.indices.data.ptr
  rowptr_ref = X_ref_cop.indptr.data.ptr

 
  unique = cp.asarray(unique, dtype=cp.int32)
  inverse = cp.asarray(inverse, dtype=cp.int32)

  num_search_leaves = unique.shape[0]

  v_q = X_q_c.data.data.ptr
  idx_q = X_q_c.indices.data.ptr
  rowptr_q = X_q_c.indptr.data.ptr

  hId = qId.data.ptr
  glob_leafIds = unique.data.ptr
  local_leafIds = inverse.data.ptr
  
  nDist = knndis.ravel().data.ptr
  nId = knnidx.ravel().data.ptr
  _, d = X_q_c.shape; 
  avgnnz = cp.mean(cp.diff(X_q_c.indptr), dtype = cp.int32)
  nq = X_q_c.shape[0]
  nq = cp.int32(nq)
  
  wrapper_sparse_queryknn_seqsearch(rowptr_ref, idx_ref, v_ref, rowptr_q, idx_q, v_q, hId, ppl, leaves, k, nDist, nId, deviceId, verbose, nq, d, avgnnz, glob_leafIds, num_search_leaves, local_leafIds)
  return knnidx, knndis, qId[0]



 
