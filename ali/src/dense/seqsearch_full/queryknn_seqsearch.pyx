from libcpp.vector cimport vector
from libcpp.string cimport string
import cython
import cupy as cp
import time


cdef extern from "queryknn_seqsearch.h" nogil:
  void query_leafknn_seqsearch(float *X_ref, float *X_q, int *QId, int ppl, int leaves, int k, float *d_knn, int *d_knn_Id, int deviceId, int verbose, int nq, int dim, int *glob_leafIds, int num_search_leaves, int *local_leafIds);

cdef wrapper_dense_queryknn_seqsearch(size_t X_ref, size_t X_q, size_t QId, int ppl, int leaves, int k, size_t NDistPtr, size_t NIdPtr, int deviceId, int verbose, int nq, int dim, size_t glob_leafIds, int num_search_leaves, size_t local_leafIds):
  query_leafknn_seqsearch(<float*> X_ref, <float*> X_q, <int*> QId, ppl, leaves, k, <float*> NDistPtr, <int*> NIdPtr, deviceId, verbose, nq,  dim, <int*> glob_leafIds, num_search_leaves, <int*> local_leafIds);




@cython.boundscheck(False)
@cython.wraparound(False)
def py_queryknn_seqsearch(X_ref, X_q, leaves, ppl, k, knndis, knnidx, deviceId, verbose, leafIds):
 
  

  qId = cp.asarray(cp.argsort(leafIds), dtype=cp.int32)
  leafIds_c = leafIds[qId]
  local_leafIds_c = cp.arange(leafIds_c.shape[0], dtype = cp.int32)
  tmp = cp.arange(0, ppl, dtype=cp.int32)
  indices = cp.tile(tmp, (leafIds_c.shape[0], 1))
  indices += cp.tile(cp.expand_dims(leafIds_c, 1), (1, indices.shape[1]))*ppl
  indices = indices.ravel() 
  X_ref_cop = X_ref[indices, :]
  del indices
   
  X_q_c = X_q[qId, :]

  ptr_X_ref = X_ref_cop.ravel().data.ptr

 
  num_search_leaves = leafIds_c.shape[0]
  num_search_leaves = cp.int32(num_search_leaves)
  ptr_X_q = X_q_c.ravel().data.ptr
   
  hId = qId.data.ptr
  glob_leafIds = leafIds_c.data.ptr
  local_leafIds = local_leafIds_c.data.ptr
  
  nDist = knndis.ravel().data.ptr
  nId = knnidx.ravel().data.ptr
  _, d = X_q_c.shape
  d = cp.int32(d) 
  nq = X_q_c.shape[0]
  nq = cp.int32(nq)
  
  wrapper_dense_queryknn_seqsearch(ptr_X_ref, ptr_X_q, hId, ppl, leaves, k, nDist, nId, deviceId, verbose, nq, d,  glob_leafIds, num_search_leaves, local_leafIds)

  return knnidx, knndis, qId[0]



 
