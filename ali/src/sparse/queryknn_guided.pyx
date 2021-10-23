import cython
import cupy as cp



cdef extern from "../../include/sparse/queryknn.h" nogil:
  void query_leafknn(int *R_ref, int *C_ref, float *V_ref, int *R_q,  int *C_q, float *V_q, int *GId, int ppl, int leaves, int k, float *d_knn, int *d_knn_Id, int deviceId, int verbose, int nq, int *leafIds, int dim, int avgnnz, int num_search_leaves);

cdef wrapper_sparse_queryknn(size_t R_ref, size_t C_ref, size_t V_ref, size_t R_q, size_t C_q, size_t V_q, size_t GId, int ppl, int leaves, int k, size_t NDistPtr, size_t NIdPtr, int deviceId, int verbose, int nq, size_t leafIds, int dim, int avgnnz, int num_search_leaves):
  query_leafknn(<int*> R_ref, <int*> C_ref, <float*> V_ref, <int*> R_q, <int*> C_q, <float*> V_q, <int*> GId, ppl, leaves, k, <float*> NDistPtr, <int*> NIdPtr, deviceId, verbose, nq, <int*> leafIds, dim, avgnnz, num_search_leaves);




@cython.boundscheck(False)
@cython.wraparound(False)
def py_queryknn(X_ref, X_q, GId, leaves, ppl, k, knndis, knnidx, deviceId, verbose, leafIds, num_search_leaves):
 
  

  qId = cp.argsort(leafIds)
  unique, inverse = cp.unique(leafIds[qId], return_inverse=True)
  indices = cp.linspace(unique, unique + ppl, ppl, endpoint=False, dtype=cp.int32).flatten('F')
  X_ref_cop = X_ref[indices, :]
  
  X_q = X_q[qId, :]

  v_ref = X_ref_cop.data.data.ptr
  idx_ref = X_ref_cop.indices.data.ptr
  rowptr_ref = X_ref_cop.indptr.data.ptr
  
  v_q = X_q.data.data.ptr
  idx_q = X_q.indices.data.ptr
  rowptr_q = X_q.indptr.data.ptr
  
  hId = qId.data.ptr
  nDist = knndis.ravel().data.ptr
  nId = knnidx.ravel().data.ptr
  _, d = X_q.shape; 
  avgnnz = cp.mean(cp.diff(X_q.indptr))
  nq = X_q.shape[0]
  
  wrapper_sparse_queryknn(rowptr_ref, idx_ref, v_ref, rowptr_q, idx_q, v_q, hId, ppl, leaves, k, nDist, nId, deviceId, verbose, nq, leafIds, d, avgnnz, num_search_leaves)

  return (knnidx, knndis)



 
