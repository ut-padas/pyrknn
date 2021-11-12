
import cupy as cp




def sparse_queriesleafknn(X_ref, X_q, leaves, ppl, k, leafIds, q_point):

  #qId = cp.asarray(cp.argsort(leafIds), dtype=cp.int32)
  #g_leafIds, inverse = cp.unique(leafIds[qId], return_inverse=True))
  #num_search_leaves = unique.shape[0]
  

  #X_q = X_q[qId, :]
  
  q = X_q[q_point, :].toarray()
  norm_q = cp.linalg.norm(q)**2
  leafId = leafIds[q_point]
  
  tmp = cp.zeros(ppl, dtype=cp.float32)
  for p in range(ppl):
    norms = norm_q + cp.linalg.norm(X_ref.data[X_ref.indptr[leafId*ppl+p]:X_ref.indptr[leafId*ppl+p+1]])**2
    norm_ref = cp.linalg.norm(X_ref.data[X_ref.indptr[leafId*ppl+p]:X_ref.indptr[leafId*ppl+p+1]])**2
    w = X_ref[leafId * ppl + p, :]
    c_tmp = w.dot(q.transpose())
    #c_tmp = q.dot(X_ref[leafId * ppl + p, :])
    tmp[p] = norms - 2 * c_tmp[0,0]
    if p < 10:
      print("p = %d, norms = %.4f, inner = %.4f"%(p, norms, c_tmp))
  idx = cp.argsort(tmp)
  dists = tmp[idx][:k]
  idx = idx[:k] + leafId * ppl
  return (idx, dists)
    
    


def sparse_queriesExact(X_ref, X_q, leaves, ppl, k, q_points):

  Q = X_q[q_points, :].toarray()
  nq = Q.shape[0]
  norm_Q = cp.tile((cp.linalg.norm(Q, axis=1)**2).reshape((nq,1)), (1, ppl))
  
  NDist = cp.zeros((nq, k+ppl), dtype = cp.float32) + 1e30
  NId = cp.zeros((nq, k+ppl), dtype = cp.int32) - 1 
 
  for l in range(leaves):
    norm_ref = cp.zeros((1,ppl))
    for p in range(ppl):
      norm_ref[0,p] = cp.linalg.norm(X_ref.data[X_ref.indptr[l*ppl+p]:X_ref.indptr[l*ppl+p+1]])**2
    
    norms = norm_Q + cp.tile(norm_ref, (nq, 1))
    w = X_ref[l*ppl:(l+1)*ppl, :]
    c_tmp = w.dot(Q.transpose())
   
    NDist[:, k:] = norms - 2 * c_tmp.transpose()
    NId[:, k:] = cp.tile(cp.arange(0,ppl)+l*ppl, (nq, 1))
   
    idx = cp.argsort(NDist, axis=1)
   
    NDist[:,:] = cp.take_along_axis(NDist, idx, axis=1) 
    NId[:,:] = cp.take_along_axis(NId, idx, axis=1) 
  
  return (NId[:, :k], NDist[:, :k])
    
    
  
def dense_queriesExact(X_ref, X_q, leaves, ppl, k, q_points):

  Q = X_q[q_points, :]
  nq = Q.shape[0]
  norm_Q = cp.tile((cp.linalg.norm(Q, axis=1)**2).reshape((nq,1)), (1, ppl))
  
  NDist = cp.zeros((nq, k+ppl), dtype = cp.float32) + 1e30
  NId = cp.zeros((nq, k+ppl), dtype = cp.int32) - 1 
 
  for l in range(leaves):
    
    norm_ref = cp.tile( (cp.linalg.norm( X_ref[l*ppl:(l+1)*ppl, :], axis=1 )**2 ).reshape((1,ppl) ), (nq, 1))
    
    #print(nq, norm_Q.shape, norm_ref.shape)
    norms = norm_Q + norm_ref
  
    w = X_ref[l*ppl:(l+1)*ppl, :]
    c_tmp = w.dot(Q.transpose())
   
    NDist[:, k:] = norms - 2 * c_tmp.transpose()
    NId[:, k:] = cp.tile(cp.arange(0,ppl)+l*ppl, (nq, 1))
   
    idx = cp.argsort(NDist, axis=1)
   
    NDist[:,:] = cp.take_along_axis(NDist, idx, axis=1) 
    NId[:,:] = cp.take_along_axis(NId, idx, axis=1) 
  
  return (NId[:, :k], NDist[:, :k])
    
    
   

