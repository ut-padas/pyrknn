
import cupy as cp




def queriesleafknn(X_ref, X_q, leaves, ppl, k, leafIds, q_point):

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
    
    
    
  
   

