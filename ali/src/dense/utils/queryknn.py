
import cupy as cp




def queriesleafknn(X_ref, X_q, leaves, ppl, k, leafIds, q_point):

  
  q = X_q[q_point, :]
  norm_q = cp.linalg.norm(q)**2
  leafId = leafIds[q_point]
  X_c_ref = X_ref[leafId*ppl:(leafId+1)*ppl, :]
   
  norm_ref = cp.linalg.norm(X_c_ref, axis=1)**2
  tmp = q.dot(X_c_ref.transpose())
  print('inner')
  print(tmp[:10])
  print('norm_q')
  print(norm_q)
  print('norm_ref')
  print(norm_ref[:10])
  
  c_tmp = -2 * tmp
  c_tmp += norm_q
  c_tmp += norm_ref
  
  idx = cp.argsort(c_tmp)
  dists = c_tmp[idx][:k]
  idx = idx[:k] + leafId * ppl
  return (idx, dists)
    
  
   

