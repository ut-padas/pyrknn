import cupy as cp 
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpspalg
import time


def Dnorms(data):

  d_norms = cp.linalg.norm(data, axis = 1)
  return d_norms


def Snorms(data):

  d_norms = cpspalg.norm(data, axis=1)
   
  return d_norms


def neighbors(data, k, test_pt):
  # return the nrighbors for the first leaf

  n,d = data.shape
  ppl = test_pt.shape[0]
  nLeaf = n // ppl
  sparse = cpsp.issparse(data)
  if sparse:
    d_norms = Snorms(data)
  else:
    d_norms = Dnorms(data)
  #print(type(d_norms))
  d_norms = cp.reshape(d_norms, (n,1))
  kN0 = cp.ones((ppl, k), dtype = cp.float32) + 1e30
  kId0 = -cp.ones((ppl, k), dtype = cp.int32)
  
  #d0 = data[:ppl, :] 
  d0 = data[test_pt, :] 
  one = cp.ones((1,ppl), dtype = cp.int32)
  tmp = d_norms[test_pt]**2
  tmp = tmp.reshape((test_pt.shape[0],1))
  norm0 = cp.matmul(tmp, one)
  
  for leaf in range(nLeaf):
    #print('Leaf = %d'%leaf)
  
    norm_i = d_norms[leaf*ppl:(leaf+1)*ppl]
    pts = data[leaf*ppl:(leaf+1)*ppl,:]
    if sparse:
      tmp = cp.matmul(d0.toarray(), pts.toarray().transpose()) 
      dists = norm0 + cp.matmul(norm_i**2, one).transpose() - 2 * tmp
    else:
      dists = norm0 + cp.matmul(norm_i**2, one).transpose() - 2 * cp.matmul(d0,pts.transpose())
    #dists[dists < 0.0] = 0.0
    #print(dists)
    Id = cp.arange(leaf*ppl, (leaf+1)*ppl, dtype = cp.int32)
    Id = Id.reshape((1, ppl))
    Id = cp.matmul(one.transpose(), Id)
    tmp = dists.argsort(axis=1)
    dists = cp.take_along_axis(dists, tmp, axis=1)
    #print(dists)
    Id = cp.take_along_axis(Id, tmp, axis=1)
    
    kN1 = dists[:, :k]
    kId1 = Id[:, :k]
      
    dtmp_c0 = cp.concatenate((kN0, kN1), axis = 1)
    dtmp_c1 = cp.concatenate((kId0, kId1), axis = 1)
    
    tmp = dtmp_c0.argsort(axis=1)
      
    dtmp_c0 = cp.take_along_axis(dtmp_c0, tmp, axis=1)
    dtmp_c1 = cp.take_along_axis(dtmp_c1, tmp, axis=1)

    kN0 = dtmp_c0[:, :k]
    kId0 = dtmp_c1[:, :k]
    #print(kN0[0,:5])    
    #print(kId0[0,:5])
    #if leaf%1000 == 0:
    #  print("leaf = %d"%leaf)
  kN0[kN0<0] = 0.0
  kN0 = kN0
  return kId0, kN0
    
     
