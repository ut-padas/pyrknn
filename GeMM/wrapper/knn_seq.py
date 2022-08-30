
import numpy as np
import cupy as cp

def f_knnSeq(R, C, V, GId, k, leaf_id, pt_id, leaf_size, dim):

   vec_i = np.zeros(dim);

   ind = GId[leaf_id * leaf_size + pt_id]
   ind = ind.get()
   ind0_i = R[ind];
   ind1_i = R[ind+1];
   
   vec_i[C[ind0_i:ind1_i]] = V[ind0_i:ind1_i]
   
   dists = np.zeros(leaf_size)
   Ids = np.zeros(leaf_size)


   for j in range(leaf_size):

     vec_j = np.zeros(dim)
     tmp = GId[leaf_id * leaf_size + j]
     tmp = tmp.get()
     ind0_j = R[tmp]
     ind1_j = R[tmp + 1]

     vec_j[C[ind0_j:ind1_j]] = V[ind0_j:ind1_j]
     tmp = np.linalg.norm(vec_i)**2 + np.linalg.norm(vec_j)**2 - 2 * np.inner(vec_i, vec_j)
     tmp = np.sqrt(tmp)
     dists[j] = tmp
     Ids[j] = j
     
   I = [x for _,x in sorted(zip(dists, Ids))]
   D = [y for y,_ in sorted(zip(dists, Ids))]

   return D, I
   
