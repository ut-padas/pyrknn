#include "norm.h"


__global__ void ComputeNorms(float* Xq, float* Norms, int dim){

  
  int pt = gridDim.x * blockIdx.y + blockIdx.x;
  
  
  int ind0_i = pt * dim;
  
  float norm_i = 0.0;
  float v;
  
  for (int n_i = 0; n_i < dim; n_i += 1){
    v = Xq[ind0_i + n_i];
    norm_i += v * v;
  } 
  Norms[pt] = norm_i;

}

__global__ void ComputeNorms_ref(float* Xref, float* Norms, int* local_leafIds, int dim){

  
  int pt = gridDim.x * blockIdx.y + blockIdx.x;
  int leafId = blockIdx.y;  

  int ind0_i = pt * dim;
  
  float norm_i = 0.0;
  float v;
  
  for (int n_i = 0; n_i < dim; n_i += 1){
    v = Xref[ind0_i + n_i];
    norm_i += v * v;
  } 
  Norms[pt] = norm_i;
}











  
  















