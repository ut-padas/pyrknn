#include "Norms.h"


__global__ void ComputeNorms(float* X, float* Norms, int dim){

  
  int pt = gridDim.x * blockIdx.y + blockIdx.x;
  
  
  
  float norm_i = 0.0;
  float v;
  int ind0_i = pt * dim; 
  for (int n_i = 0; n_i < dim; n_i += 1){
    v = X[ind0_i + n_i];
    norm_i += v * v;
  } 
  Norms[pt] = norm_i;

}

__global__ void ComputeNorms_ref(float* X, float* Norms, int* local_leafIds, int dim){

  
  int pt = gridDim.x * blockIdx.y + blockIdx.x;

  int ind0_i = pt * dim;
  
  float norm_i = 0.0;
  float v;
  
  for (int n_i = 0; n_i < dim; n_i += 1){
    v = X[ind0_i + n_i];
    norm_i += v * v;
  }
  Norms[pt] = norm_i;
  if (pt < 10) printf("Norm_ref = %.4f write at %d \n", Norms[pt], pt); 
}











  
  















