#include "norm.h"


__global__ void ComputeNorms(int * R, int* C, float* V, float* Norms){

  
  int pt = gridDim.x * blockIdx.y + blockIdx.x;
  
  
  int ind0_i = R[pt];
  int nnz = R[pt+1] - ind0_i;
  
  float norm_i = 0.0;
  float v;
  
  for (int n_i = 0; n_i < nnz; n_i += 1){
    v = V[ind0_i + n_i];
    norm_i += v * v;
  } 
  Norms[pt] = norm_i;

}

__global__ void ComputeNorms_ref(int * R, int* C, float* V, float* Norms, int* local_leafIds){

  
  int pt = gridDim.x * blockIdx.y + blockIdx.x;
  int leafId = blockIdx.y;  

  int ind0_i = R[pt];
  int nnz = R[pt+1] - ind0_i;
  
  float norm_i = 0.0;
  float v;
  
  for (int n_i = 0; n_i < nnz; n_i += 1){
    v = V[ind0_i + n_i];
    norm_i += v * v;
  } 
  Norms[pt] = norm_i;
}











  
  















