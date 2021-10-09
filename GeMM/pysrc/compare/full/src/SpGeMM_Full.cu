
#include "sfiknn.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

#define shared_size  8192
#define MaxProcperBlock  1024 


__global__ void ComputeNorms(int* R, int* C, float* V, int* G_Id, float* Norms, int ppl) {

  int ind = threadIdx.x;
  int leafId_g = blockIdx.z * gridDim.y + blockIdx.y;
  for (int row = ind; row < ppl; row += blockDim.x){
    int g_rowId = leafId_g * ppl + row;
    //changed

    int g_Id = g_rowId;


    int ind0_i = R[g_Id];

    int nnz = R[g_Id + 1] - ind0_i;
    float norm_i = 0.0;
    float v;
    for (int n_i = 0; n_i < nnz; n_i += 1) {
      v = V[ind0_i + n_i];
      norm_i += v * v;
    }
    Norms[g_Id] = norm_i;
  }
}


__global__ void compute_dist(int* R, int* C, float* V, int* G_Id,  float* Norms, int k_nn, float* K, int ppl, int bl, int sizebleaves, int M){

    int tid = threadIdx.x;
    int rowId = blockIdx.x;

    int leafId_local = blockIdx.z * gridDim.y + blockIdx.y;
    int leafId_g = bl * sizebleaves + leafId_local;

    int g_rowId = leafId_g * ppl + rowId;
    int ind0_i = R[g_rowId];
    int ind1_i = R[g_rowId + 1];
    int nnz_i = ind1_i - ind0_i;
    
    __shared__ int SM[2048];

    for (int n_i = tid; n_i < nnz_i; n_i += blockDim.x) SM[n_i] = C[ind0_i + n_i];
    __syncthreads();

    float norm_ij = 0;
    float norm_i = Norms[g_rowId];

    int tmp_0, tmp_1, ind_jk, k, ret, testInd;

    ret = 0;
    testInd = 0;

    for (int colId = tid; colId < ppl; colId += blockDim.x){

      
      int g_colId = leafId_g * ppl + colId;
      int ind0_j = R[g_colId];
      int ind1_j = R[g_colId+1];
      int nnz_j = ind1_j - ind0_j;
      norm_ij = norm_i + Norms[g_colId];
      float c_tmp = 0.0;
      if (nnz_i >0 && nnz_j > 0){
      for (int pos_k = 0; pos_k < nnz_j; pos_k++){

        k = C[ind0_j + pos_k];

        // Binary search
        for (int l = nnz_i - ret; l > 1; l -= floorf(l/2.0)){
          tmp_0 = ret + l;
          tmp_1 = nnz_i - 1;
          testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
          ret = (SM[testInd] <= k) ? testInd : ret;
        }

        tmp_0 = ret + 1;
        tmp_1 = nnz_i - 1;
        testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;

        //ret = (C[testInd + ind0_i] <= k) ? testInd : ret;
        ret = (SM[testInd] <= k) ? testInd : ret;

        //ind_jk = (C[ret + ind0_i] == k) ? ret : -1;
        ind_jk = (SM[ret] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;

      }
      }
      c_tmp = -2 * c_tmp + norm_ij;
      c_tmp = (c_tmp > 1e-8) ? c_tmp : 0.0;

      int ind_K = leafId_local * ppl * ppl + rowId * ppl + colId;
      K[ind_K] = c_tmp;
    }
    
}

__global__ void find_neighbor(float* KNN, int* KNN_Id, float* K, int* G_Id, int k_nn, int ppl, int M, int bl, int sizebleaves){



    __shared__ float Dist[4096];
    __shared__ int Dist_Id[4096];

    int rowId = blockIdx.x;
    int tid = threadIdx.x;

    int leafId_local = blockIdx.z * gridDim.y + blockIdx.y;
    int leafId_g = bl * sizebleaves + leafId_local;
    
    
    for (int n_i = tid; n_i < ppl; n_i += blockDim.x){
      Dist[n_i] = K[leafId_local * ppl * ppl + rowId * ppl + n_i]; 
      Dist_Id[n_i] = G_Id[leafId_g * ppl + n_i];
    }
    float tmp_f;
    int tmp_i;
    
    int size = ppl;
    for (int g = 2; g <= size; g *= 2){
      for (int l = g /2; l > 0; l /= 2){
      
        for (int j = tid; j < size; j += blockDim.x){

          int ixj = j ^ l;
        
          if (ixj > j){
            if ((j & g) == 0){
              if (Dist[j] > Dist[ixj]){

                tmp_f = Dist[ixj];
                Dist[ixj] = Dist[j];
                Dist[j] = tmp_f;

                tmp_i = Dist_Id[ixj];
                Dist_Id[ixj] = Dist_Id[j];
                Dist_Id[j] = tmp_i;

              }
            } else {
              if (Dist[j] < Dist[ixj]){

                tmp_f = Dist[ixj];
                Dist[ixj] = Dist[j];
                Dist[j] = tmp_f;

                tmp_i = Dist_Id[ixj];
                Dist_Id[ixj] = Dist_Id[j];
                Dist_Id[j] = tmp_i;

              }
            }
          }
        }
        __syncthreads();
      }
    }

    int ind_knn = G_Id[leafId_g * ppl + rowId]* k_nn + tid;
    if (tid < k_nn){
     Dist[tid+k_nn] = KNN[ind_knn];
     Dist_Id[tid+k_nn] = KNN_Id[ind_knn]; 
    }

    __syncthreads();


    size = 2 * k_nn;
    
    for (int g = 2; g <= size; g *= 2){
      for (int l = g /2; l > 0; l /= 2){
      
        for (int j = tid; j < size; j += blockDim.x){

          int ixj = j ^ l;
        
          if (ixj > j){
            if ((j & g) == 0){
              if (Dist[j] > Dist[ixj]){
                tmp_f = Dist[ixj];
                Dist[ixj] = Dist[j];
                Dist[j] = tmp_f;
                tmp_i = Dist_Id[ixj];
                Dist_Id[ixj] = Dist_Id[j];
                Dist_Id[j] = tmp_i;
              }
            } else {
              if (Dist[j] < Dist[ixj]){
                tmp_f = Dist[ixj];
                Dist[ixj] = Dist[j];
                Dist[j] = tmp_f;
                tmp_i = Dist_Id[ixj];
                Dist_Id[ixj] = Dist_Id[j];
                Dist_Id[j] = tmp_i;
              }
            }
          }
        }
        __syncthreads();
      }
    }

    if (tid < k_nn){
      int ind_g_knn = G_Id[leafId_g * ppl + rowId] * k_nn + tid;
      KNN[ind_g_knn] = Dist[tid];
      KNN_Id[ind_g_knn] = Dist_Id[tid]; 
    }
}



void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k_nn, float *KNN, int *KNN_Id){


  int ppl = M / leaves;

  size_t free, total, m1, m2, m3;

  cudaEvent_t t_begin;
  cudaEvent_t t_end;
  cudaEvent_t t0_dist;
  cudaEvent_t t1_dist;
  cudaEvent_t t0_merge;
  cudaEvent_t t1_merge;

  float dt_tot, dt_tmp;
  float dt_dist;
  float dt_merge;
  checkCudaErrors(cudaEventCreate(&t_begin));
  checkCudaErrors(cudaEventCreate(&t_end));
  checkCudaErrors(cudaEventCreate(&t0_dist));
  checkCudaErrors(cudaEventCreate(&t1_dist));
  checkCudaErrors(cudaEventCreate(&t0_merge));
  checkCudaErrors(cudaEventCreate(&t1_merge));

  checkCudaErrors(cudaEventRecord(t_begin, 0));



  int t_b = (ppl > 1024) ? 1024 : ppl;
  dim3 BlockNorms(t_b, 1, 1);
  dim3 GridNorms(1, leaves, 1);


  float *d_K, *d_Norms;
  checkCudaErrors(cudaMalloc((void **) &d_Norms, M * sizeof(float)));


  checkCudaErrors(cudaMemGetInfo(&free, &total));
  int numbleaves = 1;
  int sizebleaves = leaves;
  size_t size_req = ppl * ppl * sizeof(float);
  
  size_req = size_req * sizebleaves;
  
  float s1 = size_req/1e9;
  printf("Require %.4f GB \n", s1);
  float s2 = free/1e9;

  while (s1 > s2){
    numbleaves *= 2;
    sizebleaves /= 2;
    size_req = sizebleaves * ppl * ppl * sizeof(float);
    s1 = size_req/1e9;
    s2 = free/1e9;
  }

  printf("Reduced to %.4f GB \n", size_req/1e9);

  cudaEvent_t t0; 
  cudaEvent_t t1;
  
  
  dim3 BlockDist(t_b, 1, 1);
  dim3 GridDist(ppl, sizebleaves, 1);

  dim3 BlockMerge(t_b, 1, 1);
  dim3 GridMerge(ppl, sizebleaves, 1); 


  printf("# points : %d \n", M);
  printf("# leaves : %d \n", leaves);
  printf("# points/leaf : %d \n", ppl);
  printf("# sizebleaves : %d \n", sizebleaves);
  printf("# numbleaves : %d \n", numbleaves);
  printf(" Dist block (%d,%d,%d) \n", BlockDist.x, BlockDist.y, BlockDist.z); 
  printf(" Dist grid (%d,%d,%d) \n", GridDist.x, GridDist.y, GridDist.z); 
  printf(" Merge block (%d,%d,%d) \n", BlockDist.x, BlockDist.y, BlockDist.z); 
  printf(" Merge grid (%d,%d,%d) \n", GridDist.x, GridDist.y, GridDist.z); 
  printf(" Assiging %.4f GB \n", size_req/1e9);
  printf(" Bef alloc %.4f GB free from %.4f \n", free/1e9, total/1e9);
  checkCudaErrors(cudaMalloc((void **) &d_K, size_req));
  checkCudaErrors(cudaMemGetInfo(&free, &total));
  printf(" Aft alloc %.4f GB free from %.4f \n", free/1e9, total/1e9);



  ComputeNorms <<< GridNorms, BlockNorms >>> (R, C, V, G_Id, d_Norms, ppl);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));

  checkCudaErrors(cudaEventRecord(t0, 0));


  
  for (int bl = 0; bl < numbleaves; bl++){

    checkCudaErrors(cudaEventRecord(t0_dist, 0));
    compute_dist <<< GridDist, BlockDist >>> (R, C, V, G_Id, d_Norms, k_nn, d_K, ppl, bl, sizebleaves, M);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(t1_dist, 0));
    checkCudaErrors(cudaEventSynchronize(t1_dist));
    checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t0_dist, t1_dist));
    dt_dist += dt_tmp;


    find_neighbor <<< GridMerge, BlockMerge >>> (KNN, KNN_Id, d_K, G_Id, k_nn, ppl, M, bl, sizebleaves);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(t1_merge, 0));
    checkCudaErrors(cudaEventSynchronize(t1_merge));
    checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t1_dist, t1_merge));
    dt_merge += dt_tmp;

  }


  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t_end, 0));
  checkCudaErrors(cudaEventSynchronize(t_end));
  checkCudaErrors(cudaEventElapsedTime(&dt_tot, t_begin, t_end));
  


 
  printf(" Distance = %.4f (%.f ) \n", dt_dist/1000, dt_dist*100/dt_tot);
  printf(" Merge = %.4f (%.f )\n", dt_merge/1000, dt_merge*100/dt_tot);
  printf(" tot = %.4f \n", dt_tot/1000);
 
 
  checkCudaErrors(cudaFree(d_K));
  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaEventDestroy(t_begin));
  checkCudaErrors(cudaEventDestroy(t_end));
  checkCudaErrors(cudaEventDestroy(t0_merge));
  checkCudaErrors(cudaEventDestroy(t1_merge));
  checkCudaErrors(cudaEventDestroy(t0_dist));
  checkCudaErrors(cudaEventDestroy(t1_dist));


}
