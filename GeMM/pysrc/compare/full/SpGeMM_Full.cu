
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

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

    int leafId_g = bl * sizebleaves + leafId_loca;
    int leafId_local = blockIdx.z * gridDim.y + blockIdx.y;
    int leafId_g = bl * sizebleaves + leafId_local;

    int g_rowId = leafId_g * ppl + rowId;
    int ind0_i = R[g_rowId];
    int ind1_i = R[g_rowId + 1];
    int nnz_i = ind1_i - ind0_i;
    
    __shared__ int SM[2048];

    for (int n_i = tid; tid < nnz_i; tid += blockDim.x) SM[n_i] = C[ind0_i + n_i];
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
      for (int pos_k = 0; pos_k < nnz_j; pos_k++){

        k = C[ind0_j + pos_k];

        // Binary search
        for (int l = nnz_i - ret; l > 1; l -= floorf(l/2.0)){
          tmp_0 = ret + l;
          tmp_1 = nnz_i - 1;
          testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
          ret = (C[ind0_i + testInd] <= k) ? testInd : ret;
        }

        tmp_0 = ret + 1;
        tmp_1 = nnz_i - 1;
        testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;

        ret = (C[testInd + ind0_i] <= k) ? testInd : ret;

        ind_jk = (C[ret + ind0_i] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;

      }
      c_tmp = -2 * c_tmp + norm_ij;
      c_tmp = (c_tmp > 1e-8) ? c_tmp : 0.0;

      int ind_K = leafId_local * ppl * ppl + rowId * ppl + colId;
      K[ind_K] = c_tmp;
      
    }
    
}

//__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int* G_Id, int k, int M_I, int m_j, int leaf_batch_g, int M){
__global__ void find_neighbor(float* KNN, int* KNN_Id, float* K, int* G_Id, int k_nn, int ppl, int M, int bl, int sizebleaves){



    __shared__ float Dist[4096];
    __shared__ int Dist_Id[4096];

    int rowId = blockIdx.x;
    int tid = threadIdx.x;

    int leafId_local = blockIdx.z * gridDim.y + blockIdx.y;
    int leafId_g = bl * sizebleaves + leafId_local;
    
    
    for (int n_i = tid; n_i < ppl; n_i += blockDim.x){
      Dist[n_i] = K[leafId_local * ppl * ppl + rowId * ppl + n_i]    
      Dist_Id[n_i] = leafId_g * ppl + n_i;
    }

    float tmp_f;
    int tmp_i;
    
    int size = ppl
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
                Dist_Id[j] = tmp_f;
              }
            } else {
              if (Dist[j] < Dist[ixj]){
                tmp_f = Dist[ixj];
                Dist[ixj] = Dist[j];
                Dist[j] = tmp_f;
                tmp_i = Dist_Id[ixj];
                Dist_Id[ixj] = Dist_Id[j];
                Dist_Id[j] = tmp_f;
              }
            }
          }
        }
        __syncthreads();
      }
    }

    int ind_knn = (leafId_g * ppl + rowId ) * k_nn + tid;
    if (tid < k_nn){
     Dist[tid] = KNN[ind_knn];
     Dist_Id[tid] = KNN_Id[ind_knn]; 
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
                Dist_Id[j] = tmp_f;
              }
            } else {
              if (Dist[j] < Dist[ixj]){
                tmp_f = Dist[ixj];
                Dist[ixj] = Dist[j];
                Dist[j] = tmp_f;
                tmp_i = Dist_Id[ixj];
                Dist_Id[ixj] = Dist_Id[j];
                Dist_Id[j] = tmp_f;
              }
            }
          }
        }
        __syncthreads();
      }
    }

    if (tid < k_nn){
      KNN[ind_knn] = Dist[tid];
      KNN_Id[ind_knn] = Dist_Id[tid]; 
    }
}



void gen_sparse(int M, int tot_nnz, int d, int *R, int *C, float *V) {
 
    int nnz_row;
    int val;

    for (int i=0; i < M; i++){
      nnz_row = R[i+1] - R[i];
      for (int j=0; j < nnz_row; j++){
          int ind = R[i]+j; 
          val = rand()%d;
          //val = rand()%d;
          C[ind] = val;
          V[ind] = rand()%100;
        }    
      std::sort(C+R[i], C+(R[i+1]));
      /*
      printf("\n point %d\n", i);
      for (int j=R[i]; j<R[i+1]; j++) {
      printf("(%d ,%.4f) ",C[j], V[j]);
      }
      */
    }
}

void gen_R(int M, int nnzperrow, int *R, int *G_Id, int d) {  
  R[0] = 0;
  int tot_nnz = 0;
  int val;
  for (int m =1; m <= M; m++){ 
   val = 1 + rand()%(2*nnzperrow);
   //val = nnzperrow; //+ rand()%nnzperrow;
   if (val > d) val = 1; 
   tot_nnz += val;
   R[m] = tot_nnz;
   G_Id[m-1] = m-1;
  } 
  //std::random_shuffle(&G_Id[0], &G_Id[M]);
  /* 
  for (int m = 0; m < M; m++){ 
  printf("G_Id[%d] = %d \n", m , G_Id[m]);
  } 
  */
}

void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *KNN, int *KNN_Id){


  int ppl = M / leaves;

  size_t free, total, m1, m2, m3;

  cudaEvent_t t_begin;
  cudaEvent_t t_end;
  cudaEvent_t t0_dist;
  cudaEvent_t t1_dist;
  cudaEvent_t t0_merge;
  cudaEvent_t t1_merge;

  float dt_tot;
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


  float *d_K, float *d_Norms;
  checkCudaErrors(cudaMalloc((void **) &d_Norms, M * sizeof(float)));


  checkCudaErorrs(cudaMemGetInfo(&free, &total));
  int numbleaves = 1;
  int sizebleaves = leaves;
  size_t size_req = sizebleaves * ppl * ppl * sizeof(float);
  
  float s1 = size_req/1e9;
  float s2 = free/1e9;

  while (s1 > s2){
    numbleaves *= 2;
    sizebleaves /= 2;
    size_req = sizebleaves * ppl * ppl * sizeof(float);
    s1 = size_req/1e9;
    s2 = free/1e9;
  }


  float del_t1;
  cudaEvent_t t0; 
  cudaEvent_t t1;
  
  
  dim3 BlockDist(t_b, 1, 1);
  dim3 GridDist(ppl, sizebleaves, 1);

  dim3 BlockMerge(t_b, 1, 1);
  dim3 GridMerge(ppl, sizebleaves, 1); 


  checkCudaErrors(cudaMalloc((void **) &d_K, size_req));

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
    checkCudaErrors(CudaEventSynchronize(t1_dist));
    checkCudaErrors(cudaEventElapstedTime(&dt_tmp, t0_dist, t1_dist));
    dt_dist += dt_tmp;


    find_neighbor <<< GridMerge, BlockMerge >>> (KNN, KNN_Id, d_K, G_Id, k_nn, ppl, M, bl, sizebleaves);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(t1_merge, 0));
    checkCudaErrors(CudaEventSynchronize(t1_merge));
    checkCudaErrors(cudaEventElapstedTime(&dt_tmp, t1_dist, t1_merge));
    dt_merge += dt_tmp;

  }


  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t_end, 0));
  checkCudaErrors(cudaEventSynchronize(t_end));
  checkCudaErrors(cudaEventElapsedTime(&dt_tot, t_begin, t_end));
  


  printf("# points : %d \n", M);
  printf("# leaves : %d \n", leaves);
  printf("# points/leaf : %d \n", ppl);
  printf("# sizebleaves : %d \n", sizebleaves);
  printf("# numbleaves : %d \n", numbleaves);
 
  printf(" Distance = %.4f (%.f ) \n", dt_dist/1000, dt_dist*100/dt_tot);
  printf(" Merge = %.4f (%.f )\n", dt_merge/1000, dt_merge*100/dt_tot);
  printf(" tot = %.4f \n", dt_tot/1000);
 
  printf(" Dist block (%d,%d,%d) \n", BlockDist.x, BlockDist.y, BlockDist.z); 
  printf(" Dist grid (%d,%d,%d) \n", GridDist.x, GridDist.y, GridDist.z); 
  printf(" Merge block (%d,%d,%d) \n", BlockDist.x, BlockDist.y, BlockDist.z); 
  printf(" Merge grid (%d,%d,%d) \n", GridDist.x, GridDist.y, GridDist.z); 
 
  checkCudaErrors(cudaFree(d_K));
  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaEventDestroy(t_begin));
  checkCudaErrors(cudaEventDestroy(t_end));
  checkCudaErrors(cudaEventDestroy(t0_merge));
  checkCudaErrors(cudaEventDestroy(t1_merge));
  checkCudaErrors(cudaEventDestroy(t0_dist));
  checkCudaErrors(cudaEventDestroy(t1_dist));


}
