
#include "SortMerge.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <algorithm>


__global__ void SortMergeReg(float* data, float* Nhbd, int n_q, int l, int k){


  int tid = threadIdx.x;
  int row = blockIdx.x;

  

  __shared__ float SM_dist[6144];
  __shared__ float SM_k[1024];
   
  for (int tid_tmp = tid; tid_tmp < k; tid_tmp += blockDim.x){
    SM_k[tid_tmp] = Nhbd[row * k + tid_tmp];  
  }
  
  int num_batches = l/(2*blockDim.x);
  if (num_batches == 0) num_batches += 1;
  float tmp_f, s_1, s_2; 
  for (int batchId = 0; batchId < num_batches; batchId++){
  
    SM_dist[tid] = data[row * l + batchId * 2 * blockDim.x + tid];
    SM_dist[tid + blockDim.x] = data[row * l + batchId * 2 * blockDim.x + tid + blockDim.x];
    __syncthreads();
  
    for (int g = 2; g <= 2*blockDim.x; g *=2){
      for (int l = g/2; l > 0; l /=2){
        int j = tid;
        int ixj = j ^ l; 
        if (ixj < j){
          j += blockDim.x;
          ixj = j ^l;
          if (ixj < j){
            j = ixj;
            ixj = j ^l;
          }
        }
        
        if ((j & g) == 0){
          if (SM_dist[j] > SM_dist[ixj]){
            tmp_f = SM_dist[ixj];
            SM_dist[ixj] = SM_dist[j];
            SM_dist[j] = tmp_f;
          } 
        } else {
          if (SM_dist[j] < SM_dist[ixj]){
            tmp_f = SM_dist[ixj];
            SM_dist[ixj] = SM_dist[j];
            SM_dist[j] = tmp_f;
          } 
        }
        __syncthreads();
      }
    }  

    //if (row == 0 && tid < k) printf("D[%d] = %.4f \n", tid, SM_dist[tid]);
    
    // Merge with the neighbors
    for (int tid_tmp = tid; tid_tmp < k; tid_tmp += blockDim.x){ 
      s_1 = SM_k[k-1 - tid_tmp];
      s_2 = SM_dist[tid_tmp];
      SM_k[k-1 - tid_tmp] = (s_2 < s_1) ? s_2 : s_1;
      //if (row == 0) printf("D[%d] = %.4f \n", k-1 - tid_tmp, SM_k[tid_tmp]);
    }
    __syncthreads();
     
    for (int g = 2; g <= k; g *= 2){
      for (int l = g/2; l > 0; l /= 2){
        
        int j = tid;
        int ixj = j ^ l;
        if (ixj < j){
          j += k/2;
          ixj = j ^ l;
          if (ixj < j){
            j = ixj;
            ixj = j ^ l;
          }
        }
        if (tid < k/2){
        if ((j & g) == 0){
          if (SM_k[j] > SM_k[ixj]){
            tmp_f = SM_k[ixj];
            SM_k[ixj] = SM_k[j];
            SM_k[j] = tmp_f;
          }
        } else {
          if (SM_k[j] < SM_k[ixj]){
            tmp_f = SM_k[ixj];
            SM_k[ixj] = SM_k[j];
            SM_k[j] = tmp_f;
          }
        }
        }
        __syncthreads();
      }
    }
   
  }

  
  if (tid < k) Nhbd[row * k + tid] = SM_k[tid]; 
  if (tid + blockDim.x< k) Nhbd[row * k + tid+blockDim.x] = SM_k[tid + blockDim.x]; 
   
}



void SortInterface(float *data, float *Nhbd, int n_q, int l, int k){


  float dt1, dt2, dt3;
  dt1 = 0.0;
  dt2 = 0.0;
  dt3 = 0.0;

  cudaEvent_t t0,t1,t2;


  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));
  checkCudaErrors(cudaEventCreate(&t2));


  float *d_data;
  float *d_Nhbd;
  //int *d_arr, *d_arr_part;

  checkCudaErrors(cudaMalloc((void **) &d_data, sizeof(float) * n_q * l));
  checkCudaErrors(cudaMalloc((void **) &d_Nhbd, sizeof(float) * n_q * k));

  checkCudaErrors(cudaMemcpy(d_data, data, sizeof(float) * n_q * l, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Nhbd, Nhbd, sizeof(float) * n_q * k, cudaMemcpyHostToDevice));


  dim3 Grid(n_q);
  int blocksize = (l > 2048) ? 1024 : l/2;
  dim3 Block(blocksize);
  printf("size grid %d \n", Grid.x);
  printf("size block %d \n", Block.x);

  checkCudaErrors(cudaEventRecord(t0, 0));



  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  SortMergeReg <<< Grid, Block >>>(d_data, d_Nhbd, n_q, l , k);

  checkCudaErrors(cudaEventRecord(t2, 0));
  checkCudaErrors(cudaEventSynchronize(t2));

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&dt1, t0, t1));
  checkCudaErrors(cudaEventElapsedTime(&dt2, t1, t2));
  checkCudaErrors(cudaEventElapsedTime(&dt3, t0, t2));
  printf("--- Sort Timings --- \n");
  printf("Preocmp = %.4e ms \n", dt1);
  printf("Sorting = %.4e ms \n", dt2);
  printf("Total = %.4e ms \n", dt3);

  checkCudaErrors(cudaMemcpy(Nhbd, d_Nhbd, sizeof(float) * n_q * k, cudaMemcpyDeviceToHost));


  //checkCudaErrors(cudaFree(d_arr));
  //checkCudaErrors(cudaFree(d_arr_part));
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_Nhbd));


}
