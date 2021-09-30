
#include "SortMerge.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <algorithm>


__device__ void InsertSortArray(float* array, float dist, int k){

  
  int ret = 0;
  int testInd = 0;
  
  for (int l = k; l > 1; l -= l/2){
    tmp_0 = ret +l;
    tmp_1 = k-1;
     
    


  }




}




__global__ void SortMergeReg(float* data, float* Nhbd, int n_q, int l, int k){


  int tid = threadIdx.x;
  int row = blockIdx.x;

  

  __shared__ float SM_dist[6144];
  __shared__ float SM_k[1024];

  int size_sort = k;

  int num_batches_par_max = 2*blockDim.x / k ;
  int num_batches_seq = l / (2 * blockDim.x);
  if (num_batches_seq == 0) num_batches_seq += 1;

  if (tid < k) SM_k[tid] = Nhbd[row * k + tid];
  
  int batchId_par = tid / (k/2);
  int tid_sort = tid - batchId_par * k/2; 

  float s_1;
  float s_2;

  
  for (int batchId_seq = 0; batchId_seq < num_batches_seq; batchId_seq++){
  
    SM_dist[tid] = data[row * l + batchId_seq * 2 * blockDim.x + tid];
    SM_dist[tid + blockDim.x] = data[row * l + batchId_seq * 2 * blockDim.x + tid + blockDim.x];
    __syncthreads();
  
    for (int num_batches_par = num_batches_par_max; num_batches_par > num_batches_par_max-1; num_batches_par /= 2){
       
      // Bitonic sort 
			for (int g = 2; g <= k; g *= 2){
				for (int l = g/2; l > 0; l /= 2){
					
					if (batchId_par < num_batches_par){

						for (int thread = tid_sort; thread < k; thread += (k/2)){
							int ixj = thread ^ l; 
							if (ixj > thread){
								s_1 = SM_dist[batchId_par * k + thread];
								s_2 = SM_dist[batchId_par * k + ixj];
								if ((thread & g ) == 0){
									if (s_1 > s_2){
										SM_dist[batchId_par * k + thread] = s_2;
										SM_dist[batchId_par * k + ixj] = s_1;
									}
								} else {
									if (s_1 < s_2){
										SM_dist[batchId_par * k + thread] = s_2;
										SM_dist[batchId_par * k + ixj] = s_1;
									}
								}

							}
						}
						
					}
					__syncthreads();
				}
			}
        //if (row == 0 && batchId_par == 0 && num_batches_par == 1) printf("D[%d] = %.4f \n", tid_sort, SM_dist[batchId_par * k + tid_sort]);
        //if (row == 0 && batchId_par == 0 && num_batches_par == 1) printf("D[%d] = %.4f \n", tid_sort + k/2, SM_dist[batchId_par * k + k/2 + tid_sort]);
      // Merge between batches 
      int batch_pair = batchId_par ^ num_batches_par;
      if (batch_pair < num_batches_par && batchId_par < num_batches_par) {

        if (batch_pair > batchId_par){
       
          s_1 = SM_dist[batchId_par * k + k-1 - tid_sort];
          s_2 = SM_dist[batch_pair * k + tid_sort];
          
          if (s_2 < s_1) SM_dist[batchId_par * k + k-1 - tid_sort] = s_2;
          if (s_2 < s_1) SM_dist[batch_pair * k + tid_sort] = s_1;
       
        } else {
       
          s_1 = SM_dist[batch_pair * k + k/2-1 - tid_sort];
          s_2 = SM_dist[batchId_par * k + k-1 - tid_sort];
          
          if (s_2 < s_1) SM_dist[batch_pair * k + k/2-1 - tid_sort] = s_2;
          if (s_2 < s_1) SM_dist[batchId_par * k + k-1 - tid_sort] = s_1;

        }
      }
      __syncthreads();
    //if (num_batches_par == 1 && row == 0) printf("runing batch_pair = %d , batchId_par = %d , tid_sort = %d, SM_dist[%d] = %.4f \n", batch_pair, batchId_par, tid_sort, tid_sort, SM_dist[tid_sort]); 
    //if (num_batches_par == 1 && row == 0) printf("runing batch_pair = %d , batchId_par = %d , tid_sort = %d, SM_dist[%d] = %.4f \n", batch_pair, batchId_par, tid_sort + k/2, tid_sort + k/2, SM_dist[tid_sort + k/2]); 
    
      //__syncthreads();
    
    }
    //if (row == 0 && tid < k) printf("SM_k[%d] = %.4f \n", tid, SM_k[tid]);
    //if (row == 0 && tid < k) printf("SM_dist[%d] = %.4f \n", tid, SM_dist[tid]);
    
    // Merge with the neighbors 
    if (tid < k){
      s_1 = SM_k[k-1 - tid];
      s_2 = SM_dist[tid];
      SM_k[k - tid] = (s_2 < s_1) ? s_2 : s_1;
    }
    __syncthreads();
    //if (row == 0 && tid < k) printf("SM_k[%d] = %.4f \n", tid, SM_k[tid]);
    //if (row == 0 && tid < k) printf("SM_dist[%d] = %.4f \n", tid, SM_dist[tid]);
    for (int g = 2; g <= k; g *= 2){
      for (int l = g/2; l > 0; l /= 2){
        
        int ixj = tid ^ l;
        if (tid < k){  
          if (ixj> tid){
            s_1 = SM_k[tid];
            s_2 = SM_k[ixj];
            
            if ((tid & g) == 0){
              if (s_1 > s_2){
                SM_k[tid] = s_2;
                SM_k[ixj] = s_1;
              }
            } else {
              if (s_1 < s_2){
                SM_k[tid] = s_2;
                SM_k[ixj] = s_1;
              }
            }
          }
        }
        __syncthreads();
        //printf("Merging with SM_k, SM_k[%d] = %.4f \n",  
      }
    }
   
  }


  if (tid < k) Nhbd[row * k + tid] = SM_k[tid]; 
     
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
  int blocksize = (l > 1024) ? 1024 : l;
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
