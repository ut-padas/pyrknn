
#define SM_SIZE_SORT 8192
#define SM_SIZE_2 2048
#define SM_SIZE_1 1024

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <algorithm>





__global__ void SortArray(float* data, int const k, int const l, int const n_q, int* const sort_arr, int* const sort_arr_part, int const steps, float* NhbrDist){


  __shared__ float SM[SM_SIZE_2];
  __shared__ float SM_k[1024];



  int j = threadIdx.x;
  int row = blockIdx.x;
  
  for (int n=j; n < SM_SIZE_2; n+= blockDim.x) SM[n] = 1e38;
  for (int n=j; n < 1024; n+= blockDim.x) SM[n] = 1e38;
  
  float tmp_f;
  int ind_sort;
 
  int size_sort = 2 * blockDim.x;

  int num_batches = l / (size_sort);
  //int num_batches = 1;

  if (num_batches == 0) num_batches += 1;


  for (int col_batch = 0; col_batch < num_batches; col_batch++){
      
      int colId = col_batch * size_sort + j;
      SM[j] = data[row * l + j];
      SM[j+blockDim.x] = data[row * l + colId + blockDim.x];
 

    __syncthreads();
      

    for (int step = 0; step < steps; step++){
    
      int j_tmp = j;
      ind_sort = step * 2 * blockDim.x + j_tmp;
      int tid = sort_arr[ind_sort];
      int ixj = sort_arr_part[ind_sort];
      
      int min_max = (1 & tid);   
      int coupled_flag = (1 & ixj);
      
      tid = tid >> 1;
      ixj = ixj >> 1;
      
			if (coupled_flag == 1){

        ind_sort += blockDim.x;

        int tid_1 = sort_arr[ step * 2 * blockDim.x + j_tmp + blockDim.x];
        int ixj_1 = sort_arr_part[step * 2 * blockDim.x + j_tmp + blockDim.x];
        int min_max_1 = (1 & tid_1);


        tid_1 = tid_1 >> 1;
        ixj_1 = ixj_1 >> 1;


        if (min_max_1 == 1 && SM[tid_1] > SM[ixj_1]){

          tmp_f = SM[tid_1];
          SM[tid_1] = SM[ixj_1];
          SM[ixj_1] = tmp_f;

        }

        if (min_max_1 == 0 && SM[tid] < SM[ixj]){

          tmp_f = SM[tid_1];
          SM[tid_1] = SM[ixj_1];
          SM[ixj_1] = tmp_f;

        }

      }

      if (min_max == 1){
        if (SM[tid] > SM[ixj]){
          tmp_f = SM[tid];
          SM[tid] = SM[ixj];
          SM[ixj] = tmp_f;
        }
      } else {
        if (SM[tid] < SM[ixj]){
          tmp_f = SM[tid];
          SM[tid] = SM[ixj];
          SM[ixj] = tmp_f;
        }
      }

      __syncthreads();
    }
  if (j < k){
    SM_k[k-j] = (SM[j] < SM_k[k-j]) ? SM[j] : SM[k-j];
  }

  __syncthreads();
  
  for (int g = 2; g <= k; g*=2){
    for(int l = g/2; l >0; l /= 2){
      int ixj = j ^ l;
      if (ixj > j && j < k){
        if ((j &g) == 0){
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

  if (j < k) NhbrDist[row * k + j] = SM_k[j];
     
}


void PrecompSortIds(int* d_arr, int* d_arr_part, int N_true, int N_pow2, int steps, int copy_size){



  int min_max, elem, coupled_elem;
  int loc_len = ceil(N_true/2);
  int* tracker;
  tracker = (int *)malloc(sizeof(int) * N_pow2);

  for (int i = 0; i < N_pow2; i ++) tracker[i] = i;

  int step = 0;

  int *arr, *arr_part;
  arr = (int *)malloc(sizeof(int) * copy_size);
  arr_part = (int *)malloc(sizeof(int) * copy_size);


  memset(arr, 0, sizeof(int) * copy_size);
  memset(arr_part, 0, sizeof(int) * copy_size);

  int first_pair = 1;
  int prev_elem = 0;
  int tmp2;
  for (int g = 2; g <= N_pow2; g *= 2){
    for (int l = g/2; l > 0; l /= 2){
      elem = 0;
      for (int i = 0; i < N_pow2; i++){
        int ixj = i ^ l;

        if (tracker[ixj] >= N_true && tracker[i] >= N_true) continue;

        if (ixj > i){

          min_max = ((i&g) == 0 ) ? 1 : 0;

          coupled_elem = 0;

          int write_loc = elem;

          if (tracker[ixj] >= N_true || tracker[i] >= N_true) {
            coupled_elem = 1;
            if (min_max == 0 && tracker[ixj] >= N_true) {
              tmp2 = tracker[ixj];
              tracker[ixj] = tracker[i];
              tracker[i] = tmp2;
            }
            if (min_max == 1 && tracker[i] >= N_true) {
              tmp2 = tracker[ixj];
              tracker[ixj] = tracker[i];
              tracker[i] = tmp2;
            }
            if (first_pair == 1){
              prev_elem = elem;
              write_loc = elem;
              first_pair = 0;
            } else {
              write_loc = prev_elem + loc_len;
              first_pair = 1;
              elem++;
            }

            arr[step * N_true + write_loc] = (i << 1) + min_max;
            arr_part[step * N_true + write_loc] = (ixj << 1) + coupled_elem;

          } else {
            write_loc = elem;
            arr[step * N_true + write_loc] = (i << 1) + min_max;
            arr_part[step * N_true + write_loc] = (ixj << 1) + coupled_elem;
            elem++;
          }




        }
      }

      step++;
    }
  }
  checkCudaErrors(cudaMemcpy(d_arr, arr, sizeof(int)*copy_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_arr_part, arr_part, sizeof(int)*copy_size, cudaMemcpyHostToDevice));
  //checkCudaErrors(cudaDeviceSynchronize());

}




void SortInterface(float *data, float *Nhbd, int n_q, int l, int k){


  float dt1, dt2, dt3, dt4;
  dt1 = 0.0;
  dt2 = 0.0;
  dt3 = 0.0;

  cudaEvent_t t0,t1,t2,t3,t4,t5,t6,t7;


  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));
  checkCudaErrors(cudaEventCreate(&t2));


  float *d_data;
  float *d_Nhbd;
  int *d_arr, *d_arr_part;

  checkCudaErrors(cudaMalloc((void **) &d_data, sizeof(float) * n_q * l));
  checkCudaErrors(cudaMalloc((void **) &d_Nhbd, sizeof(float) * n_q * k));

  checkCudaErrors(cudaMemcpy(d_data, data, sizeof(float) * n_q * l, cudaMemcpyHostToDevice));

  int size_sort = l;

  while (size_sort > SM_SIZE_2) size_sort = ceil((size_sort)/2);
  float tmp = size_sort/2.0;
  int blocksize = ceil(tmp); 
  float tmp_f = 2 * blocksize;
  int N_pow2 = pow(2, ceil(log2(tmp_f)));
  tmp_f = N_pow2;
  int steps = log2(tmp_f) * (log2(tmp_f) + 1)/2;
  int real_size = 2 * blocksize;
  int copy_size = steps * real_size;
  dim3 Grid(n_q);
  dim3 Block(blocksize);
  printf("size grid %d \n", Grid.x);
  printf("size block %d \n", Block.x);

  checkCudaErrors(cudaMalloc((void **) &d_arr, sizeof(int) *  copy_size));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part, sizeof(int) * copy_size));
  checkCudaErrors(cudaEventRecord(t0, 0));

  

  PrecompSortIds(d_arr, d_arr_part, real_size, N_pow2, steps, copy_size);
  
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  SortArray <<< n_q, blocksize >>>(d_data, k, l, n_q, d_arr, d_arr_part, steps, d_Nhbd);
  
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


  checkCudaErrors(cudaFree(d_arr));
  checkCudaErrors(cudaFree(d_arr_part));
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_Nhbd));


}




























