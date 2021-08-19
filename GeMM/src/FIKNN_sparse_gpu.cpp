#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
#define SM_SIZE_SORT 8192

#include "FIKNN_sparse_gpu.h"




void FIKNN_sparse_gpu(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz){

  int ppl = M/leaves;

  float del_t1;
  cudaEvent_t t0;
  cudaEvent_t t1;

  int num_blocks_tri = ppl / k;

  int t_b = (ppl > SM_SIZE_1) ? SM_SIZE_1 : ppl;
  int num_batch_norm = (ppl > SM_SIZE_1) ? ppl/SM_SIZE_1 : 1;

  dim3 dimBlock_norm(t_b, 1);
  dim3 dimGrid_norm(num_batch_norm, leaves);

  printf("block Norms = (%d , %d) \n ", t_b, 1);
  printf("Grid Norms = (%d , %d) \n ", num_batch_norm, leaves);


  /*
	bool optimal_sort = true;
  int *d_sort_arr_v, *d_sort_arr_part_v;
  int *sort_arr_v, *sort_arr_part_v;

  int size_v = 2 * k;
  int blocksize_v = k;

  int steps_v = log2(size_v) * (log2(size_v) + 1) /2;
  sort_arr_v = (int *)malloc(sizeof(int) * steps_v * size_v);
  sort_arr_part_v = (int *)malloc(sizeof(int) * steps_v * size_v);

  precomp_arbsize_sortId(sort_arr_v, sort_arr_part_v, size_v, size_v, steps_v, size_v);



  checkCudaErrors(cudaMemcpy(d_sort_arr_v, sort_arr_v, sizeof(int)* steps_v * size_v, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_sort_arr_part_v, sort_arr_part_v, sizeof(int)* steps_v * size_v, cudaMemcpyHostToDevice));
  dim3 dimBlock_v(blocksize_v, 1);
  */





  float *d_Norms;


  int size_tri = (k > 32) ? 32 : k;
  int blockDim_tri = size_tri * (size_tri + 1)/2;
  if (blockDim_tri > SM_SIZE_1) blockDim_tri = SM_SIZE_1;

  dim3 dimBlock_tri(blockDim_tri, 1);
  dim3 dimGrid_tri(num_blocks_tri, leaves);

  printf("block TriPart = (%d , %d) \n ", blockDim_tri, 1);
  printf("Grid TriPart = (%d , %d) \n ", num_blocks_tri, leaves);


  dim3 dimGrid_sq(k, leaves);

  printf("Grid RecPart = (%d , %d) \n ", k, leaves);



  int size_v = 2 * k;
  dim3 dimBlock_v(size_v, 1);



  int *d_arr, *d_arr_part;

  int n_s = log2(SM_SIZE_2) *(log2(SM_SIZE_2)+1) /2;

  int copy_size = (ppl) * n_s;

  size_t free, total, m1, m2, m3;

  cudaMemGetInfo(&free, &total);
  checkCudaErrors(cudaMalloc((void **) &d_arr, sizeof(int) * copy_size));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part, sizeof(int) * copy_size));


  checkCudaErrors(cudaMemset(d_arr, 0, sizeof(int) * copy_size));
  checkCudaErrors(cudaMemset(d_arr_part, 0, sizeof(int) * copy_size));
  cudaMemGetInfo(&m1, &total);



  float * d_temp_knn;
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * ppl * leaves));

  cudaMemGetInfo(&m2, &total);
  checkCudaErrors(cudaMalloc((void **) &d_temp_knn, sizeof(float) * ppl * leaves * k));
  cudaMemGetInfo(&m3, &total);

  int steps;
  dim3 Block_GId(SM_SIZE_1);

  dim3 Grid_GId(leaves);

  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));

  checkCudaErrors(cudaEventRecord(t0, 0));

  //sort_GIds <<< Grid_GId, Block_GId >>> (G_Id, ppl);
  //checkCudaErrors(cudaDeviceSynchronize());

  FIKNN_compute_norm <<< dimGrid_norm, dimBlock_norm >>>(R, C, V, G_Id, d_Norms, ppl);
  checkCudaErrors(cudaDeviceSynchronize());
  knn_kernel_tri <<< dimGrid_tri, dimBlock_tri >>>(R, C, V, G_Id, d_Norms, k, knn, knn_Id, ppl, max_nnz);
  checkCudaErrors(cudaDeviceSynchronize());


  for (int blockInd = 0; blockInd < num_blocks_tri - 1; blockInd++){

    int size_part = ppl - blockInd *k;
    int blocksize = ceil(size_part/2);

    while (blocksize > SM_SIZE_1) blocksize = ceil(blocksize/2);

    int N_pow2 = pow(2, ceil(log2(2 * blocksize)));
    steps = log2(N_pow2) * (log2(N_pow2) +1)/2;


    steps = log2(N_pow2) * (log2(N_pow2) +1)/2;


    int real_size = 2 * blocksize;

    precomp_arbsize_sortId(d_arr, d_arr_part, real_size, N_pow2, steps, copy_size);


    dim3 dimBlock_sq(blocksize, 1);



    int size_v = ppl - (blockInd + 1) * k;

    dim3 dimGrid_v(size_v, leaves);

    knn_kernel_A <<< dimGrid_sq, dimBlock_sq >>>(R, C, V, G_Id, d_Norms, k, knn, knn_Id, ppl, max_nnz, blockInd, d_arr, d_arr_part, steps, d_temp_knn);
    checkCudaErrors(cudaDeviceSynchronize());

    knn_kernel_B <<< dimGrid_v, dimBlock_v >>> (knn, knn_Id, k, ppl, blockInd, d_temp_knn, G_Id);

    checkCudaErrors(cudaDeviceSynchronize());
  }

  checkCudaErrors(cudaDeviceSynchronize());


  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));

  printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);

  printf("\n Memory for sort %.4f GB \n", (free-m1)/1e9);
  printf("\n Memory for norm %.4f GB \n", (m1-m2)/1e9);
  printf("\n Memory for temp storage %.4f GB \n", (m2-m3)/1e9);

  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaEventDestroy(t0));
  checkCudaErrors(cudaEventDestroy(t1));

}








