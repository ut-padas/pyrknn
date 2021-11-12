
#define MAX_BLOCK_SIZE 1024
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
 
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cublas_v2.h>
#include "queryleafknn.h"
#include "norm.h"
#include "merge.h"




static const char *cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}


#define CHECK_CUBLAS(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", cudaGetErrorEnum(code), file, line);
      if (abort) exit(code);
   }
}



void query_leafknn(float *X_ref, float *X_q, int *QId, int const ppl, int const leaves, int const k, float *NDist, int *NId, int const deviceId, int const verbose, int const nq, int *glob_pointIds, int num_search_leaves, int* local_leafIds, int dim){


  float dt_dist, dt_tot, dt_norms_q, dt_norms_ref, dt_tmp, dt_mem, dt_merge;
  size_t free, total;
  
  cudaMemGetInfo(&free, &total);
  if (verbose) printf(" Available Memory : %.4f MB from %.4f \n", free/1e6, total/1e6);
  
  checkCudaErrors(cudaSetDevice(deviceId));
  cudaEvent_t t_start, t_end, t_dist, t_norms_ref, t_norms_q, t_memalloc, t_merge;
 
  checkCudaErrors(cudaEventCreate(&t_start));
  checkCudaErrors(cudaEventCreate(&t_end));
  checkCudaErrors(cudaEventCreate(&t_dist));
  checkCudaErrors(cudaEventCreate(&t_norms_ref));
  checkCudaErrors(cudaEventCreate(&t_norms_q));
  checkCudaErrors(cudaEventCreate(&t_memalloc));
  checkCudaErrors(cudaEventCreate(&t_merge));
 
  
  checkCudaErrors(cudaEventRecord(t_start, 0));
  

  cublasStatus_t status;
  cublasHandle_t handle;
  status = cublasCreate(&handle);
  float oneFloat = 1.0;
  float zeroFloat = 0.0;

  if (verbose) printf("----------------------------- start leaf queries in pyrknn -----------------------------------\n");
  
  size_t tmp_NDist_size = sizeof(float) * ppl * nq;
  size_t norm_q_size = sizeof(float) * nq;
  size_t norm_ref_size = sizeof(float) * num_search_leaves * ppl; 
  if (verbose){
  printf("==========================\n");
  printf("ppl = %d \n", ppl);
  printf("leaves = %d \n", leaves);
  printf("k = %d \n", k);
  printf("nq = %d \n", nq);
  printf("num_search_leaves = %d \n", num_search_leaves);
 
  printf("Require %.4f (GB) for tmp NDists\n", tmp_NDist_size/1e9);
  printf("Require %.4f (GB) for norm refs\n", norm_ref_size/1e9);
  printf("Require %.4f (GB) for norm queries\n", norm_q_size/1e9);
  }
  float *tmp_NDist, *Norms_ref, *Norms_q;
  int *SortInd, *StepLen, *StepStart, *tidIdMap, *tidSortDir;
  

  checkCudaErrors(cudaMalloc((void **) &tmp_NDist, tmp_NDist_size));
  checkCudaErrors(cudaMalloc((void **) &Norms_ref, norm_ref_size));
  checkCudaErrors(cudaMalloc((void **) &Norms_q, norm_q_size));

  checkCudaErrors(cudaMalloc((void **) &SortInd, sizeof(int) * SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &StepLen, sizeof(int) * 12 * SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &StepStart, sizeof(int) * 12* SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &tidIdMap, sizeof(int) * 12* SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &tidSortDir, sizeof(int) * 12* SM_SIZE_1));


  checkCudaErrors(cudaMemset(tmp_NDist, 0, tmp_NDist_size));  
 
  checkCudaErrors(cudaEventRecord(t_memalloc, 0));
  checkCudaErrors(cudaEventSynchronize(t_memalloc));
  checkCudaErrors(cudaEventElapsedTime(&dt_mem, t_start, t_memalloc));

	int size_part = ppl + k;
	int size_sort = size_part;

	while (size_sort > SM_SIZE_2) size_sort = ceil((size_sort+k)/2.0);

	float tmp = size_sort/2.0;
	int blocksize = ceil(tmp);
	float tmp_f = 2 * blocksize;
	int N_pow2 = pow(2, ceil(log2(tmp_f)));
	tmp_f = N_pow2;
	int steps = log2(tmp_f);


  
  int t_b = ppl;
  while (t_b > MAX_BLOCK_SIZE) t_b /= ceil(t_b/2.0);
 
  //int t_b = (ppl > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : ppl;
 
  dim3 BlockDist(t_b, 1, 1);
  dim3 GridDist(nq, 1, 1);
  
  dim3 BlockNorm_ref(1, 1, 1);
  dim3 GridNorm_ref(ppl, num_search_leaves, 1);
  
  dim3 BlockNorm_q(1, 1,1);
  dim3 GridNorm_q(nq, 1, 1);
 
  dim3 BlockMerge(blocksize, 1, 1);
  dim3 GridMerge(nq, 1, 1);
 
  ComputeNorms <<< GridNorm_q, BlockNorm_q >>> (X_q, Norms_q, dim);

  checkCudaErrors(cudaDeviceSynchronize());  
  checkCudaErrors(cudaEventRecord(t_norms_q, 0));
  checkCudaErrors(cudaEventSynchronize(t_norms_q));
  checkCudaErrors(cudaEventElapsedTime(&dt_norms_q, t_memalloc, t_norms_q));
  
  ComputeNorms_ref <<< GridNorm_ref, BlockNorm_ref >>> (X_ref, Norms_ref, local_leafIds, dim);

  checkCudaErrors(cudaDeviceSynchronize()); 
  checkCudaErrors(cudaEventRecord(t_norms_ref, 0));
  checkCudaErrors(cudaEventSynchronize(t_norms_ref));
  checkCudaErrors(cudaEventElapsedTime(&dt_norms_ref, t_norms_q, t_norms_ref));

  int num_gemms = nq;
  
  CHECK_CUBLAS( cublasSgemmStridedBatched( handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                           1, ppl, dim, 
                                           &oneFloat, X_q, dim, dim,
                                           X_ref, dim, ppl*dim,
                                           &zeroFloat, tmp_NDist, 1, ppl, num_gemms) );
   

  checkCudaErrors(cudaDeviceSynchronize());  
  checkCudaErrors(cudaEventRecord(t_dist, 0));
  checkCudaErrors(cudaEventSynchronize(t_dist));
  checkCudaErrors(cudaEventElapsedTime(&dt_dist, t_norms_ref, t_dist));

  S_PrecompMergeNP2 <<< 1, blocksize >>> (SortInd, StepLen, StepStart, tidIdMap, tidSortDir, steps);
  checkCudaErrors(cudaDeviceSynchronize());  
  
  S_MergeHorizNP2<<< GridMerge, BlockMerge >>> (NDist, NId, k, ppl, tmp_NDist, glob_pointIds, steps, QId, SortInd, StepLen, StepStart, tidIdMap, tidSortDir, local_leafIds, Norms_q, Norms_ref);
  
  checkCudaErrors(cudaDeviceSynchronize());  
  checkCudaErrors(cudaEventRecord(t_merge, 0));
  checkCudaErrors(cudaEventSynchronize(t_merge));
  checkCudaErrors(cudaEventElapsedTime(&dt_merge, t_dist, t_merge));


  checkCudaErrors(cudaFree(Norms_ref));
  checkCudaErrors(cudaFree(Norms_q));
  checkCudaErrors(cudaFree(tmp_NDist));
  checkCudaErrors(cudaFree(SortInd));
  checkCudaErrors(cudaFree(StepLen));
  checkCudaErrors(cudaFree(StepStart));
  checkCudaErrors(cudaFree(tidIdMap));
  checkCudaErrors(cudaFree(tidSortDir));

  checkCudaErrors(cudaEventRecord(t_end, 0));
  checkCudaErrors(cudaEventSynchronize(t_end));
  checkCudaErrors(cudaEventElapsedTime(&dt_tot, t_start, t_end));

  cudaMemGetInfo(&free, &total);
  if (verbose) printf(" Available Memory : %.4f MB from %.4f \n", free/1e6, total/1e6);
  if (verbose){
		printf("----------------- Timings ------------\n");
		printf(" Memory : %.4f (%.f %%) \n", dt_mem/1000, 100*dt_mem/dt_tot);
		printf(" Norms queries : %.4f (%.f %%) \n", dt_norms_q/1000, 100*dt_norms_q/dt_tot);
		printf(" Norms ref : %.4f (%.f %%) \n", dt_norms_ref/1000, 100*dt_norms_ref/dt_tot);
		printf(" Dist : %.4f (%.f %%) \n", dt_dist/1000, 100*dt_dist/dt_tot);
		printf(" Merge : %.4f (%.f %%) \n", dt_merge/1000, 100*dt_merge/dt_tot);
		printf("\n Total : %.4f \n", dt_tot/1000);
		printf("-----------------------------------\n");
  }



}
