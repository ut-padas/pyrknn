
#define MAX_BLOCK_SIZE 1024
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
 
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "queryknn_seqsearch.h"
#include "SeqBinSearch.h"
#include "Norms.h"
#include "merge.h"


void query_leafknn_seqsearch(int *R_ref, int *C_ref, float *V_ref, int *R_q,  int * C_q, float *V_q, int *QId, int const ppl, int const leaves, int const k, float *NDist, int *NId, int const deviceId, int const verbose, int const nq, int const dim, int const avgnnz, int *glob_leafIds, int num_search_leaves, int *local_leafIds){


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
  
  if (verbose) printf("----------------------------- start leaf queries in pyrknn -----------------------------------\n");
  
  size_t tmp_NDist_size = sizeof(float) * ppl * nq;
  size_t norm_q_size = sizeof(float) * nq;
  size_t norm_ref_size = sizeof(float) * num_search_leaves * ppl; 

  printf("==========================\n");
  printf("ppl = %d \n", ppl);
  printf("leaves = %d \n", leaves);
  printf("k = %d \n", k);
  printf("nq = %d \n", nq);
  printf("dim = %d \n", dim);
  printf("avgnnz = %d \n", avgnnz);
  printf("num_search_leaves = %d \n", num_search_leaves);
 
  printf("Require %.4f (GB) for tmp NDists\n", tmp_NDist_size/1e9);
  printf("Require %.4f (GB) for norm refs\n", norm_ref_size/1e9);
  printf("Require %.4f (GB) for norm queries\n", norm_q_size/1e9);

  float *Norms_ref, *Norms_q;
  int *SortInd, *StepLen, *StepStart, *tidIdMap, *tidSortDir;
  

  checkCudaErrors(cudaMalloc((void **) &Norms_ref, norm_ref_size));
  checkCudaErrors(cudaMalloc((void **) &Norms_q, norm_q_size));

  checkCudaErrors(cudaMalloc((void **) &SortInd, sizeof(int) * SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &StepLen, sizeof(int) * 12 * SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &StepStart, sizeof(int) * 12* SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &tidIdMap, sizeof(int) * 12* SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &tidSortDir, sizeof(int) * 12* SM_SIZE_1));


  checkCudaErrors(cudaEventRecord(t_memalloc, 0));
  checkCudaErrors(cudaEventSynchronize(t_memalloc));
  checkCudaErrors(cudaEventElapsedTime(&dt_mem, t_start, t_memalloc));

  
  int t_b = ppl;
  while (t_b > MAX_BLOCK_SIZE) t_b = ceil(t_b/2.0);

  //int t_b = (ppl > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : ppl;
  

  int size_sort = t_b + k;
	float tmp = size_sort/2.0;
	int blocksize = ceil(tmp);
	float tmp_f = 2 * blocksize;
	int N_pow2 = pow(2, ceil(log2(tmp_f)));
	tmp_f = N_pow2;
	int steps = log2(tmp_f);
  

 
  dim3 BlockDist(t_b, 1, 1);
  dim3 GridDist(nq, 1, 1);
  
  dim3 BlockNorm_ref(1, 1, 1);
  dim3 GridNorm_ref(ppl, num_search_leaves, 1);
  
  dim3 BlockNorm_q(1, 1,1);
  dim3 GridNorm_q(nq, 1, 1);
 
 
  ComputeNorms <<< GridNorm_q, BlockNorm_q >>> (R_q, C_q, V_q, Norms_q);

  checkCudaErrors(cudaDeviceSynchronize());  
  checkCudaErrors(cudaEventRecord(t_norms_q, 0));
  checkCudaErrors(cudaEventSynchronize(t_norms_q));
  checkCudaErrors(cudaEventElapsedTime(&dt_norms_q, t_memalloc, t_norms_q));
  
  ComputeNorms_ref <<< GridNorm_ref, BlockNorm_ref >>> (R_ref, C_ref, V_ref, Norms_ref, local_leafIds);

  checkCudaErrors(cudaDeviceSynchronize()); 
  checkCudaErrors(cudaEventRecord(t_norms_ref, 0));
  checkCudaErrors(cudaEventSynchronize(t_norms_ref));
  checkCudaErrors(cudaEventElapsedTime(&dt_norms_ref, t_norms_q, t_norms_ref));

  S_PrecompMergeNP2 <<< 1, blocksize >>> (SortInd, StepLen, StepStart, tidIdMap, tidSortDir, steps);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t_merge, 0));
  checkCudaErrors(cudaEventSynchronize(t_merge));
  checkCudaErrors(cudaEventElapsedTime(&dt_merge, t_norms_ref, t_merge));
 
  FusedLeafKNN_seq <<< GridDist, BlockDist >>> (R_ref, C_ref, V_ref, R_q, C_q, V_q, local_leafIds, Norms_q, Norms_ref, k, ppl, dim, QId, NDist, NId, glob_leafIds, steps, SortInd, StepLen, StepStart, tidIdMap, tidSortDir);

  checkCudaErrors(cudaDeviceSynchronize());  
  checkCudaErrors(cudaEventRecord(t_dist, 0));
  checkCudaErrors(cudaEventSynchronize(t_dist));
  checkCudaErrors(cudaEventElapsedTime(&dt_dist, t_merge, t_dist));


  checkCudaErrors(cudaFree(Norms_ref));
  checkCudaErrors(cudaFree(Norms_q));
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
