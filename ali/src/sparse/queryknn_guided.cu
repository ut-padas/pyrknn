
#define MAX_BLOCK_SIZE 1024
 
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "queryknn.h"
#include "GuidedBinSearch.h"
#include "Norms.h"


void query_leafknn(int *R_ref, int *C_ref, float *V_ref, int *R_q,  int * C_q, float *V_q, int *GId, int const ppl, int const leaves, int const k, float *d_knn, int *d_knn_Id, int const deviceId, int const verbose, int const nq, int *leafIds, int const dim, int const avgnnz, int const num_search_leaves){


  float dt_dist, dt_tot, dt_qsearch_ind, dt_norms_q, dt_norms_ref, dt_tmp, dt_mem;

  checkCudaErrors(cudaSetDevice(deviceId));
  cudaEvent_t t_start, t_end, t_qsearch, t_dist, t_norms_ref, t_norms_q, t_memalloc;
 
  checkCudaErrors(cudaEventCreate(&t_start));
  checkCudaErrors(cudaEventCreate(&t_end));
  checkCudaErrors(cudaEventCreate(&t_qsearch));
  checkCudaErrors(cudaEventCreate(&t_dist));
  checkCudaErrors(cudaEventCreate(&t_norms_ref));
  checkCudaErrors(cudaEventCreate(&t_norms_q));
  checkCudaErrors(cudaEventCreate(&t_memalloc));
 

  checkCudaErrors(cudaEventRecord(t_start, 0));
  
  if (verbose) printf("----------------------------- start leaf queries in pyrknn -----------------------------------\n");
  
  int numqsearch = avgnnz/2;
  int size_search = dim / numqsearch;
  size_t qsearch_size = sizeof(int) * numqsearch * nq;
  size_t tmp_NDist_size = sizeof(float) * k * nq;
  size_t norm_q_size = sizeof(float) * nq;
  size_t norm_ref_size = sizeof(float) * num_search_leaves; 
 
  printf("Require %.4f (GB) for qsearch \n", qsearch_size/1e9);
  printf("Require %.4f (GB) for tmp NDists\n", tmp_NDist_size/1e9);
  printf("Require %.4f (GB) for norm refs\n", norm_ref_size/1e9);
  printf("Require %.4f (GB) for norm queries\n", norm_q_size/1e9);

  int *qsearch_ind;  
  float *tmp_NDist, *Norms_ref, *Norms_q;
  

  checkCudaErrors(cudaMalloc((void **) &qsearch_ind, qsearch_size));
  checkCudaErrors(cudaMalloc((void **) &tmp_NDist, tmp_NDist_size));
  checkCudaErrors(cudaMalloc((void **) &Norms_ref, norm_ref_size));
  checkCudaErrors(cudaMalloc((void **) &Norms_q, norm_q_size));
   
  checkCudaErrors(cudaEventRecord(t_memalloc, 0));
  checkCudaErrors(cudaEventSynchronize(t_memalloc));
  checkCudaErrors(cudaEventElapsedTime(&dt_mem, t_start, t_memalloc));

  
  dim3 BlockQSearch(size_search, 1, 1);
  dim3 GridQSearch(nq, 1, 1);
  
  int t_b = (ppl > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : ppl;
 
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
  
  ComputeNorms <<< GridNorm_ref, BlockNorm_ref >>> (R_ref, C_ref, V_ref, Norms_ref);

  checkCudaErrors(cudaDeviceSynchronize()); 
  checkCudaErrors(cudaEventRecord(t_norms_ref, 0));
  checkCudaErrors(cudaEventSynchronize(t_norms_ref));
  checkCudaErrors(cudaEventElapsedTime(&dt_norms_ref, t_norms_q, t_norms_ref));

  
  ComputeSearchGuide <<< GridQSearch, BlockQSearch >>> (R_q, C_q, size_search, qsearch_ind);

  checkCudaErrors(cudaDeviceSynchronize());  
  checkCudaErrors(cudaEventRecord(t_qsearch, 0));
  checkCudaErrors(cudaEventSynchronize(t_qsearch));
  checkCudaErrors(cudaEventElapsedTime(&dt_qsearch_ind, t_norms_ref, t_qsearch));

  
  ComputeDists <<< GridDist, BlockDist >>> (R_ref, C_ref, V_ref, R_q, C_q, V_q, leafIds, Norms_q, Norms_ref, k, tmp_NDist, ppl, qsearch_ind, size_search, dim, GId);

  checkCudaErrors(cudaDeviceSynchronize());  
  checkCudaErrors(cudaEventRecord(t_dist, 0));
  checkCudaErrors(cudaEventSynchronize(t_dist));
  checkCudaErrors(cudaEventElapsedTime(&dt_dist, t_qsearch, t_dist));

  // TODO : compute the qsearch_ind before the leafknn for all the iterations

  checkCudaErrors(cudaFree(Norms_ref));
  checkCudaErrors(cudaFree(Norms_q));
  checkCudaErrors(cudaFree(qsearch_ind));
  checkCudaErrors(cudaFree(tmp_NDist));

  checkCudaErrors(cudaEventRecord(t_end, 0));
  checkCudaErrors(cudaEventSynchronize(t_end));
  checkCudaErrors(cudaEventElapsedTime(&dt_tot, t_start, t_end));

  if (verbose){
  printf("----------------- Timings ------------\n");
  printf(" Memory : %.4f (%.f %%) \n", dt_mem/1000, 100*dt_mem/dt_tot);
  printf(" Norms queries : %.4f (%.f %%) \n", dt_norms_q/1000, 100*dt_norms_q/dt_tot);
  printf(" Norms ref : %.4f (%.f %%) \n", dt_norms_ref/1000, 100*dt_norms_ref/dt_tot);
  printf(" QBinsearch : %.4f (%.f %%) \n", dt_qsearch_ind/1000, 100*dt_qsearch_ind/dt_tot);
  printf(" Dist : %.4f (%.f %%) \n", dt_dist/1000, 100*dt_dist/dt_tot);
  printf("\n Total : %.4f \n", dt_tot/1000);
  printf("-----------------------------------\n");
  }



}
