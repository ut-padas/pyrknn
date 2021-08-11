
#include "knn_seq.h"
#include "TSKNN.h"
#include "rnd_sparse.h"



int main(int argc, char **argv){

    checkCudaErrors(cudaSetDevice(0));

    int d, nnzperrow;
    float *h_V, *d_V;
    int *h_C, *d_C;
    int *h_R, *d_R;
    int *h_G_Id, *d_G_Id;
    int M = 128 * 1;
    int leaves = 1;
    d = 100000;
    int k = 4;
    nnzperrow = 4;
    int max_nnz = nnzperrow;
    int ppl = M / leaves; 
    bool var_nnz = false;    

    bool print_pt = false;    
    bool print_res = true;    
    int test_leaf = 0;    
    int test_pt = 3;

    int *d_knn_Id, *h_knn_Id, *h_knn_Id_seq;
    float *d_knn, *h_knn, *h_knn_seq;

    h_R = (int *)malloc(sizeof(int)*(M+1));
    h_G_Id = (int *)malloc(sizeof(int)*(M));

    h_knn = (float *)malloc(sizeof(float) * M *k);
    h_knn_Id = (int *)malloc(sizeof(int) * M *k);
    
    h_knn_seq = (float *)malloc(sizeof(float) * k);
    h_knn_Id_seq = (int *)malloc(sizeof(int) * k);

    // generate random data 
    gen_row(M, nnzperrow, h_R,h_G_Id, d, var_nnz);
    int tot_nnz = h_R[M];
		h_V = (float *)malloc(sizeof(float)*tot_nnz);
    h_C = (int *)malloc(sizeof(int)*tot_nnz);
    gen_col_data(M, d , h_R, h_C, h_V);   
    if (print_pt){   
      for (int i = 0; i < M; i++){
          int nnz = h_R[i+1] - h_R[i];
          for (int j = 0; j < nnz; j++){
            printf("R[%d] = %d , C[%d] = %d , V[%d] = %.4f \n", i ,h_R[i], h_R[i]+j, h_C[h_R[i] + j], h_R[i]+j, h_V[h_R[i]+j]);
          }
      }    
    }
    checkCudaErrors(cudaMalloc((void **) &d_R, sizeof(int)*(M+1)));
    checkCudaErrors(cudaMalloc((void **) &d_G_Id, sizeof(int)*(M)));
    checkCudaErrors(cudaMalloc((void **) &d_C, sizeof(int)*tot_nnz));
    checkCudaErrors(cudaMalloc((void **) &d_V, sizeof(float)*tot_nnz));
    checkCudaErrors(cudaMalloc((void **) &d_knn_Id, sizeof(int)*M*k));
    checkCudaErrors(cudaMalloc((void **) &d_knn, sizeof(float)*M*k));
 
    checkCudaErrors(cudaMemcpy(d_C, h_C, sizeof(int)*tot_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_V, h_V, sizeof(float)*tot_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_R, h_R, sizeof(int)*(M+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_G_Id, h_G_Id, sizeof(int)*(M), cudaMemcpyHostToDevice));

    printf("Random csr is generated  \n");
    cudaEvent_t t0; 
    cudaEvent_t t1;
    float del_t1;

    checkCudaErrors(cudaEventCreate(&t0));
    checkCudaErrors(cudaEventCreate(&t1));
    checkCudaErrors(cudaEventRecord(t0, 0));


    TSKNN_gpu(d_R, d_C, d_V, d_G_Id, M, leaves, k, d_knn, d_knn_Id, max_nnz);
    
    
    checkCudaErrors(cudaEventRecord(t1, 0));
    checkCudaErrors(cudaEventSynchronize(t1));
    checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));

    printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);


    checkCudaErrors(cudaMemcpy(h_knn, d_knn, sizeof(float) * M * k, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_knn_Id, d_knn_Id, sizeof(int) * M * k, cudaMemcpyDeviceToHost));  
    checkCudaErrors(cudaMemcpy(h_G_Id, d_G_Id, sizeof(int) * M, cudaMemcpyDeviceToHost)); 

    printf("print %.4f \n", h_knn[96]);
    printf("\n test for leaf %d , pt %d\n",test_leaf, test_pt);  

    f_knnSeq(h_R, h_C, h_V, h_G_Id, h_knn_seq, h_knn_Id_seq, k, test_leaf, test_pt, ppl);
    
    float acc= 0.0;  

    
    int ind;
    bool match;
    int ind0_i = h_R[h_G_Id[test_leaf * ppl + test_pt]];
    int nnz_i = h_R[h_G_Id[test_leaf * ppl + test_pt] + 1] - ind0_i;

    for (int i = 0; i < k; i++){
      ind = test_leaf * k * ppl + test_pt * k + i;
      
      match = (h_knn_Id_seq[i] == h_knn_Id[ind]);
      if (print_res){
        printf("seq ind %d,\t gpu_ind %d , \t match %d , \t v_seq %.4f, \t v_gpu %.4f , \t ind = %d\n", h_knn_Id_seq[i], h_knn_Id[ind], match, h_knn_seq[i], h_knn[ind], ind);
      }
      if (match) acc += 1.0;
    }
    
    acc /= k;    
    printf("\n\naccuracy %.4f for leaf %d\n\n", acc*100, test_leaf);
    
    checkCudaErrors(cudaFree(d_R));
    checkCudaErrors(cudaFree(d_G_Id));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_knn));
    checkCudaErrors(cudaFree(d_knn_Id));
 
    free(h_R);
    free(h_C);
    free(h_V);
    free(h_G_Id);
    free(h_knn);
    free(h_knn_Id);
    free(h_knn_seq);
    free(h_knn_Id_seq);


}
