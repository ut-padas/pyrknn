
#include "knn_seq.h"
#include "FIKNN_RecB.h"
#include "rnd_sparse.h"

int main(int argc, char **argv){
    
    checkCudaErrors(cudaSetDevice(0));

    int d;
    float *h_data, *d_data;
    int *h_G_Id, *d_G_Id;
    int M = 4096*2048;
    int leaves = 2048;
    int ppl = M / leaves;
    d = 128;
    int k = 32;
    
    bool print_pt = false;
    bool print_res = true;
    int test_leaf = 0;
    int test_pt = 3;

    int *d_knn_Id, *h_knn_Id, *h_knn_Id_seq;
    float *d_knn, *h_knn, *h_knn_seq;

    h_G_Id = (int *)malloc(sizeof(int)*(M));

    h_knn = (float *)malloc(sizeof(float) * M *k);
    h_knn_seq = (float *)malloc(sizeof(float) *k);
    h_knn_Id = (int *)malloc(sizeof(int) * M *k);
    h_knn_Id_seq = (int *)malloc(sizeof(int) *k);



    gen_rnd_dense(h_data, M, d);

    printf(" leaves = %d \n", leaves);
    printf(" points/leaf = %d \n", ppl);
    

    checkCudaErrors(cudaMalloc((void **) &d_G_Id, sizeof(int)*(M)));
    checkCudaErrors(cudaMalloc((void **) &d_data, sizeof(float)*M));
    checkCudaErrors(cudaMalloc((void **) &d_knn_Id, sizeof(int)*M*k));
    checkCudaErrors(cudaMalloc((void **) &d_knn, sizeof(float)*M*k));

    checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(float)*M, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_G_Id, h_G_Id, sizeof(int)*(M), cudaMemcpyHostToDevice));


    printf("Random csr is generated  \n");
    cudaEvent_t t0;
    cudaEvent_t t1;
    float del_t1;

    checkCudaErrors(cudaEventCreate(&t0));
    checkCudaErrors(cudaEventCreate(&t1));
    checkCudaErrors(cudaEventRecord(t0, 0));
    FIKNN_gpu_dense(d_data, d_G_Id, M, leaves, k, d_knn, d_knn_Id, d);    

    checkCudaErrors(cudaEventRecord(t1, 0));
    checkCudaErrors(cudaEventSynchronize(t1));
    checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));

    printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);

    checkCudaErrors(cudaMemcpy(h_knn, d_knn, sizeof(float) * M * k, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_knn_Id, d_knn_Id, sizeof(int) * M * k, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_G_Id, d_G_Id, sizeof(float) * M, cudaMemcpyDeviceToHost));

    printf(" \n running Seq knn \n");
    printf("\n test for leaf %d , pt %d\n",test_leaf, test_pt);

    f_knnSeq_dense(h_data, h_G_Id, h_knn_seq, h_knn_Id_seq, k, test_leaf, test_pt, ppl, d);

    float acc= 0.0;


    int ind;
    bool match;
    int counter = 0;
    int gpu_pt,seq_pt,ind_seq,ind_gpu, nnz_gpu,nnz_seq;
    int ind0_i = h_R[h_G_Id[test_leaf * ppl + test_pt]];
    int nnz_i = h_R[h_G_Id[test_leaf * ppl + test_pt] + 1] - ind0_i;


    for (int i = 0; i < k; i++){
      ind = test_leaf * k * ppl + test_pt * k + i;
      match = (h_knn_Id_seq[i] == h_knn_Id[ind]);
      if (print_res){
      printf("seq ind %d,\t gpu_ind %d , \t match %d , \t v_seq %.4f, \t v_gpu %.4f , \t ind = %d\n", h_knn_Id_seq[i], h_knn_Id[ind], match, h_knn_seq[i], h_knn[ind], ind);
      }
      if (match) acc += 1.0;
      if (counter < 2 && match==0) {
        counter++;
        gpu_pt = h_knn_Id[ind];
        seq_pt = h_knn_Id_seq[i];
        ind_gpu = h_R[gpu_pt];
        ind_seq = h_R[seq_pt];
        nnz_gpu = h_R[gpu_pt + 1]  - h_R[gpu_pt];
        nnz_seq = h_R[seq_pt + 1]  - h_R[seq_pt];


    }
    }

    acc /= k;
    printf("\n\naccuracy %.4f for leaf %d\n\n", acc*100, test_leaf);

    checkCudaErrors(cudaFree(d_R));
    checkCudaErrors(cudaFree(d_G_Id));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_V));

    free(h_R);
    free(h_C);
    free(h_V);
    free(h_G_Id);
    free(h_knn);
    free(h_knn_Id);
    free(h_knn_seq);
    free(h_knn_Id_seq);


}












