
#include "knn_seq_dense.h"
#include "FIKNN_dense.h"
#include "rnd_gen.h"

int main(int argc, char **argv){
    
    checkCudaErrors(cudaSetDevice(0));

    int d;
    float *h_data, *d_data;
    int *h_G_Id, *d_G_Id;
    int M = 1024*1024;
    int leaves = 1024;
    int ppl = M / leaves;
    d = 64;
    int k = 64;
    
    bool print_pt = false;
    bool print_res = false;
    int test_leaf = 0;
    int test_pt = 100;

    int *d_knn_Id, *h_knn_Id, *h_knn_Id_seq;
    float *d_knn, *h_knn, *h_knn_seq;

    h_G_Id = (int *)malloc(sizeof(int)*(M));
    h_data = (float *)malloc(sizeof(int)*(M * d));

    h_knn = (float *)malloc(sizeof(float) * M *k);
    h_knn_seq = (float *)malloc(sizeof(float) *k);
    h_knn_Id = (int *)malloc(sizeof(int) * M *k);
    h_knn_Id_seq = (int *)malloc(sizeof(int) *k);



    gen_rnd_dense(h_data, h_G_Id, M, d);

    printf(" leaves = %d \n", leaves);
    printf(" points/leaf = %d \n", ppl);
    

    checkCudaErrors(cudaMalloc((void **) &d_G_Id, sizeof(int)*(M)));
    checkCudaErrors(cudaMalloc((void **) &d_data, sizeof(float)*M * d));
    checkCudaErrors(cudaMalloc((void **) &d_knn_Id, sizeof(int)*M*k));
    checkCudaErrors(cudaMalloc((void **) &d_knn, sizeof(float)*M*k));

    checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(float)*M * d, cudaMemcpyHostToDevice));
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

    checkCudaErrors(cudaFree(d_G_Id));
    checkCudaErrors(cudaFree(d_data));

    free(h_data);
    free(h_G_Id);
    free(h_knn);
    free(h_knn_Id);
    free(h_knn_seq);
    free(h_knn_Id_seq);


}












