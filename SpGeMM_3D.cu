
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>





__global__ void compute_dist(int* I, int* J, float* V, float* D, int m, int log_max_nnz, int Z){ 

    int i = threadIdx.x + blockDim.x*blockIdx.x; 
    int j = blockIdx.y ;
    int z = blockIdx.z ;

    if (i>= m || j >= m || z >= Z) return;

    

    int ind0_i = I[z*(m+1) + i];
    int ind1_i = I[z*(m+1) + i+1];
    int ind0_j = I[z*(m+1) + j];
    int ind1_j = I[z*(m+1) + j+1];




    int nnz_i = ind1_i - ind0_i;
    int nnz_j = ind1_j - ind0_j;
    

    if (nnz_j == 0 || nnz_i ==0) return;


    __shared__ int sj[2000];
    //int *sj; 
    int *si;
    si = (int *)malloc(sizeof(int)*nnz_i);
    for (int n_i=0; n_i < nnz_i; n_i++) si[n_i] = J[n_i];
    //sj = (int *)malloc(sizeof(int)*nnz_j);
    for (int n_j=0; n_j < nnz_j; n_j++) sj[n_j] = J[n_j];

    //printf("arrays si : \n")
    /*
    if (i==1 && j==2){ 
    for (int n_i=ind0_i; n_i < ind1_i; n_i++) printf("i = %d , si[%d] = %d \n ",i, n_i, si[n_i]);
    printf("\n\n");
    for (int n_j=ind0_j; n_j < ind1_j; n_j++) printf("j = %d , sj[%d] = %d \n ", j, n_j, sj[n_j]);
    }
    */
    //sj = J[ind0_j:ind1_j];
    //si = J[ind0_i:ind1_i];
    float norm_ij = 0;

    for (int n_i=0; n_i < nnz_i; n_i++) norm_ij += pow(V[ind0_i + n_i], 2);
    for (int n_j=0; n_j < nnz_j; n_j++) norm_ij += pow(V[ind0_j + n_j], 2);

    float c_tmp = 0; 
    //int tmp = log2f(nnz_j);

    for (int pos_k=0; pos_k<nnz_i;pos_k++){

        int k = si[pos_k];
        int ret = 0;
        int testInd = 0;
        int tmp_0,tmp_1;
        // Binary search 
        for (int l=log_max_nnz; l > 0; l--){
            if (pow(2, l) > nnz_j) continue;
            int tmp_0 = ret+pow(2, l);
            int tmp_1 = nnz_j-1;
            testInd = min(tmp_0, tmp_1);
            ret = (sj[testInd] <= k) ? testInd : ret ;
        }
        tmp_0 = ret+1;
        tmp_1 = nnz_j-1;
        testInd = min(tmp_0, tmp_1);
        ret = (sj[testInd] <= k) ? testInd : ret;
        int ind_jk = (sj[ret] == k) ? ret : -1;

        c_tmp += (ind_jk != -1) ? V[ind0_i + pos_k]*V[ind0_j+ind_jk] : 0;
        //c_tmp += c;
    }

    //c_tmp = max(-2*c_tmp + norm_ij, 0)
    int ind = i*m+j+z*pow(m,2);
    D[ind] = c_tmp;

}


int main(int argc, char **argv)
{
  /*
  float t_1, t_2, t_3, wtime;



  t_1 = 0;
  t_2 = 0;
  t_3 = 0;
  wtime = 0;
  */

   float del_t1;
  //, del_t2, del_t3;

    checkCudaErrors(cudaSetDevice(0));

    int nnz, m, d, nnzperrow, Z;
    float *h_V, *d_V;
    int *h_J, *d_J;
    int *h_I, *d_I;
    float *h_D, *d_D; 
    cudaEvent_t t0; 
    cudaEvent_t t1;



    m = 300;
    d = 10000;
    nnzperrow = 50;
    Z = 1;
    nnz = nnzperrow*m;
    int max_nnz = d;
    int log_max_nnz = log2(max_nnz);

    h_V = (float *)malloc(sizeof(float)*nnzperrow*m);
    h_J = (int *)malloc(sizeof(int)*nnzperrow*m);
    h_I = (int *)malloc(sizeof(int)*(m+1));

    // generate random data 

    h_I[0] = 0;
    
    for (int i=0; i < m; i++){
        h_I[i+1] = h_I[i] + nnzperrow;
        int *tmp;
        //tmp_col = (int *)malloc(sizeof(int)*nnzperrow);
        //tmp_val = (float *)malloc(sizeof(float)*nnzperrow);
        for (int j=0; j < nnzperrow; j++){
            int ind = h_I[i]+j;
            int val = rand()%d;
            while (val == h_J[ind-1]) val = rand()%d;
            h_J[ind] = val;
            h_V[ind] = (float)rand()/(float)(RAND_MAX);
        }    
        std::sort(h_J+i*nnzperrow, h_J+(i+1)*nnzperrow);
    }

    h_D = (float *)malloc(sizeof(float)*m*m*Z);

    // prop of execution
    int B =  (256 > m) ? m: 256;
    dim3 dimBlock(B,1,1);	// 512 threads per thread-block
	dim3 dimGrid((m+B-1)/B, m, Z); // Enough thread-blocks to cover N


    // gen device arrays
    checkCudaErrors(cudaMalloc((void **) &d_I, sizeof(int)*(m+1)));
    checkCudaErrors(cudaMalloc((void **) &d_J, sizeof(int)*nnz));
    checkCudaErrors(cudaMalloc((void **) &d_V, sizeof(float)*nnz));
    checkCudaErrors(cudaMalloc((void **) &d_D, sizeof(float)*m*m));

    // copy to device 

    checkCudaErrors(cudaMemcpy(d_I, h_I, sizeof(int)*(m+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_J, h_J, sizeof(int)*(nnz), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_V, h_V, sizeof(float)*(nnz), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_D, 0, sizeof(float) * m * m));

    checkCudaErrors(cudaEventCreate(&t0));
    checkCudaErrors(cudaEventCreate(&t1));

    checkCudaErrors(cudaEventRecord(t0, 0));
    compute_dist<<<dimGrid, dimBlock>>>(d_I, d_J, d_V, d_D, m, log_max_nnz, Z);
    
    checkCudaErrors(cudaEventRecord(t1, 0));
    checkCudaErrors(cudaEventSynchronize(t1));
    //checkCudaErrors(cudaEventSynchronize(t1));
    checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));


    checkCudaErrors(cudaMemcpy(h_D, d_D, sizeof(float)*m*m, cudaMemcpyDeviceToHost));
    printf("\n Elapsed time (ms) : %.4f \n ", del_t1);
    /*
    printf("row pointer : \n ");
    for (int i=0; i < m+1; i++) {
        printf(" %d ", h_I[i]);
    }
    printf("\n \n ");
    printf("col indices : \n "); 
    for (int i=0; i < nnz; i++) {
        printf(" %d ", h_J[i]);
    }
    printf("\n \n ");
    printf("vals : \n "); 
    for (int i=0; i < nnz; i++) {
        printf(" %.4f ", h_V[i]);
    }
    printf("\n \n ");
    printf("out  : \n ");
    for (int i=0; i< m; i++){
        printf("\n [ ");
        for (int j=0; j < m; j++){
            printf(" %.4f ", h_D[i*m + j]);
        }
        printf(" ]");
    }
    printf("\n\n");
    */
    checkCudaErrors(cudaFree(d_I));
    checkCudaErrors(cudaFree(d_J));
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_D));
    checkCudaErrors(cudaEventDestroy(t0));
    checkCudaErrors(cudaEventDestroy(t1));
    free(h_I);
    free(h_J);
    free(h_V);
    free(h_D);


}