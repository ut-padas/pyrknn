
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>





__global__ void compute_dist(int* I, int* J, float* V, float* D, int m, int* nnz_vec, int log_max_nnz, int Z){ 

    int i = threadIdx.x + blockDim.x*blockIdx.x; 
    int j = blockIdx.y ;
    int z = blockIdx.z ;

    if (i>= m || j >= m || z >= Z) return;
    if (i == j) return;
    

    int ind0_i = I[z*(m+1) + i];
    int ind1_i = I[z*(m+1) + i+1];
    int ind0_j = I[z*(m+1) + j];
    int ind1_j = I[z*(m+1) + j+1];
    int nnz = nnz_vec[z];


    int nnz_i = ind1_i - ind0_i;
    int nnz_j = ind1_j - ind0_j;
    int ind_out = i*m + j + z*pow(m, 2);
    //float *val_out;
    //val_out = &D[ind_out];
    //printf("run 1 \n ");

    if (nnz_j == 0 || nnz_i ==0) return;

    
    __shared__ int sj[1000];
    __shared__ float v_j[1000];

    int si[1000];
    float v_i[1000];
    //si = (int *)malloc(sizeof(int)*nnz_i);
    

    //printf("run 1 \n"); 
    for (int n_i=0; n_i < nnz_i; n_i++) si[n_i] = J[ind0_i + n_i + z*nnz];
    for (int n_i=0; n_i < nnz_i; n_i++) v_i[n_i] = V[ind0_i + n_i + z*nnz]; 
        
    //sj = (int *)malloc(sizeof(int)*nnz_j);
    for (int n_j=0; n_j < nnz_j; n_j++) v_j[n_j] = V[ind0_j + n_j + z*nnz];
    sj[i] = J[ind0_j + i + z*nnz];
    //v_j[i] = V[ind0_j + j + z*nnz];
    
    __syncthreads();
    
    //printf("run 2 \n ");
    //sj = J[ind0_j:ind1_j];
    //si = J[ind0_i:ind1_i];
    float norm_ij = 0;
    
    for (int n_i=0; n_i < nnz_i; n_i++) norm_ij += pow(V[ind0_i + n_i + z*nnz], 2);
    for (int n_j=0; n_j < nnz_j; n_j++) norm_ij += pow(V[ind0_j + n_j + z*nnz], 2);
    //printf("run 3 \n ");
    float c_tmp = 0; 
    //int tmp = log2f(nnz_j);
    int ret, testInd, tmp_0, tmp_1, k, l, ind_jk;
        
    
    for (int pos_k=0; pos_k<nnz_i;pos_k++){       
        //int k = J[pos_k+ind0_i + z*nnz];
        k = si[pos_k];
        ret=0; 
        testInd = 0;
        
        // Binary search 
        for (l=log_max_nnz; l > 0; l--){
            if (pow(2, l) > nnz_j) continue;
            tmp_0 = ret+pow(2, l);
            tmp_1 = nnz_j-1;
            testInd = min(tmp_0, tmp_1);
            ret = (sj[testInd] <= k) ? testInd : ret ;
        }
        tmp_0 = ret+1;
        tmp_1 = nnz_j-1;
        testInd = min(tmp_0, tmp_1);
        ret = (sj[testInd] <= k) ? testInd : ret;
        ind_jk = (sj[ret] == k) ? ret : -1;
        //if (i==1 && j==2) printf("i = %d, j = %d , k = %d, pos_k = %d, ind_jk = %d , v_i = %.2f, v_j = %.2f \n",i,j,k,pos_k, ind_jk, v_i[pos_k], v_j[ind_jk]);
        //c_tmp += (ind_jk != -1) ? V[ind0_i + pos_k + z*nnz]*V[ind0_j+ind_jk + z*nnz] : 0;
        c_tmp += (ind_jk != -1) ? v_i[pos_k]*v_j[ind_jk] : 0;
        
        //c_tmp += c;
    }
    //c_tmp = max(-2*c_tmp + norm_ij, 0);
    //printf("D[%d, %d] = %.2f for z = %d and ind = %d \n", i, j, c_tmp, ind_out, z); 
    D[ind_out] = c_tmp;
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
    int *h_nnz_all, *d_nnz_all; 
    cudaEvent_t t0; 
    cudaEvent_t t1;



    m = 200;
    d = 10000;
    nnzperrow = 400;
    Z = 1;
    nnz = nnzperrow*m;
    int max_nnz = 2000;
    int log_max_nnz = log2(max_nnz);

    h_V = (float *)malloc(sizeof(float)*nnz*Z);
    h_J = (int *)malloc(sizeof(int)*nnz*Z);
    h_I = (int *)malloc(sizeof(int)*(m+1)*Z);
    h_nnz_all = (int *)malloc(sizeof(int) *Z);

    // generate random data 

    
    
    for (int z=0; z < Z; z++){ 
        h_I[0+z*(m+1)] = 0;
        h_nnz_all[z] = nnz;
        for (int i=0; i < m; i++){
        h_I[i+1+z*(m+1)] = h_I[i+z*(m+1)] + nnzperrow;
        
        //tmp_col = (int *)malloc(sizeof(int)*nnzperrow);
        //tmp_val = (float *)malloc(sizeof(float)*nnzperrow);
        for (int j=0; j < nnzperrow; j++){
            int ind = h_I[i]+j + z*(nnz);
            int val = rand()%d;
            while (val == h_J[ind-1]) val = rand()%d;
            h_J[ind] = val;
            h_V[ind] = j;
            //(float)rand()/(float)(RAND_MAX); 
            //
        }    
        std::sort(h_J+i*nnzperrow+ z*nnz, h_J+(i+1)*nnzperrow+z*nnz);
    }
    }
    /*
    for (int z=0; z < Z; z++){
        printf("\n z = %d \n\n", z);
    printf("row pointer : \n ");
    for (int i=0; i < m+1; i++) {
        printf(" %d ", h_I[i+z*(m+1)]);
    }
    printf("\n \n ");
    printf("col indices : \n "); 
    for (int i=0; i < nnz; i++) {
        printf(" %d ", h_J[i+z*(nnz)]);
    }
    printf("\n \n ");
    printf("vals : \n "); 
    for (int i=0; i < nnz; i++) {
        printf(" %.4f ", h_V[i+z*nnz]);
    }
    }*/
    



    h_D = (float *)malloc(sizeof(float)*m*m*Z);

    // prop of execution
    int B =  (32 > m) ? m: 32;
    dim3 dimBlock(B,1,1);	// 512 threads per thread-block
    int threadblock = (m+B-1)/B;
	  dim3 dimGrid(threadblock, m, Z); // Enough thread-blocks to cover N


    // gen device arrays
    checkCudaErrors(cudaMalloc((void **) &d_I, sizeof(int)*(m+1)*Z));
    checkCudaErrors(cudaMalloc((void **) &d_J, sizeof(int)*nnz*Z));
    checkCudaErrors(cudaMalloc((void **) &d_V, sizeof(float)*nnz*Z));
    checkCudaErrors(cudaMalloc((void **) &d_nnz_all, sizeof(int)*Z));
    checkCudaErrors(cudaMalloc((void **) &d_D, sizeof(float)*m*m*Z));

    // copy to device 

    checkCudaErrors(cudaMemcpy(d_I, h_I, sizeof(int)*(m+1)*Z, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_J, h_J, sizeof(int)*(nnz)*Z, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_nnz_all, h_nnz_all, sizeof(int)*Z, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_V, h_V, sizeof(float)*(nnz)*Z, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_D, h_D, sizeof(float)*m*m*Z, cudaMemcpyHostToDevice));

    //checkCudaErrors(cudaMemset(d_D, 0, sizeof(float) * m * m * Z));

    checkCudaErrors(cudaEventCreate(&t0));
    checkCudaErrors(cudaEventCreate(&t1));

    checkCudaErrors(cudaEventRecord(t0, 0));
    compute_dist<<<dimGrid, dimBlock>>>(d_I, d_J, d_V, d_D, m, d_nnz_all, log_max_nnz, Z);
    
    checkCudaErrors(cudaEventRecord(t1, 0));
    checkCudaErrors(cudaEventSynchronize(t1));
    //checkCudaErrors(cudaEventSynchronize(t1));
    checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));


    checkCudaErrors(cudaMemcpy(h_D, d_D, sizeof(float)*m*m*Z, cudaMemcpyDeviceToHost));
    printf("\n Elapsed time (ms) : %.4f \n ", del_t1);
    /*
    for (int z=0; z < Z; z++){
        printf(" z = %d \n\n", z);
    printf("row pointer : \n ");
    for (int i=0; i < m+1; i++) {
        printf(" %d ", h_I[i+z*(m+1)]);
    }
    printf("\n \n ");
    printf("col indices : \n "); 
    for (int i=0; i < nnz; i++) {
        printf(" %d ", h_J[i+z*(nnz)]);
    }
    printf("\n \n ");
    printf("vals : \n "); 
    for (int i=0; i < nnz; i++) {
        printf(" %.4f ", h_V[i+z*nnz]);
    }
    printf("\n \n ");
    printf("out  : \n ");
    for (int i=0; i< m; i++){
        printf("\n [ ");
        for (int j=0; j < m; j++){
            printf(" %.4f ", h_D[i*m + j + z*m*m]);
        }
        printf(" ]");
    }
    
    printf("\n");
    }
    */
    printf("\n\n");
    checkCudaErrors(cudaFree(d_I));
    checkCudaErrors(cudaFree(d_J));
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_D));
    checkCudaErrors(cudaFree(d_nnz_all));
    checkCudaErrors(cudaEventDestroy(t0));
    checkCudaErrors(cudaEventDestroy(t1));
    free(h_I);
    free(h_nnz_all);
    free(h_J);
    free(h_V);
    free(h_D);


}
