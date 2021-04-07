
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>





__global__ void compute_dist(int* R_I, int* C_I, float* V_I, int* R_J, int* C_J, float* V_J, float* K, int* ID_K, int m_i, int m_j , int max_nnz, int batchID_I, int batchID_J, int size_batch_I, int size_batch_J, int dist_max, int k_nn ){ 

    int i = threadIdx.x + blockDim.x*blockIdx.x; 
    int j = blockIdx.y ;
    //int z = blockIdx.z ;

    



    int row_ID = i / m_j;
    int col_ID = i - row_ID*m_j;

    int subbatch_I = z / size_batch_J;
    int subbatch_J = z - subbatch_I*size_batch_J;

    int ind0_i = R_I[row_ID + subbatch_I*m_i];
    int ind1_i = R_I[row_ID + subbatch_I*m_i + 1];

    int ind0_j = R_J[row_ID + subbatch_J*m_j];
    int ind1_j = R_J[row_ID + subbatch_J*m_j + 1];
 
    int nnz_i = ind1_i - ind0_i;
    int nnz_j = ind1_j - ind0_j;


    float norm_ij = 0;


    for (int n_i=0; n_i < nnz_i; n_i++) norm_ij += pow(V_I[ind0_i + n_i], 2);
    for (int n_j=0; n_j < nnz_j; n_j++) norm_ij += pow(V_J[ind0_j + n_j], 2);
    
    __syncthreads();

    __shared__ int sj[2000];
    si = (int *)malloc(sizeof(int)*nnz_i);

    for (int n_j = 0; n_j < nnz_j; n_j++) sj[n_j] = C_J[ind0_j + n_j];
    for (int n_i = 0; n_i < nnz_i; n_i++) si[n_i] = C_I[ind0_i + n_i];
    

    //printf("run 1 \n ");

    float c_tmp = 0;
    int log_max_nnz = log2(max_nnz);

        
    
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
        //c_tmp += (ind_jk != -1) ? v_i[pos_k]*v_j[ind_jk] : 0;
        c_tmp += (ind_jk != -1) ? V_I[pos_k+ind0_i]*v_j[ind0_j + ind_jk] : 0;
        
        //c_tmp += c;
    }
    
    if (batchID_I == batchID_J && subbatch_I == subbatch_J && row_ID == col_ID) c_tmp = dist_max; 

    //c_tmp = max(-2*c_tmp + norm_ij, 0);
    //printf("D[%d, %d] = %.2f for z = %d and ind = %d \n", i, j, c_tmp, ind_out, z); 
    //D[ind_out] = c_tmp;

    int g_col = subbatch_J*m_j + col_ID;

    

    // bitonic sort 
    __shared__ float sj[2000];
    __shared__ int id_k[2000];

    sj[i] = c_tmp; 
    id_k[i] = g_col;

    __syncthreads();

    int log_size = log2(m_j);
    int size = (pow(2,log_size) < m_j) ? pow(2, log_size+1) : m_j;
    
    // bitonic sort  
    
    for (int g = 2; g <= size; g *= 2){
      for (int l = g/2; l>0; l /= 2){
	int ixj = i ^ l;
	if (ixj > i){
		if ((i & g) == 0){
			if (sj[i] > sj[ixj]) std::swap(sj[i], sj[ixj]);
		} else {
			if (sj[i] < sj[ixj]) std::swap(sj[i], sj[ixj]);
		}
	}
	__syncthreads();
      }
     }

    int diff = size-m_j; 
    if (col_ID >= diff && col_ID < k_nn + diff){
	    int col_local = subbatch_J*k_nn + col_ID - diff;
	    int row_local = subbatch_I*m_i + row_ID; 
	    int ind_ij = row_local*size_batch_J*k_nn + col_local;
	    K[ind_ij] = sj[i];
	    ID_K[ind_ij] = id_k[i];
    }

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
    float *knn, *knn_local, *d_knn_local;
    int *ID_knn, *ID_knn_local, *d_ID_knn_local;
    int *h_nnz_all, *d_nnz_all; 
    cudaEvent_t t0; 
    cudaEvent_t t1;


    int M_I = 1028;
    int d = 100;
    int nnzperrow = 16;
    int size_batch_I = 64;
    int size_batch_J = 64;
    int m_i = 4;
    int m_j = 4;
    int k = 2;
    float dist_max = 1000
    int max_nnz = 2000;
    int log_max_nnz = log2(max_nnz);

    int num_I = M_I / (m_i*size_batch_I);
    int num_J = M_I / (m_j*size_batch_J);

    h_V = (float *)malloc(sizeof(float)*nnzperrow*M_I);
    h_J = (int *)malloc(sizeof(int)*nnzperrow*M_I);
    h_I = (int *)malloc(sizeof(int)*(M_I+1));

    // generate random data 

    
    
    for (int z=0; z < Z; z++){ 
        h_I[0+z*(m+1)] = 0;
        for (int i=0; i < m; i++){
	nnz_r = nnzperrow;
        h_I[i+1+z*(m+1)] = h_I[i+z*(m+1)] + nnz_r;
        
        //tmp_col = (int *)malloc(sizeof(int)*nnzperrow);
        //tmp_val = (float *)malloc(sizeof(float)*nnzperrow);
        for (int j=0; j < nnz_r; j++){
            int ind = h_I[i]+j + z*(nnz);
            int val = rand()%d;
            while (val == h_J[ind-1]) val = rand()%d;
            h_J[ind] = val;
            h_V[ind] = j;
            //(float)rand()/(float)(RAND_MAX); 
            //
        }    
        std::sort(h_J+i*nnz_r, h_J+(i+1)*nnz_r);
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
    knn = (float *)malloc(sizeof(float)*M_I*k);
    knn_local = (float *)malloc(sizeof(float)*m_i*k);
    ID_knn = (int *)malloc(sizeof(int)*M_I*k);
    ID_knn_local = (int *)malloc(sizeof(int)*m_i*size_batch_I*k);

    for (int n =0; n < *M_I*k; n++) knn_local[n] = dist_max;
    checkCudaErrors(cudaMalloc((void **) &d_C_I, sizeof(int)*nnzperrow*M_I));
    checkCudaErrors(cudaMalloc((void **) &d_V_I, sizeof(float)*(M_I+1)));
    checkCudaErrors(cudaMemcpy(d_C_I, C_I, sizeof(int)*nnzperrow*M_I, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_V_I, V_I, sizeof(float)*nnzperrow*M_I, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_knn_local, sizeof(float)*m_i*k));
    checkCudaErrors(cudaMalloc((void **) &d_ID_knn_local, sizeof(float)*m_i*size_batch_I*k));
    checkCudaErrors(cudaMemcpy(d_ID_knn_local, ID_knn_local, sizeof(int)*k*m_i, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_knn_local, knn_local, sizeof(float)*k*m_i, cudaMemcpyHostToDevice));

    for (int batch_I=0; batch_I<num_I; batch_I++) {
      for (int batch_J = 0; batch_J < num_J; batch_J++){
        R_I = (float *)malloc(sizeof(float)*size_batch_I*m_i);
        R_J = (float *)malloc(sizeof(float)*size_batch_J*m_j);
        
        for (int i =0; i<size_batch_I*m_i) R_I[i] = h_I[batch_I*size_batch_I*m_i+i];	
        for (int j =0; i<size_batch_J*m_j) R_J[j] = h_I[batch_J*size_batch_J*m_j+j];	


        //h_D = (float *)malloc(sizeof(float)*m*m*Z);

    // prop of execution
    int B =  m_i*m_j;
    dim3 dimBlock(B,1);	// 512 threads per thread-block
    dim3 dimGrid(1, size_batch_I*size_batch_J); // Enough thread-blocks to cover N


    // gen device arrays
    checkCudaErrors(cudaMalloc((void **) &d_R_I, sizeof(int)*size_batch_I*m_i));
    checkCudaErrors(cudaMalloc((void **) &d_R_J, sizeof(int)*size_batch_J*m_j));

    // copy to device 
    checkCudaErrors(cudaMemcpy(d_D, h_D, sizeof(float)*m*m*Z, cudaMemcpyHostToDevice));


    //checkCudaErrors(cudaMemset(d_D, 0, sizeof(float) * m * m * Z));

    checkCudaErrors(cudaEventCreate(&t0));
    checkCudaErrors(cudaEventCreate(&t1));

    checkCudaErrors(cudaEventRecord(t0, 0));
    compute_dist<<<dimGrid, dimBlock>>>(d_R_I, d_C_J, d_V_J, d_R_J, d_C_I, d_V_I, d_knn_local, d_ID_knn_local, m_i, m_j, max_nnz, batch_I, batch_J, size_batch_I, size_batch_J, dist_max, k);
    checkCudaErrors(cudaEventRecord(t1, 0));
    checkCudaErrors(cudaEventSynchronize(t1));
    //checkCudaErrors(cudaEventSynchronize(t1));
    checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));

    }
    checkCudaErrors(cudaMemcpy(knn_local, d_knn_local, sizeof(float)*m_i*k, cudaMemcpyDeviceToHost));
    printf("\n Elapsed time (ms) : %.4f \n ", del_t1);
    }
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
    checkCudaErrors(cudaFree(d_R_I));
    checkCudaErrors(cudaFree(d_R_J));
    checkCudaErrors(cudaFree(d_C_I));
    checkCudaErrors(cudaFree(d_C_J));
    checkCudaErrors(cudaFree(d_V_I));
    checkCudaErrors(cudaFree(d_V_J));
    checkCudaErrors(cudaFree(d_ID_knn_local));
    checkCudaErrors(cudaFree(d_knn_local));
    checkCudaErrors(cudaEventDestroy(t0));
    checkCudaErrors(cudaEventDestroy(t1));
    free(h_I);
    free(h_J);
    free(h_V);


}
