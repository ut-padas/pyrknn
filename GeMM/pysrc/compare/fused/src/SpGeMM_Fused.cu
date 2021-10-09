
#include "sfiknn.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <limits.h>


__global__ void compute_norm(int* R, int* C, float* V, int* G_Id, float* Norms, int ppl) {

  int tid = threadIdx.x;
  int leafId_g = blockIdx.x;
  
  for (int row = tid; row < ppl; row += blockDim.x){
 
    int g_rowId = leafId_g * ppl + row;
 
    int g_Id = g_rowId; 
    int ind0_i = R[g_Id];
 
    int nnz = R[g_Id + 1] - ind0_i;
    float norm_i = 0.0;
  
    for (int n_i = 0; n_i < nnz; n_i++) norm_i += V[ind0_i + n_i] * V[ind0_i + n_i];
    int ind_write = leafId_g * ppl + row;
    Norms[ind_write] = norm_i;
  }

}

__global__ void knn_kernel_tri(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int m) {

    __shared__ int SM_Id_h[4096];
    __shared__ float SM_dist_h[4096];
   
 
		int tid_x = threadIdx.x;
		int tid_y = threadIdx.y;

		int block = blockIdx.x;
		int leafId_g = blockIdx.y;

		float norm_y, norm_xy;
    int shift = (k_nn > m) ? 2 * k_nn : 2 *m;

    int ind = tid_x * blockDim.y + tid_y;
    for (int n_y = ind; n_y < 2048; n_y += blockDim.y*blockDim.x) {
      SM_dist_h[n_y] = 1e30;
      SM_Id_h[n_y] = -1;
    }

    __syncthreads();

		int k, ret, testInd, ind_jk, tmp_0, tmp_1;
	 
    int rowId = tid_y;
    int colId = tid_x;

		//for (int rowId = tid_y; rowId < m; rowId += blockDim.y){
		
			int g_rowId = leafId_g * ppl + block * m + rowId;
			int ind0_y = R[g_rowId];
			int ind1_y = R[g_rowId+1];

			int nnz_y = ind1_y - ind0_y;
			norm_y = Norms[g_rowId];

			//for (int colId = tid_x; tid_x < m; tid_x += blockDim.x){
				if (colId >= rowId){
					float c_tmp = 0.0;
					int g_colId = leafId_g * ppl + block * m + colId;
					int ind0_x = R[g_colId];
					int ind1_x = R[g_colId+1];

					int nnz_x = ind1_x - ind0_x;
					norm_xy = norm_y + Norms[g_rowId];
					
					ret = 0;
					testInd = 0;
					if (nnz_x >0 && nnz_y > 0){
						for (int pos_k = 0; pos_k < nnz_x; pos_k++){

							k = C[ind0_x + pos_k];

							// Binary search
							for (int l = nnz_y - ret; l > 1; l -= floorf(l/2.0)){
								tmp_0 = ret + l;
								tmp_1 = nnz_y - 1;
								testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
								ret = (C[ind0_y + testInd] <= k) ? testInd : ret;
							}

							tmp_0 = ret + 1;
							tmp_1 = nnz_y - 1;
							testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;

							//ret = (C[testInd + ind0_i] <= k) ? testInd : ret;
							ret = (C[ind0_y + testInd] <= k) ? testInd : ret;

							//ind_jk = (C[ret + ind0_i] == k) ? ret : -1;
							ind_jk = (C[ind0_y + ret] == k) ? ret : -1;
							c_tmp += (ind_jk != -1) ? V[ind0_x + pos_k] * V[ind0_y + ind_jk] : 0;

						}
					}
					c_tmp = -2 * c_tmp + norm_xy;
					c_tmp = (c_tmp > 1e-8) ? c_tmp : 0.0;            
					 
					SM_dist_h[rowId * shift + colId] = c_tmp;
					SM_Id_h[rowId * shift + colId] = g_colId;
   
	 				if (colId > rowId) SM_dist_h[colId * shift + rowId] = c_tmp;
					if (colId > rowId) SM_Id_h[colId * shift + rowId] = c_tmp;

				}
		//	}
	//	}

		__syncthreads();
   for (int n_x = tid_x; n_x < k_nn; n_x += blockDim.x) {
 
     SM_dist_h[tid_y * shift + n_x + m] = KNN_dist[G_Id[leafId_g * ppl + block * m + tid_y] * k_nn + n_x];
     SM_Id_h[tid_y * shift + n_x + m] = KNN_Id[G_Id[leafId_g * ppl + block * m + tid_y] * k_nn + n_x];
     
   }


    int size = 2 * k_nn;
    float tmp_f; 
    int tmp_i; 
    for (int g = 2; g <= size; g *= 2){
      for (int l = g /2; l > 0; l /= 2){

        // horizontal

        for (int x = tid_x; x < size; x += blockDim.x){

          int ixj = x ^ l;

          //int rowId = tid_y;
          if (ixj > x){
            if ((x & g) == 0){
              if (SM_dist_h[rowId * shift + x] > SM_dist_h[rowId * shift +ixj]){

                tmp_f = SM_dist_h[rowId * shift +ixj];
                SM_dist_h[rowId * shift +ixj] = SM_dist_h[rowId * shift +x];
                SM_dist_h[rowId * shift +x] = tmp_f;

                tmp_i = SM_Id_h[rowId * shift +ixj];
                SM_Id_h[rowId * shift +ixj] = SM_Id_h[rowId * shift +x];
                SM_Id_h[rowId * shift +x] = tmp_i;

              }
            } else {
              if (SM_dist_h[rowId * shift +x] < SM_dist_h[rowId * shift +ixj]){

                tmp_f = SM_dist_h[rowId * shift +ixj];
                SM_dist_h[rowId * shift +ixj] = SM_dist_h[rowId * shift +x];
                SM_dist_h[rowId * shift +x] = tmp_f;

                tmp_i = SM_Id_h[rowId * shift +ixj];
                SM_Id_h[rowId * shift +ixj] = SM_Id_h[rowId * shift +x];
                SM_Id_h[rowId * shift +x] = tmp_i;

              }
            }
          }
        }


        __syncthreads();
      }
    }


   //int g_rowId = leafId_g * ppl + block * m + tid_y; 
   for (int n_x = tid_x; n_x < k_nn; n_x += blockDim.x){
     int ind_knn = G_Id[g_rowId] * k_nn + n_x;
     KNN_dist[ind_knn] = SM_dist_h[tid_y * shift + n_x];
     KNN_Id[ind_knn] = SM_dist_h[tid_y * shift + n_x];
   }
    
  
}





__global__ void knn_kernel_sq(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int m, int blockInd, int* block_indices) {

   // square partitions

   __shared__ float SMDist_v[2048];
   __shared__ int SMId_v[2048];

   __shared__ float SMDist_h[2048];
   __shared__ int SMId_h[2048];


   int i = threadIdx.x;
   int j = threadIdx.y;


   int ind = i * blockDim.y + j;
   for (int n_i = ind; n_i < 2048; n_i+= blockDim.x*blockDim.y){
     SMDist_v[n_i] = 1e30; 
     SMDist_h[n_i] = 1e30; 
   }
   // reading block indices;
   int N = ppl / m;
      
   int b_i = block_indices[blockInd * N + 2 * blockIdx.x]; 
   int b_j = block_indices[blockInd * N + 2 * blockIdx.x +1];
   int leafId_g = blockIdx.y;
   //if (leafId_g ==0 && ind == 0) printf("blockInd = %d, b_i = %d, b_j = %d \n", blockInd, b_i, b_j);
   

   int g_rowId = leafId_g * ppl + b_i * m + i;
   int g_colId = leafId_g * ppl + b_j * m + j;
   

   int ind0_i = R[g_rowId];
   int ind1_i = R[g_rowId+1];
   int nnz_i = ind1_i - ind0_i;

   int ind0_j = R[g_colId];
   int ind1_j = R[g_colId+1];
   int nnz_j = ind1_j - ind0_j;
    
   float norm_ij = Norms[g_rowId] + Norms[g_colId];

   
		float c_tmp = 0.0;
		int tmp_0, tmp_1, ind_jk, k, ret, testInd;

		ret = 0;
		testInd = 0;


		// loop over the elements of j
		for (int pos_k = 0; pos_k < nnz_j; pos_k++){
			//k = SM_col[max_nnz * j + pos_k];
      k = C[ind0_j + pos_k];
			// Binary search
			for (int l = nnz_i - ret; l > 1; l /= 2){
				tmp_0 = ret + l;
				tmp_1 = nnz_i - 1;
				testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
				ret = (C[testInd + ind0_i] <= k) ? testInd : ret;
			}

			tmp_0 = ret + 1;
			tmp_1 = nnz_i - 1;
			testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;
			ret = (C[testInd + ind0_i] <= k) ? testInd : ret;
			ind_jk = (C[ret + ind0_i] == k) ? ret : -1;
			c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;
    }
		c_tmp = -2 * c_tmp + norm_ij;
		c_tmp = ( c_tmp > 0) ? sqrt(c_tmp) : 0.0;
   int rowId = i;
   int colId = j;
   int shift_h = rowId * 2 * k_nn;
   int shift_v = colId * 2 * k_nn;

   SMDist_v[shift_v + rowId] = c_tmp;
   SMDist_h[shift_h + colId] = c_tmp;
   SMId_v[shift_v + rowId] = G_Id[g_rowId];
   SMId_h[shift_h + colId] = G_Id[g_colId];
   
   __syncthreads();


   for (int n_i = i; n_i < k_nn; n_i += blockDim.x) {
     SMDist_v[shift_v + n_i + k_nn] = KNN_dist[G_Id[g_colId]*k_nn + n_i];
     SMId_v[shift_v + n_i + k_nn] = KNN_Id[G_Id[g_colId]*k_nn + n_i];
   }

   for (int n_j = j; n_j < k_nn; n_j += blockDim.x) {
     SMDist_h[shift_h + n_j + k_nn] = KNN_dist[G_Id[g_rowId]*k_nn + n_j];
     SMId_h[shift_h + n_j + k_nn] = KNN_Id[G_Id[g_rowId]*k_nn + n_j];
   }

   __syncthreads();

    
  // bitonic sort

	int size = 2 * k_nn;

	float tmp_f;
	int tmp_i;
	//size = 2 * m;
	//int ind = i * m + j;
	
  	
	for (int g = 2; g <= size; g *= 2){
		for (int l = g /2; l > 0; l /= 2){

			// horizontal

			for (int x = j; x < size; x += blockDim.y){

				int ixj = x ^ l;

				if (ixj > x){
					if ((x & g) == 0){
						if (SMDist_h[shift_h + x] > SMDist_h[shift_h +ixj]){

							tmp_f = SMDist_h[shift_h +ixj];
							SMDist_h[shift_h +ixj] = SMDist_h[shift_h +x];
							SMDist_h[shift_h +x] = tmp_f;

							tmp_i = SMId_h[shift_h +ixj];
							SMId_h[shift_h +ixj] = SMId_h[shift_h +x];
							SMId_h[shift_h +x] = tmp_i;

						}
					} else {
						if (SMDist_h[shift_h +x] < SMDist_h[shift_h +ixj]){

							tmp_f = SMDist_h[shift_h +ixj];
							SMDist_h[shift_h +ixj] = SMDist_h[shift_h +x];
							SMDist_h[shift_h +x] = tmp_f;

							tmp_i = SMId_h[shift_h +ixj];
							SMId_h[shift_h +ixj] = SMId_h[shift_h +x];
							SMId_h[shift_h +x] = tmp_i;

						}
					}
				}
			}

			// vertical 
			for (int y = i; y < size; y += blockDim.x){

				int ixj = y ^ l;

				if (ixj > y){
					if ((y & g) == 0){
						if (SMDist_v[shift_v + y] > SMDist_v[shift_v +ixj]){

							tmp_f = SMDist_v[shift_v +ixj];
							SMDist_v[shift_v +ixj] = SMDist_v[shift_v +y];
							SMDist_v[shift_v +y] = tmp_f;

							tmp_i = SMId_v[shift_v +ixj];
							SMId_v[shift_v +ixj] = SMId_v[shift_v +y];
							SMId_v[shift_v +y] = tmp_i;

						}
					} else {
						if (SMDist_v[shift_v +y] < SMDist_v[shift_v +ixj]){

							tmp_f = SMDist_v[shift_v +ixj];
							SMDist_v[shift_v +ixj] = SMDist_v[shift_v +y];
							SMDist_v[shift_v +y] = tmp_f;

							tmp_i = SMId_v[shift_v +ixj];
							SMId_v[shift_v +ixj] = SMId_v[shift_v +y];
							SMId_v[shift_v +y] = tmp_i;

						}
					}
				}
			}


			__syncthreads();
		}
	}

 for (int n_i = i; n_i < k_nn; n_i += blockDim.x){  
   KNN_dist[G_Id[g_colId] * k_nn + n_i] = SMDist_v[shift_v + n_i];
   KNN_Id[G_Id[g_colId] * k_nn + n_i] = SMId_v[shift_v + n_i];
 }


 for (int n_j = j; n_j < k_nn; n_j += blockDim.y){  
   KNN_dist[G_Id[g_rowId] * k_nn + n_j] = SMDist_h[shift_h + n_j];
   KNN_Id[G_Id[g_rowId] * k_nn + n_j] = SMId_h[shift_h + n_j];
 }

 	
}


void par_block_indices(int N, int* d_arr)
{


  int* vals;
  vals = (int *)malloc(sizeof(int) * N * (N-1));

  int elem = -1;
  /*
  do {

     for (int i = 0; i < N; i++)
     {
       if (bitmask[i]) {
       elem++;
       vals[elem] = i;
       }
     }
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
  */
  for (int m = 0; m < N; m++){
    for (int n = m+1; n < N; n++){
      elem++;
      vals[elem] = m;
      elem++;
      vals[elem] = n;
    }
  }

  int *arr;
  arr = (int *)malloc(sizeof(int) * N * (N-1));  


  int* track;
  track = (int *) malloc(sizeof(int) * (N) * (N-1));
  memset(track, 0, sizeof(int) * N * (N-1));

  int* row_mems;
  row_mems = (int *) malloc(sizeof(int) * (N-1));
  memset(row_mems, 0, sizeof(int) * (N-1));

  for (int mems = 0; mems < (N/2) * (N-1); mems++){

    int i = vals[2*mems];
    int j = vals[2*mems+1];
    int row = 0;
    //printf("(%d, %d) -> ", i,j);
    bool ex_f = false;
    while (! ex_f) {
      if (track[row * N + i] == 0 && track[row * N + j] == 0){
        int start = row_mems[row];

        arr[row * N + start] = i;
        arr[row * N + start+1] = j;
        track[row * N + i] = 1;
        track[row * N + j] = 1;
        row_mems[row] +=2;
        ex_f = true;
        //printf("at row %d loc = %d \n", row, start);
      }
      row++;
    }
  }
 
  checkCudaErrors(cudaMemcpy(d_arr, arr, sizeof(int)*N * (N-1), cudaMemcpyHostToDevice));  
  

}







void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id){
 
	int ppl = M/leaves;

  cudaEvent_t t_begin, t_end, t_tri, t0_sq,t1_sq, t_ext;

  float dt_tmp, dt_tot, dt_tri, dt_sq;
   

  checkCudaErrors(cudaEventCreate(&t_begin));
  checkCudaErrors(cudaEventCreate(&t_end));
  checkCudaErrors(cudaEventCreate(&t_tri));
  checkCudaErrors(cudaEventCreate(&t0_sq));
  checkCudaErrors(cudaEventCreate(&t1_sq));
  checkCudaErrors(cudaEventCreate(&t_ext));


  checkCudaErrors(cudaEventRecord(t_begin, 0));



	//int m = 8192 / max_nnz;
  
  //int tmp = sqrt(ppl);
  //printf("tmp %d \n", tmp);
  //m = min(m, tmp);
  //m = min(m, ppl);

  int m = (k > 32) ? 32 : k;
 
  while (k*m > 1536) m /= 2;
 
  //int m = ppl / partsize;

  size_t free, total;
  cudaMemGetInfo(&free, &total);
  float tmp_f = free / sizeof(float);
  int log_size = log2(tmp_f);
  double arr_len = pow(2, log_size); 

  float del_t1;
  cudaEvent_t t0; 
  cudaEvent_t t1;
  int blocks = m*m;
  
  int num_blocks_tri = ppl / m;
  int t_b = (ppl > 1024) ? 1024 : ppl;
  dim3 dimBlock_tri(m, m);	
  dim3 dimGrid_tri(num_blocks_tri, leaves); 
  
  int num_blocks_sq = m * (m-1) /2;
  dim3 dimBlock_sq(m, m);	
  dim3 dimGrid_sq(num_blocks_tri/2, leaves); 
  
  dim3 dimBlock_norm(t_b);	
  dim3 dimGrid_norm(leaves); 
  
  float *d_Norms;
  
  int *d_block_indices;
  
  checkCudaErrors(cudaMalloc((void **) &d_block_indices, sizeof(int) * 2 * num_blocks_sq));
  par_block_indices(num_blocks_tri, d_block_indices);

 
  printf("# leaves : %d \n", leaves);
  printf("# points/leaf : %d \n", ppl);
  printf(" block (tri) = (%d,%d) \n", dimBlock_tri.x, dimBlock_tri.y);
  printf(" grid (tri) = (%d, %d) \n", dimGrid_tri.x, dimGrid_tri.y);
  printf(" block (sq) = (%d,%d) \n", dimBlock_sq.x, dimBlock_sq.y);
  printf(" grid (sq) = (%d, %d) \n", dimGrid_sq.x, dimGrid_sq.y);
  printf(" block (norm) = (%d,%d) \n", dimBlock_norm.x, dimBlock_norm.y);
  printf(" grid (norm) = (%d, %d) \n", dimGrid_norm.x, dimGrid_norm.y);
  printf(" # points = %d \n" , M);
 
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * ppl * leaves));


  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));
  

  compute_norm <<< dimGrid_norm, dimBlock_norm >>>(R, C, V, G_Id, d_Norms, ppl);
  
  checkCudaErrors(cudaEventRecord(t_ext, 0));
  knn_kernel_tri <<< dimGrid_tri, dimBlock_tri >>>(R, C, V, G_Id, d_Norms, k, knn, knn_Id, ppl, m);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t_tri, 0));
  checkCudaErrors(cudaEventSynchronize(t_tri));
  
  for (int blockInd = 0; blockInd < num_blocks_tri - 1; blockInd++){  
    //checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(t0_sq, 0));
    
    //checkCudaErrors(cudaDeviceSynchronize());
    knn_kernel_sq <<< dimGrid_sq, dimBlock_sq >>>(R, C, V, G_Id, d_Norms, k, knn, knn_Id, ppl, m ,blockInd, d_block_indices);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(t1_sq, 0));
    checkCudaErrors(cudaEventSynchronize(t1_sq));
    checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t0_sq, t1_sq));
    dt_sq += dt_tmp;
  } 
  
  //size_t free, total;
  cudaMemGetInfo(&free, &total);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t_end, 0));
  checkCudaErrors(cudaEventSynchronize(t_end));
  checkCudaErrors(cudaEventElapsedTime(&dt_tot, t_begin, t_end));
  checkCudaErrors(cudaEventElapsedTime(&dt_tri, t_ext, t_tri));

  //std::cout<<"Free memory before copy dev 0: "<<free<<" Device: "<< total <<std::endl;


  printf("============================ \n");
  printf(" Tri part = %.4f (%.f %%) \n", dt_tri/1000, 100*dt_tri/dt_tot);
  printf(" sq part = %.4f (%.f %%) \n", dt_sq/1000, 100*dt_sq/dt_tot);
  printf("\n Elapsed time (s) : %.4f \n ", dt_tot/1000);
  printf("============================ \n");
 
  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaEventDestroy(t0));
  checkCudaErrors(cudaEventDestroy(t1));

}




