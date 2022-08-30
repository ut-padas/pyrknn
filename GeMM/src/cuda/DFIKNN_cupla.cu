
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
#define SM_SIZE_SORT 8192

#include "FIKNN_dense.h"

__global__ void computeNorms(float* data, int* G_Id, float* Norms, int ppl, int dim) {

  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int leaf_id_g = blockIdx.y;

  int g_rowId = leaf_id_g * ppl + row;
  int g_Id = G_Id[g_rowId];


  //int ind0_i = R[g_Id];

  //int nnz = R[g_Id + 1] - ind0_i;
  float norm_i = 0.0;

  for (int n_i = 0; n_i < dim; n_i += 1) {
    norm_i += data[g_Id * dim + n_i] * data[g_Id * dim + n_i];
  }
  Norms[g_Id] = norm_i;
}



__global__ void sortRecPart(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* G_Id, bool init){
      

  __shared__ float SM_dist[SM_SIZE_1];
  __shared__ int SM_Id[SM_SIZE_1];

  int tid = threadIdx.x;

  int col = blockIdx.x;
  int leaf_id_g = blockIdx.y;

  int colId_leaf = (init) ? col : col + k_nn * (blockInd + 1);
  //int size_part = (init) ? ppl : ppl - (blockInd + 1) * (k_nn);
  int size_part = (init) ? ppl : ppl - (blockInd + 1) * (k_nn);

  if (tid < k_nn){
    int ind_tmp = (init) ? leaf_id_g * ppl * k_nn + col * k_nn + tid : leaf_id_g * k_nn * size_part + tid * size_part + col;
    //int ind_tmp = leaf_id_g * k_nn * size_part + tid * size_part + colId_leaf;
    SM_dist[tid] = d_temp_knn[ind_tmp];

    int block = col / k_nn;
    int rowId_g = (init) ? leaf_id_g * ppl + block * k_nn + tid : leaf_id_g * ppl + k_nn * blockInd + tid;
    SM_Id[tid] = G_Id[rowId_g];

  } else {

    int ind_pt_knn = leaf_id_g * ppl + colId_leaf;
    int ind_pt_knn_g = G_Id[ind_pt_knn];

    int ind_knn = ind_pt_knn_g * k_nn + tid - k_nn;
    SM_dist[tid] = KNN[ind_knn];
    SM_Id[tid] = KNN_Id[ind_knn];

  }


  __syncthreads();

  if (tid < k_nn){
    float val = SM_dist[tid];
    int index = SM_Id[tid];
    for (int ind_check = 0; ind_check < k_nn; ind_check++){
      if (val == SM_dist[ind_check + k_nn] && index == SM_Id[ind_check + k_nn]){
        SM_dist[tid] = 1e30;
        SM_Id[tid] = -1;
        break;
      }
    }


  }
  __syncthreads();


  // sort
  float tmp_f;
  int tmp_i;

  int size = 2 * k_nn;
  for (int g = 2; g <= size; g *= 2){
    for (int l = g/2; l > 0; l /= 2){

      int ixj = tid ^ l;

      if (ixj > tid){
        if ((tid & g) == 0){
          if (SM_dist[tid] > SM_dist[ixj]){

            tmp_f = SM_dist[ixj];
            SM_dist[ixj] = SM_dist[tid];
            SM_dist[tid] = tmp_f;

            tmp_i = SM_Id[ixj];
            SM_Id[ixj] = SM_Id[tid];
            SM_Id[tid] = tmp_i;

          }
        } else {
          if (SM_dist[tid] < SM_dist[ixj]){

            tmp_f = SM_dist[ixj];
            SM_dist[ixj] = SM_dist[tid];
            SM_dist[tid] = tmp_f;

            tmp_i = SM_Id[ixj];
            SM_Id[ixj] = SM_Id[tid];
            SM_Id[tid] = tmp_i;

          }
        }
      }
      __syncthreads();


    }
  }
  int ind_pt = leaf_id_g * ppl + colId_leaf;
  int ind_pt_g = G_Id[ind_pt];
  int write_ind = ind_pt_g * k_nn + tid;


  if (tid < k_nn) {

    KNN[write_ind] = SM_dist[tid];
    KNN_Id[write_ind] = SM_Id[tid];


  }

}




