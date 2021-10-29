
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048

#include "merge.h"


__global__ void S_PrecompMergeNP2(int* SortInd, int* StepLen, int* StepStart, int* tidIdMap, int* tidSortDir, int steps){

  int tid = threadIdx.x;
  int size_sort = 2 * blockDim.x;

  int tid1 = 2 * tid + 1;
  int tid2 = tid1 + 1;

  __shared__ int sortind[SM_SIZE_1];
  
  sortind[tid] = 0;

  int shift = tid * steps;

  StepLen[shift + 0] = size_sort;
  StepStart[shift + 0] = 0;
 
  int g_tmp = size_sort;
  int ind = 0;
  int len = g_tmp;
  int direction = -1;
  tidSortDir[shift +0] = direction;
  tidIdMap[shift +0] = tid;
  int loc;
 
  for (int s = 1; s < steps; s++){
    g_tmp /= 2;

    ind = ind << 1;

    if (min(tid1, tid2) > g_tmp){
      ind++;
      tid1 -= g_tmp;
      tid2 -= g_tmp;
      loc = g_tmp + StepStart[shift + s-1];
      len -= g_tmp;
      g_tmp = len;

    } else {
      tid2 = (tid2 > g_tmp) ? g_tmp : tid2;
      tid1 = (tid1 > g_tmp) ? g_tmp : tid1;
      loc = StepStart[shift + s-1];
      direction *= -1;
      len = g_tmp;
    }
    StepLen[shift +s] = len;
    StepStart[shift +s] = loc;
    tidSortDir[shift +s] = direction;

  }
  sortind[tid] = ind;
  SortInd[tid] = ind;
  tidIdMap[shift + 0] = tid;
  __syncthreads();

  for (int s = 1; s < steps; s++){
    int diff = steps - s - 1;
    int tmp = ind >> diff;
    int testInd = tid;
    int tmp2 = sortind[testInd] >> diff;

    int pos = -1;
    while (testInd > -1){
      tmp2 = sortind[testInd] >> diff;
      if (tmp2 == tmp){
        pos++;
      } else{
        break;
      }
      testInd--;
    }
    tidIdMap[shift +s] = pos;
  }


}

__global__ void S_MergeHorizNP2(float* KNN, int* KNN_Id, int const k_nn, int const ppl, float* tmp_knnDist, int* glob_leafIds, int  const steps, int* QId, int* SortInd, int* step_len, int* step_start, int* tid_idmap, int* tid_sortdir, int* local_leafIds, float* Norms_q, float* Norms_ref){

  __shared__ float SM_dist[SM_SIZE_2];
  __shared__ int SM_Id[SM_SIZE_2];

  int const tid = threadIdx.x;
  int const q = blockIdx.x;
  int const nq = gridDim.x;
  int const leafId_l = local_leafIds[q];
  int const leafId_g = glob_leafIds[leafId_l];

  int size_part = ppl;
  int size_sort = 2 * blockDim.x;
  int shift = tid * steps;
  float norm_q = Norms_q[q];
 

  for (int n=tid; n < SM_SIZE_2; n += blockDim.x){
    SM_dist[n] = 1e30;
    SM_Id[n] = -1;
  }

  float tmp_f;
  int tmp_i;
  int ind_sort;

  int num_batches = size_part / (size_sort - k_nn);
  if (size_part > 0 && num_batches == 0) num_batches += 1;

  for (int col_batch = 0; col_batch < num_batches; col_batch++){
    for (int j_tmp = tid; j_tmp < size_sort; j_tmp += blockDim.x){

      int colId_leaf = (col_batch == 0) ? j_tmp - k_nn : col_batch * (size_sort - k_nn) + j_tmp - k_nn;

      if (col_batch == 0 && j_tmp < k_nn){
       
        int ind_q = QId[q];
        int ind_read = ind_q * k_nn + j_tmp;
        
        SM_dist[j_tmp] = KNN[ind_read];
        SM_Id[j_tmp] = KNN_Id[ind_read];
      
      } else if (colId_leaf < ppl && j_tmp >= k_nn){

        int ind_tmp = q + colId_leaf * nq;
        int g_colId_J = leafId_g * ppl + colId_leaf;
       
        //SM_dist[j_tmp] = tmp_knnDist[ind_tmp];
        if (q == 0 && colId_leaf < 10) printf("inner = %.4f , norm_ref = %.4f, norm_q = %.4f at %d , readfrom %d \n", tmp_knnDist[ind_tmp], Norms_ref[leafId_l * ppl + colId_leaf], norm_q, colId_leaf, leafId_l * ppl + colId_leaf);
        SM_dist[j_tmp] = -2 * tmp_knnDist[ind_tmp] + norm_q + Norms_ref[leafId_l * ppl + colId_leaf];
        SM_Id[j_tmp] = g_colId_J;
      }
    }
    __syncthreads();

    for (int j_tmp = tid; j_tmp < size_sort; j_tmp += blockDim.x) {

      if (j_tmp >= k_nn){
        int index = SM_Id[j_tmp];
        for (int ind_check = 0; ind_check < k_nn; ind_check++){
          if (index == SM_Id[ind_check]){
            SM_Id[j_tmp] = -1;
            SM_dist[j_tmp] = 1e30;
            break;
          }
        }
      }

    }
    __syncthreads();

    for (int s = 1; s < steps+1; s++){
      int diff = steps - s;
      int startloc = step_start[shift + diff];
      int arr_len = step_len[shift +diff];
      int tid_new = tid_idmap[shift +diff];
      int dir = tid_sortdir[shift +diff];
      for (int sl = s-1; sl > -1; sl--){

        int l = 1 << sl;
        int b = (tid_new+0.1) / l;
        int r = tid_new - b * l;
        int j = b * 2 * l + r;

        int ixj = j ^ l;
        j += startloc;
        ixj += startloc;
        bool cond = (dir == -1);
        bool gen_cond = (ixj - startloc < arr_len && arr_len != 1);
        if (gen_cond){
          if (cond){
            if (SM_dist[j] > SM_dist[ixj]){

              tmp_f = SM_dist[j];
              SM_dist[j] = SM_dist[ixj];
              SM_dist[ixj] = tmp_f;

              tmp_i = SM_Id[j];
              SM_Id[j] = SM_Id[ixj];
              SM_Id[ixj] = tmp_i;
            }
          } else {
            if (SM_dist[j] < SM_dist[ixj]){
              tmp_f = SM_dist[j];
              SM_dist[j] = SM_dist[ixj];
              SM_dist[ixj] = tmp_f;

              tmp_i = SM_Id[j];
              SM_Id[j] = SM_Id[ixj];
              SM_Id[ixj] = tmp_i;
            }

          }
        }
        __syncthreads();
      }
    }

  }
  for (int j_tmp = tid; j_tmp < k_nn; j_tmp += blockDim.x){
      
      int write_ind = QId[q] * k_nn + j_tmp;
      KNN[write_ind] = SM_dist[j_tmp];
      KNN_Id[write_ind] = SM_Id[j_tmp];

  }
}



