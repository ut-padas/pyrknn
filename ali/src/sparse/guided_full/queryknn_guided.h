#include <stdio.h>
#include <stdlib.h>
#include <algorithm>



void query_leafknn_guided(int *R_ref, int *C_ref, float *V_ref, int *R_q,  int *C_q, float *V_q, int *GId, int ppl, int const leaves, int const k, float *d_knn, int *d_knn_Id, int const deviceId, int const verbose, int const nq, int *local_leafIds, int const dim, int const avgnnz, int const num_search_leaves, int *glob_leafIds);
