#include <stdio.h>
#include <stdlib.h>
#include <algorithm>



void query_leafknn_seqsearch(float *X_ref, float *X_q, int *GId, int ppl, int const leaves, int const k, float *d_knn, int *d_knn_Id, int const deviceId, int const verbose, int const nq, int const dim, int *glob_leafIds, int num_search_leaves, int *local_leafIds);
