#ifndef KNN_SPARSE_GPU_HPP
#define KNN_SPARSE_GPU_HPP

// implemented in kernel_leaf.cu
void find_knn(int *ID, int *rowPtrP, int *colIdxP, float *valP, 
    int n, int d, int nnzP, int *seghead, int nLeaf, int m, int maxPoint,
    int *nborID, float *nborDist, int k, int LD);

void create_tree_next_level(int *ID, int *rowPtrP, int *colIdxP, float *valP, 
    int n, int d, int nnz, int *seghead, int *segHeadNext, int nNode,
    float *valX, float *median);

#endif //KNN_SPARSE_GPU_HPP
