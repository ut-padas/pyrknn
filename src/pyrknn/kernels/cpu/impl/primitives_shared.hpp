#ifndef PRIMITIVES_CPU_HPP
#define PRIMITIVES_CPU_HPP

/** Use STL, and vector. */
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <algorithm>
#include <parallel/algorithm>
#include <omp.h>
//#include<ompUtils.h>

#include<cblas.h>

#include <cassert>
#include <queue>
#include <cmath>
#include <utility>
#include <iostream>
#include <numeric>


//TODO: Remove conditionally on POWER systems
//#include <mkl.h>
//#include <gsknn.h>

#include <limits>


#include <random>
#include <string>

#include "util.hpp"
//#include<gsknn_ref.h>
//#include<gsknn_ref_stl.hpp>



#define KNN_MAX_BLOCK_SIZE 1024
#define KNN_MAX_MATRIX_SIZE 2e7L


//TODO: Define conditonally on POWER systems
#define ARCH_POWER9 1

using namespace std;

template <typename T>
using idx_type = int; //typename vector<T>::size_type;

template <typename T>
using neigh_type = typename std::pair<T, idx_type<T>>;

typedef std::vector<unsigned> ivec;

/* reorder kernels*/
template<typename index_t, typename value_t>
void map_1D(index_t* idx, value_t *val, size_t length, value_t* output){
    //Permute
    #pragma omp parallel for
    for(int i = 0; i < length; ++i){
        output[i] = val[idx[i]];
    }
}

template<typename index_t, typename value_t>
void map_2D(index_t* idx, value_t *val, size_t length, size_t dim, value_t* output){
    //Permute
    #pragma omp parallel for
    for(auto i = 0; i < length; ++i){
        #pragma omp simd
        for(auto j = 0; j < dim; ++j){
            output[dim*i + j] = val[dim*idx[i] + j];
        }
    }
}

template<typename index_t, typename value_t>
void reindex_1D(index_t* idx, value_t* val, size_t length, value_t* buffer){
    //Fill buffer
    #pragma omp parallel for simd schedule(static)
    for(auto i = 0; i < length; ++i){
        buffer[i] = val[i];
    }

    //Permute
    #pragma omp parallel for simd schedule(static)
    for(auto i = 0; i < length; ++i){
        val[i] = buffer[idx[i]];
    }
}

template<typename index_t, typename value_t>
void reindex_2D(index_t* idx, value_t* val, size_t length, size_t dim, value_t* buffer){
    //Fill buffer
    #pragma omp parallel for schedule(static)
    for(auto i = 0; i < length; ++i){
        #pragma omp simd
        for(auto j = 0; j < dim; ++j){
            buffer[dim*i + j] = val[dim*i + j];
        }
    }

    //Permute
    #pragma omp parallel for schedule(static)
    for(auto i = 0; i < length; ++i){
        #pragma omp simd
        for(auto j = 0; j < dim; ++j){
            buffer[dim*i + j] = buffer[dim*idx[i] + j];
        }
    }
}

template<typename index_t, typename value_t>
void arg_sort(index_t* idx, value_t* val, size_t length){
    
    #pragma omp parallel for simd schedule(static)
    for(auto i = 0; i < length; ++i){
        idx[i] = i;
    }

    auto compare = [&val](size_t a, size_t b){return val[a] < val[b];};
    //std::stable_sort(std::execution::par_unseq, idx, idx+length, compare);
    __gnu_parallel::stable_sort(idx, idx+length, compare);
}

void find_interval(int* starts, int* sizes, unsigned char* index, int len, int nleaves, unsigned char* leaf_ids){

    #pragma omp parallel for
    for(auto i = 0; i < nleaves; ++i){
        unsigned char leaf = leaf_ids[i];
        const auto p = std::equal_range(index, index+len, leaf);

        int start = std::distance(index, std::get<0>(p));
        int end = std::distance(index, std::get<1>(p));
        starts[i] = start;
        sizes[i] = end - start;
    }

}

template<typename index_t, typename value_t>
void bin_queries(size_t n, int levels, value_t* proj, value_t* medians, index_t* idx, value_t* rbuffer){

    for(auto l = 0; l < levels; ++l){
        #pragma omp simd
        for(auto i = 0; i < n; ++i){
            const auto base = 2*idx[i] + 1;
            idx[i] = base + (proj[l*n+i] < medians[idx[i]]);
        }
    }
}


template<typename index_t, typename value_t>
void bin_queries_pack(size_t n, int levels, value_t* proj, value_t* medians, index_t* idx, value_t* rbuffer){
    for(auto l = 0; l < levels; ++l){
        //#pragma omp parallel for schedule(static) simd
        for(auto i = 0; i < n; i+=4){
            #pragma omp simd
            for(auto k = 0; k < 4; ++k){
                rbuffer[i+k] = medians[idx[i]];
            }
            #pragma omp simd
            for(auto k = 0; k < 4; ++k){
                const auto base = 2*idx[i] + 1;
                idx[i+k] = base + (proj[l*n+i+k] < rbuffer[i+k]);
            }
        }
    }
}

//reorder for pointer to array
template <typename T>
void hgather(T *ID, const unsigned int *perm, const unsigned int n)
{
    // Data type must be 4 bytes (e.g., float, int/unsigned).
    assert(sizeof(T) == sizeof(float));

    //Create temporary copy to store permutation
    T *copy = new T[n];

    par::hcopy(n, (float *)ID, (float *)copy);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        ID[i] = copy[perm[i]];

    //Delete temporary copy to store permutation
    delete[] copy;
}

//reorder for pointer to array, without allocation
template <typename T>
void hgather(T *ID, const int *perm, const size_t n, T *copy)
{
    // Data type must be 4 bytes (e.g., float, int/unsigned).
    assert(sizeof(T) == sizeof(float));

    //Create temporary copy to store permutation

    par::hcopy(n, (float *)ID, (float *)copy);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        ID[i] = copy[perm[i]];
}

void hgather(float* data, const int* perm, const int n, const int d){
    float* copy = new float[n];
    for(int i = 0; i < d; i++){
        par::hcopy(n, (float*) data + n*i, (float*) copy);

        #pragma omp parallel for
        for(int j = 0; j < n; j++){
            data[i*n + j] = copy[perm[j]];
        }
    }
}

//reorder for vectors
void hgather(ivec &order, const ivec &perm)
{
    size_t n = perm.size();
    ivec copy(n);
    par::hcopy(n, (float *)order.data(), (float *)copy.data());
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        order[i] = copy[perm[i]];
}

//reorder for vectors

template<typename T>
void hgather(T* order, const ivec &perm)
{
    size_t n = perm.size();
    ivec copy(n);
    par::hcopy(n, (float *)order, (float *)copy.data());

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        order[i] = copy[perm[i]];
}

template <typename T>
void print(const T *ptr, const size_t n, std::string name)
{
  std::cout << name << ":\n";
  for (size_t i = 0; i < n; i++)
    std::cout << ptr[i] << " ";
  std::cout << std::endl;
}


/* Tree Build */
//Assume local tree has less than 2 billion elements
void build_tree(float *X, unsigned *order, unsigned *firstPt, const unsigned int n, const size_t L)
{
    // starting indices of points in all nodes
    std::vector<ivec> firstPoint(L + 1);
    firstPoint[0].push_back(0);
    firstPoint[0].push_back(n); // from 0 to total
    for (size_t i = 0; i < L; i++)
    {
        unsigned nNode = 1 << i;
        firstPoint[i + 1].resize(nNode * 2 + 1, n); // the last entry is total
#pragma omp parallel for
        for (unsigned j = 0; j < nNode; j++)
        {
            firstPoint[i + 1][2 * j] = firstPoint[i][j];
            firstPoint[i + 1][2 * j + 1] = (firstPoint[i][j] + firstPoint[i][j + 1]) / 2;
        }
    }

    // return offsets of leaf nodes
    par::hcopy(firstPoint[L].size(), (float *)firstPoint[L].data(), (float *)firstPt);
    //std::cout << firstPoint[L].size() << std::endl;
    // initial ordering
    par::hiota(order, order + n, 0);
    //print(order, n, "order");
    // permutation of one level
    ivec perm(n);
    for (size_t i = 0; i < L; i++)
    {
        // apply accumulated permutation of previous levels
        hgather(X + i * n, order, n);

        // initialize permuation
        par::hiota(perm.begin(), perm.end(), 0);

        // partition nodes at this level
        const ivec &offset = firstPoint[i];
        unsigned nNode = 1 << i;

        //print(order, n, "order");
        //print(X+i*n, n, "data");

        #pragma omp parallel for
        for (unsigned j = 0; j < nNode; j++)
        {
            std::nth_element(&perm[offset[j]], &perm[(offset[j] + offset[j + 1]) / 2], &perm[offset[j + 1]],
                             [&X, i, n](unsigned a, unsigned b) { return X[a+i*n] < X[b+i*n]; });
        }
        hgather(order, perm);
    }
}

void choose2(int *out, int N)
{
    out[0] = rand() % N;
    out[1] = rand() % N;
    while (out[0] == out[1])
    {
        out[1] = rand() % N;
    }
}

unsigned int intlog2(uint64_t n)
{
#define S(k)                     \
    if (n >= (UINT64_C(1) << k)) \
    {                            \
        i += k;                  \
        n >>= k;                 \
    }

    unsigned int i = -(n == 0);
    S(32);
    S(16);
    S(8);
    S(4);
    S(1);
    return i;

#undef S
}




class maxheap_comp
{
public:
    __inline__ bool operator()(const std::pair<float, int> &a, const std::pair<float, int> &b)
    {
        const float x = a.first;
        const float y = b.first;
        return (x < y && !isinf(x)) || (!isnan(x) && (isnan(y) || isinf(y))); 
    }
};

void sqnorm(float *a, size_t n, int dim, float* b, bool threads)
{
    int one = 1;

    if(threads){
        bool omptest = n * static_cast<long>(dim) > 10000L;

        #pragma omp parallel if (omptest)
        {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i)
            {
                b[i] = cblas_sdot(dim, &(a[dim * i]), one, &(a[dim * i]), one);
            }
        }
    }
    else{
        for(size_t i = 0; i < n; ++i){
            float total = 0.0;
            #pragma omp simd reduction(+: total)
            for(int j = 0; j < dim; ++j){
                float elem = a[i*dim+j];
                total += elem*elem;
            }
            b[i] = total;
        }
    }
}


void compute_distances(float *ref, float *query, int n, int m, int dim, float* dist, float* sqnorm_r, float* sqnorm_q, bool use_norms){

    const float alpha = -2.0;
    const float beta = 0.0;

    //Form vector outer-product
    int offset = 0;
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, m, dim, alpha, ref, dim, query, dim, beta, dist, n);

    //Form (or reuse) vector norms
    bool dealloc_sqnorm_r = false;
    if(!sqnorm_r){
        sqnorm_r = new float[n];
        dealloc_sqnorm_r = true;
    }

    if(!use_norms){
        sqnorm(ref, n, dim, sqnorm_r, false);
    }

    bool dealloc_sqnorm_q = false;
    if(!sqnorm_q){
        sqnorm_q = new float[n];
        dealloc_sqnorm_q = true;
    }
    //always update query norms
    sqnorm(query, m, dim, sqnorm_q, false);

    //Add norms to distance matrix
    for(int i = 0; i < m; ++i){
        int row_start = i*n;
        #pragma omp simd
        for(int j = 0; j < n; ++j){
            dist[row_start+j] += sqnorm_q[i] + sqnorm_r[j];
        }
    }

    //Cleanup sqnorms
    if(dealloc_sqnorm_r){
        delete [] sqnorm_r;
    } 
    if(dealloc_sqnorm_q){
        delete [] sqnorm_q;
    }

    //Clean negative distances
    /*
    for(int i = 0; i < m; ++i){

        #pragma omp simd
        for(int j=0; j < n; ++j){
            float d = dist[i];
            dist[i] = d > 0 ? d : 0.0; 
        }
    }
    */
}




void print(float* v, int n, int m=1){
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            std::cout << v[i*n + j] << " ";
        }
        std::cout << std::endl;
    }
}

void direct_knn_base(int* rid, float* ref, float* query, int n, int m, int dim, int k, int* knn_ids, float* knn_dist, int blocksize)
{
    using knn_pair = pair<float, int>;

    //Check bounary
    k = (k < n) ? k : n;
    blocksize = (blocksize < m) ? blocksize : m;
    const int iters = static_cast<int>( ceil( static_cast<double>(m) / static_cast<double>(blocksize) ) );
    //std::cout << k << " " << blocksize << " " << iters <<std::endl;
    
    //Allocate distance matrix
    float* dist = (float*) malloc(sizeof(float)*blocksize*n);

    float* sqnorm_r = (float*) malloc(sizeof(float)*n);
    float* sqnorm_q = (float*) malloc(sizeof(float)*blocksize);

    priority_queue<knn_pair, vector<knn_pair>, maxheap_comp> maxheap;

    //Compute all norms 
    sqnorm(ref, n, dim, sqnorm_r, false);

    int current_blocksize = blocksize;

    //for each query block
    for(int i = 0; i < iters; ++i){
        float* current_query = query + i*blocksize*dim;
        if( (i == iters-1) && (m%blocksize) ){
            current_blocksize = m % blocksize;
        }

        //void compute_distances(float *ref, float *query, int n, int m, int dim, float* dist, float* sqnorm_r, float* sqnorm_q, bool use_norms){
        compute_distances(ref, current_query, n, current_blocksize, dim, dist, sqnorm_r, sqnorm_q, true);

        //for each query in block
        for(int h = 0; h < current_blocksize; ++h){
            while(!maxheap.empty()) maxheap.pop();
            int query_idx = i*blocksize + h;
            
            //fill the heap
            for(int j = 0; j < k; ++j){
                maxheap.push(make_pair(dist[h*n+j], j));
            }

            //iterative over the rest (maintain top-K)
            for(int j = k; j < n; ++j){
                maxheap.push(make_pair(dist[h*n+j], j));
                maxheap.pop();
            }

            //Read results back to output
            for(int j = k-1; j >=0; j--){
                knn_pair result = maxheap.top();
                knn_dist[query_idx*k + j] = result.first;
                knn_ids[query_idx*k + j] = rid[result.second];
                maxheap.pop();
            }
        }
    }
    free(sqnorm_q);
    free(sqnorm_r);
    free(dist);
}




template<typename I>
void batched_relabel_impl(I* gids, int** __restrict__ qid_list, int* __restrict__ mlist, int k, int** __restrict__ knn_ids_list, float** __restrict__ knn_dist_list, I*  output_ids, float* __restrict__ output_dist, int nleaves, int cores){

    //neighbor_list[idx, :] = global_ids[NL[:, :]]
    //neighbor_dist[idx, :] = ND[:, :]

    #pragma omp parallel for schedule(static)
    for(int l = 0; l < nleaves; ++l){

            int* __restrict__ qids = qid_list[l];
            int* __restrict__ knn_ids = knn_ids_list[l];
            float* __restrict__ knn_dist = knn_dist_list[l];
            int m = mlist[l];

            for(int i = 0; i < m; ++i){
                int idx = qids[i];
                #pragma omp simd
                for(int j = 0; j < k; ++j){
                    output_ids[idx*k + j] = gids[knn_ids[i*k + j]];
                    //output_ids[idx*k + j] = knn_ids[i*k + j];
                    output_dist[idx*k + j] = knn_dist[i*k + j];
                }
            }

    } //for each leaf
}

template<typename I>
void batched_relabel(I* gids, int** qid_list, int* mlist, int k, int** knn_ids_list, float** knn_dist_list, I* output_ids, float* output_dist, int nleaves, int cores){

    batched_relabel_impl(gids, qid_list,  mlist, k, knn_ids_list, knn_dist_list, output_ids, output_dist, nleaves, cores);

}

void batched_direct_knn_base(int** rid_list, float** ref_list, float** query_list, int* nlist, int* mlist, int dim, int k, int** knn_ids_list, float** knn_dist_list, int blocksize, int nleaves, const int cores)
{

    omp_set_num_threads(cores);
    #pragma omp parallel for 
    for(int l = 0; l < nleaves; ++l){

        const int n = nlist[l];
        const int m = mlist[l];

        int* knn_ids = knn_ids_list[l];
        float* knn_dist = knn_dist_list[l];
        int* rid = rid_list[l];
        float* ref = ref_list[l];
        float* query = query_list[l];

        direct_knn_base(rid, ref, query, n, m, dim, k, knn_ids, knn_dist, blocksize);
    }
}


#ifndef ARCH_POWER9

void GSKNN(
           int *rgids,
           float *R, float* Q,
           int n, //refernce length
           int d, //shared dimension
           int m, //query length
           int k, //number of neighbors
           int* neighbor_list,
           float* neighbor_dist)
{

    int maxt = omp_get_max_threads();

    float* sqnormr = new float[n];
    float* sqnormq = new float[m];
    int* qgids = new float[m];

    #pragma omp simd
    for (int z = 0; z < localn; ++z)
        qgids[z] = z;

    sqnorm(R, n, d, sqnormr);
    sqnorm(Q, m, d, sqnormq);

    heap_t *heap = heapCreate_s(m, k, 1.79E+30);
    sgsknn(n, m, d, k, R, sqnormr, rgids, Q, sqnormq, qgids, heap);

    #pragma omp parallel if (m > 128 * maxt)
    {
        #pragma omp for
        for (idx_type<T> j = 0; j < m; ++j)
        {
            #pragma omp simd
            for (idx_type<T> i = 0; i < k; ++i)
            {
                neighbor_dist[j * k + i] = heap->D_s[j * heap->ldk + i];
                neighbor_list[j * k + i] = rgids[heap->I[j * heap->ldk + i]];
            }
        }
    }

    delete[] sqnormr;
    delete[] sqnormq;
}

//TODO: Make this a clean batchedGSKNN call for the a2a case.

void batchedGSKNN(
                  int **rgids,
                  float **R, float **Q,
                  int *n,
                  int d,
                  int *m,
                  int k,
                  int **neighbor_list,
                  float **neighbor_dist,
                  int nleaves,
                  const int blocksize,
                  const int cores)
{

    omp_set_num_threads(cores);
    //printf("Started C++ Section \n");
    int maxt = (int) omp_get_max_threads();
    //printf("MAXT %d\n", maxt);
    //Allocate neighborlist & neighbor_dist
    //neighbor_list = new idx_type<T>*[nleaves];
    //neighbor_dist = new T*[nleaves];

    #pragma omp parallel for
    for (int l = 0; l < nleaves; ++l)
    {

        //    printf("NUMT %d\n", omp_get_num_threads());
        //    printf("Starting New Leaf %d \n", l);
        const int localm = m[l];
        const int localn = n[l];

        //neighbor_list[l] = new idx_type<T>[k*localm];
        //neighbor_dist[l] = new T[k*localm];

        float *localR = R[l];
        float *localQ = Q[l];
        int *local_rgids = rgids[l];

        int *local_qgids = new int[localn];
        #pragma omp simd
        for (int z = 0; z < localn; ++z)
            local_qgids[z] = z;

        int nblocks = localm / blocksize;
        int iters = static_cast<int>(ceil( static_cast<double>(localm) / static_cast<double>(blocksize)));

        float *sqnormr = new float[localn];
        float *sqnormq = new float[blocksize];

        sqnorm(localR, localn, d, sqnormr);

        //Loop over all blocks
        for (int i = 0; i < iters; ++i)
        {
            float *currquery = localQ + i * blocksize * d;
            int current_blocksize = blocksize;

            if ((i == iters - 1) && (localm % blocksize))
            {
                current_blocksize = localm % blocksize;
            }

            const size_t offset = i * blocksize;
            sqnorm(currquery, current_blocksize, d, sqnormq);

            heap_t *heap = heapCreate_s(current_blocksize, k, 1.79E+30);
            sgsknn(localn, current_blocksize, d, k, localR, sqnormr, local_qgids, currquery, sqnormq, local_qgids, heap);

            auto D = heap->D_s;
            auto I = heap->I;
            auto ldk = heap->ldk;
            for (int j = 0; j < current_blocksize; ++j)
            {
                #pragma omp simd
                for (int h = 0; h < k; ++h)
                {
                    neighbor_dist[l][(j + offset) * k + h] = D[j * ldk + h];
                    neighbor_list[l][(j + offset) * k + h] = rgids[l][I[j * ldk + h]];
                }
            } //end copy

            heapFree_s(heap);
        } //end loop over blocks

        delete[] local_qgids;
        delete[] sqnormr;
        delete[] sqnormq;

    } //end loop over leaves

} //end function

#endif

void sort_select(const float *value, const int *ID, int n, float *kval, int *kID, int k)
{
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&value](int i, int j) { return value[i] < value[j]; });

    for (int i = 0; i < k; i++)
    {
        int j = idx[i];
        kval[i] = value[j];
        kID[i] = ID[j];
    }
}

template <typename key_t, typename value_t>
inline bool less_key(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b)
{
    return (a.first < b.first);
}

template <typename key_t, typename value_t>
inline bool less_value(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b)
{
    return (a.second < b.second);
}

template <typename key_t, typename value_t>
inline bool equal_key(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b)
{
    return (a.first == b.first);
}

template <typename key_t, typename value_t>
inline bool equal_value(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b)
{
    return (a.second == b.second);
}

template <typename T>
void merge_neighbor_cpu(T *D1, unsigned int *I1, T *D2, unsigned int *I2, const unsigned int n, const int k, const int cores)
{
    omp_set_num_threads(cores);

#pragma omp parallel
    {

        std::vector<std::pair<unsigned int, T>> aux(2 * k);

#pragma omp for
        for (size_t i = 0; i < n; ++i)
        {
            size_t offset = i * k;
            //Merge twolists
            for (size_t j = 0; j < k; ++j)
            {
                aux[j] = std::make_pair(I1[offset + j], D1[offset + j]);
                aux[k + j] = std::make_pair(I2[offset + j], D2[offset + j]);
            }

            //Sort according to index
            std::sort(aux.begin(), aux.end(), less_key<unsigned int, T>);
            auto last = std::unique(aux.begin(), aux.end(), equal_key<unsigned int, T>);
            std::sort(aux.begin(), last, less_value<unsigned int, T>);

            //Copy Results back to output
            for (size_t j = 0; j < k; ++j)
            {
                auto current = aux[j];
                I1[offset + j] = current.first;
                D1[offset + j] = current.second;
            }
        }
    }
}

#endif /* define PRIMITIVES_CPU_HPP */
