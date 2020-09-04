#ifndef PRIMITIVES_CPU_HPP
#define PRIMITIVES_CPU_HPP

/** Use STL, and vector. */
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <algorithm>

#include <omp.h>
//#include<ompUtils.h>
//#include<blas.h>
#include<cassert>
#include<queue>
#include<cmath>
#include<utility>
#include<iostream>
#include<numeric>
#include<mkl.h>
#include<limits>
#include<gsknn.h>
//#include<gsknn_ref.h>
//#include<gsknn_ref_stl.hpp>

#define KNN_MAX_BLOCK_SIZE 1024
#define KNN_MAX_MATRIX_SIZE 2e7L

using namespace std;

template<typename T>
using idx_type = int;//typename vector<T>::size_type;


template<typename T>
using neigh_type = typename std::pair<T, idx_type<T>>;


void choose2(int* out, int N){
    out[0] = rand() % N;
    out[1] = rand() % N;
    while(out[0] == out[1]){
        out[1] = rand() % N;
    }
}


unsigned int intlog2(uint64_t n){
    #define S(k) if (n >= (UINT64_C(1) << k)) { i+=k; n >>=k; }
    
    unsigned int i = -(n == 0); S(32); S(16); S(8); S(4); S(1); return i;

    #undef S

}

//sequential partition algorithm
template<typename T>
unsigned int partition(T* array, const unsigned int N, const T elem, const unsigned int ind){
    unsigned int start = 0;
    unsigned int idx = 0;
    unsigned int p = 0;
    while(idx <= N){
        T item = array[idx];
        
        if(item < elem){
            array[idx] = array[start];
            array[start] = item;
            start++;     
        }
        if(idx == ind){
            p = start;
        }

        idx++;
    }
    return p;
}

template<typename T>
unsigned int parallel_partition(T* array, const unsigned int N, const T elem, const unsigned int idx, unsigned int* workspace){

    bool dealloc_workspace = false;

    if(!workspace) {
        workspace = new unsigned int[N];
        dealloc_workspace = true;
    }
    
    unsigned int scan_a = 0;
    unsigned int scan_b = 0;
    #pragma omp simd reduction(inscan +:scan_a, scan_b)
    for(unsigned int i = 0; i < N; ++i){
        workspace[i] = (array[i] > elem) ? scan_a : scan_b;
        #pragma omp scan exclusive(scan_a, scan_b)
        scan_a += (array[i] > elem)
        scan_b += (array[i] < elem)
    }

    #pragma omp parallel for
    for(unsigned int i = 0; i < N; ++i){
        workspace[i] = array[workspace[i]]
    }

    unsigned int k = workspace[idx];

    #pragma omp parallel for
    for(unsigned int i = 0; i < N; ++i){
        array[i] = workspace[i];
    }

    if(dealloc_workspace){
        delete [] workspace;
    }

    return k;
}

//sequential quicksort algorithm
template<typename T>
T quickselect(T* array, const unsigned int N, const unsigned int k){
    
     unsigned int start = 0;
     unsigned int stop = N;
     unsigned int idx = 0;
     T elem;
     int iter = 0;
     while(start <= stop){
        iter++;
        printf("Iter %d\n", iter);
        idx = rand() % N;
        elem = array[idx];
        auto p = partition((float*) array+start, stop-start, elem, idx);
        printf("Selected %d, Loc %d \n", idx, p);
        if (k == p){
            return elem;
        }
        if (k > p){
            start = p;
        }
        else{
            stop = p;
        }
        quickselect((float*) array+start, stop-start, k); 
     }
    return -1;
}


template<typename T>
void tree_build(int offset, int* gids, T* data ,int n, int d, int* seghead, int nNode, int maxsize, T* valX, float* median, T* y){

    set_num_threads(4); //TODO Replace with Singleton Enviornment Call (Or fix/remove PARLA import )
    
    unsigned int nthreads = get_num_threads();

    //Get randomized projections
    //TODO: Time against MKL_BATCHED_GEMM
    #pragma omp parallel for (if nNode > 8)
    for(unsigned int i = 0; i < nNode; ++i){
        unsigned int size = seghead[i+1] - seghead[i];
        cblas_sgemv(CblasRowMajor, CblasNoTrans, size, d, 1.0, data + offset*n + seghead[i]*d, size, x+i*d, 1, 0.0, y+seghead[i], 1);
    }

    //Allocate array of local ids (These are used to permute gids and data)
    std::vector<unsigned int> lids(n);
    //unsigned int lids[nthreads][maxsize]; //TODO: Change this to malloc so we don't run into memory problems on root nodes

    //Find median and permute gids
    #pragma omp parallel for
    for(unsigned int i = 0; i < nNode; ++i){
        
        unsigned int size = seghead[i+1] - seghead[i]; //Size of current node

        //unsigned int tid = omp_get_thread_num();

        //Reset array of local ids
        for(unsigned int j = 0; j < size; ++j){
            lids[seghead[i]+j] = j;
        }

        //Perform quickselect on each node and get median
        std::nth_element(&lids[ seghead[i] ], &lids[ seghead[i] ]+size, [] (const unsigned int &a, const unsigned int &b) const noexcept { return *(y+seghead[i]+a) < *(y+seghead[i]+b) } );
        median[i] = y[seghead[i]+lids[seghead[i] + size/2]];

        //Copy to new data array
        for(unsigned int j = 0; j < size; ++j){
            for(unsigned int k = 0; k < d; ++k){
                data[(1-offset)*n + seghead[i]*d + lids[seghead[i] + j]*d + k] = data[offset*n + seghead[i]*d + j*d + k];
            }
        }
        for(unsigned int j = 0; j < size; ++j){
            lids[seghead[i]+j] = gids[seghead[i]+lids[seghead[i]+j]];
        }
        for(unsigned int j = 0; j < size; ++j){
            gids[j] = lids[j];
        }

    }
}



/*
//Nearest Neighbor Kernel (from GOFMM Not Fused)
//This actually wasn't parallel in GOFMM and it uses a pretty costly sort
//TODO(p2)[Will] Really need to use GSKNN here or from Bo Xiao's code
template<typename T>
bool NeighborSearch(
    idx_type<T> kappa, 
    T *Q, 
    T *R,
    const idx_type<T> N,
    const idx_type<T> d, 
    vector<idx_type<T>> neighbor_list,
    vector<T> neighbor_dist)
{
   auto DRQ = Distances(R, Q, N, d);

    //Loop over query points
    for(idx_type<T> j = 0; j < N; ++j){
        vector<idx_type<T>> candidate_list( N );
        vector<T> candidate_dist( N );
        for(idx_type<T> i = 0; i < N; ++i){
            candidate_list[ i ] ;
            candidate_dist[ i ] = DRQ[ j, i ];
         }

         sort(candidates_list.begin(), candidates_list.end(), 
            [] (const idx_type<T> a, const idx_type<T> b){ return candidate_list[a] < candidate_list[b]});
         sort(candidates_dist.begin(), candidates_dist.end());

        //Fill in the neighbor list
        for( idx_type<T> i = 0; i < kappa; i++){
            neighbor_list = candidates_list[ i ];
            neighbor_dist = candidates_dist[ i ];
        }
    }

    return true;
} 
*/
//Kernels from Bo Xiao's Code 

template<typename T>
class maxheap_comp {
    public:
        bool operator() (const std::pair<T, idx_type<T>> &a, const std::pair<T, idx_type<T>> &b) {
            double diff = fabs(a.first-b.first)/a.first;
            if( std::isinf(diff) || std::isnan(diff) ) {      // 0/0 or x/0
                return a.first < b.first;
            }
            if( diff < 1.0e-8 ) {
                return a.second < b.second;
            }
            return a.first < b.first;
        }
};

template<typename T>
T getBlockSize(const T n, const T m){
    T blocksize;

    if( m > KNN_MAX_BLOCK_SIZE || n > 10000L){ 
       blocksize = std::min((T)KNN_MAX_BLOCK_SIZE, m); //number of query points handled in a given iteration
       
       if(n * blocksize > (T)KNN_MAX_MATRIX_SIZE) blocksize = std::min((T)(KNN_MAX_MATRIX_SIZE/n), blocksize); //Shrink block size if n is huge.
       
       blocksize = std::max(blocksize, omp_get_max_threads()); //Make sure each thread has some work.
     } else {
        blocksize = m;
     }
     return blocksize;
}


template<typename T>
T getNumLocal(const T rank, const T size, const T num){
    if(rank < num % size)
        return (idx_type<T>) std::ceil( (double) num / (double) size );
    else
        return num / size;
}


template<typename T>
void sqnorm(T *a, idx_type<T> n, idx_type<T> dim, T *b){
    int one = 1;
    bool omptest = n*(long)dim > 10000L;

    #pragma omp parallel if (omptest)
    {
        #pragma omp for schedule(static)
        for(idx_type<T> i = 0; i < n; ++i){
            b[i] = sdot(&dim, &(a[dim*i]), &one, &(a[dim*i]), &one);

        }
    }
}


template<typename T>
void compute_distances(T *R, T *Q, 
                  idx_type<T> n, idx_type<T> m, idx_type<T> dim, T* dist, 
                  T* sqnormr, T* sqnormq, bool useSqnormrInput)
{
    T alpha = -2.0;
    T beta = 0.0;

    int iN = (int) n;
    int iM = (int) m;

    idx_type<T> maxt = omp_get_max_threads();
    bool omptest = (m > 4 * maxt || (m >= maxt && n > 128)) && n < 100000;

    #pragma omp parallel if( omptest )
    {
        idx_type<T> t = omp_get_thread_num();
        idx_type<T> numt = omp_get_num_threads();
        idx_type<T> npoints = getNumLocal(t, numt, m);

        int offset = 0;
        for(idx_type<T> i = 0; i < t; ++i) offset += getNumLocal(i, numt, m);
        
        sgemm("T", "N", &iN, &npoints, &dim, &alpha, R, &dim, Q + (dim*offset), 
               &dim, &beta, dist+(offset*n), &iN);                    
    }


    bool dealloc_sqnormr = false;
    bool dealloc_sqnormq = false;

    if(!sqnormr && !useSqnormrInput) {
        sqnormr = new T[n];
        dealloc_sqnormr = true;
    }

    if(!sqnormq){
        sqnormq = new T[m];
        dealloc_sqnormq = true;
    }

    if(!useSqnormrInput)
        sqnorm(R, n, dim, sqnormr);

    sqnorm(Q, m, dim, sqnormq);

    if( m > maxt || n > 10000) {
        idx_type<T> blocksize = (n > 10000) ? m/maxt/2 : 128;
        #pragma omp parallel for
        for(idx_type<T> i = 0; i < m; ++i){
            idx_type<T> in = i*n;
            idx_type<T> j;

            #pragma ivdep
            for(j = 0; j < n; ++j){
                idx_type<T> inpj = in + j;
                dist[inpj] += sqnormq[i] + sqnormr[j];
            }
        }
    }
    else{
        for(idx_type<T> i = 0; i < m; ++i){
            idx_type<T> in = i*n;
            idx_type<T> j;
            
            #pragma ivdep
            for(j = 0; j < n; ++j){
                idx_type<T> inpj = in + j;
                dist[inpj] += sqnormq[i] + sqnormr[j];
            }
        }
    }

    if(dealloc_sqnormr)
        delete [] sqnormr;

    if(dealloc_sqnormq)
        delete [] sqnormq;

    //force distances to be greater than 0 to mitigate rounding errors
    
    #pragma omp parallel for
    for(idx_type<T> i = 0; i < m*n; ++i){
        if(dist[i] < 0.0) dist[i] = 0.0;
    }

}



//Direct Query (Low Memory)
template<typename T>
void directKLowMem(const idx_type<T> *gids, 
                   T *R, T *Q,
                   const idx_type<T> n,
                   const idx_type<T> d, 
                   const idx_type<T> m, 
                   const idx_type<T> k, 
                   idx_type<T>* neighbor_list,
                   T* neighbor_dist){

    register idx_type<T> num_neighbors = (k < n) ? k : n;

    vector<pair< T, idx_type<T> >> result;
    result.reserve(m*k);
    
    //Split Query into smaller pieces
    idx_type<T> blocksize = getBlockSize(n, m);

    bool dealloc_dist = false;
    bool dealloc_sqnormr = false;
    bool dealloc_sqnormq = false;

    T *dist;
    T *sqnormr;
    T *sqnormq;

    idx_type<T> maxt = (idx_type<T>) omp_get_max_threads();

    assert(blocksize > 0);

    idx_type<T> nblocks = m / blocksize;
    idx_type<T> iters = (idx_type<T>) ceil((double) m/ (double) blocksize);

    if(!dist) {
        dist = new T[n*blocksize];
        dealloc_dist = true;
    }

    if(!sqnormr) {
        sqnormr = new T[n];
        dealloc_sqnormq = true;
    }

    if(!sqnormq) {
        sqnormq = new T[blocksize];
        dealloc_sqnormq = true;
    }

    bool useSqnormrInput = false;

    //Loop over all blocks
    for(idx_type<T> i = 0; i < iters; ++i){
        T *currquery = Q + i*blocksize*d;
        if ( (i == iters -1) && (m%blocksize) ){
            idx_type<T> lastblocksize = m%blocksize;

            compute_distances(R, currquery, n, lastblocksize, d, dist, sqnormr, sqnormq, useSqnormrInput);

            #pragma omp parallel
            {
                priority_queue<neigh_type<T>, vector<neigh_type<T>>, maxheap_comp<T>> maxheap;
                
                #pragma omp for
                for(idx_type<T> h = 0; h < lastblocksize; ++h) {
                    while(!maxheap.empty()) maxheap.pop();
                    int querynum = i*blocksize + h;
                    for(idx_type<T> j = 0; j < num_neighbors; ++j)
                        maxheap.push( make_pair< T, idx_type<T> >( (T) dist[h*n+j], (idx_type<T>) j) );

                    for(idx_type<T> j = num_neighbors; j < n; ++j){
                        maxheap.push( make_pair< T, idx_type<T> >( (T) dist[h*n+j], (idx_type<T>) j) );
                        maxheap.pop();
                    }
                    
                    for(size_t j = num_neighbors-1; j >=0; --j){
                        result[querynum*k+j] = maxheap.top();
                        maxheap.pop();
                    }
                }
            }
        } //end if last block
        else {
            compute_distances(R, currquery, n, blocksize, d, dist, sqnormr, sqnormq, useSqnormrInput);
            
            #pragma omp parallel
            {
                priority_queue< neigh_type<T>, vector< neigh_type<T> >, maxheap_comp<T>> maxheap;
                #pragma omp for
                for(idx_type<T> h = 0; h < blocksize; ++h){

                    while(!maxheap.empty()) maxheap.pop();
                    idx_type<T> querynum = i*blocksize + h;
                    for(idx_type<T> j = 0; j < num_neighbors; j++)
                        maxheap.push( make_pair< T, idx_type<T> >( (T) dist[h*n+j], (idx_type<T>) j) );

                    for(idx_type<T> j = num_neighbors; j < n; ++j){
                        maxheap.push( make_pair< T, idx_type<T> >( (T) dist[h*n+j], (idx_type<T>) j) );
                        maxheap.pop();
                    }
                    for(idx_type<T> j = num_neighbors-1; j >= 0; --j){
                        result[querynum*k+j] = maxheap.top();
                        maxheap.pop();
                    }
                } //end for h
            } //end parallel region
        } //end block
        useSqnormrInput = true;
    }
    if( num_neighbors < k){
        //pad with bogus values
        #pragma omp parallel if( m > 128 * maxt)
        {
            #pragma omp for schedule(static)
            for(idx_type<T> i = 0; i < m; ++i){
                for(idx_type<T> j = num_neighbors; j < k; ++j){
                    result[i*k+j].first = 3.4028E38;
                    result[i*k+j].second = 0;
                }
            }
        }
    }
            
    //copy results over to neighbor_dist and neighbor_list
    #pragma omp parallel if (m > 128 * maxt)
    {
        #pragma omp for
        for(idx_type<T> i = 0; i < m; ++i){
            #pragma ivdep
            for(idx_type<T> j = 0; j < k; ++j){
                neighbor_dist[i*k+j] = result[i*k+j].first;
                neighbor_list[i*k+j] = gids[result[i*k+j].second];
            }
        }
    }

    if(dealloc_dist) delete[] dist;
    if(dealloc_sqnormq) delete [] sqnormr;
    if(dealloc_sqnormq) delete [] sqnormq;
}  

template<typename T>
void GSKNN(idx_type<T> *rgids,
           idx_type<T> *qgids,  
           T *R, T *Q,
           const idx_type<T> n,   //refernce length
           const idx_type<T> d,   //shared dimension
           const idx_type<T> m,   //query length
           const idx_type<T> k,   //number of neighbors
           idx_type<T>* neighbor_list,
           T* neighbor_dist){

    
    bool dealloc_sqnormr = false;
    bool dealloc_sqnormq = false;

    T *dist;
    T *sqnormr;
    T *sqnormq;

    idx_type<T> maxt = (idx_type<T>) omp_get_max_threads();

    if(!sqnormr) {
        sqnormr = new T[n];
        dealloc_sqnormq = true;
    }

    if(!sqnormq) {
        sqnormq = new T[m];
        dealloc_sqnormq = true;
    }

    sqnorm(R, n, d, sqnormr);
    sqnorm(Q, m, d, sqnormq);
    heap_t *heap = heapCreate_s(m, k, 1.79E+30);

    sgsknn(n, m, d, k, R, sqnormr, rgids, Q, sqnormq, qgids, heap);

    //printf("%f \n", (float) heap->ldk);
    #pragma omp parallel if (m > 128 * maxt)
    {
        #pragma omp for
        for(idx_type<T> j = 0; j < m; ++j){
            #pragma unroll
            for(idx_type<T> i = 0; i < k; ++i){
                neighbor_dist[j*k+i] = heap->D_s[j*heap->ldk + i];
                neighbor_list[j*k+i] = rgids[heap->I[j*heap->ldk + i]];
            }
        }
    }

    if(dealloc_sqnormq) delete [] sqnormr;
    if(dealloc_sqnormq) delete [] sqnormq;
}

template<typename T>
void blockedGSKNN(idx_type<T> *rgids, 
                  idx_type<T> *qgids, 
                  T *R, T *Q,
                  const idx_type<T> n,
                  const idx_type<T> d,
                  const idx_type<T> m,
                  const idx_type<T> k,
                  idx_type<T> *neighbor_list,
                  T *neighbor_dist){

    register idx_type<T> num_neighbors = (k < n) ? k : n;

    //Split Query into smaller pieces
    idx_type<T> blocksize = getBlockSize(n, m);

    bool dealloc_dist = false;
    bool dealloc_sqnormr = false;
    bool dealloc_sqnormq = false;

    T *sqnormr;
    T *sqnormq;

    idx_type<T> maxt = (idx_type<T>) omp_get_max_threads();

    assert(blocksize > 0);

    idx_type<T> nblocks = m / blocksize;
    idx_type<T> iters = (idx_type<T>) ceil((double) m/ (double) blocksize);

    if(!sqnormr) {
        sqnormr = new T[n];
        dealloc_sqnormq = true;
    }

    if(!sqnormq) {
        sqnormq = new T[blocksize];
        dealloc_sqnormq = true;
    }

    bool useSqnormrInput = false;

    //Loop over all blocks
    for(idx_type<T> i = 0; i < iters; ++i){
        T *currquery = Q + i*blocksize*d;
        if ( (i == iters -1) && (m%blocksize) ){

            idx_type<T> lastblocksize = m%blocksize;

            if(!useSqnormrInput)
                sqnorm(R, n, d, sqnormr);
            sqnorm(Q, lastblocksize, d, sqnormq);
            #pragma omp parallel
            {
                idx_type<T> t = omp_get_thread_num();
                idx_type<T> numt = omp_get_num_threads();
                idx_type<T> npoints = getNumLocal(t, numt, lastblocksize);

                idx_type<T> offset = 0;
                for(idx_type<T> i = 0; i < t; ++i) offset += getNumLocal(i, numt, m);

                heap_t *heap = heapCreate_s(m, k, 1.79E+30); //local heap
                sgsknn(n, npoints, d, k, R, sqnormr, rgids, Q+(d*offset), sqnormq + offset, qgids+offset, heap);
                
                ///copy over results
                #pragma omp for
                for(idx_type<T> j = 0; j < npoints; ++j){
                    #pragma unroll
                    for(idx_type<T> h = 0; h < k; ++h){
                        neighbor_dist[(j+offset)*k+h] = heap->D_s[j*heap->ldk + h];
                        neighbor_list[(j+offset)*k+h] = rgids[heap->I[j*heap->ldk+h]];
                    }
                }

            }
        } //end if last block
        else {

            if(!useSqnormrInput)
                sqnorm(R, n, d, sqnormr);
           
            sqnorm(Q, blocksize, d, sqnormq); 
            #pragma omp parallel
            {
                idx_type<T> t = omp_get_thread_num();
                idx_type<T> numt = omp_get_num_threads();
                idx_type<T> npoints = getNumLocal(t, numt, blocksize);
                //printf("Thread %d, npoints=%d \n", t, npoints);
                idx_type<T> offset = 0;
                for(idx_type<T> i = 0; i < t; ++i) offset += getNumLocal(i, numt, m);

                heap_t *heap = heapCreate_s(m, k, 1.79E+30); //local heap
                sgsknn(n, npoints, d, k, R, sqnormr, rgids, Q+(d*offset), sqnormq + offset, qgids+offset, heap);
                
                ///copy over results
                #pragma omp for
                for(idx_type<T> j = 0; j < npoints; ++j){
                    #pragma unroll
                    for(idx_type<T> h = 0; h < k; ++h){
                        neighbor_dist[(j+offset)*k+h] = heap->D_s[j*heap->ldk + h];
                        neighbor_list[(j+offset)*k+h] = rgids[heap->I[j*heap->ldk+h]];
                    }
                }
            }
        } //end block
        useSqnormrInput = true;
    }
    
    if(dealloc_sqnormq) delete [] sqnormr;
    if(dealloc_sqnormq) delete [] sqnormq;

}


template<typename T>
void batchedDirectKNN(idx_type<T> **gids,
                 T **R, T **Q,
                 const idx_type<T> *n,
                 const idx_type<T>  d,
                 const idx_type<T> *m,
                 const idx_type<T>  k,
                       idx_type<T> **neighbor_list,
                       T **neighbor_dist,
                 const idx_type<T>  nleaves){


    idx_type<T> maxt = (idx_type<T>) omp_get_max_threads();

    #pragma omp parallel for
    for(idx_type<T> l; l < nleaves; ++l){

        const idx_type<T> localm = m[l];
        const idx_type<T> localn = n[l];

        const T* localR = R[l];
        const T* localQ = Q[l];

        const idx_type<T> blocksize = getBlockSize(localn, localm);
        const register idx_type<T> num_neighbors = (k < localn) ? k : localn;
        
        idx_type<T> nblocks = localm / blocksize;
        idx_type<T> iters = (idx_type<T>) ceil( (double) localm / (double) blocksize );

        bool dealloc_sqnormr = false;
        bool dealloc_sqnormq = false;
        bool dealloc_dist = false;

        T *sqnormr;
        T *sqnormq;
        T *dist;

        if(!dist) {
            dist = new T[localn*blocksize];
            dealloc_dist = true;
        }

        if(!sqnormr) {
            sqnormr = new T[localn];
            dealloc_sqnormq = true;
        }

        if(!sqnormq) {
            sqnormq = new T[blocksize];
            dealloc_sqnormq = true;
        }

       bool useSqnormrInput = false;

       //Loop over all blocks
       for(idx_type<T> i = 0; i < iters; ++i){
            T *currquery = localQ + i*blocksize*d;

            
            //Handle the edge case if blocksize does not divide localn evenly
            if ( (i == iters - 1) && (m%blocksize) ) {
               const idx_type<T> lastblocksize = m%blocksize;            
                
               compute_distances(localR, currquery, localn, blocksize, d, dist, sqnormr, sqnormq, useSqnormrInput);
                
               priority_queue< neigh_type<T>, vector< neigh_type<T> >, maxheap_comp<T> > maxheap;

               for(idx_type<T> h = 0; h < lastblocksize; ++h){
                    while(!maxheap.empty()) maxheap.pop();
                    const idx_type<T> querynum = i*blocksize + h;
                    for(idx_type<T> j = 0; j < num_neighbors; ++j)
                        maxheap.push( make_pair<T, idx_type<T> >( (T) dist[h*localn+j], (idx_type<T>) j) );
                    
                    for(idx_type<T> j = num_neighbors; j < localn; ++j){
                        maxheap.push( make_pair< T, idx_type<T> >( (T) dist[h*localn+j], (idx_type<T>) j) );
                        maxheap.pop();
                    }
                    for(idx_type<T> j = num_neighbors-1; j >=0; --j){
                        neighbor_dist[l][querynum*k+j] = maxheap.top().first;
                        neighbor_list[l][querynum*k+j] = maxheap.top().second;
                        maxheap.pop();
                    }
                } // end loop over query points in block
            } //end last block
            //This is the normal case (interior blocks)
            else{
               compute_distances(localR, currquery, localn, blocksize, d, dist, sqnormr, sqnormq, useSqnormrInput);
:q
                
               priority_queue< neigh_type<T>, vector< neigh_type<T> >, maxheap_comp<T> > maxheap;

               for(idx_type<T> h = 0; h < blocksize; ++h){
                    while(!maxheap.empty()) maxheap.pop();
                    const idx_type<T> querynum = i*blocksize + h;
                    for(idx_type<T> j = 0; j < num_neighbors; ++j)
                        maxheap.push( make_pair<T, idx_type<T> >( (T) dist[h*localn+j], (idx_type<T>) j) );
                    
                    for(idx_type<T> j = num_neighbors; j < localn; ++j){
                        maxheap.push( make_pair< T, idx_type<T> >( (T) dist[h*localn+j], (idx_type<T>) j) );
                        maxheap.pop();
                    }

                    for(idx_type<T> j = num_neighbors-1; j >=0; --j){
                        neighbor_dist[l][querynum*k+j] = maxheap.top().first;
                        neighbor_list[l][querynum*k+j] = maxheap.top().second;
                        maxheap.pop();
                    }
               } // end loop over query points in block
            } //end block
            useSqnormrInput = true;
        } //end loop over blocks

        if(dealloc_dist) delete[] dist;
        if(dealloc_sqnormq) delete [] sqnormr;
        if(dealloc_sqnormq) delete [] sqnormq;
    } //end loop over leaves
} //end function


template<typename T>
void batchedRef(idx_type<T> **rgids,
             idx_type<T> **qgids,
             T **R, T **Q,
             const idx_type<T> *n,
             const idx_type<T>  d,
             const idx_type<T> *m,
             const idx_type<T>  k,
             idx_type<T>** neighbor_list,
             T** neighbor_dist, 
             const idx_type<T>  nleaves
            ){

    idx_type<T> maxt = (idx_type<T>) omp_get_max_threads();

    #pragma omp parallel for
    for(idx_type<T> l=0; l < nleaves; ++l){
        
        const idx_type<T> localm = m[l];
        const idx_type<T> localn = n[l];

        idx_type<T> blocksize = getBlockSize(localn, localm);

        T* localR = (T*)R[l];
        T* localQ = (T*)Q[l];
        idx_type<T>* local_rgids= (idx_type<T>*) rgids[l];
        
        idx_type<T>* local_qgids= new idx_type<T>[blocksize];
        for(idx_type<T> z = 0; z < blocksize; ++z) local_qgids[z] = z;
        
        idx_type<T> nblocks = localm / blocksize;
        idx_type<T> iters = (idx_type<T>) ceil( (double) localm / (double) blocksize );

        bool dealloc_sqnormr = false;
        bool dealloc_sqnormq = false;
        bool dealloc_dist = false;

        T *sqnormr = new T[localn];
        T *sqnormq = new T[blocksize];

        dealloc_sqnormr = true;
        dealloc_sqnormq = true;

       bool useSqnormrInput = false;
       
       //Loop over all blocks
       for(idx_type<T> i = 0; i < iters; ++i){
            T *currquery = localQ + i*blocksize*d;
            idx_type<T> current_blocksize = blocksize;

            //Handle the edge case if blocksize does not divide localn evenly
            if ( (i == iters - 1) && (localm%blocksize) ) {
                current_blocksize = localm%blocksize;
            } //end last block

           const idx_type<T> offset = i*blocksize;
           
            if(!useSqnormrInput){
                sqnorm((T*) localR, (idx_type<T>) localn, (idx_type<T>) d, (T *)sqnormr);
            }
           sqnorm((T*) currquery, (idx_type<T>) current_blocksize, (idx_type<T>) d, (T*) sqnormq);

           //sgsknn_ref_stl(localn, current_blocksize, d, k, localR, sqnormr, (idx_type<T>*) local_rgids, currquery, sqnormq, (idx_type<T>*) (local_qgids), (T*) (neighbor_dist[l]+offset*k), (idx_type<T>*) (neighbor_list[l]+offset*k));
           
           useSqnormrInput = true;
        } //end loop over blocks
        //printf("Leaf %d: Ending Loop \n", l);

        delete [] local_qgids;
        if(dealloc_sqnormr) delete [] sqnormr;
        if(dealloc_sqnormq) delete [] sqnormq;
//        printf("Finished dealloc\n");
    } //end loop over leaves

} //end function



template<typename T>
void batchedGSKNN(idx_type<T> **rgids,
             idx_type<T> **qgids,
             T **R, T **Q,
             const idx_type<T> *n,
             const idx_type<T>  d,
             const idx_type<T> *m,
             const idx_type<T>  k,
             idx_type<T>** neighbor_list,
             T** neighbor_dist, 
             const idx_type<T>  nleaves, 
             const int cores
            ){

    omp_set_num_threads(cores);
    //printf("Started C++ Section \n");
    idx_type<T> maxt = (idx_type<T>) omp_get_max_threads();
    //printf("MAXT %d\n", maxt);
    //Allocate neighborlist & neighbor_dist
    //neighbor_list = new idx_type<T>*[nleaves];
    //neighbor_dist = new T*[nleaves];

    #pragma omp parallel for
    for(idx_type<T> l=0; l < nleaves; ++l){
        
    //    printf("NUMT %d\n", omp_get_num_threads());
    //    printf("Starting New Leaf %d \n", l);
        const idx_type<T> localm = m[l];
        const idx_type<T> localn = n[l];

        idx_type<T> blocksize = getBlockSize(localn, localm);

        //neighbor_list[l] = new idx_type<T>[k*localm];
        //neighbor_dist[l] = new T[k*localm];

        T* localR = (T*)R[l];
        T* localQ = (T*)Q[l];
        idx_type<T>* local_rgids= (idx_type<T>*) rgids[l];
        //idx_type<T>* local_qgids= (idx_type<T>*) qgids[l];
        idx_type<T>* local_qgids= new idx_type<T>[localn];
        for(idx_type<T> z = 0; z < localn; ++z) local_qgids[z] = z;
        
        //Verify all of these exist and can be accessed
        /*
        for(idx_type<T> z=0; z < d*localn; ++z){
            std::cout << "R " << localR[z] <<std::endl;
        }
        for(idx_type<T> z=0; z < d*localm; ++z){
            std::cout << "Q " << localQ[z] <<std::endl;
        }
        for(idx_type<T> z=0; z < localm; ++z){
            std::cout << "Qgids" << local_qgids[z] <<std::endl;
        }

        for(idx_type<T> z=0; z < localm; ++z){
            local_qgids[z] = z;
            std::cout << "Qgids" << local_qgids[z] <<std::endl;
        }

        for(idx_type<T> z=0; z < localn; ++z){
            std::cout << "Rgids" << local_rgids[z] <<std::endl;
        }
        */

        
        //idx_type<T>* qgids = new idx_type<T>[localm];

        idx_type<T> nblocks = localm / blocksize;
        idx_type<T> iters = (idx_type<T>) ceil( (double) localm / (double) blocksize );

        bool dealloc_sqnormr = false;
        bool dealloc_sqnormq = false;
        bool dealloc_dist = false;

        T *sqnormr = new T[localn];
        T *sqnormq = new T[blocksize];

        dealloc_sqnormr = true;
        dealloc_sqnormq = true;

       bool useSqnormrInput = false;

       //printf("Leaf %d : Allocated Space. Blocksize = %d\n LOCALM = %d LOCALN = %d \n Starting Loop\n", l, blocksize, localm, localn);
       //Loop over all blocks
       for(idx_type<T> i = 0; i < iters; ++i){
            //printf("Starting Block %d\n", i);
            T *currquery = localQ + i*blocksize*d;
            idx_type<T> current_blocksize = blocksize;

            //Handle the edge case if blocksize does not divide localn evenly
            if ( (i == iters - 1) && (localm%blocksize) ) {
                current_blocksize = localm%blocksize;
                //printf("On Last Block. blocksize=%d", current_blocksize);            
            } //end last block

           const idx_type<T> offset = i*blocksize;
      //     printf("Leaf %d; Block %d; offset %d \n",l, i, offset); 
           if(!useSqnormrInput){
                //printf("Calculating R2 \n");
                sqnorm((T*) localR, (idx_type<T>) localn, (idx_type<T>) d, (T *)sqnormr);
            }
           //printf("Calculating Q2. current_blocksize = %d\n", current_blocksize);
           sqnorm((T*) currquery, (idx_type<T>) current_blocksize, (idx_type<T>) d, (T*) sqnormq);

           //printf("Finished Squared\n");
           heap_t *heap = heapCreate_s(current_blocksize, k, 1.79E+30);
           //printf("Leaf %d; block %d; Calling GSKNN\n", l, i);
           //sgsknn(localn, current_blocksize, d, k, localR, sqnormr, (idx_type<T>*) local_rgids, currquery, sqnormq, (idx_type<T>*) (qgids), heap);
           sgsknn(localn, current_blocksize, d, k, localR, sqnormr, (idx_type<T>*) (local_qgids), currquery, sqnormq, (idx_type<T>*) (local_qgids), heap);
           //printf("Leaf %d; block %d; FINISHED GSKNN\n", l, i); 
          
 
           //copy over results
           auto D = heap->D_s;
           auto I = heap->I;
           auto ldk = heap->ldk;
           //printf("Leaf %d, Starting copy\n", l);
           #pragma omp parallel for  //nested parallelism?
           for(idx_type<T> j = 0; j < current_blocksize; ++j){
                #pragma ivdep
                for(idx_type<T> h = 0; h < k; ++h){
                    neighbor_dist[l][(j+offset)*k+h] = D[j*ldk+h];
                    neighbor_list[l][(j+offset)*k+h] = rgids[l][I[j*ldk+h]];
                    //printf("Leaf %d; block %d; Seeing Index %d at (%d, %d, %d)\n", l, i, I[j*ldk+h], j, ldk, h);
                }
           } //end copy 
           heapFree_s( heap );
           //printf("Leaf %d, Ending Copy\n", l);
           useSqnormrInput = true;
        } //end loop over blocks
        //printf("Leaf %d: Ending Loop \n", l);

        delete [] local_qgids;
        if(dealloc_sqnormr) delete [] sqnormr;
        if(dealloc_sqnormq) delete [] sqnormq;
//        printf("Finished dealloc\n");
    } //end loop over leaves

} //end function


/*
//Distance Kernel
template<typename T>
float* Distances(float* Q, float* R){
    ///cblas_sgemm(LAYOUT, TRANSA, TRANSB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    T* D = &T(0);
    return D;
}
*/


void sort_select(const float *value, const int *ID, int n, float *kval, int *kID, int k) {
      std::vector<int> idx(n);
      std::iota(idx.begin(), idx.end(), 0);
      std::stable_sort(idx.begin(), idx.end(),
          [&value](int i, int j) {return value[i]<value[j];});
      
        for (int i=0; i<k; i++) {
            int j = idx[i];
            kval[i] = value[j];
            kID[i] = ID[j];
      }
}

/**
template<typename T>
void merge_neighbor(const T *D2PtrL, const int *IDl, int kl, 
                    const T *D2PtrR, const int *IDr, int kr, 
                    T *nborDistPtr, int *nborIDPtr,  int k) {

      std::vector<T> D2(kl+kr);
      std::vector<int> ID(kl+kr);
      std::memcpy(D2.data(), D2PtrL, sizeof(T)*kl);
      std::memcpy(D2.data()+kl, D2PtrR, sizeof(T)*kr);
      std::memcpy(ID.data(), IDl, sizeof(int)*kl);
      std::memcpy(ID.data()+kl, IDr, sizeof(int)*kr);

      // (sort and) unique
      std::vector<int> idx(kl+kr);
      std::iota(idx.begin(), idx.end(), 0);
      std::stable_sort(idx.begin(), idx.end(),
           [&ID](int i, int j) {return ID[i] < ID[j];});
      std::stable_sort(ID.begin(), ID.end());
      std::vector<int> idx2(kl+kr);
      std::iota(idx2.begin(), idx2.end(), 0);
      std::unique(idx2.begin(), idx2.end(),
                       [&ID](int i, int j) {return ID[i]==ID[j];});
      ID.erase(std::unique(ID.begin(), ID.end()), ID.end());
      std::vector<T> value(ID.size());
      for (size_t i=0; i<ID.size(); i++) {
        int j = idx2[i];
        value[i] = D2[ idx[j] ];
      }
      // call k-select
      sort_select(value.data(), ID.data(), ID.size(), nborDistPtr, nborIDPtr, k);
}
**/
/**
template<typename T>
void merge_neighbor_gofmm(const T *D2PtrL, const int *IDl, 
                    const T *D2PtrR, const int *IDr,
                    T *nborDistPtr, int *nborIDPtr,  int k) {

      // Enlarge temporary buffer if it is too small.
      
      std::vector<std::pair<T, int>> aux;
      aux.resize(2*k);

      // Merge two lists into one.
      for ( sizeType i = 0; i < k; i++ ) 
      {
        aux[                 i ] = std::make_pair<T, int>(D2PtrL[i], ;
        aux[ num_neighbors + i ] = B[ i ];
      }

      // First sort according to the index.
      std::sort( aux.begin(), aux.end(), less_value<T, TINDEX> );
      auto last = std::unique( aux.begin(), aux.end(), equal_value<T, TINDEX> );
      std::sort( aux.begin(), last, less_key<T, TINDEX> );

      // Copy the results back from aux.
      for ( sizeType i = 0; i < num_neighbors; i++ ) 
      {
        A[ i ] = aux[ i ];
      }

}; // end MergeNeighbors()
**/

template<typename key_t, typename value_t>
inline bool less_key(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b ){
    return (a.first < b.first);
}

template<typename key_t, typename value_t>
inline bool less_value(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b){
    return (a.second < b.second);
}

template<typename key_t, typename value_t>
inline bool equal_key(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b ){
    return (a.first == b.first);
}

template<typename key_t, typename value_t>
inline bool equal_value(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b){
    return (a.second == b.second);
}


template<typename T>
void merge_neighbor_cpu(T* D1, unsigned int* I1, T* D2, unsigned int* I2, const unsigned int n, const int k, const int cores){
    omp_set_num_threads(cores);

    #pragma omp parallel
    {

        std::vector<std::pair<unsigned int, T>> aux(2*k);

        #pragma omp for
        for(size_t i = 0; i < n; ++i){
            size_t offset = i * k;
            //Merge twolists
            for(size_t j = 0; j < k; ++j){
                aux[ j]   = std::make_pair(I1[offset + j], D1[offset + j]);
                aux[ k+j] = std::make_pair(I2[offset + j], D2[offset + j]);
            }

            //Sort according to index
            std::sort(aux.begin(), aux.end(), less_key<unsigned int, T>);
            auto last = std::unique(aux.begin(), aux.end(), equal_key<unsigned int, T> );
            std::sort(aux.begin(), last, less_value<unsigned int, T>);

            //Copy Results back to output
            for(size_t j = 0; j < k; ++j){
                auto current = aux[j];
                I1[offset+j] = current.first;
                D1[offset+j] = current.second;
            }
        }
        
    }
    
}

template<typename T>
void test(){
    const int n = 10;
    #pragma omp parallel for
    for(idx_type<T> l = 0; l < n; ++l){
        std::cout << "Number of Current Threads: " << omp_get_num_threads() << std::endl;
        std::cout << "Iteration of Loop: "<< l << std::endl;
    }
}

template<typename T>
std::vector<T> sampleWithoutReplacement(idx_type<T> l, std::vector<T> v)
{
    if( l >= v.size() )
    {
        return v;
    }

    std::random_device rd;
    std::mt19937 generator( rd() );

    std::shuffle(v.begin(), v.begin() + l, generator);
    vector<T> ret(v.begin(), v.begin() + l);

    return ret;
}


/** use default stl allocator */
template<class T, class Allocator = std::allocator<T> >
vector<T> Sum(idx_type<T> d, idx_type<T> n, vector<T, Allocator> &X, vector<idx_type<T>> &gids )
{
  bool do_general_stride = ( gids.size() == n );

  /** assertion */
  if ( !do_general_stride ) assert( X.size() == d * n );

  /** declaration */
  int n_split = omp_get_max_threads();
  std::vector<T> sum( d, 0.0 );
  std::vector<T> temp( d * n_split, 0.0 );

  /** compute partial sum on each thread */
  #pragma omp parallel for num_threads( n_split )
  for ( idx_type<T> j = 0; j < n_split; j ++ )
    for ( idx_type<T> i = j; i < n; i += n_split )
      for ( idx_type<T> p = 0; p < d; p ++ )
        if ( do_general_stride )
          temp[ j * d + p ] += X[ gids[ i ] * d + p ];
        else
          temp[ j * d + p ] += X[ i * d + p ];

  /** reduce all temporary buffers */
  for ( idx_type<T> j = 0; j < n_split; j ++ )
    for ( idx_type<T> p = 0; p < d; p ++ )
      sum[ p ] += temp[ j * d + p ];

  return sum;
}; /** end Sum() */


// Multi-core version can use std::reduce. std::reduce is only available in c++17;
template<class T>
T Accumulate(std::vector<T> &v, T & sum_glb)
{
  /* Initialize global sum to zero. */
  // sum_glb = static_cast<T>(0);
  return std::accumulate(v.begin(), v.end(), sum_glb);
  // return std::reduce(std::execution::par, v.begin(), v.end());
}

template<class T>
T Reduce(std::vector<T> &v, T & sum_glb)
{
  #pragma omp parallel for reduction(+:sum_glb)
  for (idx_type<T> i = 0; i < v.size(); i++){
    sum_glb += v[i];
  }
  return sum_glb;
}


/**
 *  @brief Parallel prefix scan
 */ 
template<typename TA, typename TB>
void Scan( std::vector<TA> &A, std::vector<TB> &B )
{
  assert( A.size() == B.size() - 1 );

  /** number of threads */
  idx_type<TA> p = omp_get_max_threads();

  /** problem size */
  idx_type<TB> n = B.size();

  /** step size */
  idx_type<TB> nb = n / p;

  /** private temporary buffer for each thread */
  std::vector<TB> sum( p, (TB)0 );

  /** B[ 0 ] = (TB)0 */
  B[ 0 ] = (TB)0;

  /** small problem size: sequential */
  if ( n < 100 * p ) 
  {
    idx_type<TB> beg = 0;
    idx_type<TB> end = n;
    for ( idx_type<TB> j = beg + 1; j < end; j ++ ) 
      B[ j ] = B[ j - 1 ] + A[ j - 1 ];
    return;
  }

  /** parallel local scan */
  #pragma omp parallel for schedule( static )
  for ( idx_type<TB> i = 0; i < p; i ++ ) 
  {
    idx_type<TB> beg = i * nb;
    idx_type<TB> end = beg + nb;
    /** deal with the edge case */
    if ( i == p - 1 ) end = n;
    if ( i != 0 ) B[ beg ] = (TB)0;
    for ( idx_type<TB> j = beg + 1; j < end; j ++ ) 
    {
      B[ j ] = B[ j - 1 ] + A[ j - 1 ];
    }
  }

  /** sequential scan on local sum */
  for ( idx_type<TB> i = 1; i < p; i ++ ) 
  {
    sum[ i ] = sum[ i - 1 ] + B[ i * nb - 1 ] + A[ i * nb - 1 ];
  }

  #pragma omp parallel for schedule( static )
  for ( idx_type<TB> i = 1; i < p; i ++ ) 
  {
    idx_type<TB> beg = i * nb;
    idx_type<TB> end = beg + nb;
    /** deal with the edge case */
    if ( i == p - 1 ) end = n;
    TB sum_ = sum[ i ];
    for ( idx_type<TB> j = beg; j < end; j ++ ) 
    {
      B[ j ] += sum_;
    }
  }

}; /** end Scan() */


template<typename TA, typename TB>
std::vector<TB> Scan( std::vector<TA> &A )
{
  std::vector<TB> B = std::vector<TB>( A.size(),static_cast<TB>(0) );
  Scan(A,B);
  return B;
}


/**
 *  @brief Select the kth element in x in the increasing order.
 *
 *  @para  
 *
 *  @TODO  The mean function is parallel, but the splitter is not.
 *         I need something like a parallel scan.
 */ 
template<typename T>
T Select(idx_type<T> n, idx_type<T> k, std::vector<T> &x )
{

  /** assertion */
  // size_t n = x.size()
  assert( k <= n && n == x.size());

  /** Early return */
  if ( n == 1 )
  {
    return x[ 0 ];
  }

  T mean = std::accumulate(x.begin(), x.end(), static_cast<T>(0)) / x.size();

  std::vector<T> lhs, rhs;
  std::vector<idx_type<T>> lflag( n, 0 );
  std::vector<idx_type<T>> rflag( n, 0 );
  std::vector<idx_type<T>> pscan( n + 1, 0 );

  /** mark flags */
  #pragma omp parallel for
  for ( idx_type<T> i = 0; i < n; i ++ )
  {
    if ( x[ i ] > mean ) rflag[ i ] = 1;
    else                 lflag[ i ] = 1;
  }
  
  /** 
   *  prefix sum on flags of left hand side 
   *  input:  flags
   *  output: zero-base index
   **/
  Scan( lflag, pscan );

  /** resize left hand side */
  lhs.resize( pscan[ n ] );

  #pragma omp parallel for 
  for (idx_type<idx_type<T>> i = 0; i < n; i ++ )
  {
	  if ( lflag[ i ] ) 
      lhs[ pscan[ i ] ] = x[ i ];
  }

  /** 
   *  prefix sum on flags of right hand side 
   *  input:  flags
   *  output: zero-base index
   **/
  Scan( rflag, pscan );

  /** resize right hand side */
  rhs.resize( pscan[ n ] );

  #pragma omp parallel for 
  for (idx_type<T> i = 0; i < n; i ++ )
  {
	  if ( rflag[ i ] ) 
      rhs[ pscan[ i ] ] = x[ i ];
  }

  idx_type<T> nlhs = lhs.size();
  idx_type<T> nrhs = rhs.size();

  if ( nlhs == k || nlhs == n || nrhs == n ) 
  {
    return mean;
  }
  else if ( nlhs > k )
  {
    rhs.clear();
    return Select( nlhs, k, lhs );
  }
  else
  {
    lhs.clear();
    return Select( nrhs, k - nlhs, rhs );
  }

}; /** end Select() */


template<typename T>
T Select( idx_type<T> k, std::vector<T> &x )
{
  return Select(x.size(), k, x);
}


template<typename T>
std::vector< std::vector<uint64_t> > MedianThreeWaySplit( std::vector<T> &v, T tol )
{
  uint64_t n = v.size();
  T median = Select( n, 0.5 * n, v );

  T left = median;
  T right = median;
  T perc = 0.0;

  while ( left == median || right == median )
  {
    if ( perc == 0.5 ) 
    {
      break;
    }
    perc += 0.1;
    left = Select( n, ( 0.5 - perc ) * n, v );
    right = Select( n, ( 0.5 + perc ) * n, v );
  }

  /** Split indices of v into 3-way: lhs, rhs, and mid. */
  std::vector< std::vector<uint64_t> > three_ways( 3 );
  std::vector<uint64_t> & lhs = three_ways[ 0 ];
  std::vector<uint64_t> & rhs = three_ways[ 1 ];
  std::vector<uint64_t> & mid = three_ways[ 2 ];
  for ( uint64_t i = 0U; i < v.size(); i ++ )
  {
    //if ( std::fabs( v[ i ] - median ) < tol ) mid.push_back( i );
    if ( v[ i ] >= left && v[ i ] <= right ) 
    {
      mid.push_back( i );
    }
    else if ( v[ i ] < median ) 
    {
      lhs.push_back( i );
    }
    else 
    {
      rhs.push_back( i );
    }
  }
  return three_ways;
}; /* end MedianTreeWaySplit() */



/** @brief Split values into two halfs accroding to the median. */ 
template<typename T>
std::vector< std::vector<uint64_t> > MedianSplit(std::vector<T> &v)
{
  std::vector< std::vector<uint64_t> > three_ways = MedianThreeWaySplit( v, (T)1E-6 );
  std::vector< std::vector<uint64_t> > two_ways( 2 );
  two_ways[0] = three_ways[0 ];
  two_ways[1] = three_ways[1 ];  
  std::vector<uint64_t> & lhs = two_ways[ 0 ];
  std::vector<uint64_t> & rhs = two_ways[ 1 ];
  std::vector<uint64_t> & mid = three_ways[ 2 ];
  for ( std::vector<uint64_t>::iterator it = mid.begin(); it != mid.end(); ++it )
  {
    if ( lhs.size() < rhs.size() )
    {
      lhs.push_back( *it );
    }
    else 
    {
      rhs.push_back( *it );
    }
  }
  return two_ways;
}; /* end MedianSplit() */

#endif /* define PRIMITIVES_CPU_HPP */
