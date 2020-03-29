#include "util_gpu.hpp"
#include "merge_gpu.hpp"
#include "timer_gpu.hpp"
#include "knn_handle.hpp"

#include <thrust/random.h>
#include <limits>       // std::numeric_limits


// implemented in kernel_leaf.cu
void find_knn(int *ID, int *rowPtrP, int *colIdxP, float *valP, 
    int n, int d, int nnzP, int *seghead, int nLeaf, int m, int maxPoint,
    int *nborID, float *nborDist, int k, int LD);


// implemented in kernel_tree.cu
void create_tree_next_level(int *ID, int *rowPtrP, int *colIdxP, float *valP, 
    int n, int d, int nnz, int *seghead, int *segHeadNext, int nNode,
    float *valX, float *median,
    float&, float&, float&);


template <typename T>
T diff(T *x, T *y, int n) {
  dptr<T> xptr(x), yptr(y);
  dvec<T> err(n);
  thrust::transform(xptr, xptr+n, yptr, err.begin(), thrust::minus<T>());
  auto iter = thrust::make_transform_iterator(err.begin(), thrust::square<T>());
  return thrust::reduce(iter, iter+n);
}


void dprint(int m, int n, int nnz, dvec<int> &rowPtr, dvec<int> &colIdx, dvec<float> &val,
    const std::string &name) {
  dprint(m, n, nnz, 
      thrust::raw_pointer_cast(rowPtr.data()),
      thrust::raw_pointer_cast(colIdx.data()),
      thrust::raw_pointer_cast(val.data()),
      name);
}


struct prg: public thrust::unary_function<unsigned int, float> {
  float a, b;

  __host__ __device__
  prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

  __host__ __device__
  float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};


// m: blocking size in distance calculation
void spknn(int *hRowPtr, int *hColIdx, float *hVal, int n, int d, int nnz, int level,
    int *hNborID, float *hNborDist, int k, int m=64) {
 
  //print(n, d, nnz, hRowPtr, hColIdx, hVal, "host P");
  
  const int nLeaf = 1<<(level-1);
  const int maxPoint = (n+nLeaf-1)/nLeaf;
  const int nExtra = maxPoint*nLeaf - n;
  
  // copy data to GPU
  dvec<int> dRowPtr(n+1+nExtra);
  dvec<int> dColIdx(nnz+nExtra);
  dvec<float> dVal(nnz+nExtra);

  thrust::copy(hRowPtr, hRowPtr+n+1, dRowPtr.begin());
  thrust::copy(hColIdx, hColIdx+nnz, dColIdx.begin());
  thrust::copy(hVal, hVal+nnz, dVal.begin());
  
  // insert artificial points at infinity
  thrust::sequence(dRowPtr.begin()+n, dRowPtr.end(), hRowPtr[n]);
  thrust::fill(dColIdx.begin()+nnz, dColIdx.end(), 0); // the first coordinate is infinity
  thrust::fill(dVal.begin()+nnz, dVal.end(), std::numeric_limits<float>::max()); 

  // update # points and # nonzeros
  n += nExtra;
  nnz += nExtra;

  //dprint(n, d, nnz, dRowPtr, dColIdx, dVal, "device P");
  
  
  std::cout<<"\n========================"
           <<"\nPoints"
           <<"\n------------------------"
           <<"\n# points: "<<n
           <<"\n# dimensions: "<<d
           <<"\n# artificial points: "<<nExtra
           <<"\n# points/leaf: "<<maxPoint
           <<"\n# leaf nodes: "<<nLeaf
           <<"\n------------------------"
           <<"\nsparsity: "<<100.*nnz/n/d<<" %"
           <<"\nmemory: "<<(n+1)/1.e9*4+nnz/1.e9*4*2<<" GB"
           <<"\nmem projection: "<<d/1.e9*4*2*nLeaf<<" GB"
           <<"\n========================\n"
           <<std::endl;


  // -----------------------
  //    Setup for SPKNN
  // -----------------------
  // compute number of points in every node
  dvec<int> nPoints[level];
  nPoints[0].resize(1, n); // root
  for (int i=1; i<level; i++) {
    nPoints[i].resize( 1<<i );
    // even index gets half from parent
    dvec<int> evenIdx( 1<<(i-1) );
    thrust::sequence(evenIdx.begin(), evenIdx.end(), 0, 2);
    auto TWO  = thrust::make_constant_iterator<int>(2);
    auto evenElm = thrust::make_permutation_iterator(nPoints[i].begin(), evenIdx.begin());
    thrust::transform(nPoints[i-1].begin(), nPoints[i-1].end(), TWO, evenElm, thrust::divides<int>());
    // odd index gets the rest half
    auto OddElm = thrust::make_permutation_iterator(nPoints[i].begin()+1, evenIdx.begin());
    thrust::transform(nPoints[i-1].begin(), nPoints[i-1].end(), evenElm, OddElm, thrust::minus<int>());
    //print(nPoints[i], "# points");
  }

  // compute segment starts for all levels
  dvec<int> seghead[level];
  for (int i=0; i<level; i++) {
    seghead[i].resize( (1<<i)+1, 0 );
    thrust::inclusive_scan(nPoints[i].begin(), nPoints[i].end(), seghead[i].begin()+1);
    //print(seghead[i], "segment heads");
  }

  // generate random arrays
  dvec<float> dX[level-1];
  for (int i=0; i<level-1; i++) {
    int nNode = 1<<i;
    dX[i].resize( nNode*d );
    auto start = thrust::make_counting_iterator<int>( (nNode-1)*d );
    thrust::transform(start, start+nNode*d, dX[i].begin(), prg());
    //print(dX[i], "X");
  }

  // output median in tree construction
  dvec<float> median[level-1];
  for (int i=0; i<level-1; i++)
    median[i].resize( 1<<i );
  

  // -----------------------
  // output
  // -----------------------
  dvec<int> dNborID(n*k, std::numeric_limits<int>::max());
  dvec<float> dNborDist(n*k, std::numeric_limits<float>::max());

  // -----------------------
  // timing
  // -----------------------
  float t_build = 0., t_knn = 0., t_merge = 0.;
  TimerGPU t;

  // -----------------------
  // Start SPKNN
  // -----------------------
  // build tree
  float t_gemm = 0., t_sort = 0., t_mat = 0.;
  int nTree = 1;
  for (int tree=0; tree<nTree; tree++) {

    /*
    // create a copy of the input
    dvec<int> dRowPtrCpy = dRowPtr;
    dvec<int> dColIdxCpy = dColIdx;
    dvec<float> dValCpy  = dVal;
    int *d_rowPtr = thrust::raw_pointer_cast(dRowPtrCpy.data());
    int *d_colIdx = thrust::raw_pointer_cast(dColIdxCpy.data());
    float *d_val  = thrust::raw_pointer_cast(dValCpy.data());
    */
    //std::cout<<"Created a copy of points."<<std::endl;
    int *d_rowPtr = thrust::raw_pointer_cast(dRowPtr.data());
    int *d_colIdx = thrust::raw_pointer_cast(dColIdx.data());
    float *d_val  = thrust::raw_pointer_cast(dVal.data());
    
    // create tree
    dvec<int> ID(n); // global ID of all points
    thrust::sequence(ID.begin(), ID.end(), 0);
    t.start();
    for (int i=0; i<level-1; i++) {
      create_tree_next_level(
          thrust::raw_pointer_cast(ID.data()),
          // CSR format of sparse data
          d_rowPtr, 
          d_colIdx,
          d_val,
          n, d, nnz, 
          thrust::raw_pointer_cast(seghead[i].data()), 
          thrust::raw_pointer_cast(seghead[i+1].data()), 
          1<<i, // # tree nodes
          thrust::raw_pointer_cast(dX[i].data()), // random projection
          thrust::raw_pointer_cast(median[i].data()), // output
          t_gemm, t_sort, t_mat);
    }
    t.stop();
    t_build += t.elapsed_time();
    std::cout<<"Finished tree construction"<<std::endl;

    //tprint(ID, "ID");
    //dprint(n, d*(1<<(level-1)), nnz, d_rowPtr, d_colIdx, d_val, "reordered P");

#if 0
    // check ID
    // compute B = Perm * A
    int *rowPtrB, *colIdxB, nnzB;
    float *valB;
    
    int *rowPtrP = thrust::raw_pointer_cast(dRowPtr.data());
    int *colIdxP = thrust::raw_pointer_cast(dColIdx.data());
    float *valP  = thrust::raw_pointer_cast(dVal.data());
    permute_sparse_matrix(n, d, nnz, rowPtrP, colIdxP, valP,
        ID, rowPtrB, colIdxB, valB, nnzB,
        info, handle, descr);
    
    assert(nnz == nnzB);
    int errRowPtr = diff(rowPtrB, d_rowPtr, n+1);
    float errVal  = diff(valB, d_val, nnz);
    std::cout<<"Error of row pointer: "<<errRowPtr
             <<"\nError of value: "<<errVal
             <<std::endl;

    // free allocation from calling GEMM_SSS
    CHECK_CUDA( cudaFree(rowPtrB) )
    CHECK_CUDA( cudaFree(colIdxB) )
    CHECK_CUDA( cudaFree(valB) )
#endif
    

    // compute neighbors at leaf level
    dvec<int> curNborID(n*k, -1);
    dvec<float> curNborDist(n*k, -1.);
    int *curID = thrust::raw_pointer_cast(curNborID.data());
    float *curDist = thrust::raw_pointer_cast(curNborDist.data());
    

    int *pts  = thrust::raw_pointer_cast(nPoints[level-1].data());
    int maxPoint = thrust::reduce(thrust::device, pts, pts+nLeaf, 0, thrust::maximum<int>());
    int *segPtr = thrust::raw_pointer_cast(seghead[level-1].data());
    int offset = 0;
    int LD = k; // leading dimension
    // int LD = 2*k;
    //if (t==1) {offset = 0;}
    //else {offset = k;}
    int *preID = thrust::raw_pointer_cast(dNborID.data()+offset);
    float *preDist = thrust::raw_pointer_cast(dNborDist.data()+offset);

    t.start();
    find_knn(thrust::raw_pointer_cast(ID.data()), 
        d_rowPtr, d_colIdx, d_val,
        n, d, nnz, segPtr, nLeaf, m, maxPoint,
        curID, curDist, k, LD);
    t.stop();
    t_knn += t.elapsed_time();

    //dprint(3, k, curDist, "curDist");
    //dprint(3, k, curID, "curID");


    // update previous results
    t.start();
    merge_neighbors_gpu(preDist, preID, curDist, curID, n, k, k);
    t.stop();
    t_merge += t.elapsed_time();
  }
  

  std::cout<<"\n==========================="
           <<"\n    Sparse KNN Timing"
           <<"\n---------------------------"
           <<"\n* build tree: "<<t_build<<" s"
           <<"\n\t- gemm: "<<t_gemm<<" s"
           <<"\n\t- sort: "<<t_sort<<" s"
           <<"\n\t- matrix: "<<t_mat<<" s"
           <<"\n* KNN: "<<t_knn<<" s"
           <<"\n* merge: "<<t_merge<<" s"
           <<"\n===========================\n"
           <<std::endl;

  // -----------------------
  // Copy to CPU
  // -----------------------
  thrust::copy(dNborID.begin(), dNborID.end()-nExtra*k, hNborID);
  thrust::copy(dNborDist.begin(), dNborDist.end()-nExtra*k, hNborDist);
}

