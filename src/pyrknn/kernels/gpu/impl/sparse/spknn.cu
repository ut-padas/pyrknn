#include "util_gpu.hpp"
#include "timer_gpu.hpp"
#include "knn_handle.hpp"
#include "op_gpu.hpp"
#include "reorder.hpp"
#include "gemm.hpp"
#include "orthogonal.hpp"
#include "merge.hpp"
#include "print.hpp"
#include "matrix.hpp"
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>

#include <limits>       // std::numeric_limits

void build_tree(dvec<float> &P, int n, int level, dvec<int> &perm, float&);

void leaf_knn(int *dID, SpMat, ivec&, ivec&, fvec&, int maxPoint, 
    int *curID, float *curDist, int LD, int k, int m, fvec&, ivec&,
    float&, float&, float&, float&, float&, float&, float&, float&);
void sfi_leafknn(int* dR, int* dC, float* dV, int* d_ID, int n, int nLeaf, int k, float* curDist, int* curID); 

    
void get_submatrices(ivec &dID, int *curNborID, float *curNborDist, int LD,
    ivec &dRowPtr, ivec &dColIdx, fvec &dVal, int n, int d, int nnz, 
    int maxPoint, int nLeaf, int blkLeaf, int nBlock, SpMat *subBlock, int &maxNNZ,
    int *IDPtr[], int *nborIDPtr[], float *nborDistPtr[], int k) {

    ivec seghead(nBlock+1, 0);
    thrust::sequence(seghead.begin(), seghead.end(), 0, blkLeaf);
    auto NLEAF = thrust::make_constant_iterator<int>(nLeaf);
    auto NPOINT = thrust::make_constant_iterator<int>(maxPoint);
    thrust::transform(seghead.begin(), seghead.end(), NLEAF, seghead.begin(), thrust::minimum<int>());
    thrust::transform(seghead.begin(), seghead.end(), NPOINT, seghead.begin(), thrust::multiplies<int>());
    //tprint(seghead, "start of leaf points");


    int cumNNZ[nBlock+1];
    auto rowVal = thrust::make_permutation_iterator(dRowPtr.begin(), seghead.begin());
    thrust::copy(rowVal, rowVal+nBlock+1, cumNNZ);
    assert(cumNNZ[0] == 0  && cumNNZ[nBlock] == nnz);
    //print(cumNNZ, nBlock+1, "cumulative nnz");
   
    int P_rows[nBlock], P_nnz[nBlock], *P_rowPtr[nBlock], *P_colIdx[nBlock];
    float *P_val[nBlock];
    maxNNZ = 0;
    for (int blk=0; blk<nBlock; blk++) {
      int offset = blk*blkLeaf;
      int blkSize = std::min(blkLeaf, nLeaf-offset);
      
      P_rows[blk] = blkSize*maxPoint; // # points per batch
      P_nnz[blk] = cumNNZ[blk+1] - cumNNZ[blk];
      maxNNZ = std::max(maxNNZ, P_nnz[blk]);
      
      P_colIdx[blk] = thrust::raw_pointer_cast(dColIdx.data())+cumNNZ[blk];
      P_val[blk] = thrust::raw_pointer_cast(dVal.data())+cumNNZ[blk];
      
      CHECK_CUDA( cudaMalloc(&P_rowPtr[blk], (P_rows[blk]+1)*sizeof(int)) );

      iptr rowPtrP(P_rowPtr[blk]);
      thrust::constant_iterator<int> PreNNZ(cumNNZ[blk]);
      thrust::transform(dRowPtr.begin()+offset*maxPoint, 
          dRowPtr.begin()+(offset+blkSize)*maxPoint+1, 
          PreNNZ, rowPtrP, thrust::minus<int>());

      subBlock[blk] = 
        SpMat(P_rows[blk], d, P_nnz[blk], P_rowPtr[blk], P_colIdx[blk], P_val[blk], blkSize);
      //tprint(dRowPtr, "Matrix row pointer");
      //tprint(rowPtrP, "Submatrix row pointer");
      //dprint(P_rows[blk], d, P_nnz[blk], P_rowPtr[blk], P_colIdx[blk], P_val[blk], "submatrix");


      IDPtr[blk] = thrust::raw_pointer_cast(dID.data())+offset*maxPoint;
      nborIDPtr[blk] = curNborID + offset*maxPoint*LD;
      nborDistPtr[blk] = curNborDist + offset*maxPoint*LD; 
    }
}

    
void create_BDSpMat(SpMat matrix) {
    
  int n = matrix.rows(), d = matrix.cols(), nnz = matrix.nnz, nLeaf = matrix.nNodes;
  int *rowPtr = matrix.rowPtr;
  int *colIdx = matrix.colIdx;

  assert(n%nLeaf == 0);
  ivec pIdx(nLeaf);
  ivec nodeIdx(nnz);
  thrust::sequence(pIdx.begin(), pIdx.end(), 0, n/nLeaf);
  auto zero = thrust::make_counting_iterator<int>(0);
  auto nnzIdx = thrust::make_permutation_iterator(iptr(rowPtr), pIdx.begin());
  thrust::upper_bound(nnzIdx+1, nnzIdx+nLeaf, zero, zero+nnz, nodeIdx.begin());
  // compute shift
  thrust::constant_iterator<int> DIM(d);
  thrust::transform(nodeIdx.begin(), nodeIdx.end(), DIM, nodeIdx.begin(), thrust::multiplies<int>());
  iptr colPtr(colIdx);
  thrust::transform(colPtr, colPtr+nnz, nodeIdx.begin(), colPtr, thrust::plus<int>());
}


// column pointers point to 'colIdx' array
void destroy_BDSpMat(SpMat *subBlock, int nBlock, dvec<int>& colIdx) {

  thrust::constant_iterator<int> DIM(subBlock[0].cols());
  thrust::transform(colIdx.begin(), colIdx.end(), DIM, colIdx.begin(), thrust::modulus<int>());
    
  // free temporary resource
  for (int blk=0; blk<nBlock; blk++)
    CHECK_CUDA( cudaFree(subBlock[blk].rowPtr) );
}


int current_time_nanoseconds(){
    struct timespec tm;
    clock_gettime(CLOCK_REALTIME, &tm);
    return tm.tv_nsec;
}


/*
void build_sparse_tree(bool copy, int *hID, int *hRowPtr, int *hColIdx, float *hVal, 
    int* pdRowPtr, int* pdColIdx, float* pdVal, int nExtra, int n, int d, int nnz, int level, int device) {

    TimerGPU t, t2, t3;
    float copy_t = 0.0;
    float t_tsort = 0.0;

    const int nLeaf = 1<<level;
    const int maxPoint = (n+nLeaf-1)/nLeaf;

    thrust::device_ptr<int> dRowPtr = thurst::device_pointer_cast(pdRowPtr);
    thrust::device_ptr<int> dColIdx = thurst::device_pointer_cast(pdColIdx);
    thrust::device_ptr<float> dVal = thurst::device_pointer_cast(pdVal);

    //copy over local ids
    thrust::copy(hID, hID+n, dID.begin());
    
    //copy matrix from host 
    if (pull){
      thrust::copy(hRowPtr, hRowPtr+n+1, dRowPtr);
      thrust::copy(hColIdx, hColIdx+nnz, dColIdx);
      thrust::copy(hVal, hVal+nnz, dVal);
      cudaDeviceSynchronize();
    }

    n += nExtra;
    nnz += nExtra;

  //random seed
  int seed = current_time_nanoseconds();

  dvec<int> perm(n);
  //cluster points
  {
    //Allocate projection matrix
    dvec<float> R(d*level);
    dvec<float> P(n*level);

    //Initialize random numbers (gaussian)
    thrust::counting_iterator<int> start(tree*d*level);
    thrust::transform(start, start+d*level, R.begin(), prg(seed));

    //Compute projection
    //TODO: Check device_vector lengths
    GEMM_SDD(n, level, d, pdRowPtr, pdColIdx, pdVal, nnz, R, P);

    //Compute permutation
    build_tree(P, n, level, perm, t_tsort);

  }

  gather(dRowPtr, dColIdx, dVal, n, d, nnz, perm);
  gather(dID, perm);

  thrust::copy(dID, dID+n, hID);

  cudaDeviceSynchronize();
}
*/



// m: blocking size in distance calculation
void spknn(int *hID, int *hRowPtr, int *hColIdx, float *hVal, 
    int n, int d, int nnz, int level, int nTree,
    int *hNborID, float *hNborDist, int k, 
    int blkLeaf, int blkPoint, int device) {
 
  cudaSetDevice(device);
  
  //print(n, d, nnz, hRowPtr, hColIdx, hVal, "host P");
  
  TimerGPU t, t2, t3;

  const int nLeaf = 1<<level;
  const int maxPoint = (n+nLeaf-1)/nLeaf;
  const int nExtra = maxPoint*nLeaf - n;
  float t_copy = 0.;
  float t_neigh = 0.;
  float t_art = 0.;

  // copy coordinate data to GPU
  t.start();
  dvec<int> dID(n+nExtra);
  thrust::copy(hID, hID+n, dID.begin());
  
  dvec<int> dRowPtr(n+1+nExtra);
  dvec<int> dColIdx(nnz+nExtra);
  dvec<float> dVal(nnz+nExtra);
  thrust::copy(hRowPtr, hRowPtr+n+1, dRowPtr.begin());
  thrust::copy(hColIdx, hColIdx+nnz, dColIdx.begin());
  thrust::copy(hVal, hVal+nnz, dVal.begin());
  cudaDeviceSynchronize();
  t.stop(); t_copy += t.elapsed_time();

  //Initialize Result Output
  t.start();
  dvec<int> dNborID((n+nExtra)*k*2, std::numeric_limits<int>::max());
  dvec<float> dNborDist((n+nExtra)*k*2, std::numeric_limits<float>::max());
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iter  = thrust::make_transform_iterator(zero, firstKCols(k, 2*k));
  auto permI = thrust::make_permutation_iterator(dNborID.begin(), iter);
  auto permD = thrust::make_permutation_iterator(dNborDist.begin(), iter);
  //thrust::copy(hNborID, hNborID+n*k, permI);
  //thrust::copy(hNborDist, hNborDist+n*k, permD);
  //cudaDeviceSynchronize();
  t.stop(); t_neigh += t.elapsed_time();

  
  //tprint((n+nExtra), 2*k, dNborID, "nborID");
  //tprint((n+nExtra), 2*k, dNborDist, "nborDist");

  // insert artificial points at infinity
  t.start();
  thrust::sequence(dID.begin()+n, dID.end(), -nExtra, 1); // negative id
  thrust::sequence(dRowPtr.begin()+n, dRowPtr.end(), hRowPtr[n]);
  thrust::fill(dColIdx.begin()+nnz, dColIdx.end(), 0); // the first coordinate is infinity
  thrust::fill(dVal.begin()+nnz, dVal.end(), std::numeric_limits<float>::max()); 
  cudaDeviceSynchronize();
  t.stop(); t_art = t.elapsed_time();

  // update # points and # nonzeros
  n += nExtra;
  nnz += nExtra;

  //tprint(dRowPtr, "rowPtr");
  //dprint(n, d, nnz, dRowPtr, "device P");
  
  std::cout<<"\n========================"
           <<"\nPoints"
           <<"\n------------------------"
           <<"\n# points: "<<n
           <<"\n# dimensions: "<<d
           <<"\n# nonzeros: "<<nnz
           <<"\nmem: "<<(n+1)/1.e9*4+nnz/1.e9*4*2<<" GB"
           <<"\n------------------------"
           <<"\n# artificial points: "<<nExtra
           <<"\n# points/leaf: "<<maxPoint
           <<"\n# leaf nodes: "<<nLeaf
           <<"\n------------------------"
           <<"\nmem output: "<<n/1.e9*k*4*4<<" GB"
           <<"\nmem orthogonal bases: "<<d/1.e9*level*4<<" GB"
           <<"\nmem projection: "<<n/1.e9*level*4<<" GB"
           <<"\n========================\n"
           <<std::endl;


  // -----------------------
  // timing
  // -----------------------
  float t_tree = 0., t_knn = 0., t_merge = 0.;
  float t_sdd = 0., t_tsort = 0.; 
  float t_msort = 0., t_mcopy = 0., t_unique = 0.;
  float t_trans = 0., t_dist = 0., t_gemm = 0., t_sort = 0.;
  float t_sss = 0., t_den = 0., t_nnz = 0.;
  float t_orth = 0.;
  float t_rand = 0.;
  float t_org  = 0.;
  float sparse = 0.;
  float t_scatter = 0.;
  float t_sub = 0.;
  float t_bd = 0.;
  float t_alloc = 0.;
  
  t2.start();

  // local ordering
  dvec<int> order(n);
  thrust::sequence(order.begin(), order.end(), 0);

  // -----------------------
  // random seed
  // -----------------------
  int seed = current_time_nanoseconds();
 
  // -----------------------
  // Start SPKNN
  // -----------------------
  for (int tree=0; tree<nTree; tree++) {
  
    // cluster points
    {
      dvec<int> perm(n);
      // compute permutation
      {
        // allocate memory for random arrays and projections
        t.start();
        dvec<float> R(d*level);
        dvec<float> P(n*level);
        t.stop(); t_alloc += t.elapsed_time();
      
        // generate random basis

        t.start();
        thrust::counting_iterator<int> start(tree*d*level);
        thrust::transform(start, start+d*level, R.begin(), prg(seed));
        cudaDeviceSynchronize();
        t.stop(); t_rand += t.elapsed_time();

        //thrust::transform(start, start+d*level, R.begin(), dist(rng));
        //thrust::counting_iterator<int> zero(0);
        //thrust::transform(zero, zero+d*level, R.begin(), prg(current_time_nanoseconds()));

        // Orthogonalize random basis
        t.start(); 
        //orthogonal(R, d, level);
        //cudaDeviceSynchronize();
        t.stop(); t_orth += t.elapsed_time();
        //tprint(level, d, R, "random projection");

        // gemm - compute projection
        t.start();
        GEMM_SDD(n, level, d, dRowPtr, dColIdx, dVal, nnz, R, P);
        cudaDeviceSynchronize();
        t.stop(); t_sdd += t.elapsed_time();

        // compute tree permutation
        t.start();
        build_tree(P, n, level, perm, t_tsort);
        cudaDeviceSynchronize();
        t.stop(); t_tree += t.elapsed_time();
      }

      // shuffle points into tree order
      t.start();
      gather(dRowPtr, dColIdx, dVal, n, d, nnz, perm);
      gather(dID, perm);
      gather(order, perm);
      cudaDeviceSynchronize();
      t.stop(); t_org += t.elapsed_time();
    }

    //tprint(perm, "permutation");
    //tprint(dID, "reordered ID");
    //dprint(n, d, nnz, dRowPtr, dColIdx, dVal, "reordered P");


    // compute neighbors at leaf level
    int *curNborID = thrust::raw_pointer_cast(dNborID.data());
    float *curNborDist = thrust::raw_pointer_cast(dNborDist.data());
    int*dR = thrust::raw_pointer_cast(dRowPtr.data());
    int* dC = thrust::raw_pointer_cast(dColIdx.data());
    int* d_ID = thrust::raw_pointer_cast(dID.data());
    float* dV = thrust::raw_pointer_cast(dVal.data());
    t.start();
    /*
    {
      t3.start();
      int nBlock = (nLeaf+blkLeaf-1) / blkLeaf;
      int *IDPtr[nBlock], *nborIDPtr[nBlock];
      float *nborDistPtr[nBlock]; 
      SpMat subBlock[nBlock];
      t3.stop(); t_alloc += t3.elapsed_time();

      int maxNNZ; 
      t3.start();
      get_submatrices(dID, curNborID, curNborDist, 2*k, dRowPtr, dColIdx, dVal, 
          n, d, nnz, maxPoint, nLeaf, blkLeaf, nBlock, subBlock, maxNNZ,
          IDPtr, nborIDPtr, nborDistPtr, k);
      cudaDeviceSynchronize();
      t3.stop(); t_sub += t3.elapsed_time();

      // allocate memory for the transpose
      //std::cout<<"[Transpose] R: "<<d/1.e9*blkLeaf*4+maxNNZ/1.e9*4*2<<" GB"<<std::endl;
      t3.start();
      dvec<int> R_rowPtr(d*blkLeaf+1);
      dvec<int> R_colIdx(maxNNZ);
      dvec<float> R_val(maxNNZ);
      cudaDeviceSynchronize();
      t3.stop(); t_alloc += t3.elapsed_time();

      // allocate memory for distance
      //std::cout<<"Allocate distance and index array: "<<maxPoint/1.e9*blkPoint*blkLeaf*4*2<<" GB\n";
      t3.start();
      dvec<float> Dist(maxPoint*blkPoint*blkLeaf);
      dvec<int> tmp(maxPoint*blkPoint*blkLeaf); // auxiliary memory for sorting
      cudaDeviceSynchronize();
      t3.stop(); t_alloc += t3.elapsed_time();

      // process a batch of nodes
      for (int blk=0; blk<nBlock; blk++) {
        //int blkSize = std::min(blkLeaf, nLeaf-blk*blkLeaf);
        t3.start();
        create_BDSpMat(subBlock[blk]);
        cudaDeviceSynchronize();
        t3.stop(); t_bd += t3.elapsed_time();
        
        leaf_knn(IDPtr[blk], subBlock[blk], R_rowPtr, R_colIdx, R_val, 
            maxPoint, nborIDPtr[blk], nborDistPtr[blk], 
            2*k, k, blkPoint, Dist, tmp, t_trans, t_dist, t_sss, t_gemm, t_nnz, t_den, t_sort, sparse);
      }

      //std::cout<<"Sparsity: "<<sparse/nBlock/maxPoint*blkPoint<<std::endl;
      t3.start();
      destroy_BDSpMat(subBlock, nBlock, dColIdx);
      t3.stop(); t_bd += t3.elapsed_time();
    }
    */
      t3.start();

      sfi_leafknn(dR, dC, dV, d_ID, n, nLeaf, k, curNborDist, curNborID);

      //std::cout<<"Sparsity: "<<sparse/nBlock/maxPoint*blkPoint<<std::endl;
      t3.stop(); t_knn += t.elapsed_time();


    //tprint(n, 2*k, dNborID, "[Before scatter] curID");
    //tprint(n, 2*k, dNborDist, "[Before scatter] curDist");
    /* 
    t.start();
    scatter(curNborID, n, 2*k, k, order);
    scatter(curNborDist, n, 2*k, k, order);
    cudaDeviceSynchronize();
    t.stop(); t_scatter += t.elapsed_time();
    */
    //tprint(n, 2*k, dNborDist, "[After scatter] curDist");
    //tprint(n, 2*k, dNborID, "[After scatter] curID");


    // update previous results
    t.start();
    //merge_neighbors(dNborDist, dNborID, n-nExtra, 2*k, k, t_msort, t_mcopy, t_unique);
    t.stop();
    t_merge += t.elapsed_time();
  
    //tprint(n, 2*k, dNborDist, "[After merge] Dist");
    //tprint(n, 2*k, dNborID, "[After merge] ID");
  }
  t2.stop(); float t_kernel = t2.elapsed_time();
 
  printf("\n===========================");
  printf("\n    Sparse KNN Timing");
  printf("\n---------------------------");
  printf("\n* Random Gen: %.2e s (%.0f %%)", t_rand, 100.*t_rand/t_kernel);
  printf("\n* Orthogonalize: %.2e s (%.0f %%)", t_orth, 100.*t_orth/t_kernel);
  printf("\n* GEMM SDD: %.2e s (%.0f %%)", t_sdd, 100.*t_sdd/t_kernel);
  printf("\n* Build tree: %.2e s (%.0f %%)", t_tree, 100.*t_tree/t_kernel);
  printf("\n* Gather: %.2e s (%.0f %%)", t_org, 100.*t_org/t_kernel);
  printf("\n* Scatter: %.2e s (%.0f %%)", t_scatter, 100.*t_scatter/t_kernel);

  //printf("\n  - sort: %.2e s (%.0f %%)", t_tsort, 100.*t_tsort/t_tree);
  printf("\n* Leaf KNN: %.2e s (%.0f %%)", t_knn, 100.*t_knn/t_kernel);
  printf("\n  - transpose: %.2e s (%.0f %%)", t_trans, 100.*t_trans/t_knn);
  printf("\n  - bd: %.2e s (%.0f %%)", t_bd, 100.*t_bd/t_knn);
  printf("\n  - dist: %.2e s (%.0f %%)", t_dist, 100.*t_dist/t_knn);
  printf("\n    ^ sss: %.2e s (%.0f %%)", t_sss, 100.*t_sss/t_dist);
  printf("\n      ~ gemm: %.2e s (%.0f %%)", t_gemm, 100.*t_gemm/t_sss);
  printf("\n      ~ malloc: %.2e s (%.0f %%)", t_nnz, 100.*t_nnz/t_sss);
  printf("\n    ^ den: %.2e s (%.0f %%)", t_den, 100.*t_den/t_dist);
  printf("\n  - sort: %.2e s (%.0f %%)", t_sort, 100.*t_sort/t_knn);
  printf("\n* Merge: %.2e s (%.0f %%)", t_merge, 100.*t_merge/t_kernel);
  printf("\n* Alloc: %.2e s (%.0f %%)", t_alloc, 100.*t_alloc/t_kernel);
  printf("\n* Moving empty Neighbors: %.2e s (%.0f %%)", t_neigh, 100.*t_neigh/t_kernel);
  printf("\n* Artificial Points: %.2e s (%.0f %%)", t_art, 100.*t_art/t_kernel);
  //printf("\n  - sort: %.2e s (%.0f %%)", t_msort, 100.*t_msort/t_merge);
  //printf("\n  - copy: %.2e s (%.0f %%)", t_mcopy, 100.*t_mcopy/t_merge);
  //printf("\n  - unique: %.2e s (%.0f %%)", t_unique, 100.*t_unique/t_merge);
  //printf("\n---------------------------");
  //printf("\n! Sorting: %.2e s (%.0f %%)", t_sort, 100.*t_sort/t_kernel);
  printf("\n===========================\n");

  // -----------------------
  // Copy to CPU
  // -----------------------
  t.start();
  thrust::copy(permI, permI+(n-nExtra)*k, hNborID);
  thrust::copy(permD, permD+(n-nExtra)*k, hNborDist);
  cudaDeviceSynchronize();
  t.stop(); t_copy += t.elapsed_time();
  printf("\n* Copy: %.2e s", t_copy);
}

