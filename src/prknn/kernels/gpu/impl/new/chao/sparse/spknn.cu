#include "util_gpu.hpp"
#include "timer_gpu.hpp"
#include "knn_handle.hpp"
#include "op_gpu.hpp"
#include "reorder.hpp"
#include "gemm.hpp"
#include "orthogonal.hpp"
#include "merge.hpp"
#include "print.hpp"

#include <thrust/random.h>
#include <limits>       // std::numeric_limits


void build_tree(dvec<float> &P, int n, int level, dvec<int> &perm, float&);

void leaf_knn(int *dID, int *dRowPtr, int *dColIdx, float *dVal, 
    int n, int d, int nnz, int nLeaf, int maxPoint, 
    int *curID, float *curDist, int LD, int k, int m);

    
void get_submatrices(ivec &dID, int *curNborID, float *curNborDist, int LD,
    ivec &dRowPtr, ivec &dColIdx, fvec &dVal, 
    int n, int nnz, int maxPoint, int nLeaf, int blkLeaf, int nBlock,
    int P_rows[], int P_nnz[], int *P_rowPtr[], int *P_colIdx[], float *P_val[], 
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
    
    for (int blk=0; blk<nBlock; blk++) {
      int offset = blk*blkLeaf;
      int blkSize = std::min(blkLeaf, nLeaf-offset);
      
      P_rows[blk] = blkSize*maxPoint; // # points per batch
      P_nnz[blk] = cumNNZ[blk+1] - cumNNZ[blk];
      
      P_colIdx[blk] = thrust::raw_pointer_cast(dColIdx.data())+cumNNZ[blk];
      P_val[blk] = thrust::raw_pointer_cast(dVal.data())+cumNNZ[blk];
      
      CHECK_CUDA( cudaMalloc(&P_rowPtr[blk], (P_rows[blk]+1)*sizeof(int)) );

      iptr rowPtrP(P_rowPtr[blk]);
      thrust::constant_iterator<int> PreNNZ(cumNNZ[blk]);
      thrust::transform(dRowPtr.begin()+offset*maxPoint, 
          dRowPtr.begin()+(offset+blkSize)*maxPoint+1, 
          PreNNZ, rowPtrP, thrust::minus<int>());

      //tprint(dRowPtr, "Matrix row pointer");
      //tprint(rowPtrP, "Submatrix row pointer");


      IDPtr[blk] = thrust::raw_pointer_cast(dID.data())+offset*maxPoint;
      nborIDPtr[blk] = curNborID + offset*maxPoint*LD;
      nborDistPtr[blk] = curNborDist + offset*maxPoint*LD;
   
      //dprint(P_rows[blk], d, P_nnz[blk], P_rowPtr[blk], P_colIdx[blk], P_val[blk], "submatrix");
    }
}

    
void create_BDSpMat(int *rowPtr, int *colIdx, int n, int d, int nnz, int nLeaf) {
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


void revert_BDSpMat(dvec<int>& colIdx, int d) {
  thrust::constant_iterator<int> DIM(d);
  thrust::transform(colIdx.begin(), colIdx.end(), DIM, colIdx.begin(), thrust::modulus<int>());
}


int current_time_nanoseconds(){
    struct timespec tm;
    clock_gettime(CLOCK_REALTIME, &tm);
    return tm.tv_nsec;
}


// m: blocking size in distance calculation
void spknn(int *hID, int *hRowPtr, int *hColIdx, float *hVal, 
    int n, int d, int nnz, int level, int nTree,
    int *hNborID, float *hNborDist, int k, 
    int blkLeaf, int blkPoint, int device) {
 
  //print(n, d, nnz, hRowPtr, hColIdx, hVal, "host P");
  
  const int nLeaf = 1<<level;
  const int maxPoint = (n+nLeaf-1)/nLeaf;
  const int nExtra = maxPoint*nLeaf - n;
  
  // copy data to GPU
  dvec<int> dID(n+nExtra);
  thrust::copy(hID, hID+n, dID.begin());
  
  dvec<int> dRowPtr(n+1+nExtra);
  dvec<int> dColIdx(nnz+nExtra);
  dvec<float> dVal(nnz+nExtra);
  thrust::copy(hRowPtr, hRowPtr+n+1, dRowPtr.begin());
  thrust::copy(hColIdx, hColIdx+nnz, dColIdx.begin());
  thrust::copy(hVal, hVal+nnz, dVal.begin());
  
  dvec<int> dNborID((n+nExtra)*k*2, std::numeric_limits<int>::max());
  dvec<float> dNborDist((n+nExtra)*k*2, std::numeric_limits<float>::max());
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iter  = thrust::make_transform_iterator(zero, firstKCols(k, 2*k));
  auto permI = thrust::make_permutation_iterator(dNborID.begin(), iter);
  auto permD = thrust::make_permutation_iterator(dNborDist.begin(), iter);
  thrust::copy(hNborID, hNborID+n*k, permI);
  thrust::copy(hNborDist, hNborDist+n*k, permD);
  
  //tprint((n+nExtra), 2*k, dNborID, "nborID");
  //tprint((n+nExtra), 2*k, dNborDist, "nborDist");

  // insert artificial points at infinity
  thrust::sequence(dID.begin()+n, dID.end(), -nExtra, 1); // negative id
  thrust::sequence(dRowPtr.begin()+n, dRowPtr.end(), hRowPtr[n]);
  thrust::fill(dColIdx.begin()+nnz, dColIdx.end(), 0); // the first coordinate is infinity
  thrust::fill(dVal.begin()+nnz, dVal.end(), std::numeric_limits<float>::max()); 

  // update # points and # nonzeros
  n += nExtra;
  nnz += nExtra;

  //tprint(dID, "device ID");
  //dprint(n, d, nnz, dRowPtr, dColIdx, dVal, "device P");
  
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


  // local ordering
  dvec<int> order(n);
  thrust::sequence(order.begin(), order.end(), 0);

  // -----------------------
  // timing
  // -----------------------
  float t_tree = 0., t_knn = 0., t_merge = 0.;
  float t_orth = 0., t_sdd = 0., t_sort = 0.;
  TimerGPU t, t1;
  
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
      t.start();
      // compute permutation
      {
        // allocate memory for random arrays and projections
        dvec<float> R(d*level);
        dvec<float> P(n*level);
      
        // generate random bases
        thrust::counting_iterator<int> start(tree*d*level);
        thrust::transform(start, start+d*level, R.begin(), prg(seed));

        //thrust::counting_iterator<int> zero(0);
        //thrust::transform(zero, zero+d*level, R.begin(), prg(current_time_nanoseconds()));
        t1.start();
        orthogonal(R, d, level);
        t1.stop(); t_orth += t1.elapsed_time();
        //tprint(level, d, R, "random projection");

        // gemm
        t1.start();
        GEMM_SDD(n, level, d, dRowPtr, dColIdx, dVal, nnz, R, P);
        t1.stop(); t_sdd += t1.elapsed_time();
        std::cout<<"Finished GEMM.\n"<<std::endl;

        // compute permutation
        build_tree(P, n, level, perm, t_sort);
      }
      t.stop(); t_tree += t.elapsed_time();
      //std::cout<<"Finished tree construction.\n"<<std::endl;

      // shuffle
      gather(dRowPtr, dColIdx, dVal, n, d, nnz, perm);
      gather(dID, perm);
      gather(order, perm);
    }
    //tprint(perm, "permutation");
    //tprint(dID, "reordered ID");
    //dprint(n, d, nnz, dRowPtr, dColIdx, dVal, "reordered P");

    // compute neighbors at leaf level
    t.start();
    
    //tprint(n, 2*k, dNborID, "[Before leaf knn] curID");
    //tprint(n, 2*k, dNborDist, "[Before leaf knn] curDist");


    int nBlock = (nLeaf+blkLeaf-1) / blkLeaf;

    int P_rows[nBlock], P_nnz[nBlock];
    int *P_rowPtr[nBlock], *P_colIdx[nBlock];
    float *P_val[nBlock];
    int *IDPtr[nBlock], *nborIDPtr[nBlock];
    float *nborDistPtr[nBlock];
    
    int *curNborID = thrust::raw_pointer_cast(dNborID.data()+k);
    float *curNborDist = thrust::raw_pointer_cast(dNborDist.data()+k);
    get_submatrices(dID, curNborID, curNborDist, 2*k, dRowPtr, dColIdx, dVal, 
        n, nnz, maxPoint, nLeaf, blkLeaf, nBlock,
        P_rows, P_nnz, P_rowPtr, P_colIdx, P_val, IDPtr, nborIDPtr, nborDistPtr, k);

    // process a batch of nodes
    for (int blk=0; blk<nBlock; blk++) {
      int blkSize = std::min(blkLeaf, nLeaf-blk*blkLeaf);

      create_BDSpMat(P_rowPtr[blk], P_colIdx[blk], P_rows[blk], d, P_nnz[blk], blkSize);
      
      leaf_knn(IDPtr[blk], P_rowPtr[blk], P_colIdx[blk], P_val[blk], P_rows[blk], 
          d, P_nnz[blk], blkSize, maxPoint, nborIDPtr[blk], nborDistPtr[blk], 2*k, k, blkPoint);
    }
    t.stop(); t_knn += t.elapsed_time();

    // free temporary resource
    for (int blk=0; blk<nBlock; blk++)
      CHECK_CUDA( cudaFree(P_rowPtr[blk]) );
    
    revert_BDSpMat(dColIdx, d);


    //tprint(n, 2*k, dNborID, "[Before scatter] curID");
    //tprint(n, 2*k, dNborDist, "[Before scatter] curDist");
  
    scatter(curNborID, n, 2*k, k, order);
    scatter(curNborDist, n, 2*k, k, order);

    //tprint(n, 2*k, dNborDist, "[After scatter] curDist");
    //tprint(n, 2*k, dNborID, "[After scatter] curID");

    // update previous results
    t.start();
    //int *preID = thrust::raw_pointer_cast(dNborID.data());
    //float *preDist = thrust::raw_pointer_cast(dNborDist.data());
    //merge_neighbors_gpu(preDist, preID, curDist, curID, n-nExtra, k, k);

    std::cout<<"Merge neighbors\n";
    merge_neighbors(dNborDist, dNborID, n-nExtra, 2*k, k);
    t.stop();
    t_merge += t.elapsed_time();
  
    //tprint(n, 2*k, dNborDist, "[After merge] Dist");
    //tprint(n, 2*k, dNborID, "[After merge] ID");
  }
  

  std::cout<<"\n==========================="
           <<"\n    Sparse KNN Timing"
           <<"\n---------------------------"
           <<"\n* Build tree: "<<t_tree<<" s"
           //<<"\n\t- rand: "<<t_rand<<" s"
           <<"\n\t- orthogonal: "<<t_orth<<" s"
           <<"\n\t- gemm_sdd: "<<t_sdd<<" s"
           <<"\n\t- sort: "<<t_sort<<" s"
           <<"\n* Leaf KNN: "<<t_knn<<" s"
           <<"\n* Merge: "<<t_merge<<" s"
           <<"\n===========================\n"
           <<std::endl;

  // -----------------------
  // Copy to CPU
  // -----------------------
  thrust::copy(permI, permI+(n-nExtra)*k, hNborID);
  thrust::copy(permD, permD+(n-nExtra)*k, hNborDist);
  //thrust::copy(dNborID.begin(), dNborID.end(), hNborID);
  //thrust::copy(dNborDist.begin(), dNborDist.end(), hNborDist);
}

