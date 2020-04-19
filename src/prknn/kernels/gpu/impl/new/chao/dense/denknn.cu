#include "denknn.hpp"
#include "orthogonal.hpp"
#include "gemm.hpp"
#include "reorder.hpp"
#include "merge.hpp"
#include "op_gpu.hpp"
#include "timer_gpu.hpp"
#include "util_gpu.hpp"
#include "print.hpp"


void build_tree(fvec&, int, int, ivec&, int, float&);

void leaf_knn(const ivec&, const fvec&, int, int, int, int*, float*, int, int, int,
    float&, float&);

int current_time_nanoseconds() {
    struct timespec tm;
    clock_gettime(CLOCK_REALTIME, &tm);
    return tm.tv_nsec;
}

void merge_neighbors_python(float* nborD1, int *nborI1, const float* nborD2, const int *nborI2, int m, int n, int k, int device){
    cudaSetDevice(device);


    dvec<int> dNborID(m*k*2, std::numeric_limits<int>::max());
    dvec<float> dNborDist(m*k*2, std::numeric_limits<float>::max());

    auto zero = thrust::make_counting_iterator<int>(0);
    auto iter = thrust::make_transform_iterator(zero, firstKCols(k, 2*k));
    auto leftKColsID = thrust::make_permutation_iterator(dNborID.begin(), iter);
    auto leftKColsDist = thrust::make_permutation_iterator(dNborDist.begin(), iter);

    auto rightKColsID = thrust::make_permutation_iterator(dNborID.begin()+k, iter);
    auto rightKColsDist = thrust::make_permutation_iterator(dNborDist.begin()+k, iter);

    {
        dvec<int> tmpNborID(m*k);
        dvec<float> tmpNborDist(m*k);

        thrust::copy(nborI2, nborI2+m*k, tmpNborID.begin());
        thrust::copy(nborD2, nborD2+m*k, tmpNborDist.begin());

        thrust::copy_n(tmpNborID.begin(), m*k, rightKColsID);
        thrust::copy_n(tmpNborDist.begin(), m*k, rightKColsDist);


        thrust::copy_n(nborI1, m*k, tmpNborID.begin());
        thrust::copy_n(nborD1, m*k, tmpNborDist.begin());
        thrust::copy_n(tmpNborID.begin(), m*k, leftKColsID);
        thrust::copy_n(tmpNborDist.begin(), m*k, leftKColsDist);
    }

    float t_msort = 0;
    float t_mcopy = 0;
    float t_unique = 0;

    merge_neighbors(dNborDist, dNborID, m, 2*k, k, t_msort, t_mcopy, t_unique);

    {
        dvec<int> tmpNborID(m*k);
        dvec<float> tmpNborDist(m*k);
        thrust::copy_n(leftKColsID, m*k, tmpNborID.begin());
        thrust::copy_n(leftKColsDist, m*k, tmpNborDist.begin());
        thrust::copy_n(tmpNborID.begin(), m*k, nborI1);
        thrust::copy_n(tmpNborDist.begin(), m*k, nborD1);
    }

} 

void denknn(const int* hID, const float *hP, int n, int d, int level, int nTree,
    int *hNborID, float *hNborDist, int k, int blkPoint, int device) {
  
  cudaSetDevice(device);

  const int nLeaf = 1<<level;
  const int nPoint = (n+nLeaf-1)/nLeaf; // # points per leaf node
  const int nExtra = nPoint*nLeaf - n;
  const int N = n + nExtra;

  TimerGPU t0; 
  float t_alloc = 0., t_copy1 = 0., t_copy2 = 0.;

  // copy data to GPU
  t0.start();
  dvec<int> dID(N);
  dvec<float> dP(N*d);
  dvec<int> dNborID(N*k*2, std::numeric_limits<int>::max());
  dvec<float> dNborDist(N*k*2, std::numeric_limits<float>::max());
  t0.stop(); t_alloc = t0.elapsed_time();

  t0.start();
  thrust::copy(hID, hID+n, dID.begin());  
  thrust::copy(hP, hP+n*d, dP.begin()); // both in row-major
  t0.stop(); t_copy1 = t0.elapsed_time();

  t0.start();
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter = thrust::make_transform_iterator(zero, firstKCols(k, 2*k));
  auto leftKColsID = thrust::make_permutation_iterator(dNborID.begin(), iter);
  auto leftKColsDist = thrust::make_permutation_iterator(dNborDist.begin(), iter);
  {
    dvec<int> tmpNborID(n*k);
    dvec<float> tmpNborDist(n*k);
    thrust::copy_n(hNborID, n*k, tmpNborID.begin());
    thrust::copy_n(hNborDist, n*k, tmpNborDist.begin());
    thrust::copy_n(tmpNborID.begin(), n*k, leftKColsID);
    thrust::copy_n(tmpNborDist.begin(), n*k, leftKColsDist);
  }
  t0.stop(); t_copy2 = t0.elapsed_time();

  // insert artificial points at infinity
  thrust::sequence(dID.begin()+n, dID.end(), -nExtra, 1); // negative id
  thrust::fill(dP.begin()+n*d, dP.end(), -std::numeric_limits<float>::max());
  
  //tprint(N, d, dP, "Points on GPU");

  std::cout<<"\n========================"
           <<"\nPoints"
           <<"\n------------------------"
           <<"\n# points: "<<N
           <<"\n# dimensions: "<<d
           <<"\n# artificial points: "<<nExtra
           <<"\n# points/leaf: "<<nPoint
           <<"\n# leaf nodes: "<<nLeaf
           <<"\n------------------------"
           <<"\nmem points: "<<N/1.e9*d*4<<" GB"
           <<"\nmem output: "<<N/1.e9*k*4*4<<" GB"
           <<"\nmem distance: "<<N/1.e9*blkPoint*2*4<<" GB"
           <<"\nmem orthogonal bases: "<<d/1.e9*level*4<<" GB"
           <<"\nmem projection: "<<N/1.e9*level*4<<" GB"
           <<"\n------------------------"
           <<"\nmalloc time: "<<t_alloc<<" s"
           <<"\ncopy data time: "<<t_copy1<<" s"
           <<"\ncopy neighbor time: "<<t_copy2<<" s"
           <<"\n========================\n"
           <<std::endl;

  
  // -----------------------
  // timing
  // -----------------------
  float t_tree = 0., t_knn = 0., t_merge = 0.;
  float t_tsort = 0.;
  float t_dist = 0., t_lsort = 0.;
  float t_msort = 0., t_mcopy = 0., t_unique;
  TimerGPU t, t2; t2.start();

  // local ordering
  dvec<int> order(N);
  thrust::sequence(order.begin(), order.end(), 0);

  // allocate memory for random arrays and projections
  int nBases = std::min(d, level);
  dvec<float> R(d*nBases); // column-major
  dvec<float> Y(N*level); // column-major
      
  // -----------------------
  // random seed
  // -----------------------
  int seed = current_time_nanoseconds();

  // -----------------------
  // Start KNN
  // -----------------------
  for (int tree=0; tree<nTree; tree++) {
  
    // cluster points
    {
      dvec<int> perm(N);
      // compute permutation
      {
        // generate random bases
        thrust::counting_iterator<int> start(tree*R.size());
        thrust::transform(start, start+R.size(), R.begin(), prg(seed));

        orthogonal(R, d, nBases);
        //tprint(level, d, R, "random projection");

        // gemm
        GEMM(N, level, d, dP, R, Y);
        //std::cout<<"Finished GEMM.\n"<<std::endl;

        // compute permutation
        t.start();
        build_tree(Y, N, level, perm, nBases, t_tsort);
        t.stop(); t_tree += t.elapsed_time();
      }
      //std::cout<<"Finished tree construction.\n"<<std::endl;

      // shuffle
      gather(dP, N, d, perm);
      gather(dID, perm);
      gather(order, perm);
    }

    // compute neighbors at leaf level
    t.start();
    int *curNborID = thrust::raw_pointer_cast(dNborID.data()+k);
    float *curNborDist = thrust::raw_pointer_cast(dNborDist.data()+k);    
    leaf_knn(dID, dP, nPoint, d, nLeaf, curNborID, curNborDist, k, 2*k, blkPoint, t_dist, t_lsort);
    t.stop(); t_knn += t.elapsed_time();

    // shuffle results
    scatter(curNborID, N, 2*k, k, order);
    scatter(curNborDist, N, 2*k, k, order);

    t.start();
    //std::cout<<"Merge neighbors\n";
    merge_neighbors(dNborDist, dNborID, n, 2*k, k, t_msort, t_mcopy, t_unique);
    t.stop(); t_merge += t.elapsed_time();
  }
  t2.stop();
  float t_kernel = t2.elapsed_time();
  float t_sort = t_tsort+t_lsort+t_msort;

  printf("\n===========================");
  printf("\n    Dense KNN Timing");
  printf("\n---------------------------");
  printf("\n* Build tree: %.2e s (%.0f %%)", t_tree, 100.*t_tree/t_kernel);
  printf("\n  - sort: %.2e s (%.0f %%)", t_tsort, 100.*t_tsort/t_tree);
  printf("\n* Leaf KNN: %.2e s (%.0f %%)", t_knn, 100.*t_knn/t_kernel);
  printf("\n  - sort: %.2e s (%.0f %%)", t_lsort, 100.*t_lsort/t_knn);
  printf("\n  - dist: %.2e s (%.0f %%)", t_dist, 100.*t_dist/t_knn);
  printf("\n* Merge: %.2e s (%.0f %%)", t_merge, 100.*t_merge/t_kernel);
  printf("\n  - sort: %.2e s (%.0f %%)", t_msort, 100.*t_msort/t_merge);
  printf("\n  - copy: %.2e s (%.0f %%)", t_mcopy, 100.*t_mcopy/t_merge);
  printf("\n  - unique: %.2e s (%.0f %%)", t_unique, 100.*t_unique/t_merge);
  printf("\n---------------------------");
  printf("\n! Sorting: %.2e s (%.0f %%)", t_sort, 100.*t_sort/t_kernel);
  printf("\n===========================\n");
  

  // -----------------------
  // Copy results back to CPU
  // -----------------------
  {
    dvec<int> tmpNborID(n*k);
    dvec<float> tmpNborDist(n*k);
    thrust::copy_n(leftKColsID, n*k, tmpNborID.begin());
    thrust::copy_n(leftKColsDist, n*k, tmpNborDist.begin());
    thrust::copy_n(tmpNborID.begin(), n*k, hNborID);
    thrust::copy_n(tmpNborDist.begin(), n*k, hNborDist);
  }
}


