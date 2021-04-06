#include "spknn.hpp"
#include "matrix.hpp"
#include "NLA.hpp"
#include "merge.hpp"
#include "timer.hpp"
#include "util_eigen.hpp"
#include "util.hpp"

#include <random>
#include <string>
#include <iostream>


template <typename T>
void gather(T *ID, const ivec &perm) {
  // Data type must be 4 bytes (e.g., float, int/unsigned).
  assert(sizeof(T)==sizeof(float));
  size_t n = perm.size();
  T *copy = new T[n];
  par::copy(n, (float*)ID, (float*)copy);
#pragma omp parallel for
  for (size_t i=0; i<n; i++)
    ID[i] = copy[ perm[i] ];
  delete[] copy;
}


void gather(ivec &order, const ivec &perm) {
  size_t n = perm.size();
  ivec copy(n);
  par::copy(n, (float*)order.data(), (float*)copy.data());
#pragma omp parallel for
  for (size_t i=0; i<n; i++)
    order[i] = copy[ perm[i] ];
}


template <typename T>
void scatter(matrix<T> &nbor, const ivec &perm, size_t from, size_t to) {
  size_t n = to - from;
  size_t k = nbor.cols();
  T *copy = new T[n*k];
  std::copy(nbor.data()+from*k, nbor.data()+to*k, copy);
#pragma omp barrier
  for (size_t i=0; i<n; i++) {
    T *src = copy+i*k;
    T *dst = nbor.data()+perm[i+from]*k;
    std::copy(src, src+k, dst);
  }
  delete[] copy;
}


template <typename T>
void scatter(matrix<T> &nbor, const ivec &perm) {
#pragma omp parallel 
  {
    size_t from, to; 
    par::compute_range(nbor.rows(), from, to);
    scatter(nbor, perm, from, to);
  }
}


template <typename T>
void scatter(T *ptr, const ivec &perm) {
#pragma omp parallel
  {
    unsigned m = perm.size();
    unsigned from, to;
    par::compute_range(m, from, to);
    unsigned n = to - from;
    T *copy = new T[n];
    std::copy(ptr+from, ptr+to, copy);
#pragma omp barrier
    for (unsigned i=0; i<n; i++) {
      ptr[ perm[i+from] ] = copy[i];
    }
    delete[] copy;
  }
}


void print(const ivec &x, std::string name) {
  std::cout<<name<<":\n";
  for (size_t i=0; i<x.size(); i++)
    std::cout<<x[i]<<" ";
  std::cout<<std::endl;
}


template <typename T>
void print(const T *ptr, const size_t n, std::string name) {
  std::cout<<name<<":\n";
  for (size_t i=0; i<n; i++)
    std::cout<<ptr[i]<<" ";
  std::cout<<std::endl;
}


template <typename T>
void print(const T *ptr, unsigned n, int k, std::string name) {
  std::cout<<name<<":\n";
  for (unsigned i=0; i<n; i++) {
    for (int j=0; j<k; j++)
      std::cout<<ptr[i*k+j]<<" ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
}


void build_tree(fMatrix &X, ivec &order, ivec &firstPt) {
  unsigned n = X.rows();
  int L = X.cols();

  // starting indices of points in all nodes
  std::vector<ivec> firstPoint(L+1);
  firstPoint[0].push_back(0);
  firstPoint[0].push_back(n); // from 0 to total
  for (int i=0; i<L; i++) {
    unsigned nNode = 1<<i;
    firstPoint[i+1].resize(nNode*2+1, n); // the last entry is total
#pragma omp parallel for
    for (unsigned j=0; j<nNode; j++) {
      firstPoint[i+1][2*j] = firstPoint[i][j];
      firstPoint[i+1][2*j+1] = (firstPoint[i][j] + firstPoint[i][j+1])/2;
    }
    //std::cout<<"level["<<i+1<<"]: "<<firstPoint[i+1]<<std::endl;
    //print(firstPoint[i+1], "offset");
  }

  // return offsets of leaf nodes
  par::copy(firstPoint[L].size(), (float*)firstPoint[L].data(), (float*)firstPt.data());
  
  // initial ordering    
  par::iota(order.begin(), order.end(), 0);

  // permutation of one level 
  ivec perm(n);
  for (int i=0; i<L; i++) {
    // apply accumulated permutation of previous levels
    gather(X.data()+i*n, order);
    // initialize permuation
    par::iota(perm.begin(), perm.end(), 0);
    // partition nodes at this level
    const ivec &offset = firstPoint[i];
    unsigned nNode = 1<<i;
#pragma omp parallel for
    for (unsigned j=0; j<nNode; j++) {
      std::nth_element( &perm[offset[j]], &perm[(offset[j]+offset[j+1])/2], &perm[offset[j+1]], 
          [&X,i](unsigned a, unsigned b) {return X(a, i) < X(b, i);});
      //std::sort(&perm[offset[j]], &perm[offset[j+1]],
        //  [&X,i](unsigned a, unsigned b) {return X(a, i) < X(b, i);});
    }
    gather(order, perm);
#if 0
    print(perm, "permutation");
    print(order, "order");
    gather(X.data()+i*n, perm);
    print(X.data()+i*n, n, "X");
#endif
  }
}


void leaf_knn(const unsigned *ID, const Points &P, const ivec &startIdx,
    iMatrix &nborID, fMatrix &nborDist, int blkPoint, dvec &t) {

  double t_dist = 0, t_prod = 0, t_nbor = 0., t_sort = 0.;
  double t_rank = 0., t_symm = 0.;

  //print(P, "P");
  
  // compute norm squared
  Timer t0; t0.start();
  fvec norm(P.rows());
  compute_row_norm(P, norm, t[8]);
  t0.stop(); t[7] += t0.elapsed_time();
  
  // loop over leaf nodes
  int nNbor = nborID.cols();
  int nNode = startIdx.size()-1;
  
#pragma omp declare reduction \
  (maxtime : double : omp_out = omp_in > omp_out ? omp_in : omp_out) \
  initializer (omp_priv=0)

#pragma omp parallel for \
  reduction(maxtime: t_dist, t_prod, t_nbor, t_sort, t_rank, t_symm) \
  //schedule(guided)
  
  for (int i=0; i<nNode; i++) {
    unsigned from = startIdx[i];
    unsigned to = startIdx[i+1]; // exclusive
    Points Q = P.subset(from, to);
    fvec Q_nrm(norm.begin()+from, norm.begin()+to);
    //print(Q, "Q");

    // find knn
    Timer timer, t1; timer.start();
    unsigned nPoint = to-from;
    assert(nPoint >= nNbor);
    fMatrix Dt(nPoint, nPoint);
    dvec tDist(5, 0.);
    compute_distance(Q, Q_nrm, Dt, tDist); // distance transpose
    timer.stop(); t_dist += timer.elapsed_time();
    //print(Dt, "D");
    t_prod += tDist[0];
    t_rank += tDist[1];
    t_symm += tDist[2];

    timer.start();
    ivec perm(nPoint);
    std::iota(perm.begin(), perm.end(), 0);
    for (unsigned j=0; j<nPoint; j++) {
      t1.start();
      std::nth_element(perm.begin(), perm.begin()+nNbor-1, perm.end(),
          [&Dt, j, nPoint](unsigned a, unsigned b) {return Dt(a,j) < Dt(b,j);});
      t1.stop(); t_sort += t1.elapsed_time();
      //std::sort(perm.begin(), perm.end(),
        //  [&Dt, j](size_t a, size_t b) {return Dt(a,j) < Dt(b,j);});
      // output
      for (int k=0; k<nNbor; k++) {
        assert(perm[k] < nPoint);
        nborID(from+j,k) = ID[from+perm[k]];
        nborDist(from+j,k) = Dt(perm[k], j);
      }
      //print(perm, "permutation");
      //print(Dt.data()+j*nPoint, nPoint, "distance");
      //print(nborDist.data()+(from+j)*nNbor, nNbor, "neighbor");
    }
    timer.stop(); t_nbor += timer.elapsed_time();
    
    //print(Dt, "Dt");
    //std::cout<<"# threads: "<<omp_get_num_threads()<<std::endl;
    //std::cout<<"thread ["<<omp_get_thread_num()<<"]: dist="<<t_dist
      //<<", product="<<t_prod<<", neighbor="<<t_nbor<<std::endl;
  }
  t[1] += t_dist; // assign reduction result
  t[2] += t_nbor;
  t[3] += t_prod;
  t[4] += t_sort;
  t[5] += t_rank;
  t[6] += t_symm;
  //print(nborID, "neighbor ID");
  //print(nborDist, "neighbor Distance");
}


void spknn
(unsigned *ID, int *rowPtr, int *colIdx, float *val, unsigned n, unsigned d, unsigned nnz, 
 unsigned *nborID, float *nborDist, int k, int level, int nTree, int blkPoint, int cores) {

  omp_set_num_threads(cores);


  double t_kernel, t_merge = 0.;
  dvec t_tree(10, 0);
  dvec t_leaf(10, 0);
  dvec t_shuffle(10, 0);
  
  Timer t0, t, t1; t0.start();

  Points P(n, d, nnz, rowPtr, colIdx, val);
  //print(P, "P");

  // local ordering
  ivec order(n);
  par::iota(order.begin(), order.end(), 0);
      
  for (int tree=0; tree<nTree; tree++) {
  
    // compute starting index of leaf nodes
    ivec startIdx((1<<level)+1);

    // recursive partitioning
    t.start();
    {
      // compute new ordering
      ivec perm(n);
      par::iota(perm.begin(), perm.end(), 0);
      {
        // random arrays and projections
        t1.start();
        fMatrix R(d, level);
        R.rand();

        /* 
        #pragma omp parallel
        {
            std::mt19937_64 generator;
            std::normal_distribution<float> distribution(0.0, 1.0);
            //std::uniform_real_distribution<float> distribution(0.0, 1.0);
            #pragma omp for collapse(2)
            for(int a=0; a<d; ++a){
                for(int b=0; b<level; ++b){
                    float sample = distribution(generator);
                    R(a, b) = sample;
                }
            }
        }
        */

        t1.stop(); t_tree[5] += t1.elapsed_time();
        //print(R, "R");
        
        t1.start();
        orthonormalize(R);
        t1.stop(); t_tree[1] += t1.elapsed_time();
        //print(R, "R");

        // X = P * R;
        t1.start();
        fMatrix X(n, level);
        GEMM_SDD(P, R, X);
        t1.stop(); t_tree[2] += t1.elapsed_time();
        //print(X, "X");

        // compute new ordering
        t1.start();
        build_tree(X, perm, startIdx);
        t1.stop(); t_tree[3] += t1.elapsed_time();
      }

      // shuffle
      t1.start();
      gather(P, perm, t_tree);
      gather(ID, perm);
      gather(order, perm);
      t1.stop(); t_tree[4] += t1.elapsed_time();
      //print(P, "P"); 
      //print(perm, "permutation");
      //print(ID, n, "ID");
    }
    t.stop(); t_tree[0] += t.elapsed_time();
    //std::cout<<"Finished partitioning ..."<<std::endl;

    // exact local search
    t.start();
    iMatrix newNborID(n, k, true /*row major*/);
    fMatrix newNborDist(n, k, true /*row major*/);
    leaf_knn(ID, P, startIdx, newNborID, newNborDist, blkPoint, t_leaf);
    //print(newNborID, "neighbor ID before shuffle");
    t.stop(); t_leaf[0] += t.elapsed_time();

    // reverse shuffle
    t.start();
    scatter(newNborID, order);
    scatter(newNborDist, order);
    t.stop(); t_shuffle[0] += t.elapsed_time();
    //print(newNborID, "new neighbor ID");
    //print(newNborDist, "new neighbor Distance");

    // update previous results
    t.start();
    merge_neighbor_cpu(nborDist, nborID, newNborDist.data(), newNborID.data(), n, k, cores);
    t.stop(); t_merge += t.elapsed_time();
    //print(nborID, n, k, "merge neighbor ID");
    //print(nborDist, n, k, "merge neighbor Distance");
  }

  // recover original order before exit
  t.start();
  scatter(P, order, t_shuffle);
  scatter(ID, order);
  t.stop(); t_shuffle[1] = t.elapsed_time();
  //print(P, "P after scatter");
  //print(ID, n, "ID after scatter");
  
  // stop timing
  t0.stop(); t_kernel = t0.elapsed_time();

    /*
  std::cout<<"\n========================"
           <<"\nPoints"
           <<"\n------------------------"
           <<"\n# points: "<<n
           <<"\n# dimensions: "<<d
           <<"\n# nonzeros: "<<nnz
           <<"\n------------------------"
           <<"\n# points/leaf: "<<n/(1<<level)
           <<"\n# leaf nodes: "<<(1<<level)
           <<"\n------------------------"
           <<"\nmem points: "<<(n+1)/1.e9*4+nnz/1.e9*4*2<<" GB"
           <<"\nmem output (peak): "<<n/1.e9*k*4*3<<" GB"
           <<"\nmem orthogonal bases: "<<d/1.e9*level*4<<" GB"
           <<"\nmem projection: "<<n/1.e9*level*4<<" GB"
           <<"\n========================\n"
           <<std::endl;
  printf("\n===========================");
  printf("\n    Sparse KNN Timing");
  printf("\n---------------------------");
  printf("\n# threads: %d", cores);
  printf("\ntotal time: %.2f s", t_kernel);
  printf("\n---------------------------");
  printf("\n* Partition: %.2e s (%.0f %%)", t_tree[0], 100.*t_tree[0]/t_kernel);
  //printf("\n  - rand: %.2e s (%.0f %%)", t_tree[5], 100.*t_tree[5]/t_tree[0]);
  printf("\n  - qr: %.2e s (%.0f %%)", t_tree[1], 100.*t_tree[1]/t_tree[0]);
  printf("\n  - gemm: %.2e s (%.0f %%)", t_tree[2], 100.*t_tree[2]/t_tree[0]);
  printf("\n  - tree: %.2e s (%.0f %%)", t_tree[3], 100.*t_tree[3]/t_tree[0]);
  printf("\n  - shuffle: %.2e s (%.0f %%)", t_tree[4], 100.*t_tree[4]/t_tree[0]);
  printf("\n    > product: %.2e s (%.0f %%)", t_tree[6], 100.*t_tree[6]/t_tree[4]);
  printf("\n* Leaf KNN: %.2e s (%.0f %%)", t_leaf[0], 100.*t_leaf[0]/t_kernel);
  printf("\n  - norm: %.2e s (%.0f %%)", t_leaf[7], 100.*t_leaf[7]/t_leaf[0]);
  printf("\n    > gemv: %.2e s (%.0f %%)", t_leaf[8], 100.*t_leaf[8]/t_leaf[7]);
  printf("\n  - dist: %.2e s (%.0f %%)", t_leaf[1], 100.*t_leaf[1]/t_leaf[0]);
  printf("\n    > product: %.2e s (%.0f %%)", t_leaf[3], 100.*t_leaf[3]/t_leaf[1]);
  printf("\n    > rank-1: %.2e s (%.0f %%)", t_leaf[5], 100.*t_leaf[5]/t_leaf[1]);
  printf("\n    > symmetrize: %.2e s (%.0f %%)", t_leaf[6], 100.*t_leaf[6]/t_leaf[1]);
  printf("\n  - nbor: %.2e s (%.0f %%)", t_leaf[2], 100.*t_leaf[2]/t_leaf[0]);
  printf("\n    > sort: %.2e s (%.0f %%)", t_leaf[4], 100.*t_leaf[4]/t_leaf[2]);
  printf("\n* Merge: %.2e s (%.0f %%)", t_merge, 100.*t_merge/t_kernel);
  printf("\n* Shuffle nbor: %.2e s (%.0f %%)", t_shuffle[0], 100.*t_shuffle[0]/t_kernel);
  //printf("\n  - copy: %.2e s (%.0f %%)", t_shuffle[2], 100.*t_shuffle[2]/t_shuffle[0]);
  printf("\n* Shuffle points: %.2e s (%.0f %%)", t_shuffle[1], 100.*t_shuffle[1]/t_kernel);
  printf("\n  - product: %.2e s (%.0f %%)", t_shuffle[6], 100.*t_shuffle[6]/t_shuffle[1]);
  printf("\n  - copy: %.2e s (%.0f %%)", t_shuffle[7], 100.*t_shuffle[7]/t_shuffle[1]);
  //printf("\n  - csr_create: %.2e s (%.0f %%)", t_shuffle[8], 100.*t_shuffle[8]/t_shuffle[1]);
  printf("\n===========================\n\n"); 
    */ 
}


