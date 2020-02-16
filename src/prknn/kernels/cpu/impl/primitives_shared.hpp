#ifndef PRIMITIVES_CPU_HPP
#define PRIMITIVES_CPU_HPP

/** Use STL, and vector. */
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <algorithm>

#include <omp.h>

using namespace std;

namespace hmlp
{
namespace combinatorics
{

template<typename T>
using idx_type = typename vector<T>::size_type;

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


}; /* end namespace combinatorics */
}; /* end namespace hmlp */

#endif /* define PRIMITIVES_CPU_HPP */
