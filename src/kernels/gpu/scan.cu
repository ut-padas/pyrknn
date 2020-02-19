#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "simple.hpp"

void Exclusive_scan(int *array_out, int *array_in, int N)
{
    thrust::exclusive_scan(thrust::device, array_in, array_in + N, array_out);
}

void Sort(float *array_in, int N)
{
    thrust::sort(thrust::device, array_in, array_in + N);
}


void Sort_by_key(float *keys, float *values, int N)
{
    thrust::sort_by_key(thrust::device, keys, keys + N, values);
}

// void Partition(int *in, int N, int i)
// {
//     int val = in[i];
//     struct less
// 	{
//     __device__
// 	bool operator()(const int &x)
//         {
//             return x<val;
//         }
//     };
//     thrust::partition(thrust::device, in, in + N, less());
// }
