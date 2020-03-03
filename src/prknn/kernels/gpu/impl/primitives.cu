#include "primitives.hpp"

#include<stdio.h>
#include<stdlib.h>

#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

//Simple vector addition kernel to test Cython compilation

/*The CUDA Kernel*/
__global__ void vector_add_kernel(float *out, float *a, float *b, size_t n){
    for(size_t i = 0; i < n; ++i){
        out[i] = a[i] + b[i];
    }
}

/*Impl of function to be wrapped by Cython*/
/*Assume given data is on device*/
void vector_add(float *out, float *a, float *b, size_t n){
    vector_add_kernel<<<1, 1>>>(out, a, b, n);
}


//Example Reduction Kernel

__inline__ __device__
float warpReduceSum(float val){
	unsigned mask = 0xffffffff;
	for(size_t offset = warpSize/2; offset >0; offset /=2){
		val += __shfl_down_sync(mask, val, offset);
	}
	return val;
}

__global__
void device_reduce_warp_atomic_kernel(const float *in, float *out, const size_t n){
	float sum = float(0);
	for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
		sum += in[i];
	}
	sum = warpReduceSum(sum);
	if((threadIdx.x & (warpSize-1)) == 0)
		atomicAdd(out, sum);
}

void device_reduce_warp_atomic(const float *in, float *out, const size_t n){
	size_t threads=256;
    size_t maxblocks = 2048;
	size_t blocks = min((n+threads-1)/threads, maxblocks);
	
	cudaMemsetAsync(out, 0, sizeof(float));
	device_reduce_warp_atomic_kernel<<<blocks,threads>>>(in, out, n);
}


//Median finding with Kelley Cutting

//Loop through multiple reductions
//val = [f, g]  (here n = 2)
//Skips the first element (that's where I'm storing the median estimate for comparison)
__inline__ __device__
float* warpReduceSumMultiple(float* val, const size_t n){
    unsigned mask = 0xffffffff;
    for(size_t offset = warpSize/2; offset > 0; offset /= 2){
        //I don't think unrolling here will actually do anything but lets see just for fun
        #pragma unroll
        for(size_t idx = 0; idx < n; ++idx){
            val[ idx ] += __shfl_down_sync(mask, val[ idx ], offset);
        }
    }
    return val;
}

/*
__inline__ __device__
float* warpReduceSumTwo(float f, float g){
    unsigned mask = 0xffffffff;
    for(size_t offset = warpSize/2; offset > 0; offset /= 2){
        f += __shfl_down_sync(mask, f, offset);
        g += __shfl_down_sync(mask, g, offset);
    }
    return val;
}
*/

__global__
void device_kelley_reduce_kernel(const float* arr, const size_t n, float* val){
    const int D = 2;
    float update[D]; 
    float temp_v[D] = {float(0), float(0)};
    const float y = val[ 0 ];
    const float eps = 1e-5;
    for(size_t i = blockIdx.x *blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        const float x = arr[ i ];
        const float diff = (y-x);
        //compute objective
        update[0] = abs(diff);

        //compute subgradient
        //todo replace 0 with rand()? How does the subgradient matter?
        update[1] = ( diff > eps) ? float(1.0) : float(1.0);
        //bool swit = (diff > eps) ? false : true;
        update[1] = ( diff < -1*eps) ? float(0) : update[1];

        temp_v[0] += update[0]; //update f
        temp_v[1] += update[1]; //update g
    }
    warpReduceSumMultiple(&temp_v[0], D);
    if ((threadIdx.x & (warpSize-1)) == 0){
        atomicAdd(&val[ 1 ], temp_v[0]);
        atomicAdd(&val[ 2 ], temp_v[1]);
    }
}


__global__ 
void small_update_kernel(float* val){
    float numerator = val[1] - val[4] + val[3]*val[5] - val[0]*val[2];
    float denominator = val[5] - val[2];
    val[6] = numerator / denominator;

}

float device_kelley_cutting(float *arr, const size_t n){
    const size_t MAXITER = 7;
    const size_t threads = 256;
    const size_t maxblocks = 2048;
    const size_t blocks = min((n+threads-1)/threads, maxblocks);

    const size_t wss = 9;
    const float THRES = 1e-6;
    const float MAXGAP = 10000;
    //Initialize and allocate workspace
    //d_w = [Y_r, F_r, G_r, Y_l, F_l, G_l, Y1, F1, G1]

    float* d_w;
    cudaMalloc((void**)&d_w, sizeof(float)*wss);
    cudaMemset(d_w, float(0), sizeof(float)*wss);
    
    float* w = (float*) calloc (9, sizeof(float));
     
    //Find the max and min of arr
    thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>> extrema_tuple;
    auto d_a = thrust::device_pointer_cast(arr);
    extrema_tuple = thrust::minmax_element(d_a, d_a+n);
    
    //Set Y_l to min
    w[3] = extrema_tuple.first[0];
    //printf("MIN %f\n", w[3]);
    //Set Y_r to max
    w[0] = extrema_tuple.second[0];
    //printf("MAX %f\n", w[0]);

    //Copy to device
    cudaMemcpy(d_w, w, sizeof(float)*wss, cudaMemcpyHostToDevice);

    //Reduce left
    device_kelley_reduce_kernel<<<blocks, threads>>>(arr, n, d_w+3);

    //Reduce right
    device_kelley_reduce_kernel<<<blocks, threads>>>(arr, n, d_w);

    float g = 0;    

    cudaMemcpy(w, d_w, sizeof(float)*wss, cudaMemcpyDeviceToHost);
    w[2] = 2*w[2] - n;
    w[5] = 2*w[5] - n;
    cudaMemcpy(d_w, w, sizeof(float)*wss, cudaMemcpyHostToDevice);
    /*
    printf("YR %f\n", w[0]);
    printf("OBJ %f\n", w[1]);
    printf("SUBGRAD (COUNT) %f\n", w[2]);


    printf("YL %f\n", w[3]);
    printf("OBJ %f\n", w[4]);
    printf("SUBGRAD (COUNT) %f\n", w[5]);
    */
    float median;

    size_t iter = 0;
    bool no_median = true;
    bool not_converged = true;
    bool req_sort = true;

    float lessL = 0;
    float lessR = 0;

    while(not_converged && iter < MAXITER){
    
        small_update_kernel<<<1, 1>>>(d_w);

        
        device_kelley_reduce_kernel<<<blocks, threads>>>(arr, n, d_w+6);

        cudaMemcpy(w, d_w, sizeof(float)*wss, cudaMemcpyDeviceToHost);

        w[8] = 2*w[8] - n;
        g =  w[8];

        /*
        printf("Iteration %d\n", iter);
        printf("SPLITPOINT %f\n", w[6]);
        printf("OBJ %f\n", w[7]);
        printf("SUBGRAD (COUNT) %f\n", g);
        */

        no_median = true;

        //TODO: Write a CUDA kernel for this swap? How can I avoid the two copies here? Should I?
        
        //YL is updated
        if (g <= -0.5){
            #pragma unroll
            for (size_t i = 0; i < wss/3; ++i){
                w[i+3] = w[i+6];
            }
            no_median = g < -1 ? false : true;
            //printf("TO THE LEFT\n");
        }

        //YR is updated
        if (g >= 0.5 and no_median){
            #pragma unroll
            for (size_t i = 0; i < wss/3; ++i){
                w[i] = w[i+6];
            }
            no_median = g > 1 ? false : true;
            //printf("TO THE RIGHT\n");
        }
        
        median = w[6];
        w[7] = float(0);
        w[8] = float(0);
        
        cudaMemcpy(d_w, w, sizeof(float)*wss, cudaMemcpyHostToDevice);        

        if (no_median || abs(w[0] - w[3]) < THRES || w[2] - w[5] < MAXGAP){
            not_converged=false;
            if(n%2 != 0 && (w[5] == 0 || w[2] == 0)){
                req_sort = false;
                //printf("MEDIAN FOUND EXACTLY");
            }
            else if(n%2 == 0 && (w[5] == 1 || w[2] == -1)){
                req_sort=false;
                //printf("MEDIAN FOUND EXACTLY");
            }
            else{
                //printf("Need to partition the gap %f\n", w[2] - w[5]);
            }
        }

        ++iter;
            
    }

    if(w[5] >= -1){
        median = w[3];
    } 

    if(w[3] <= 1){
        median = w[0];
    }
    /*
    if(req_sort){

        size_t* d_idx;
        //allocate space for idx keys
        cudaMalloc((void**)&d_idx, sizeof(size_t)*n);
        auto idx = thrust::device_pointer_cast(d_idx);
        auto copy = [=] __device__ ( double x ){ return (x < w[3] ) ? 1 : 
        thrust::transform(idx, idx+n, );
          
        
        //scan right
        //partition right
        
        //scan left
        //partition left

        //parition middle

        //sort middle
        
    }
    else{
        //thrust partition median
        
        



    }
    */
    //printf("==========================\n");
    //printf("LEFT BOUND: %f:%f \n", w[3], w[5]);
    //printf("RIGHT BOUND: %f:%f \n", w[0], w[2]);
    cudaFree(d_w);
    free(w);

    return median;
}






