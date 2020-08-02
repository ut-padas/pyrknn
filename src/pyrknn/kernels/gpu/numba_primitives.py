from numba import cuda, int32
import numpy as np

SECTION_SIZE = 2048 #should be twice the maximum number of threads

@cuda.jit
def numba_add(x, y, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]


@cuda.jit
def exclusive_scan(l, result, ps):
    # The length of l and result should be the same
    s = cuda.shared.array(shape=(SECTION_SIZE),dtype=int32) # The size and type of the arrays must be known at compile time
    pos = cuda.grid(1)

    # load the data to shared memory
    # if pos*2 < len(l):
    #     s[cuda.threadIdx.x*2] = l[pos*2]
    # else:
    #     s[cuda.threadIdx.x*2] = 0

    # if pos*2 + 1 < len(l):
    #     s[cuda.threadIdx.x*2 + 1] = l[pos*2 + 1]
    # else:
    #     s[cuda.threadIdx.x*2 + 1] = 0
    s[cuda.threadIdx.x*2] = l[pos*2] * (pos*2 < len(l))
    s[cuda.threadIdx.x*2 + 1] = l[pos*2 + 1] * (pos*2 + 1 < len(l))
    
    # Up sweep
    stride = 1
    while stride < SECTION_SIZE:
        cuda.syncthreads()
        idx = (cuda.threadIdx.x+1)*2*stride - 1
        if idx < SECTION_SIZE:
            s[int(idx)] += s[int(idx-stride)]
        stride *= 2
    
    # Down sweep
    stride = SECTION_SIZE//4
    while stride > 0:
        cuda.syncthreads()
        idx = (cuda.threadIdx.x+1)*2*stride - 1
        if (idx + stride) < SECTION_SIZE:
            s[int(idx + stride)] += s[int(idx)]
        stride //= 2

    # Copy the results
    cuda.syncthreads()
    if pos*2 + 1 < len(result):
        result[pos*2+1] = s[cuda.threadIdx.x*2]
    if pos*2 + 2 < len(result):
        result[pos*2 + 2] = s[cuda.threadIdx.x*2 + 1]
    # The very first element of result and ps should be zero
    if pos == 0:
        result[0] = 0
        ps[0] = 0

    if cuda.threadIdx.x == 0 and cuda.blockIdx.x < len(ps):
        ps[cuda.blockIdx.x] = s[SECTION_SIZE - 1]
    
    return


@cuda.jit
def exclusive_scan_inplace(l, ps):
    # The length of l and result should be the same
    s = cuda.shared.array(shape=(SECTION_SIZE),dtype=int32) # The size and type of the arrays must be known at compile time
    pos = cuda.grid(1)

    # load the data to shared memory
    # if pos*2 < len(l):
    #     s[cuda.threadIdx.x*2] = l[pos*2]
    # else:
    #     s[cuda.threadIdx.x*2] = 0

    # if pos*2 + 1 < len(l):
    #     s[cuda.threadIdx.x*2 + 1] = l[pos*2 + 1]
    # else:
    #     s[cuda.threadIdx.x*2 + 1] = 0
    s[cuda.threadIdx.x*2] = l[pos*2] * (pos*2 < len(l))
    s[cuda.threadIdx.x*2 + 1] = l[pos*2 + 1] * (pos*2 + 1 < len(l))
    
    # Up sweep
    stride = 1
    while stride < SECTION_SIZE:
        cuda.syncthreads()
        idx = (cuda.threadIdx.x+1)*2*stride - 1
        if idx < SECTION_SIZE:
            s[int(idx)] += s[int(idx-stride)]
        stride *= 2
    
    # Down sweep
    stride = SECTION_SIZE//4
    while stride > 0:
        cuda.syncthreads()
        idx = (cuda.threadIdx.x+1)*2*stride - 1
        if (idx + stride) < SECTION_SIZE:
            s[int(idx + stride)] += s[int(idx)]
        stride //= 2

    # Copy the results
    cuda.syncthreads()
    if pos*2 + 1 < len(l):
        l[pos*2+1] = s[cuda.threadIdx.x*2]
    if pos*2 + 2 < len(l):
        l[pos*2 + 2] = s[cuda.threadIdx.x*2 + 1]
    # The very first element of l and ps should be zero
    if pos == 0:
        l[0] = 0
        ps[0] = 0

    if cuda.threadIdx.x == 0 and cuda.blockIdx.x < len(ps):
        ps[cuda.blockIdx.x] = s[SECTION_SIZE - 1]
    return


@cuda.jit
def block_add(l,s):
    # Each thread will process two of the elements
    # Used as a subroutine of scan function
    elem = s[cuda.blockIdx.x]
    pos = cuda.grid(1)

    if pos*2 + 1 < len(l):
        l[pos*2 + 1] += elem
    if pos*2 + 2 < len(l):
        l[pos*2 + 2] += elem
    return


@cuda.jit
def less(array_in,array_out,index):
    pos = cuda.grid(1)
    value = array_in[index]

    if pos < len(array_in):
        array_out[pos] = int(array_in[pos] < value)
    return

@cuda.jit
def greater(array_in,array_out,index):
    pos = cuda.grid(1)
    value = array_in[index]

    if pos < len(array_in):
        array_out[pos] = int(array_in[pos] > value)
    return

@cuda.jit
def eq(array_in,array_out,index):
    pos = cuda.grid(1)
    value = array_in[index]

    if pos < len(array_in):
        array_out[pos] = int(array_in[pos] == value)
    return

@cuda.jit
def geq(array_in,array_out,index):
    pos = cuda.grid(1)
    value = array_in[index]

    if pos < len(array_in):
        array_out[pos] = int((array_in[pos] >= value) and (pos != index))
    return


@cuda.jit
def negate(array):
    pos = cuda.grid(1)

    if pos < len(array):
        array[pos] = int(not array[pos])
    return

@cuda.jit
def set_value(array,i,val):
    if cuda.threadIdx.x == 0:
        array[i] = val
    return

@cuda.jit
def placing_less(array_in,array_out,indices,index):
    pos = cuda.grid(1)
    value = array_in[index]
    
    if pos < len(array_in):
        if array_in[pos] < value:
            array_out[indices[pos]] = array_in[pos]
    return


@cuda.jit
def placing_geq(array_in,array_out,indices,index):
    pos = cuda.grid(1)
    value = array_in[index]
    
    if pos < len(array_in):
        if array_in[pos] >= value and pos != index:
            array_out[indices[pos]] = array_in[pos]
    return

@cuda.jit
def mv_value(array_in,array_out,srci,dsti):
    pos = cuda.grid(1)

    if pos == 0:
        array_out[dsti] = array_in[srci]
    return


def scan(l):
    # The function will return a block of memory on GPU
    result_gpu = cuda.device_array_like(l)
    nthreads = 1024
    nblocks = (len(l)+nthreads*2-1)//(nthreads*2)
    assert(nblocks<=65536)
    s = cuda.device_array(nblocks)
    if len(l)<=2048:
        # base case here
        exclusive_scan[1,nthreads](l,result_gpu,s)
        return result_gpu
    
    exclusive_scan[nblocks,nthreads](l,result_gpu,s) # total number of threads should be greater than the number of elements in array
    s = scan(s)
    block_add[nblocks,nthreads](result_gpu,s)
    return result_gpu


def scan_inplace(l):
    # The function runs scan operation but use inplace exclusive scan
    nthreads = 1024
    nblocks = (len(l)+nthreads*2-1)//(nthreads*2)
    assert(nblocks<=65536)
    s = cuda.device_array(nblocks)
    if len(l)<=2048:
        # base case here
        exclusive_scan_inplace[1,nthreads](l,s)
        return
    
    exclusive_scan_inplace[nblocks,nthreads](l,s) # total number of threads should be greater than the number of elements in array
    scan_inplace(s)
    block_add[nblocks,nthreads](l,s)
    return


# This method use extra memory and does not do things in place.
def partition(l,index):
    # Partition the elements in array l into two arrays by less than or greater than value
    # The two result arrays will stay in GPU
    nthreads = 1024
    nblocks = (len(l)+nthreads-1)//(nthreads)
    assert(nblocks<=65536)

    temp_arr = cuda.device_array(len(l)+1,dtype=np.int32)
    set_value[1,1](temp_arr,-1,0)
    less[nblocks,nthreads](l,temp_arr,index)
    left_indices = scan(temp_arr)
    last_elem = left_indices[-1:].copy_to_host()
    if last_elem[0] != 0:
        left_values = cuda.device_array(shape=int(last_elem[0]),dtype=np.int32)
        placing_less[nblocks,nthreads](l,left_values,left_indices,index)
    else:
        left_values = []

    geq[nblocks,nthreads](l,temp_arr,index)
    set_value[1,1](temp_arr,-1,0)
    right_indices = scan(temp_arr)
    last_elem = right_indices[-1:].copy_to_host()
    if last_elem[0] != 0:
        right_values = cuda.device_array(len(l)-len(left_values),dtype=np.int32)
        mv_value[1,1](l,right_values,index,0)
        placing_geq[nblocks,nthreads](l,right_values[1:],right_indices,index)
    else:
        right_values = []

    n = len(left_values)
    l[:n] = cuda.to_device(left_values)
    l[n:] = cuda.to_device(right_values)

    return n

# Depreciated. Should implement in-place partition
def merge(array1, array2):
    ret = cuda.device_array(len(array1)+len(array2),dtype=np.float32)
    ret[:len(array1)] = cuda.to_device(array1)
    ret[len(array1):] = cuda.to_device(array2)
    return ret


def partition_inplace(l,index):
    '''
    l should reside in GPU. Otherwise, there will be a lot of memory movement. 

    This function implements inplace partition using inplace scan and store the
    result in l. 

    This function use 3N memory.

    The function return how many points to the left of the pivot (including the pivot).
    '''
    n = len(l)
    nthreads = 1024
    nblocks = (n+nthreads-1)//(nthreads)
    assert(nblocks<=65536)

    # Placing the left values
    indices = cuda.device_array(n+1,dtype=np.int32)
    less[nblocks,nthreads](l,indices,index)
    set_value[1,1](indices,-1,0)
    scan_inplace(indices)
    last_elem = indices[-1:].copy_to_host()
    nleft = int(last_elem[0])
    left_values = cuda.device_array(shape=nleft,dtype=np.float32)
    if nleft != 0:
        placing_less[nblocks,nthreads](l,left_values,indices,index)

    # Placing the right values
    geq[nblocks,nthreads](l,indices,index)
    set_value[1,1](indices,-1,0)
    scan_inplace(indices)
    last_elem = indices[-1:].copy_to_host()
    right_values = cuda.device_array(n-nleft,dtype=np.float32)
    mv_value[1,1](l,right_values,index,0)
    if last_elem[0] != 0:
        placing_geq[nblocks,nthreads](l,right_values[1:],indices,index)
    #print(right_values.copy_to_host())

    # Copy the results
    if nleft != 0:
        l[:nleft] = cuda.to_device(left_values)
    if nleft != n:
        l[nleft:] = cuda.to_device(right_values)

    return nleft+1



def quickSelect(k,l):
    '''
    Randomized Selection. 

    After the function is done, the k the smallest element will reside in the kth position.
    The method is not optimized when we have a large number of elements that are equal.
    '''
    n = len(l)
    if n == 1:
        return
    p = n//2
    #print("p is ",p)
    nleft = partition_inplace(l,p)

    if nleft == k:
        return
    elif nleft > k:
        quickSelect(k, l[:nleft-1])
    else:
        quickSelect(k-nleft,l[nleft:])
    return



@cuda.jit
def print_array(l):
    pos = cuda.grid(1)
    if pos < len(l):
        print(l[pos])
    return
