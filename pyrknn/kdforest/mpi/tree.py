from . import error as ErrorType
from . import util as Primitives

import os
import gc 
import time 

import numpy as np
import scipy.sparse as sp
from collections import defaultdict, deque 

from numba import njit, prange, set_num_threads

set_num_threads(16)

from mpi4py import MPI

def rprint(tag, obj, comm=MPI.COMM_WORLD, rank=0):
    if comm.Get_rank() == rank:
        print(tag, obj, flush=True)

@njit(fastmath=True, parallel=True)
def pack_sparse_sizes(data_ptr, data_col, data_val, requests):
    n = requests.shape[0]
    l = np.empty(n, dtype=data_ptr.dtype)

    for i in prange(n):
        rid = requests[i]
        l[i] = data_ptr[rid+1] - data_ptr[rid]
    return l

@njit(parallel=True, fastmath=True)
def pack_sparse_values(nptr, data_ptr, data_col, data_val, requests):
    n = requests.shape[0]
    nnz = nptr[-1]
    nc = np.empty(nnz, dtype=data_col.dtype)
    nv = np.empty(nnz, dtype=data_val.dtype)

    for i in prange(n):
        rid = requests[i]

        nstart = nptr[i]
        nend = nptr[i+1]

        start = data_ptr[rid]
        end = data_ptr[rid+1]

        nv[nstart:nend] = data_val[start:end]
        nc[nstart:nend] = data_col[start:end]

    return nc, nv

def numpy_exsum(array):
    array = np.insert(array, 0, 0)
    prefix = np.cumsum(array)
    return prefix

def pack_sparse(data, requests):
    n = requests.shape[0]
    d = data.shape[1]
    data_ptr = data.indptr
    data_col = data.indices
    data_val = data.data
    lengths = pack_sparse_sizes(data_ptr, data_col, data_val, requests)
    nptr = numpy_exsum(lengths)
    nc, nv = pack_sparse_values(nptr, data_ptr, data_col, data_val, requests)
    #new_data = sp.sparse.csr_matrix((nv, nc, nptr), shape=(n, d))
    return nptr, nc, nv, lengths

@njit(fastmath=True, parallel=True)
def label_id(global_ids, size_prefix):
    N = len(global_ids)
    labels = np.empty(N, dtype=np.int32)
    for i in prange(N):
        current_id = global_ids[i]
        idx = np.searchsorted(size_prefix, current_id, side='right')-1
        labels[i] = idx
    return labels


@njit(fastmath=True, parallel=True)
def reorder(array, p):
    N = p.shape[0]
    new = np.empty(N, dtype=array.dtype)
    for i in prange(N):
        new[i] = array[p[i]]
    return new

@njit(fastmath=True, parallel=True)
def reorder_2(array, p):
    N = p.shape[0]
    d = array.shape[1]
    new = np.empty((N, d), dtype=array.dtype)
    for i in prange(N):
        for j in range(d):
            new[i, j] = array[p[i], j]
    return new


@njit(fastmath=True, parallel=True)
def reorder_3(array, p, start=0):
    d = array.shape[0]
    N = array.shape[1]
    new = np.empty_like(array)
    #print(array)
    for j in prange(start, d):
        for i in range(N):
            #print(i, p[i], array[j, p[i]])
            new[j, i] = array[j, p[i]]
    return new

@njit(fastmath=True, parallel=True)
def ordered_sizes(mpi_size, keys, prec=np.float32):
    starts = np.zeros(mpi_size, dtype=prec)
    stops = np.zeros(mpi_size, dtype=prec)
    n = keys.shape[0]

    # TODO: Test parallel version
    for i in prange(n):
        if i > 0 and keys[i] != keys[i-1]:
            starts[keys[i]] = i
            stops[keys[i-1]] = i
        if i == n-1:
            stops[keys[i]] = i+1

    return starts, stops-starts


@njit(fastmath=True, parallel=True)
def global_2_local(gl, size_prefix, rank):
    local = gl - size_prefix[rank]
    return local

@njit(fastmath=True, parallel=True)
def get_nnz_index(lptr, rstarts, rsizes):
    nnz_starts = lptr[rstarts]
    nnz_ends = lptr[rstarts+rsizes]
    nnz_sizes = nnz_starts - nnz_ends


def gather_sparse(comm, data, requests, size_prefix, rank, sizes, starts, rsizes, rstarts):
    #Assume data is tuple of CSR ndarrays
    data_ptr = data.indptr 
    data_col = data.indices
    data_val = data.value 

    #Gather Requests
    timer.push("Collect Points: Gather Requests")
    requests = global_2_local(requests, size_prefix, rank)
    lptr, nc, nv, lengths = pack_sparse(data, requests)
    timer.pop("Collect Points: Gather Requests")

    #TODO: Overlap prefix sum with col/val A2As?

    #Send requested size (unsummed row ptr)
    timer.push("Collect Points: Allocate Row Size List")
    #TODO: Check length on this
    recv_l = np.zeros(data_ptr.shape[0], dtype=data_ptr.dtype)
    timer.pop("Collect Points: Allocate Row Size List")

    timer.push("Collect Points: Communicate Row Size List")
    if data_ptr.dtype == np.int32:
        comm.Alltoallv([lengths, rsizes, rstarts, MPI.INT], [
                    recv_l, sizes, starts, MPI.INT])
    else:
        comm.Alltoallv([lengths, rsizes, rstarts, MPI.LONG], [
            recv_l, sizes, starts, MPI.LONG])
    timer.pop("Collect Points: Communicate Row Size List")

    #Get local indexing for received data
    timer.push("Collect Points: Prefix Sum (row ptr)")
    nptr = numpy_exsum(recv_data)
    timer.pop("Collect Points: Prefix Sum (row ptr)")

    #Get start/stops for nnz indexed arrays
    timer.push("Collect Points: NNZ Idx")
    nnz_starts, nnz_sizes = get_nnz_index(lptr, rstarts, rsizes)
    timer.pop("Collect Points: NNZ Idx")

    timer.push("Collect Points: NNZ recv")
    nnz_rsizes, nnz_rstarts = exchange_send_info(comm, sizes)
    timer.pop("Collect Points: NNZ recv")

    #Send column indices
    timer.push("Collect Points: Allocate Col Idx")
    #TODO: Check length on this
    n = nnz_rsizes[-1] + nnz_rstarts[-1]
    recv_c = np.zeros(n, dtype=data_col.dtype)
    timer.pop("Collect Points: Allocate Col Idx")

    timer.push("Collect Points: Communicate Col Idx")
    if data_col.dtype == np.int32:
        comm.Alltoallv([nc, nnz_sizes, nnz_starts, MPI.INT], [
            recv_c, nnz_rsizes, nnz_rstarts, MPI.INT])
    else:
        comm.Alltoallv([nc, nnz_sizes, nnz_starts, MPI.LONG], [
            recv_c, nnz_rsizes, nnz_rstarts, MPI.LONG])
    timer.pop("Collect Points: Communicate Col Idx")

    #Send values
    timer.push("Collect Points: Allocate Values")
    recv_v = np.zeros(n, dtype=data_val.dtype)
    timer.pop("Collect Points: Allocate Values")

    timer.push("Collect Points: Communicate Values")
    comm.Alltoallv([nv, nnz_sizes, nnz_starts, MPI.FLOAT], [
        recv_v, nnz_rsizes, nnz_rstarts, MPI.FLOAT])
    timer.pop("Collect Points: Communicate Values")

    #return nptr, recv_c, recv_v

    #Convert/Wrap in SciPy Datatype
    new_data = sp.csr_matrix((recv_v, recv_c, nptr), shape=data.shape)

    return new_data







    


def gather_dense(comm, data, requests, size_prefix, rank, sizes, starts, rsizes, rstarts):
    timer = Primitives.Profiler()

    # Gather Requests
    timer.push("Collect Points: Gather Requests")
    #requests = np.asarray(requests, dtype=np.int32)
    requests = global_2_local(requests, size_prefix, rank)
    #print(rank, "Updated requests", requests, flush=True)
    requested_data = reorder_2(data, requests)
    #requested_data = data[requests]
    timer.pop("Collect Points: Gather Requests")

    # Send requested data
    timer.push("Collect Points: Communicate Data")

    timer.push("Collect Points: Communicate Data - Allocate")
    #recv_data = np.zeros_like(data)
    recv_data = np.zeros((data.shape[0], data.shape[1]), dtype=data.dtype)
    #print(rank, "check emp", recv_data)
    timer.pop("Collect Points: Communicate Data - Allocate")

    timer.push("Collect Points: Communicate Data - AlltoAll")
    #requested_data[0] = 10
    #print(rank, "Requested Data", requested_data, flush=True)
    #print(rank, "send", rsizes, rstarts, flush=True)
    #print(rank, "recv", sizes, starts, flush=True)
    d = data.shape[1]
    #print(rank, "D", d)
    comm.Alltoallv([requested_data, rsizes*d, rstarts*d, MPI.FLOAT], [
                   recv_data, sizes*d, starts*d, MPI.FLOAT])
    #print(rank, "Recv Data", recv_data, flush=True)
    timer.pop("Collect Points: Communicate Data - AlltoAll")

    timer.pop("Collect Points: Communicate Data")

    return recv_data

def collect(comm, requested_global_ids, data, size_prefix, dtype=np.int64):

    sparse_flag = isinstance(data, sp.csr_matrix)
    dense_flag = isinstance(data, np.ndarray)

    timer = Primitives.Profiler()
    mpi_size = comm.Get_size()
    rank = comm.Get_rank()

    N = data.shape[0]

    if data.ndim == 2:
        d = data.shape[1]
    else:
        d = 1

    # Organize new locally owned ids, determine which process each coordinate needs to be requested from
    # TODO: Replace this with a parallel reduce by key implementation

    timer.push("Collect Points: Organize Points")

    timer.push("Collect Points: Organize Points - label")
    labels = label_id(requested_global_ids, size_prefix)
    timer.pop("Collect Points: Organize Points - label")
    #print(rank, "labels", labels, flush=True)

    timer.push("Collect Points: Organize Points - sort")
    p = np.argsort(labels, kind='stable')
    timer.pop("Collect Points: Organize Points - sort")

    timer.push("Collect Points: Organize Points - reorder")
    #labels = reorder(labels, p)
    #requested_global_ids = reorder(requested_global_ids, p)
    labels = labels[p]
    requested_global_ids = requested_global_ids[p]
    timer.pop("Collect Points: Organize Points - reorder")

    timer.pop("Collect Points: Organize Points")

    # Compute sending sizes
    timer.push("Collect Points: Find Request Sizes")
    starts, sizes = ordered_sizes(
        mpi_size, labels, prec=requested_global_ids.dtype)
    #print(rank, "s", sizes, flush=True)
    timer.pop("Collect Points: Find Request Sizes")

    # Exchange recving sizes
    timer.push("Collect Points: Exchange Recv Sizes")
    sizes = np.asarray(sizes, dtype=np.int32)
    rsizes, rstarts = exchange_send_info(comm, sizes)
    #print(rank, "rs", rsizes, flush=True)
    #print(rank, "ordered ids", requested_global_ids, flush=True)
    timer.pop("Collect Points: Exchange Recv Sizes")

    # Send id requests
    timer.push("Collect Points: Communicate Requests")
    #print(rank, "Number of requests", np.sum(rsizes), flush=True)
    requests = np.zeros(int(np.sum(rsizes)), dtype=requested_global_ids.dtype)
    if requests.dtype == np.int32:
        comm.Alltoallv([requested_global_ids, sizes, starts, MPI.INT], [
                       requests, rsizes, rstarts, MPI.INT])
    else:
        comm.Alltoallv([requested_global_ids, sizes, starts, MPI.LONG], [
                       requests, rsizes, rstarts, MPI.LONG])
    #print(rank, "requested ids", requests, flush=True)
    timer.pop("Collect Points: Communicate Requests")

    if dense_flag:
        recv_data = gather_dense(comm, data, requests, size_prefix, rank, sizes, starts, rsizes, rstarts)
    elif sparse_flag:
        recv_data = gather_sparse(comm, data, requests, size_prefix, rank, sizes, starts, rsizes, rstarts)
    else:
        raise Exception()
        #TODO: More specific error handling

    return recv_data, requested_global_ids


def redistribute(comm, global_ids, result, size_prefix):
    timer = Primitives.Profiler()
    timer.push("Redistribute")

    N = result[0].shape[0]
    k = result[0].shape[1]

    # Check result sizes match
    assert(N == result[1].shape[0])
    assert(k == result[1].shape[1])

    neighbor_ids = result[0]
    neighbor_dist = result[1]

    # Check datatype consistency (important for mpi4py, not using automatic interface)
    assert(global_ids.dtype == neighbor_ids.dtype)
    assert(neighbor_dist.dtype == np.float32)

    mpi_size = comm.Get_size()
    rank = comm.Get_rank()

    # Allocate storage to recieve all to all results
    # These will be the output
    # TODO: Move this allocate outside of the function, to reuse memory between iterations
    timer.push("Redistribute: Allocate")
    result_gids = np.zeros(N, dtype=neighbor_ids.dtype)
    result_ids = np.zeros_like(neighbor_ids)
    result_dist = np.zeros_like(neighbor_dist)
    timer.pop("Redistribute: Allocate")

    # Label each result id with its destination process
    timer.push("Redistribute: Label")
    labels = label_id(global_ids, size_prefix)
    timer.pop("Redistribute: Label")

    # TODO: Replace the below with a reduce/scan by key implementation

    # Sort the labels to get contiguous bands to send
    # TODO: Replace with a parallel sort
    timer.push("Redistribute: Sort")
    p = np.argsort(labels)
    timer.pop("Redistribute: Sort")

    # Reorder everything to get contiguous bands to send
    timer.push("Redistribute: Reorder")
    labels = reorder(labels, p)
    #neighbor_ids = reorder(neighbor_ids, p)
    #neighbor_dist = reorder(neighbor_dist, p)
    #global_ids = reorder(global_ids, p)
    neighbor_ids = neighbor_ids[p]
    neighbor_dist = neighbor_dist[p]
    global_ids = global_ids[p]
    timer.pop("Redistribute: Reorder")

    # Compute start/stop indicies
    timer.push("Redistribute: Compute Targets")
    starts, sizes = ordered_sizes(mpi_size, labels, prec=neighbor_ids.dtype)
    timer.pop("Redistribute: Compute Targets")

    # Exchange sizes with receiving processes
    timer.push("Redistribute: Exchange Recv")
    rsizes, rstarts = exchange_send_info(comm, sizes)
    timer.pop("Redistribute: Exchange Recv")

    # Communicate global ids
    timer.push("Redistribute: Alltoall GIDs")
    if global_ids.dtype == np.int32:
        comm.Alltoallv([global_ids, sizes, starts, MPI.INT], [
                       result_gids, rsizes, rstarts, MPI.INT])
    else:
        comm.Alltoallv([global_ids, sizes, starts, MPI.LONG], [
                       result_gids, rsizes, rstarts, MPI.LONG])
    timer.pop("Redistribute: Alltoall GIDs")

    # Adjust sizes to be k-stride
    sizes = sizes * k
    rsizes = rsizes * k

    starts = starts * k
    rstarts = rstarts * k

    # Communicate neighbor ids
    timer.push("Redistribute: Alltoall IDs")
    if neighbor_ids.dtype == np.int32:
        comm.Alltoallv([neighbor_ids, sizes, starts, MPI.INT], [
                       result_ids, rsizes, rstarts, MPI.INT])
    else:
        comm.Alltoallv([neighbor_ids, sizes, starts, MPI.LONG], [
                       result_ids, rsizes, rstarts, MPI.LONG])
    timer.pop("Redistribute: Alltoall IDs")

    # Communicate neighbor distances
    timer.push("Redistribute: Alltoall Dist")
    comm.Alltoallv([neighbor_dist, sizes, starts, MPI.FLOAT], [
        result_dist, rsizes, rstarts, MPI.FLOAT])
    timer.pop("Redistribute: Alltoall Dist")

    # Output results need to be sorted by global id

    # TODO: Replace with parallel sort
    timer.push("Redistribute: Sort Output")
    lids = np.argsort(result_gids)
    timer.pop("Redistribute: Sort Output")

    timer.push("Redistribute: Reorder Output")
    #result_gids = reorder(result_gids, lids)
    #result_ids = reorder(result_ids, lids)
    #result_dist = reorder(result_dist, lids)
    result_gids = result_gids[lids]
    result_ids = result_ids[lids, :]
    result_dist = result_dist[lids, :]
    timer.pop("Redistribute: Reorder Output")

    timer.pop("Redistribute")

    return result_gids, (result_ids, result_dist)


#@njit(fastmath=True)
def balance_partition(rank, mpi_size, left_size, left_prefix, right_size, right_prefix, size_prefix):

    starts = np.zeros(mpi_size, dtype=size_prefix.dtype)
    sizes = np.zeros(mpi_size, dtype=size_prefix.dtype)

    # TODO: This shouldn't be necessary. Test without, change datatypes before hand if necessary.
    size_prefix = np.asarray(size_prefix, dtype=size_prefix.dtype)
    left_prefix = np.asarray(left_prefix, dtype=size_prefix.dtype)
    right_prefix = np.asarray(right_prefix, dtype=size_prefix.dtype)

    # Left Buffer
    # ======================================

    # Compute the initial send idx of left buffer
    before_in_bin = left_prefix[rank]
    start_idx = np.searchsorted(size_prefix, before_in_bin, side='right')-1

    # Compute the final send idx of left buffer
    after_in_bin = left_prefix[rank] + left_size
    last_idx = np.searchsorted(size_prefix, after_in_bin, side='right')-1

    # Correction for last bin (we overcount above)
    if after_in_bin == size_prefix[last_idx]:
        last_idx -= 1

    print(rank, "(sending bins)", start_idx, last_idx)
    print(rank, "(inbin/size_of_last)", after_in_bin, size_prefix[last_idx])
    r = 1
    local_start = 0
    to_send = left_size
    for idx in range(start_idx, last_idx+1):
        remain = size_prefix[idx+1] - left_prefix[rank]
        #rprint("l idx", idx, rank=r)
        #rprint("l remain", remain, rank=r)
        #rprint("l size_prefix[idx]", size_prefix[idx], rank=r)
        #rprint("l size_prefix[idx+1]", size_prefix[idx+1], rank=r)
        #rprint("l left_prefix[rank]", left_prefix[rank], rank=r)
        #rprint("l to_send", to_send, rank=r)
        #rprint("l local_start", local_start, rank=r)
        #rprint("l total_sent", before_in_bin+local_start, rank=r)
        send_size = min(remain, to_send)
        to_send -= send_size

        sizes[idx] = send_size
        starts[idx] = local_start

        local_start += send_size

        #print(rank, send_size, idx, flush=True)
    #print(rank, start_idx, last_idx)

    # Right Buffer
    # ======================================

    # Adjust counts (normalize right starting to 0)
    middle = mpi_size//2
    size_prefix = size_prefix[middle:] - size_prefix[middle]

    # Compute the initial send idx of right buffer
    before_in_bin = right_prefix[rank]
    start_idx = np.searchsorted(size_prefix, before_in_bin, side='right')-1

    # Compute the final send idx of right buffer
    after_in_bin = right_prefix[rank] + right_size
    last_idx = np.searchsorted(size_prefix, after_in_bin, side='right')-1

    # Correction for last bin (we overcount above)
    if after_in_bin == size_prefix[last_idx]:
        last_idx -= 1

    #local_start = 0
    to_send = right_size
    for idx in range(start_idx, last_idx+1):
        remain = size_prefix[idx+1] - right_prefix[rank]
        #rprint("idx", idx, rank=r)
        #rprint("remain", remain, rank=r)
        #rprint("size_prefix[idx]", size_prefix[idx], rank=r)
        #rprint("size_prefix[idx+1]", size_prefix[idx+1], rank=r)
        #rprint("left_prefix[rank]", left_prefix[rank], rank=r)
        #rprint("to_send", to_send, rank=r)
        #rprint("local_start", local_start, rank=r)
        #rprint("total_sent", before_in_bin+local_start, rank=r)
        send_size = min(remain, to_send)
        to_send -= send_size

        sizes[middle+idx] = send_size
        starts[middle+idx] = local_start

        local_start += send_size

        #print(rank, send_size, middle+idx, flush=True)

    #print(rank, sizes, starts)
    return sizes, starts


def exchange_send_info(comm, sizes):
    # Get recv sizes and starts for all-to-allv calls from send sizes

    dtype = sizes.dtype
    
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    #print(rank, "dtype", dtype)

    temp_sizes = np.ones(mpi_size, dtype=dtype)
    temp_starts = np.arange(mpi_size, dtype=dtype)
    recv = np.zeros(mpi_size, dtype=dtype)
    #print("exchange")
    # Switch mpi4py call based on data type.
    # TODO: There is a better way to do this in mpi4py 3.10 which was just released, switch or keep for compatibility?
    if dtype == np.int32:
        #print(rank, "This should print ::: ", sizes, temp_sizes, temp_starts)
        comm.Alltoallv([sizes, temp_sizes, temp_starts, MPI.INT],
                       [recv, temp_sizes, temp_starts, MPI.INT])
    else:
        #print(rank, "This shouldn't print ::: ", sizes, temp_sizes, temp_starts)
        comm.Alltoallv([sizes, temp_sizes, temp_starts, MPI.LONG],
                       [recv, temp_sizes, temp_starts, MPI.LONG])

    temp = np.zeros(mpi_size+1, dtype=dtype)
    temp[1:] = recv
    starts = np.cumsum(temp)[:-1]

    return recv, starts


class RKDT:

    verbose = False

    def __init__(self, levels=0, leafsize=512, data=None, location="CPU", sparse=False, comm=None, cores=None):

        #TODO: Make this automatic with psutils
        if cores is None:
            self.cores = 8
        else:
            self.cores = cores

        # Set precision threshold (testing with lower than MAXINT for safety)
        precision_threshold = 1000000000

        # Store tree parameters
        self.max_levels = levels
        self.leafsize = leafsize
        self.location = location
        self.sparse = sparse

        # Setup MPI Communicator
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()
        self.rank = rank

        # Assign each tree object a unique id
        self.loc_id = id(self)

        # Share same id as on rank 0
        self.glb_id = 0
        if rank == 0:
            self.glb_id = id(self)
        self.glb_id = self.comm.bcast(self.glb_id)

        # Get size of locally owned points.
        # This balance will be maintained over the tree construction
        self.local_size = data.shape[0]

        # Set precision for local ids based on local size
        self.lprec = np.int32
        # TODO: Currently we only support a local index of INT32
        # TODO: - Would need to change CUDA and GSKNN kernels
        # Likely not worth the effort? 2 billion points in a single MPI Process seems excessive

        # TODO: Make this parallel with MPI if it becomes a scaling issue? (Reduce, Exscan, etc.)

        # Gather local arrays
        local_size_list = self.comm.allgather(self.local_size)
        local_size_list.insert(0, 0)
        self.global_size = np.sum(local_size_list)

        # Set precision for global ids based on global size
        self.gprec = np.int64
        if(self.global_size < precision_threshold):
            self.gprec = np.int32

        # Fix datatype of size_list to match needed precision
        local_size_list = np.asarray(local_size_list, dtype=self.gprec)

        # Compute exclusive prefix sum of local sizes
        # To be used in partition balancing
        self.prefix_sizes = np.cumsum(local_size_list)
        self.prefix_sizes = self.prefix_sizes

        # Allocate local and global ids
        self.local_ids = np.arange(self.local_size, dtype=self.lprec)

        start_idx = self.prefix_sizes[rank]
        self.global_ids = np.arange(
            start_idx, start_idx+self.local_size, dtype=self.gprec)

        #print(rank, self.global_ids, flush=True)

        # Ensure data is in float32 precision
        self.host_data = None
        if self.sparse:
            N, d = data.shape
            local_value = np.asarray(data.data, dtype=np.float32)
            local_rowptr = np.asarray(data.indptr, dtype=self.lprec)
            local_colidx = np.asarray(data.indices, dtype=self.lprec)

            self.host_data = sp.csr_matrix(
                (local_value, local_colidx, local_rowptr), size=(N, d))
        else:
            self.host_data = np.asarray(data, dtype=np.float32)

        self.dim = self.host_data.shape[1]

        # Assumes all trees have same location and datatype
        # TODO: Change this, have primitives take self.location, self.sparse as input (default: host, False)
        #Primitives.set_env(self.location, self.sparse)
        #print(self.local_size, self.leafsize)
        #print(np.ceil(np.log2(np.ceil(self.local_size/self.leafsize))), self.max_levels)
        
        # Update max tree levels by leafsize and level parameter
        self.dist_levels = int(np.floor(np.log2(self.mpi_size)))
        self.local_levels = int(
            min(np.ceil(np.log2(np.ceil(self.local_size/self.leafsize))), self.max_levels))

        self.built = False

        print(self.rank, "Initialized Tree", flush=True)

    """
    def __del__(self):
        del self.global_ids
        del self.local_ids
        del self.host_data
    """

    @classmethod
    def set_verbose(self, v):
        self.verbose = v

    def __str__(self):
        msg = f"Tree: (id: {self.glb_id}, llevels: {self.local_levels}, dlevels: {self.dist_levels}) \n"
        if self.verbose:
            msg += f"Built: {self.built} \n"
            msg += f"Shape: (local_size: {self.local_size}, global_size: {self.global_size}, dim: {self.dim} )\n"
            msg += f"Index: (start: {self.global_ids[0]}, stop: {self.global_ids[-1]}) \n"
        return msg

    def generate_projection_vectors(self, levels):
        timer = Primitives.Profiler()

        # Assume processes share the same random seed to avoid communication

        timer.push("Projection: Rand")
        vectors = np.random.randn(self.dim, levels)
        vectors = np.asarray(vectors, dtype=np.float32, order='F')
        timer.pop("Projection: Rand")

        timer.push("Projection: QR")
        vectors = np.linalg.qr(vectors)[0]
        timer.pop("Projection: QR")

        self.spill = None
        #print("HERE")
        print("projection construction: ", levels, self.dim)
        #print(self.rank, "Check vectors: ", vectors, flush=True)
        print("projection vector layout: ", vectors.flags)
        if levels > self.dim:
            index = np.arange(self.dim, dtype=self.lprec)
            spill = np.random.randint(
                low=0, high=self.dim, size=levels-self.dim, dtype=self.lprec)
            self.spill = np.concatenate((index, spill), axis=0)
            #print("HERE: ", self.spill)
        print(self.rank, vectors, self.spill, flush=True)
        self.vectors = vectors

    def distributed_build(self):
        timer = Primitives.Profiler()
        comm = self.comm
        local_rank = comm.Get_rank()
        global_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        timer.push("Dist Build:")
        if mpi_size > 1:
            timer.push("Build: Generate Projection")
            # Generate orthogonal projection vectors
            self.generate_projection_vectors(self.dist_levels)
            timer.pop("Build: Generate Projection")

            timer.push("Dist Build: Compute Projection")
            # Compute projection upfront.
            # If levels > dim, append projection with resampled random vectors
            vectors = self.vectors
            if self.spill is not None:
                print("USING SPILL", self.spill, flush=True)
                print("before", vectors.shape)
                vectors = vectors[self.spill]
                print("after", vectors.shape)
            print("after", vectors.shape, flush=True)
            print("data", self.host_data.shape, flush=True)
            proj = self.host_data @ vectors
            #proj = proj.reshape(self.local_size, self.dist_levels)
            timer.pop("Dist Build: Compute Projection")
            
            #print(global_rank, "proj", proj)
            #print(global_rank, "proj shape", proj.shape)

            # For each level in the distributed tree
            #   Each node is its own MPI communicator
            #   - Find median of node (k-select)
            #   - Compute balanced sending/recv targets
            #   - Send remaining projection and global ids
            #   - Split communicator between children
            for l in range(self.dist_levels):

                timer.push("Dist Build: Get Global Size")
                local_rank = comm.Get_rank()
                global_size = np.array(0, dtype=self.gprec)
                local_size = np.array(self.local_size, dtype=self.gprec)
                comm.Allreduce(local_size, global_size, op=MPI.SUM)
                timer.pop("Dist Build: Get Global Size")
                print(global_rank, "Current Comm Size", mpi_size, flush=True)

                # Compute median
                # Get permutation vector for median partition (lids)
                timer.push("Dist Build: Distributed Select")
                lids = np.arange(self.local_size, dtype=self.lprec)

                # Always grab the first element of projection.
                # We shrink it each time in the loop to pass less data

                current_proj = np.copy(proj[:, l])

                median, local_split = Primitives.dist_select(
                    global_size/2, current_proj, lids, comm)
                timer.pop("Dist Build: Distributed Select")

                # Reorder left and right of projection to prepare for communication
                timer.push("Dist Build: Reorder Projection")
                #proj = reorder_2(proj, lids)
                proj = proj[lids, :]
                timer.pop("Dist Build: Reorder Projection")

                # Reorder left and right of global ids to prepare for communication
                timer.push("Dist Build: Reorder Global Ids")
                #self.global_ids[:] = reorder(self.global_ids, lids)
                self.global_ids = self.global_ids[lids]
                timer.pop("Dist Build: Reorder Global Ids")

                timer.push("Dist Build: Compute Targets")

                # nleft - number of local points to the left of global median
                nleft = local_split
                # nright - number of local points to the right of global median
                nright = self.local_size - local_split
               
                #print(global_rank, "(lids)", lids, flush=True)
                #print(global_rank, "(gids)", self.global_ids, flush=True)
                print(global_rank, "(median)", median, proj, flush=True)
                print(global_rank, "(nleft/nright)", nleft, nright, flush=True)
            
                timer.push("Dist Build: Compute Targets - Prefix Sums")

                # Perform prefix sum on nleft and gather
                nleft_buffer = np.array(nleft, dtype=self.gprec)
                nleft_sum = np.array(0, dtype=self.gprec)
                comm.Exscan(nleft_buffer, nleft_sum)
                nleft_prefix = np.zeros(mpi_size, dtype=self.gprec)
                comm.Allgather(nleft_sum, nleft_prefix)

                # Perform prefix sum on nright and gather
                nright_buffer = np.array(nright, dtype=self.gprec)
                nright_sum = np.array(0, dtype=self.gprec)
                comm.Exscan(nright_buffer, nright_sum)
                nright_prefix = np.zeros(mpi_size, dtype=self.gprec)
                comm.Allgather(nright_sum, nright_prefix)
                timer.pop("Dist Build: Compute Targets - Prefix Sums")

                #print(global_rank, "(left prefix / right prefix)", nleft_prefix, nright_prefix, flush=True)
                #print(global_rank, "(prefix sizes)", self.prefix_sizes, flush=True)

                # Compute what I need to send to where (to keep load balance and setup child nodes)
                timer.push("Dist Build: Compute Targets - Balance")
                sizes, starts = balance_partition(
                    local_rank, mpi_size, nleft, nleft_prefix, nright, nright_prefix, self.prefix_sizes)
                timer.pop("Dist Build: Compute Targets - Balance")

                #print(global_rank, "starts/sizes", starts, sizes, flush=True)

                # Get the size of what's being sent to me
                timer.push("Dist Build: Compute Targets - Exchange Recv")
                rsizes, rstarts = exchange_send_info(comm, sizes)
                timer.pop("Dist Build: Compute Targets - Exchange Recv")

                #print(global_rank, "rstarts/rsizes", rstarts, rsizes, flush=True)

                timer.pop("Dist Build: Compute Targets")

                # AlltoAllv Global ID vector
                timer.push("Dist Build: Communicate IDs")

                timer.push("Dist Build: Communicate IDs - Allocate")
                recv_gids = np.zeros(len(self.global_ids), dtype=self.gprec)
                timer.pop("Dist Build: Communicate IDs - Allocate")
                
                timer.push("Dist Build: Communicate IDs - alltoall")
                # TODO: There is a better way to do this in mpi4py 3.10 which was just released, switch or keep for compatibility?
                if self.gprec == np.int32:
                    comm.Alltoallv([self.global_ids, sizes, starts, MPI.INT], [
                                   recv_gids, rsizes, rstarts, MPI.INT])
                else:
                    comm.Alltoallv([self.global_ids, sizes, starts, MPI.LONG], [
                                   recv_gids, rsizes, rstarts, MPI.LONG])
                timer.pop("Dist Build: Communicate IDs - alltoall")

                timer.pop("Dist Build: Communicate IDs")
                #print(global_rank, "recv_ids", recv_gids, flush=True)


                timer.push("Dist Build: Communicate Projections")

                # AlltoAllv Projection Vector
                timer.push("Dist Build: Communicate Projections - Allocate")
                recv_proj = np.zeros_like(proj)
                # Get remaining levels (this is the data stride)
                rl = recv_proj.shape[1]
                #print(global_rank, "rl", rl, flush=True)
                #print(global_rank, "send_proj", proj, proj.shape, proj.dtype, proj.flags, flush=True)
                timer.pop("Dist Build: Communicate Projections - Allocate")

                timer.push("Dist Build: Communicate Projections - alltoall")
                comm.Alltoallv([proj, sizes*rl, starts*rl, MPI.FLOAT], [
                    recv_proj, rsizes*rl, rstarts*rl, MPI.FLOAT])
                timer.pop("Dist Build: Communicate Projections - alltoall")

                print(global_rank, median, "recv_proj check", recv_proj[:, l], flush=True)
                timer.pop("Dist Build: Communicate Projections")

                # update global ids
                self.global_ids = recv_gids

                # Set color for comm split (between children)
                if(local_rank >= mpi_size//2):
                    color = 1
                else:
                    color = 0

                #Check recieved projection
                if color:
                    cond = np.min(recv_proj[:, l]) >= median
                    count = np.sum(recv_proj[:, l] < median)
                else:
                    cond = np.max(recv_proj[:, l]) <= median
                    count = np.sum(recv_proj[:, l] > median)
                
                lproj = proj[:local_split, l]
                rproj = proj[local_split:, l]

                lcond = (np.max(lproj) <= median)
                rcond = (np.min(rproj) >= median)

                lcount = np.sum(lproj > median)
                rcount = np.sum(rproj < median)

                print(global_rank, l, "Validate Send Proj Left", lcond, lcount, lproj, flush=True)
                print(global_rank, l, "Validate Send Proj Right", rcond, rcount, rproj, flush=True)
                print(global_rank, l, "Validate Correct Proj: ", cond, count, recv_proj[:, l].shape[0], flush=True)
                #assert(cond)

                proj = recv_proj

                # split communicator
                comm = comm.Split(color, local_rank)

                # update mpi_size
                mpi_size = comm.Get_size()

        self.built = True
        timer.pop("Dist Build:")

    def collect_data(self):
        timer = Primitives.Profiler()
        timer.push("Collect Points:")
        data, gids = collect(self.comm, self.global_ids, self.host_data,
                                 self.prefix_sizes, dtype=self.gprec)
        timer.pop("Collect Points:")
        self.host_data = data
        self.global_ids = gids

    def redistribute_results(self, results):
        return redistribute(self.comm, self.global_ids, results, self.prefix_sizes)

    def local_exact(self, Q, k, index='local'):
        query_size = Q.shape[0]

        # Compute exact results in local indexing
        # NOTE: type should be int32 for GSKNN and CUDA support
        assert(self.local_ids.dtype == self.lprec)
        result = Primitives.direct_knn(self.local_ids, self.host_data, Q, k, cores=self.cores)

        # Sort result
        # NOTE: GSKNN does not return sorted results
        result = Primitives.merge_neighbors(result, result, k)

        # If we want results to be in the local index, we're done
        if index == 'local':
            return result
        # Otherwise remap result_ids to global indexing
        elif index == 'global':
            result_ids = result[0]
            result_dist = result[1]

            result_ids = reorder(self.global_ids, result_ids)

            # Check datatype (Do I trust the numba dispatching?)
            assert(result_ids.dtype == self.gprec)

            result = (result_ids, result_dist)

            return result

    # Compute exact nearest neighbors over distributed tree
    def distributed_exact(self, Q, k):
        # TODO: Convert to real reduction, not serialized
        mpi_size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        query_size = Q.shape[0]

        # Allocate storage for local reduction on rank 0
        recvbuff_list = np.empty([mpi_size, query_size, k], dtype=self.gprec)
        recvbuff_dist = np.empty([mpi_size, query_size, k], dtype=np.float32)

        # Compute exact result in local indexing
        result = Primitives.direct_knn(self.local_ids, self.host_data, Q, k, cores=self.cores)
        result = Primitives.merge_neighbors(result, result, k)
        
        # Data type security (ran into problems here before, this is just a sanity check)
        result_ids = np.asarray(result[0], dtype=self.lprec)
        result_dist = np.asarray(result[1], dtype=np.float32)

        # Convert local to global indexing
        #TODO: Use numba
        result_ids = self.global_ids[result_ids]
        assert(result_ids.dtype == self.gprec)

        print(rank, "locally owned results", result_ids, self.global_ids, flush=True)

        # Gather all information back to rank 0
        self.comm.Gather(result_ids, recvbuff_list, root=0)
        self.comm.Gather(result_dist, recvbuff_dist, root=0)

        # Merge  mpi_size blocks sequentially on rank 0
        result = None
        if rank == 0:
            for i in range(0, mpi_size):
                neighbors = (recvbuff_list[i], recvbuff_dist[i])
                print("merge", i, neighbors, flush=True)
                if result:
                    result = Primitives.merge_neighbors(result, neighbors, k)
                else:
                    result = neighbors
        
        #Sort results back into original id ordering
        #p = np.argsort( 


        return result