from pyrknn.kdforest.mpi.tree import *
from pyrknn.kdforest.mpi.util import *
from pyrknn.kdforest.mpi.forest import *

from scipy.sparse import random

from mpi4py import MPI
import numpy as np

import time
import platform

import os

from sklearn.datasets import load_svmlight_file
from joblib import Memory

from scipy.sparse import csr_matrix
from scipy.sparse import random

import argparse

parser = argparse.ArgumentParser(description="Test Sparse KNN")
parser.add_argument('-dim', type=int, default=10000)
parser.add_argument('-iter', type=int, default=10)
parser.add_argument('-dataset', default="url")
parser.add_argument('-bs', type=int, default=64)
parser.add_argument('-bl', type=int, default=128)
parser.add_argument('-cores', type=int, default=56)
parser.add_argument('-use_gpu', type=bool, default=0)
parser.add_argument('-levels', type=int, default=13)
parser.add_argument('-k', type=int, default=32)
parser.add_argument('-leafsize', type=int, default=512)
parser.add_argument('-ltrees', type=int, default=3)
parser.add_argument('-merge', type=int, default=1)
parser.add_argument('-overlap', type=int, default=1)
parser.add_argument('-seed', type=int, default=15)
parser.add_argument('-nq', type=int, default=10)
parser.add_argument('-nnz', type=int, default=10)
args = parser.parse_args()

mem = Memory("./mycache")

@mem.cache()
def get_url_data():
    t = time.time()
    data = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/url/url_combined", n_features=3231961)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data[0]

@mem.cache()
def get_avazu_data():
    t = time.time()
    data_app = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/avazu/avazu-app", n_features=1000000)    
    print(data_app[0].shape)
    data_site = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/avazu/avazu-site", n_features=1000000)
    print(data_site[0].shape)
    data = sp.vstack([data_app[0], data_site[0]])
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data

@mem.cache()
def get_kdd12_data():
    t = time.time()
    data = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/kdd/kdd12", n_features=54686452)
    print(data[0].shape)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data[0]

def unique_byrow(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind = np.unique(b, return_index=True)
    b = np.zeros_like(a)
    np.put(b, ind, a.flat[ind])
    return b

def gen_random_sparse_csr(N,M,avg_nnz,idtype=np.int32,vltype=np.float32):
    '''
    Generate a real random sparse matrix of N-by-M dimensions with avz_nnz
    nonzeros per row:  total memory ~ N*avz_nnz
    The values are normally distributed
    The nonzeros are uniformly distributed
    
    Does not allow empty rows, (at least )
    
    
    Parameters
    ----------
    N : int
        number of rows
    M : int
        number of columns (full matrix)
    avg_nnz : int
        average number of nonzeros _per_row_
    idtype : type, optional
        type for indeces The default is np.int32.
    vltype : type, optional
        type for matrix values. The default is np.float32.
    Returns
    -------
    X : sparse N-by-M  matrix
        CSR format.
        
    Example:
    N=10
    avg_nnz_per_row = 4
    M = 9
    idtype = np.int32
    vltype = np.float32
    X = gen_random_sparse_csr(N,M,avg_nnz_per_row)
    print(X.toarray())
    '''

    # generate random connectivity matrix
    con = np.random.randint(1,M+1,size=(N,2*avg_nnz), dtype=idtype)
    con = unique_byrow(con)  # detect duplicates
    con = np.sort(con,axis=1)  # sort indices
    flt = np.random.randint(2,size=(N,2*avg_nnz))
    flt = np.sort(flt,axis=1)
    con = con*flt
    cols = con.ravel()  # get column
    cols = cols[cols>0] 
    nnz_arr = np.sum(con>0,axis=1) # nonzeros per rows

    nnz = np.sum(nnz_arr)        
    rows = np.block( [0, np.cumsum(nnz_arr)])
    cols -=1
    vals = np.random.randn(nnz).astype(vltype)
    X =csr_matrix((vals, cols, rows), shape=(N,M) )
    return X



if args.dataset == "url":
    get_data = get_url_data 
elif args.dataset == "avazu":
    get_data = get_avazu_data
elif args.dataset == "kdd":
    get_data = get_kdd12_data

if args.use_gpu:
    location = "GPU"
else:
    location = "HOST"

def run():
    sparse = True
    k = args.k

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print(rank, ", ", os.getpid(), flush=True)
    time.sleep(5)

    #Read data
    if rank == 0:
        print("Starting to Read Data", flush=True)

    N = 2**22
    d = args.dim
    avg_nnz = args.nnz
    
    nq = 200
    np.random.seed(10)
    #X = random(N, d, density=avg_nnz//d)
    X = gen_random_sparse_csr(N,d,avg_nnz,idtype=np.int32,vltype=np.float32)
    X = X.tocsr()
    Q = X[:nq]

    #if rank>0:
    #    np.random.seed(None)
    #    X = random(N, d, density=avg_nnz/d)
    #    #X = gen_random_sparse_csr(N,d,avg_nnz,idtype=np.int32,vltype=np.float32)
    #    X = X.tocsr()

    t = 0
    if rank == 0:
        print("Finished Reading Data: ", X.shape, flush=True)
        print("Reading data took: ", t," seconds", flush=True)

    N, d = X.shape

    #Convert Q to the correct datatype
    q_data = np.asarray(Q.data, dtype=np.float32)
    q_indices = np.asarray(Q.indices, dtype=np.int32)
    q_indptr = np.asarray(Q.indptr, dtype=np.int32)
    Q = sp.csr_matrix( (q_data, q_indices, q_indptr), shape=(nq, d))

    #Convert X to the correct datatype
    X_data = np.asarray(X.data, dtype=np.float32)
    X_indices = np.asarray(X.indices, dtype=np.int32)
    X_indptr = np.asarray(X.indptr, dtype=np.int32)
    X = sp.csr_matrix( (X_data, X_indices, X_indptr), shape=X.shape)


    if rank == 0:
        print("Finished Preprocessing Data", flush=True)

    t = time.time()

    timer = Profiler()
    record = Recorder() 

    #Compute true solution with brute force on nq subset
    #C = X.copy()
    tree = RKDT(data=X, levels=0, leafsize=2048, location="HOST")
    truth = tree.distributed_exact(Q, k)
    #truth = (np.zeros([nq, k],  dtype=np.int32), np.zeros([nq, k], dtype=np.float32))
    t = time.time() - t

    if rank == 0:
        print("Exact Search took: ", t, " (s)", flush=True)

    #print("Truth:", truth)

    #timer.reset()
    #record.reset()

    #np.random.seed(100)
    np.random.seed(args.seed)
    forest = RKDForest(data=X, levels=args.levels, leafsize=args.leafsize, location=location)

    if args.overlap:
        approx = forest.overlap_search(k, ntrees=args.iter, ltrees = args.ltrees, truth=truth, cores=args.cores, blocksize=args.bs, blockleaf=args.bl, merge_flag=args.merge)
    else:
        approx = forest.all_search(k, ntrees=args.iter, ltrees = args.ltrees, truth=truth, cores=args.cores, blocksize=args.bs, blockleaf=args.bl, merge_flag=args.merge)

    if rank == 0:
        timer.print()
        print("=======")
        record.print()
        
run()








