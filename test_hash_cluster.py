from pyrknn.kdforest.mpi.tree import *
from pyrknn.kdforest.mpi.util import *
from pyrknn.kdforest.mpi.forest import *

from mpi4py import MPI

import numpy as np

import time
import platform
import pickle 

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize
from mlxtend.data import loadlocal_mnist

from kmodes.kmodes import KModes 
from scipy.spatial.distance import cdist

from collections import defaultdict

def project(data, samples=1):
    #GOFMM median split
    #np.random.seed(10)
    if samples > data.shape[0]:
        samples = data.shape[0]
    samples = np.random.choice(data.shape[0], samples, replace=False)
    dic = 1 - (1-cdist(data[samples, :], data, metric='hamming').T)**2
    temp = np.sum(dic, axis=1)
    lip = np.argmax(temp)
    dip = 1 - (1-cdist(data[lip:lip+1, :], data, metric='hamming').T)**2
    liq = np.argmax(dip)
    diq = 1 - (1-cdist(data[liq:liq+1, :], data, metric='hamming').T)**2
    return dip - diq

def mode_split(ids, data, branch):
    N = data.shape[0]
    km = KModes(n_clusters=branch, init='Huang')
    cluster_labels = km.fit_predict(data[ids])
    nclusters = len(np.unique(cluster_labels))
    children_ids = []
    #print(cluster_labels, np.sum(cluster_labels))
    for i in range(nclusters):
        loc = np.where(cluster_labels==i)
        children_ids.append(ids[loc])
    return children_ids

def mode_tree_2(data, levels, branch):
    ids = np.arange(data.shape[0])
    node_list = [ids]
    for level in range(levels):
        #print(node_list)
        new_node_list = []
        for node in node_list:
            children = mode_split(node, data, branch)
            for child in children:
                new_node_list.append(child)
        node_list = new_node_list

    label = np.zeros(data.shape[0])
    size = 0
    for node, i in zip(node_list, range(len(node_list))):
        label[node] = i
        size += len(node)
    #print(size)
    return label
        
def gofmm_tree(data, levels):       
    N = data.shape[0]
    labels = np.zeros(N)
    ids = np.arange(N)
    for level in range(levels):
        Nper = N//(2**level)
        mid = Nper//2
        for leaf in range(2**level):
            start_idx = leaf*Nper
            end_idx = (leaf+1)*Nper
            local_data = data[start_idx:end_idx]

            local_project = project(local_data)
            
            lids = np.argsort(local_project.ravel())
            local_ids = ids[start_idx:end_idx]
            ids[start_idx:end_idx] = local_ids[lids]

            data[start_idx:end_idx] = local_data[lids]
            labels[start_idx:start_idx+mid] = 0+2*leaf
            labels[start_idx+mid:end_idx] = 1+2*leaf
    return labels, data, ids


def pick_bin(data, labels, q_id, partitions):
    N = data.shape[0]
    p = np.argsort(labels,kind='stable')
    rlabels = labels[p]
    lids = np.arange(N)
    lids[p] = np.arange(N)

    new_id = lids[q_id]
    reordered_data = data[p]
    break_list = np.linspace(0, N, partitions, dtype=np.int32)
    loc = np.searchsorted(break_list, new_id)
    print(rlabels[break_list[loc-1]:break_list[loc]], np.unique(rlabels[break_list[loc-1]:break_list[loc]]).shape[0])
    return reordered_data[break_list[loc-1]:break_list[loc]], p[break_list[loc-1]:break_list[loc]]


def bin_clusters(labels, partitions):
    #Break the clusters into even bins after sorting by them
    N = labels.shape[0]
    p = np.argsort(labels, kind='stable')

    nlabels = np.copy(labels)
    break_list = np.linspace(0, N, partitions+1,dtype=np.int32)
    for i in range(partitions):
        start = break_list[i]
        #print(start)
        end = break_list[i+1]
        #print(end)
        ids_in_part = p[start:end]
        nlabels[ids_in_part] = i

        #print(len(np.unique(labels[ids_in_part])))
        #print(labels[ids_in_part])

    #print(nlabels)
    return nlabels

#Check bin variance
def get_cluster_variation(labels, data):
    #Get median distance within cluster
    nclusters = len(np.unique(labels))
    var_list = []
    for i in range(nclusters):
        loc = np.where(labels==i)
        data_in_cluster = data[loc]
        distance = cdist(data_in_cluster, data_in_cluster)
        stat = np.nanmedian(distance)
        var_list.append(stat)
    return var_list

#Check cluster size
def get_cluster_size(labels):
    #Get size list of clusters
    nclusters = len(np.unique(labels))
    size_list = []
    for i in range(nclusters):
        loc = np.where(labels==i)
        size_list.append(labels[loc].shape[0])
    return size_list 


def get_ratio(size_list):
    #Get size ratio of min and max cluster
    total = np.sum(size_list)
    clusters = len(size_list)
    print(np.min(size_list), np.max(size_list), total/clusters)
    return np.min(size_list)/np.max(size_list), 
#Balance clusters

def balance_clusters(labels, partitions):
    #Attempt to balance roughly by adding large to small
    size_list = np.asarray(get_cluster_size(labels))
    p = np.argsort(size_list, kind='stable')
    size_list = size_list[p]

    nclusters = len(np.unique(labels))

    new_clusters = np.zeros(partitions)
    new_labels = np.copy(labels)
    for i in range(nclusters):
        idx = np.argmin(new_clusters)
        new_clusters[idx] += size_list[i]
        loc = np.where(labels==i)
        new_labels[loc] = idx

    return new_labels

    
def balance_clusters_old(labels, nclusters):
    #This is incorrect
    #set target size
    thres = labels.shape[0]//nclusters

    #Count the size of each cluster
    size_list = []
    initial_clusters = len(np.unique(labels))

    for i in range(initial_clusters):
        locs = np.where(labels==i)
        size_list.append(labels[locs].shape[0])

    print("Before", size_list)
    print("Size", len(size_list), np.sum(size_list))

    new_c_size = 0
    new_c = []
    cluster_list = []
    for size, c in zip(size_list, range(len(size_list))):
        if new_c_size + size < thres:
            new_c_size += size
            new_c.append(c)
        else:
            cluster_list.append(new_c)
            new_c_size = 0
            new_c = []

    #print("Cluster List", cluster_list)
    #for cl, i in zip(cluster_list, range(len(cluster_list))):
    #    for c in cl:
    #        labels[labels==c] = i 
    
    size_list = []

    final_clusters = len(np.unique(labels))
    for i in range(final_clusters):
        locs = np.where(labels==i)
        size_list.append(labels[locs].shape[0])

    #print("After", size_list)
    #print("Size", len(size_list), np.sum(size_list))

    return labels

def search_cluster(q=None, q_id=None, data=None, X=None, labels=None, itr=1):
    labels = labels.ravel()
    q_cluster = labels[q_id]
    lids = np.where(labels==q_cluster)

    cluster = data[lids]

    lgids = gids[lids]

    approx = single_knn(gids, cluster, q, k)
    a, b = approx 
    a = lgids[a]
    approx = (a, b)

    return approx, cluster.shape[0]

def compute_truth(data, q, k):
    truth = single_knn(gids, data, q, k)
    truth = merge_neighbors(truth, truth, k)

def shuffle_hash(H):
    return np.random.shuffle(H)

def default_label(data=None, X=None, hash=None, H=None, levels=None, display=None, clusters=None, itr=1):
    return hash[:, itr]-1

def gofmm_label(q=None, q_id=None, data=None, X=1, hash=None, H=1, display=True, levels=6, clusters=None, itr=1):
    #Attempt to use GOFMM split
    treehash = np.copy(hash[:, itr*H:(itr+1)*H])
    clusters, dd, lids = gofmm_tree(treehash, levels)
    p = np.arange(N)
    p[lids] = np.arange(N)
    labels = clusters[p]
    return labels

def mode_tree_label(q=None, q_id=None, data=None, X=1, hash=None, H=1, display=True, levels=6, clusters=None, itr=1):
    #Hierarchical K-Modes clustering 
    branch = 2
    treehash = np.copy(hash[:, itr*H:(itr+1)*H])
    labels = mode_tree_2(treehash, levels, branch)
    return labels 

def random_label(q=None, q_id=None, data=None, X=1, hash=None, H=1, display=True, levels=3, clusters=None, itr=1):
    #Random partitioning baseline
    cluster_size = data.shape[0]//clusters
    labels = np.zeros(data.shape[0])
    for i in range(clusters):
        labels[(i+1)*cluster_size:] += 1

    np.random.shuffle(labels)
    return labels

def mode_label(q=None, q_id=None, data=None, X=1, hash=None, H=1, display=True, levels=3, clusters=None, itr=1):
    #standard 1 level k modes
    treehash = hash[:, itr*H:(itr+1)*H]
    km = KModes(n_clusters=clusters, init='Huang')
    labels = km.fit_predict(treehash)
    return labels

def mode_label_small(q=None, q_id=None, data=None, X=1, hash=None, H=1, display=True, levels=3, clusters=None, itr=1):
    #The unweighted kmodes on the number of unique keys
    treehash = hash[:, itr*H:(itr+1)*H]
    l = np.unique(treehash, axis=0)
    print("Number of unique ids:", len(l), treehash.shape[0])
    km = KModes(n_clusters=clusters*3, init='Huang')
    labels = km.fit_predict(l)

    relabel = dict()
    for t, i in zip(l, labels):
        relabel[t.tobytes()] = i
    out_labels = np.zeros(treehash.shape[0])
    for i in range(treehash.shape[0]):
        out_labels[i] = relabel[treehash[i].tobytes()]
    return out_labels

def hash_label(q=None, q_id=None, data=None, X=1, hash=None, H=1, display=True, levels=3, clusters=None, itr=1):
    #Single bin hash
    a = np.random.randn(data.shape[1], 1)
    proj = data @ a
    labels = np.mod(np.floor(proj), clusters)
    print(labels)
    return labels.T

def many_queries(function=None, Q=None, qids=None, data=None, X=8, hash=None, H=1, levels=5, clusters=32, max_it=10, display=True):
    accuracy_list = [None]*max_it
    labels = []
    search_list = [0]*max_it
    for i in range(max_it):
        print("Building label set... ", i)
        label = function(data=data, X=X, hash=hash, H=H, levels=levels, clusters=clusters, itr=i)

        size_list = get_cluster_size(label)
        var_list = get_cluster_variation(label, data)
        #print("Before- Size: ", get_ratio(size_list), np.max(size_list), np.min(size_list))
        #print(size_list)
        #print("After- Intranode Distance: ", np.nanmean(var_list))

        #label = balance_clusters(label, clusters)
        #label = bin_clusters(label, clusters)
        size_list = get_cluster_size(label)
        var_list = get_cluster_variation(label, data)

        #print("After- Size: ", get_ratio(size_list), np.max(size_list), np.min(size_list))
        #print(size_list)
        #print("After- Intranode Distance: ", np.nanmean(var_list))

        labels.append(label)
    #print("Q", Q)
    for q, q_id in zip(Q, qids):
        q = q.reshape(1, d)
        #print("Searching q ... ", q_id)
        truth = single_knn(gids, data, q, k)
        truth = merge_neighbors(truth, truth, k)

        result = None
        for i in range(max_it):
            approx, size = search_cluster(q=q, q_id=q_id, data=data, labels=labels[i])
            search_list[i] += size / Q.shape[0]
            if result is None:
                result = merge_neighbors(approx, approx, k)
            else:
                result = merge_neighbors(result, approx, k)
            accuracy_probe = accuracy_metric(truth, result, k_list, id_only=True)
            if display:
                print(accuracy_probe)
            accuracy_probe = np.asarray(accuracy_probe) / Q.shape[0]

            if accuracy_list[i] is None:
                accuracy_list[i] = accuracy_probe
            else:
                accuracy_list[i] += accuracy_probe
    
    print("Average Search Size", np.sum(search_list))
    return accuracy_list


def display(title, accuracy_list):
    print(title)
    for a in accuracy_list:
        print(a)

N = 2**15
X = 64
ls = N//X
d = 15
k = 1024
dataset = "gauss15d"

drefname = dataset+"_"+str(ls)+"_hash.bin.npy"
dname = dataset+"_"+str(N//2)+"_data.bin.npy"
fname = dataset+"_"+str(N//2)+"_hash.bin.npy"

k_list = [16, 64, 128, 256, 512, 1024]
hash = np.load(fname)
data = np.load(dname)
hash_ref = np.load(drefname)

nq = 500
N = min(N, data.shape[0])
nq = min(nq, N)
hash = hash[:N]
hash_ref = hash_ref[:N]

gids = np.arange(N, dtype=np.int32)
data = data
Q = data[:nq]
qids = np.arange(nq)

max_it = 3

#output = many_queries(function=default_label, Q=Q, qids=qids, data=data, hash=hash_ref, X=X, H=10, levels=6, clusters=64, max_it=max_it, display=False)
#display("Default", output)

#output = many_queries(function=hash_label, Q=Q, qids=qids, data=data, hash=hash, X=X, H=20, levels=6, clusters=3, max_it=max_it, display=False)
#display("Hash", output)

output = many_queries(function=mode_label, Q=Q, qids=qids, data=data, hash=hash, X=X, H=20, levels=6, clusters=64, max_it=max_it, display=False)
display("KModes", output)

#output = many_queries(function=mode_tree_label, Q=Q, qids=qids, data=data, hash=hash, X=X, H=20, levels=6, clusters=64, max_it=max_it, display=False)
#display("KModes", output)

#output = many_queries(function=random_label, Q=Q, qids=qids, data=data, hash=hash, X=X, H=20, levels=6, clusters=64, max_it=max_it, display=False)
#display("Random", output)

#output = many_queries(function=gofmm_label, Q=Q, qids=qids, data=data, hash=hash_ref, X=X, H=5, levels=6, clusters=64, max_it=max_it, display=False)
#display("GOFMM", output)
