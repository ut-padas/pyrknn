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

def project(data, samples=5):
    np.random.seed(10)
    if samples > data.shape[0]:
        samples = data.shape[0]
    samples = np.random.choice(data.shape[0], samples, replace=False)
    dic = 1 - (1-cdist(data[samples, :], data, metric='hamming').T)**2
    #print(data[samples, :])
    #print("DIC", dic)
    temp = np.sum(dic, axis=1)
    lip = np.argmax(temp)
    dip = 1 - (1-cdist(data[lip:lip+1, :], data, metric='hamming').T)**2
    liq = np.argmax(dip)
    diq = 1 - (1-cdist(data[liq:liq+1, :], data, metric='hamming').T)**2
    return dip - diq

def mode_tree(data, levels):
    N = data.shape[0]
    km = KModes(n_clusters=2, init='Huang')
    ids = np.arange(N)
    labels = np.zeros(N)
    for level in range(levels):
        Nper = N//(2**level)
        for leaf in range(2**level):
            start_idx = leaf*Nper
            end_idx = (leaf+1)*Nper
            local_data = data[start_idx:end_idx]
            local_clusters = km.fit_predict(local_data)
            sort_ids = np.argsort(local_clusters)

            local_ids = ids[start_idx:end_idx]
            ids[start_idx:end_idx] = local_ids[sort_ids]

            data[start_idx:end_idx] = local_data[sort_ids]
            labels[start_idx:end_idx] = local_clusters[sort_ids]+2*leaf
            #print(level, leaf, labels[start_idx:end_idx])
    #print(labels)
    return labels, data, ids 

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
            #print(lids, lids.ravel())
            local_ids = ids[start_idx:end_idx]
            ids[start_idx:end_idx] = local_ids[lids]

            data[start_idx:end_idx] = local_data[lids]
            labels[start_idx:start_idx+mid] = 0+2*leaf
            labels[start_idx+mid:end_idx] = 1+2*leaf
            #print(level, leaf, labels[start_idx:end_idx], data[start_idx:end_idx])
    #print(labels)
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

results = dict()
levels = 6
kk = 2**levels
k_list = [16, 64, 128, 256, 512, 1024]
for X in [8, 64]:
    for H in [20]:
        #Load data and hash
        N = 2**15
        ls = N//X
        #H = 10
        max_tree = 100
        d = 1
        k = 1024
        q_id = 1000
        global cores 
        cores = 56
        trials = 1
        if X == kk:
            split_types = ["gofmm", "default"]
        else:
            split_types = ["gofmm"]
        """
        if X != kk:
            split_types = ["truncated_kmodes"]
        else:
            split_types = ["truncated_kmodes", "random", "default"]
        """
        max_it = min(max_tree//H, 10)

        print("MAX", max_it)
        dataset = "gauss15d"
        dname = dataset+"_"+str(N//X)+"_data.bin.npy"
        fname = dataset+"_"+str(N//X)+"_hash.bin.npy"
        
        output_storage = defaultdict(list)

        hash = np.load(fname)
        data = np.load(dname)
        N = min(N, data.shape[0])

        gids = np.arange(N, dtype=np.int32)
        q = data[q_id:q_id+1, :]
        print(q)
        print(q.shape)
        #Compute True Nearest Neighbors
        truth = single_knn(gids, data, q, k)
        truth = merge_neighbors(truth, truth, k)


        if "default" in split_types:
            #Compute normal tree split
            print("Default")
            print("-----------")
            mid = N//2
            result = None
            for l in range(max_it):
                #Form clusters

                q_cluster = hash[q_id, l]
                clust = True
                gclust = True
                #print("Cluster: ", q_cluster)
                if clust:
                    slids = np.argsort(hash[:, l], kind='stable')
                    shash = hash[slids, l]
                    lids = np.where(shash==q_cluster)
                    cluster = data[slids][lids]
                    lgids = gids[slids][lids]
                    thash = shash[lids]
                    #print("std", np.std(thash), np.max(thash), np.min(thash))
                elif gclust:
                    lids = np.where(hash[:, l]==q_cluster)
                    cluster = data[lids]
                    lgids = gids[lids]
                    thash = hash[lids, l]
                    #print("std", np.std(thash), np.max(thash), np.min(thash))
                else:  
                    lids = np.argsort(hash[:, l], kind='stable')
                    lgids = gids[lids]
                    clustered = data[lids]
                    thash = hash[lids, l]

                    if(q_cluster == 1):
                        cluster = clustered[:mid]
                        thash = thash[:mid]
                        lgids = lgids[:mid]
                    else:
                        cluster = clustered[mid:]
                        lgids = lgids[mid:]
                        thash = thash[mid:]
                #print("std", np.std(thash), np.max(thash), np.min(thash), cluster.shape[0])
                #print(thash)
                #print(np.sum(thash==q_cluster))
                
                #Search Clusters
                approx_tree = single_knn(gids, cluster, q, k)
                #print(lgids)
                #print(lgids.max())
                #Convert to local ids
                a, b = approx_tree
                a = lgids[a]
                approx_tree = (a, b)
                
                if result is None:
                    result = merge_neighbors(approx_tree, approx_tree, k)
                else:
                    result = merge_neighbors(result, approx_tree, k)
                #print(result)
                #Check Accuracy
                accuracy_list = accuracy_metric(truth, result, k_list, id_only=True)
                print(accuracy_list)
                output_storage["default"].append(accuracy_list)

        if "random" in split_types:
            #Compute random split
            print("Random")
            print("-----------")
            mid = N//X
            result = None
            for l in range(max_it):
                #Form clusters
                p = np.random.permutation(N)
                lgids = gids[p]
                clustered = data[p]
                q_cluster = p[p][q_id] < mid 

                cluster = clustered[:mid]
                lgids = lgids[:mid]

                #Search Clusters
                approx_tree = single_knn(gids, cluster, q, k)

                #Convert to local ids
                a, b = approx_tree
                a = lgids[a]
                approx_tree = (a, b)
                
                if result is None:
                    result = merge_neighbors(approx_tree, approx_tree, k)
                else:
                    result = merge_neighbors(result, approx_tree, k)

                #Check Accuracy
                accuracy_list = accuracy_metric(truth, result, k_list, id_only=True)
                output_storage["random"].append(accuracy_list)
                print(accuracy_list)

        if "truncated_kmodes" in split_types:
            output_storage["truncated_kmodes"] = [0]*max_it
            output_storage["truncated_kmodes_size"] = [0]*max_it
            #Compute kmode split
            print("Trun_KMODES")
            print("-----------")
            mid = N//2
            result = None
            for t in range(trials):
                result = None
                print("Trial:", t)
                for l in range(max_it):
                    treehash = hash[:, l*H:(l+1)*H]

                    km = KModes(n_clusters=2*kk, init='Huang')
                    tdata = np.array(treehash, dtype=np.int64)# np.random.choice(2, (N, 1))
                    clusters = km.fit_predict(tdata)
                    print(clusters[q_id])
                    cluster, lgids = pick_bin(data, clusters, q_id, kk+1)
                    approx_tree = single_knn(gids, cluster, q, k)

                    #Convert to local ids
                    a, b = approx_tree
                    a = lgids[a]
                    approx_tree = (a, b)
                    
                    if result is None:
                        result = merge_neighbors(approx_tree, approx_tree, k)
                    else:
                        result = merge_neighbors(result, approx_tree, k)

                    #Check Accuracy
                    accuracy_list = accuracy_metric(truth, result, k_list, id_only=True)
                    output_storage["truncated_kmodes"][l] += np.asarray(accuracy_list) / trials
                    output_storage["truncated_kmodes_size"][l] += cluster.shape[0] / trials  
                    print(accuracy_list, cluster.shape[0])


        if "kmodes" in split_types:
            output_storage["kmodes"] = [0]*max_it
            output_storage["kmodes_size"] = [0]*max_it
            #Compute kmode split
            print("KMODES", kk, X, H)
            print("-----------")
            result = None
            for t in range(trials):
                result = None
                print("Trial:", t)
                for l in range(max_it):
                    treehash = hash[:, l*H:(l+1)*H]

                    km = KModes(n_clusters=kk, init='Huang')
                    tdata = np.array(treehash, dtype=np.int64)# np.random.choice(2, (N, 1))
                    clusters = km.fit_predict(tdata)
                    q_c_id = clusters[q_id]
                    loc = np.where(clusters==q_c_id)

                    #print(np.sum(treehash[loc]==(1-q_c_id)), np.sum(treehash[loc]==treehash[q_id]), print(treehash[loc].shape[0]))

                    cluster = data[loc]
                    lgids = gids[loc]
                    #print(loc)
                    #print(lgids)
                    #Search Clusters
                    approx_tree = single_knn(gids, cluster, q, k)

                    #Convert to local ids
                    a, b = approx_tree
                    a = lgids[a]
                    approx_tree = (a, b)
                    
                    if result is None:
                        result = merge_neighbors(approx_tree, approx_tree, k)
                    else:
                        result = merge_neighbors(result, approx_tree, k)

                    #Check Accuracy
                    accuracy_list = accuracy_metric(truth, result, k_list, id_only=True)
                    output_storage["kmodes"][l] += np.asarray(accuracy_list) / trials
                    output_storage["kmodes_size"][l] += cluster.shape[0] / trials  
                    print(accuracy_list, cluster.shape[0])

        if "gofmm" in split_types:
            #Compute kmode split
            output_storage["gofmm"] = [0]*max_it
            print("GOFMM", X, H)
            print("-----------")
            mid = N//2
            result = None
            for t in range(trials):
                result = None
                for l in range(max_it):
                    treehash = hash[:, l*H:(l+1)*H]
                    temp = np.copy(treehash)
                    clusters, dd, lids = mode_tree(temp, 6)
                    p = np.arange(N)
                    dd = np.asarray(dd, dtype=np.float32)
                    p[lids] = np.arange(N)
                    q_c_id = clusters[p[q_id]]

                    #print(p[q_id], q_id)
                    #print("datapoint", dd[p[q_id]], treehash[q_id])
                    #print("id", q_id, lids[p[q_id]])
                    #print("cluster", q_c_id)
                    #print("clusters", clusters)


                    select = np.where(clusters==q_c_id)[0]
                    #print(select)
                    #print("cluster labels", clusters[select])
                    #print("Contains?", np.where(select==p[q_id]))
                    temp_id = np.where(select==p[q_id])[0]
                    #print("Lids id", )
                    cluster = data[lids[select]]
                    #print(cluster[temp_id])
                    lgids = lids[select]
                    #print("Lids id", lgids[temp_id])
                    #Search Clusters
                    approx_tree = single_knn(gids, cluster, q, k)
                    #print(lgids)
                    #print(approx_tree)
                    ee = lgids
                    #Convert to local ids
                    a, b = approx_tree
                    a = ee[a]
                    approx_tree = (a, b)
                    
                    if result is None:
                        result = merge_neighbors(approx_tree, approx_tree, k)
                    else:
                        result = merge_neighbors(result, approx_tree, k)
                    #print(result[0])
                    #Check Accuracy
                    accuracy_list = accuracy_metric(truth, result, k_list, id_only=True)
                    output_storage["gofmm"][l] += np.asarray(accuracy_list) / trials
                    print(accuracy_list, cluster.shape[0])


        results[str(H)+"_"+str(ls)] = output_storage

outname = str(kk)+"_"+dataset+"_convergence_modes.pkl"
with open(outname,"wb") as f:
    pickle.dump(results,f)

print(results)