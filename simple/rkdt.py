import numpy as np
import utils as ut
import matplotlib.pyplot as plt

def leaf_knn(X,gids,m,knnidx,knndis,k,init,overlap=0):
    '''
    X - point coordinates in original ordering
    p - permutation array 
    m - points per leaf
    knnidx - array with current k-neighbors, updated upon exit
    knndis - k-nearest distances, updated upon exit
    k - number of nearest neighbors
    init - whether this is the first iteration or not 
           (to initialize neighbors)
    overlap - whether you want to do a spill-tree search 


    output
    knnidx - array with current k-neighbors, updated upon exit
    knndis - k-nearest distances, updated upon exit
    '''
    n = len(gids)
    offsets = np.arange(0,n,m)
    for i in range(len(offsets)):
        st = offsets[i]
        en = min(st+m, n)
        ls =  gids[st:en]    # leaf set
        ov=overlap
        lss = gids[max(st-ov,0):min(n,en+ov)]
        D = ut.l2sparse(X[ls,:],X[lss,:])
        T=np.tile(lss,(en-st,1))
        S = np.argsort(D,axis=1)
        T=np.take_along_axis(T,S,axis=1)
        D=np.take_along_axis(D,S,axis=1)
        
        if init:
            knnidx[ls,:]=T[:,:k]
            knndis[ls,:]=D[:,:k]
            continue;
 
        kit=np.block([knnidx[ls,:],T[:,:k]])
        kdt=np.block([knndis[ls,:],D[:,:k]])                
        ut.merge_knn(kdt,kit,k)
        knndis[ls,:]=kdt[:,:k]
        knnidx[ls,:]=kit[:,:k]


def vis(X,gids,level, colors, knnidx, point=None,fnm=None):
    n = X.shape[0]
    segsize = n>>level
    offsets = np.arange(0,n,segsize)
    fix, ax = plt.subplots()
    for i in range(len(offsets)):
        st = offsets[i]
        en = min(st+segsize,n)
        scale = 100

        ci = 0.2989 * colors(i)[0] + 0.5870 * colors(i)[1] + 0.1140 * colors(i)[2]
        print(f'Color {colors(i)} and ci={ci}')
        cisc = np.array([colors(i)])
        #cisc = np.array([ci])
        ax.scatter(X[gids[st:en],0],X[gids[st:en],1],c=cisc,
                   s=scale,alpha=0.3,edgecolors='none')

    if point is not None:
        ax.scatter(X[knnidx[point,1:],0],X[knnidx[point,1:],1],
                   marker="x",c='black',alpha=0.2)
        scale = 200        
        ax.scatter(X[point,0],X[point,1],marker="s",c='black',alpha=0.9)

        
    ax.grid(False)
    ax.axis('equal')
    plt.axis('off')
    plt.show
    if fnm is not None:
        plt.savefig(fnm, bbox_inches='tight')
    
    
        
def rkdt_a2a_it(X,gids,levels,knnidx,knndis,K,maxit,monitor=None,
                overlap=0,visualize=False):
    n = X.shape[0]
    perm = np.empty_like(gids)
    for t in range(maxit):
        gids[:] = np.arange(0,n).astype(type(gids[0]))
        perm[:] = np.arange(0,n).astype(type(gids[0]))
        P,_ = ut.orthoproj(X,levels)
        for i in range(0,levels):
            segsize = n>>i
            ut.segpermute(P[:,i],segsize,perm)
            P[:,:]=P[perm,:]
            gids[:]=gids[perm]

    
        leaf_knn(X,gids,segsize,knnidx,knndis,K,t==0,overlap)
        if monitor is not None:
            if monitor(t,knnidx,knndis):
                break

    if visualize is True:
        cset = ['hot','Accent','tab10','tab20']
        for i in range(1,levels):
            colors = plt.cm.get_cmap(cset[i-1],1<<i)
            finm=None
            finm="Ctree%d.pdf"%i
            vis(X,gids,i,colors,knnidx,fnm=finm,point=None)

        pnt = 384
        finm="Cpoint.pdf"
        vis(X,gids,i,colors,knnidx,fnm=finm,point=pnt)

        

            
            
    









    
    
    
