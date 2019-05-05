from sklearn.cluster import KMeans, SpectralClustering
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def bounding_box(data):
    nr_dimensions = data.shape[1]
    min_vals = np.zeros(nr_dimensions)
    max_vals = np.zeros(nr_dimensions)
    for d in range(nr_dimensions):
        min_vals[d] = min(data[:,d])
        max_vals[d] = max(data[:,d])
    return min_vals, max_vals

def within_cluster_scatter(clusters, mu):
    K = len(mu)
    res = sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])
    return res

def get_clusters_from_labels(labels, data, nrClusters):
    return [c for c in [data[labels==i] for i in range(nrClusters)]]

def get_cluster_centers(clusters):
    return [np.mean(c, axis=0).tolist() for c in clusters]

def clustering_algorithm(data, K, algorithm='kmeans'):
    if algorithm == 'kmeans':
        km = KMeans(n_clusters=K, random_state=0).fit(data)
        return km.labels_
    
    if algorithm == 'spectral':
        sc = SpectralClustering(K,eigen_solver='arpack' ,affinity="nearest_neighbors", random_state=0).fit(data)
        return sc.labels_


def get_gap_statistic(data, use_spectral=False, cluster_algorithm='kmeans', K_max=10, verbose=False):
    
    B = 10 # number of reference 
    Ks = range(1,K_max+1) 
    wk_real_data = np.zeros(len(Ks))
    wk_reference_data = np.zeros(len(Ks))
    sdk = np.zeros(len(Ks))
    nrDataPoints = len(data)
    min_vals, max_vals = bounding_box(data)  

    for iK, K in enumerate(Ks):
        
        if verbose:
            print("Calculating for K = {} of {} ...".format(K,K_max))

        # We generate 10 reference data sets with K clusters and check Wk
        wk_bs = np.zeros(B);
        for i in range(B):
            
            dimension = data.shape[1]
            rand_vals = []
            
            for d in range(dimension):
                randv = np.random.uniform(min_vals[d], max_vals[d], nrDataPoints)
                rand_vals.append(randv)
                
            random_Xdat = np.transpose(np.array(rand_vals))

            km = KMeans(n_clusters=K, random_state=0).fit(random_Xdat)
            clusters_unif = get_clusters_from_labels(km.labels_, random_Xdat, K);
            wk_unif = within_cluster_scatter(clusters_unif, km.cluster_centers_)
            wk_bs[i] = np.log(wk_unif)

        wk_reference_data[iK] = np.mean(wk_bs)

        sdk[iK] = np.sqrt(np.mean((wk_bs - wk_reference_data[iK])**2))

        # Calculate Wk for real data
        km = KMeans(n_clusters=K, random_state=0).fit(data)
        clusters_real = get_clusters_from_labels(km.labels_, data, K);
        wk_real_log = np.log(within_cluster_scatter(clusters_real, km.cluster_centers_))

        wk_real_data[iK] = wk_real_log

    sk = sdk*np.sqrt(1+1/B)
    return wk_real_data, wk_reference_data, sk, Ks

def get_optimal_k(wk, wk_ref, sk, Ks):
    gaps = np.zeros(len(Ks))
    for iK, K in enumerate(Ks):
        gaps[iK]  = wk_ref[iK] - wk[iK]
        
    return np.argmax(gaps) + 1
        
def plot_wk(Ks, wk, wk_ref):
    f, (ax1,ax2) = plt.subplots(1, 2)
    f.set_figwidth(15)

    ax1.plot(Ks, wk, label=r"$log W_k$")
    ax1.plot(Ks, wk_ref, label=r"$\frac{1}{B}\sum^{B}_{b}\log W_{kb}$")
    ax1.legend(loc='upper right')
    ax1.set_xlabel("K")


    ax2.plot(Ks, np.exp(wk))
    ax2.set_xlabel("K")
    ax2.set_ylabel(r"$W_k$")

    plt.show()
    