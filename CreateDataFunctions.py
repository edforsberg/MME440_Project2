import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import math
from sklearn.utils import shuffle

def CreateCircleCluster(nrDataPoints,radius,center,noise,classNr,nrFeatures,grid):
    firstCircleFeature = random.randint(0,(nrFeatures-1))
    secondCircleFeature = (firstCircleFeature + 1) % nrFeatures
    Xdata = []
    Ydata = []

    for i in range(nrDataPoints):
        features = []
        randR = random.random()*2*math.pi
        for k in range(nrFeatures):
            if k == firstCircleFeature:
                features.append(center[0]+ math.cos(randR)*radius+random.random()*noise)
            elif k == secondCircleFeature:
                features.append(center[1]+ math.sin(randR)*radius+random.random()*noise)
            else:
                features.append(grid[k]*random.random())
        Xdata.append(features)
        Ydata.append(classNr)
    Xdata = np.array(Xdata)
    Ydata = np.array(Ydata)
    return Xdata,Ydata

def ThreeCirclesData():
    A,Ac = CreateCircleCluster(1000,0.1,(0,0),0.2,0,2,(10,10))
    B,Bc = CreateCircleCluster(1000,3,(0,0),0.2,1,2,(10,10))
    Mat = np.concatenate((A, B))
    MatC =np.concatenate((Ac, Bc))
    C,Cc = CreateCircleCluster(1000,7,(0,0),0.2,2,2,(10,10))
    Mat = np.concatenate((Mat, C))
    MatC = np.concatenate((MatC, Cc))
    return Mat,MatC

def GenerateGaussianData(means, stds, datapoints):
    nrClasses = len(means)
    nrFeatures = len(means[0])
    Xdata = []
    Ydata = []
    for i in range(nrClasses):
        for j in range(datapoints[i]):
            features = []
            for k in range(nrFeatures):
                features.append(np.random.normal(
                    loc=means[i][k], scale=stds[i][k], size=None))
            Xdata.append(features)
            Ydata.append(i)
    Xdata = np.array(Xdata)
    Ydata = np.array(Ydata)
    Xdata, Ydata = shuffle(Xdata, Ydata)
    return Xdata, Ydata

def GenerateUniformData(lows, highs, datapoints):
    nrClasses = len(lows)
    nrFeatures = len(lows[0])
    Xdata = []
    Ydata = []
    for i in range(nrClasses):
        for j in range(datapoints[i]):
            features = []
            for k in range(nrFeatures):
                features.append(np.random.uniform(
                    low = lows[i][k], high = highs[i][k], size=None))
            Xdata.append(features)
            Ydata.append(i)
    Xdata = np.array(Xdata)
    Ydata = np.array(Ydata)
    Xdata, Ydata = shuffle(Xdata, Ydata)
    return Xdata, Ydata
    

def Data_gussian_cluster(nrFeatures = 2, nrClasses = 4, nr_data_points = 10):
    mislabelProportion = 0
    nrMislabelPoints = 40
    means = [[0, 0], [-3.5, 1.5], [5, 0], [0, -5]]
    sigma = [[1, 1]] * nrClasses
    dataPoints = [nr_data_points] * nrClasses
    Xdat, Ydat = GenerateGaussianData(means, sigma, dataPoints)
    return Xdat, Ydat

def Data_gussian_cluster_hard():
    mislabelProportion = 0
    nrMislabelPoints = 40
    means = [[0, 0], [-3.5, 1.5], [5, 0], [0, -5],[-3, -1.5], [-3.5, 1.5], [4, 2], [1, 5]]
    sigma = [[1, 0.8],[1.2, 1],[1, 1.5],[1, 1]]*2
    dataPoints = [nr_data_points] * nrClasses
    Xdat, Ydat = GenerateGaussianData(means, sigma, dataPoints)
    return Xdat, Ydat

def Gaussian_cluster_grid(nr_features = 2, grid_size = 4, nr_data_points = 10, scale=5, sigma=1):
    positions = [(d*scale).flatten().tolist() for d in np.mgrid[[slice(1,grid_size+1)]*nr_features]]
    means = list(list(a) for a in zip(*positions))
    sigmas = [[sigma]*nr_features] * grid_size**nr_features
    dataPoints = [nr_data_points] * grid_size**nr_features
    Xdat, Ydat = GenerateGaussianData(means, sigmas, dataPoints)
    return Xdat, Ydat
    
def Data2():
    nrFeatures = 2
    mislabelProportion = 0
    nrClasses = 6
    nrMislabelPoints = 40
    means = [[1, 3], [-6, 2], [-2, 6], [-5, 4], [-8,0], [4,3]]
    sigma = [[1, 1.4], [1.1, 1.1], [0.9, 1.2], [0.7, 1.1], [1.1,1.2], [1.0,1.0]]
    dataPoints = [10000, 2000, 2000, 500, 800, 3000]

    Xdat, Ydat = GenerateGaussianData(means, sigma, dataPoints)
    return Xdat, Ydat

def Data_circ_cluster():    
    nrVars = 2
    radius = 6
    nrClasses = 8  
    sigma = np.ones([nrClasses, nrVars])
    means = np.ones([nrClasses, nrVars])
    for i in range (0,nrClasses):
        means[i,0] = np.cos(i*np.pi*2/nrClasses)*radius
        means[i,1] = np.sin(i*np.pi*2/nrClasses)*radius
    dataPoints = [1000,1000,1000,1000,1000,1000,1000,1000]#np.ones([nrClasses,1])*1000
    Xdat, Ydat = GenerateGaussianData(means, sigma, dataPoints)
    return Xdat, Ydat

def Data_uniform_cluster():
    nrFeatures = 2
    mislabelProportion = 0
    nrClasses = 3
    nrMislabelPoints = 40
    lows = [[0, 0], [5, 0], [0, -5]]
    highs = [[6, 5], [0, 5], [6, 4]]
    dataPoints = [10000, 2000, 2000]

    Xdat, Ydat = GenerateUniformData(lows, highs, dataPoints)
    return Xdat, Ydat
  
def Data_one_cluster():    
    nrVars = 2
    radius = 6
    nrClasses = 1  
    sigma = np.ones([nrClasses, nrVars])
    means = np.ones([nrClasses, nrVars])
    for i in range (0,nrClasses):
        means[i,0] = np.cos(i*np.pi*2/nrClasses)*radius
        means[i,1] = np.sin(i*np.pi*2/nrClasses)*radius
    dataPoints = [1000]
    Xdat, Ydat = GenerateGaussianData(means, sigma, dataPoints)
    return Xdat, Ydat

def Data_separated_clusters():    
    nrVars = 2
    radius = 12
    nrClasses = 8  
    sigma = np.ones([nrClasses, nrVars])
    means = np.ones([nrClasses, nrVars])
    for i in range (0,nrClasses):
        means[i,0] = np.cos(i*np.pi*2/nrClasses)*radius
        means[i,1] = np.sin(i*np.pi*2/nrClasses)*radius
    dataPoints = [1000,1000,1000,1000,1000,1000,1000,1000]#np.ones([nrClasses,1])*1000
    Xdat, Ydat = GenerateGaussianData(means, sigma, dataPoints)
    return Xdat, Ydat

def Data_many_clusters():    
    nrVars = 2
    radius = 6
    nrClasses = 8  
    sigma = np.ones([nrClasses, nrVars])
    means = np.ones([nrClasses, nrVars])
    for i in range (0,nrClasses):
        means[i,0] = np.cos(i*np.pi*2/nrClasses)*radius
        means[i,1] = np.sin(i*np.pi*2/nrClasses)*radius
    dataPoints = [1000,1000,1000,1000,1000,1000,1000,1000]#np.ones([nrClasses,1])*1000
    Xdat, Ydat = GenerateGaussianData(means, sigma, dataPoints)
    return Xdat, Ydat

def create_rings_of_clusters(classes_per_ring = 8, nr_rings = 3, radius=12, points_per_cluster=100):  
    nrVars = 2
    sigma = np.ones([classes_per_ring*nr_rings, nrVars])
    means = np.ones([classes_per_ring*nr_rings, nrVars])
    for i in range(1,nr_rings+1):
        for j in range(classes_per_ring):
            means[j + classes_per_ring*(i-1),0] = np.cos(j*np.pi*2/classes_per_ring)*radius*i
            means[j + classes_per_ring*(i-1),1] = np.sin(j*np.pi*2/classes_per_ring)*radius*i
            
    dataPoints = [points_per_cluster]*nr_rings*classes_per_ring
    
    Xdat, Ydat = GenerateGaussianData(means, sigma, dataPoints)
    return Xdat, Ydat
