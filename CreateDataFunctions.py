import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from sklearn.utils import shuffle


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
    

def Data_gussian_cluster():
    nrFeatures = 2
    mislabelProportion = 0
    nrClasses = 4
    nrMislabelPoints = 40
    means = [[0, 0], [-3.5, 1.5], [5, 0], [0, -5]]
    sigma = [[1, 1], [1, 1], [1, 1], [1, 1]]
    dataPoints = [10000, 2000, 2000, 500]

    Xdat, Ydat = GenerateGaussianData(means, sigma, dataPoints)
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
