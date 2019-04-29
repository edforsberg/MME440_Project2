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
    

def Data1():
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
    nrClasses = 4
    nrMislabelPoints = 40
    lows = [[0, 0], [-3.5, 1.5], [5, 0], [0, -5]]
    highs = [[3, 2], [0, 5], [6, 4], [6, 2]]
    dataPoints = [10000, 2000, 2000, 500]

    Xdat, Ydat = GenerateUniformData(lows, highs, dataPoints)
    return Xdat, Ydat

