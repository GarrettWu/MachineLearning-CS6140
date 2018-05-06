#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:24:02 2018

@author: Garrett
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# reader
def dataset_reader(file):
    return np.array(pd.read_csv(file, header=None), dtype=np.float64)

# shuffle and normalize data
def data_process(dataset):
    np.random.shuffle(dataset)
    
    X = dataset[:, 0:-1]
    y = dataset[:, -1]
    
    # normalize the class labels to start from 0
    y -= 1
#    X = normalize(X)
    
    return X, y

# normalize X data
def normalize(X):
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    
    std[std==0] = 1
    
    X_norm = (X - mean) / std
    
    return X_norm

# squared error between two points
def squared_error(x, center):
    diff = x - center
    return diff.dot(diff)

# SSE for points
def sum_of_squared_errors(X, y, centers):
    s = 0
    
    for i in range(X.shape[0]):
        se = squared_error(X[i], centers[y[i]])
        s += se
    
    return s

# Entropy for classes (or clusters)
def entropy(y, n_classes):
    N = y.shape[0]
    if N == 0:
        return 0
    
    e = 0
    for i in range(n_classes):
        py = np.sum(y == i) / N
        
        if py != 0:
            e -= py * np.log2(py) 
    
    return e

# Mutual Information of y classes and c clusters
def mutual_information(y, n_classes, c, n_clusters):
    N = y.shape[0]
    
    Hy = entropy(y, n_classes)
    
    Hyc = 0
    for i in range(n_clusters):
        logic = (c == i)
        
        Nc = np.sum(logic)
        
        pc = Nc / N
        yc = y[logic]
                
        ec = entropy(yc, n_classes)
        
        Hyc += pc * ec
        
    return Hy - Hyc
            
# Normalized Mutual Information of y classes and c clusters
def normalized_mutual_information(y, n_classes, c, n_clusters):
    mi = mutual_information(y, n_classes, c, n_clusters)
    Hy = entropy(y, n_classes)
    Hc = entropy(c, n_clusters)
    
    return 2 * mi / (Hy + Hc)

def multivariate_gaussian(x, mu, sigma):
#    k = x.shape[0]
#    numerator = np.exp( - 1/2 * (x - mu).T.dot(np.linalg.pinv(sigma)).dot(x - mu) )
#    denominator = np.sqrt( (2 * np.pi) ** k * np.linalg.det(sigma) )
#    return numerator / denominator
    return multivariate_normal.pdf(x, mean=mu, cov=sigma, allow_singular=True)

def plot_measure_vs_k(measure_name, measure, k):
    # Data for plotting
    s = np.array(measure)
    t = np.array(k)
    
    fig, ax = plt.subplots()
    ax.plot(t, s)
    
    ax.set(xlabel='k', ylabel=measure_name,
           title=(measure_name+' vs k'))
    ax.grid()
    
    plt.show()