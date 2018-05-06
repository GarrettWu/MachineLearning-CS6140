#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:24:02 2018

@author: Garrett
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reader
def dataset_reader(file):
    return np.array(pd.read_csv(file, header=None), dtype=np.float64)

# shuffle and normalize data
def data_process(dataset):
    np.random.shuffle(dataset)
    
    X = dataset[:, 0:-1]
    y = dataset[:, -1]
    
    # normalize the class labels to 1 / -1
    y[y == 0] = -1
    
    end = int(0.8 * X.shape[0])
    
    X_train = X[:end]
    X_test = X[end:]
    y_train = y[:end]
    y_test = y[end:]
     
#    X_train, mean, std = normalize_train(X_train)
#    X_test = (X_test - mean) / std
    
    return X_train, y_train, X_test, y_test

# normalize X data
def normalize_train(X):
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    
    std[std==0] = 1
    
    X_norm = (X - mean) / std
    
    return X_norm, mean, std

# sigmoid function
def sigmoid(a): 
    sig = 1 / (1 + np.exp(-a))

    return sig

# squared error between two points
def squared_error(t, o):
    diff = t - o
    return diff.dot(diff) / 2

# evaluate the accuracy
def evaluate(y, y_hat, class_id=1):      
    accuracy = np.equal(y, y_hat).sum() / y.size
    
    return accuracy


def plot_error_vs_t(error_name, error, T):
    # Data for plotting
    s = np.array(error)
    t = np.arange(T) + 1
    
    fig, ax = plt.subplots()
    ax.plot(t, s)
    
    ax.set(xlabel='T', ylabel=error_name,
           title=(error_name+' vs T'))
    ax.grid()
    
    plt.show()