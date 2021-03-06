#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:32:58 2018

CS6140 Assignment2 Gradient Descent Problem2 

@author: Garrett
"""
from sklearn import linear_model
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reader
def dataset_reader(file):
    return np.array(pd.read_csv(file, header=None), dtype=np.float64)

# normalize X data using z-score and then add x0
def normalize(X):
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    X_norm = (X - mean) / std
    X_norm = add_x0(X_norm)
    
    return X_norm, mean, std

# normalize X testing data using mean and deviation of training data
def test_normalize(X, mean, std):
    X_norm = (X - mean) / std
    X_norm = add_x0(X_norm)
    
    return X_norm

# add x0 to data
def add_x0(X):
    return np.column_stack([np.ones([X.shape[0], 1]), X])
    
# predict y_hat using X and w
def predict(X, w):
    return X.dot(w)
    
# sum of squared errors
def sse(X, y, w):
    y_hat = predict(X, w)
    return ((y_hat - y) ** 2).sum()

# root mean squared error
def rmse(X, y, w):
    return math.sqrt(sse(X, y, w) / y.size)

# cost function of regression
def cost_function(X, y, w):
    return sse(X, y, w) / 2
    
# derivative vector of the cost function
def cost_derivatives(X, y, w):
    y_hat = predict(X, w)
    return (y_hat - y).dot(X)

def plot_rmse(rmse_sequence):
    # Data for plotting
    s = np.array(rmse_sequence)
    t = np.arange(s.size)
    
    fig, ax = plt.subplots()
    ax.plot(t, s)
    
    ax.set(xlabel='iterations', ylabel='rmse',
           title='rmse trend')
    ax.grid()
    
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, shadow=True)
    plt.show()
    

# implement gradient descent to calculate w
def gradient_descent(X, y, w, learningrate, tolerance, maxIteration=1000):
    rmse_sequence = []
    
    last = float('inf')
    for i in range(maxIteration):
        w = w - learningrate * cost_derivatives(X, y, w)
        cur = rmse(X, y, w)
        diff = last - cur
        last = cur
        rmse_sequence.append(cur)
        if diff < tolerance:
#            print(i)
            break
        
    plot_rmse(rmse_sequence)
    return w
    
# k fold validation
def k_fold_validation(dataset, learningrate, tolerance, folds=10):
    np.random.shuffle(dataset)
    
    end = 0
    size = math.floor(dataset.shape[0] / folds)
    
    rmse_train = []
    rmse_test = []
    sse_train = []
    sse_test = []
        
    for k in range(folds):
        start = end
        end = start + size
        dataset_test = dataset[start: end]
        
        left = dataset[0: start]
        right = dataset[end: ]
        dataset_train = np.vstack([left, right])
        
        X_train = dataset_train[:, 0:-1]
        y_train = dataset_train[:, -1]
        X_train, mean, std = normalize(X_train)
        
        X_test = dataset_test[:, 0:-1]
        y_test = dataset_test[:, -1]
        X_test = test_normalize(X_test, mean, std)
        
        w = np.ones(X_train.shape[1], dtype=np.float64) * 0
        w = gradient_descent(X_train, y_train, w, learningrate, tolerance)
                
        rmse_train.append(rmse(X_train, y_train, w))
        rmse_test.append(rmse(X_test, y_test, w))
        
        sse_train.append(sse(X_train, y_train, w))
        sse_test.append(sse(X_test, y_test, w))
        
    
    print('RMSE for training data:')
    print(rmse_train)
    print('Mean:')
    print(np.mean(rmse_train))
    print('RMSE for testing data:')
    print(rmse_test)
    print('Mean:')
    print(np.mean(rmse_test))
    
    print()
    print('SSE for training data:')
    print('Mean:')
    print(np.mean(sse_train))
    print('Standard Deviation:')
    print(np.std(sse_train))
    
    print('SSE for testing data:')
    print('Mean:')
    print(np.mean(sse_test))
    print('Standard Deviation:')
    print(np.std(sse_test))
        
def test_housing():
    print('Housing:')
    dataset = dataset_reader('housing.csv')
    
    k_fold_validation(dataset, 0.1e-3, 0.5e-2)

    print()
    
def test_yacht():
    print('Yacht:')
    dataset = dataset_reader('yachtData.csv')
    k_fold_validation(dataset, 0.1e-2, 0.1e-2)
    print()
    
def test_concrete():
    print('Concrete:')
    dataset = dataset_reader('concreteData.csv')
    k_fold_validation(dataset, 0.7e-3, 0.1e-3)
    print()
    
def main():
    
    test_housing()
    test_yacht()
    test_concrete()
    
    

if __name__ == '__main__':
    main()