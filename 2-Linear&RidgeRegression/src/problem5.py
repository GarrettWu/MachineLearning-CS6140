#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:32:58 2018

CS6140 Assignment2 Gradient Descent Problem5

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

# add polynomial features
def add_p(dataset, p):
    X = dataset[:, : -1]
    y = dataset[:, -1]
    y = y[:, None]
    
    copy = X.copy()
    for pow in range(2, p + 1):
        new_feature = np.power(copy, pow)
        X = np.hstack([X, new_feature])
    dataset = np.hstack([X, y])
    return dataset
    
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

# implement normal equation to calculate w
def normal_equation(X, y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    
# k fold validation
def k_fold_validation(dataset, folds=10):
    np.random.shuffle(dataset)
    
    end = 0
    size = math.floor(dataset.shape[0] / folds)
    
    rmse_train = []
    rmse_test = []
    
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
        
        w = np.zeros(X_train.shape[1], dtype=np.float64)
        w = normal_equation(X_train, y_train)
                
        rmse_train.append(rmse(X_train, y_train, w))
        rmse_test.append(rmse(X_test, y_test, w))
        

    return np.mean(rmse_train), np.mean(rmse_test)
    
        
def test_sinusoid():
    print('Sinusoid:')
    original_train = dataset_reader('sinData_train.csv')
    original_test = dataset_reader('sinData_Validation.csv')

    
    ps = list(range(1, 65))
    rmses_train = []
    rmses_test = []
  
    for p in ps:
        dataset_train = add_p(original_train, p)
        dataset_test = add_p(original_test, p)
        
        X_train = dataset_train[:, 0:-1]
        X_train = add_x0(X_train)
        y_train = dataset_train[:, -1]
        
        X_test = dataset_test[:, 0:-1]
        X_test = add_x0(X_test)
        y_test = dataset_test[:, -1]
        
        w = normal_equation(X_train, y_train)
        
        rmse_train = rmse(X_train, y_train, w)
        rmse_test = rmse(X_test, y_test, w)
        
        rmses_train.append(rmse_train)
        rmses_test.append(rmse_test)

    
    print('training: ')
    print(rmses_train)
    print('testing: ')
    print(rmses_test)
    
    s_train = np.array(rmses_train)
    s_test = np.array(rmses_test)
    t = np.array(ps)
    
    fig, ax = plt.subplots()
    ax.plot(t, s_train, label = 'train RMSE')
    ax.plot(t, s_test, label = 'test RMSE')
    
    ax.set(xlabel='max(p)', ylabel='rmse',
           title='rmse for max ps')
    ax.grid()
    
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, shadow=True)
    plt.show()

    print()
    
def test_yacht():
    print('Yacht:')
    original_dataset = dataset_reader('yachtData.csv')
    
    ps = list(range(1, 50))
    rmses_train = []
    rmses_test = []
    
    for p in ps:
        dataset = add_p(original_dataset, p)
                
        rmse_train, rmse_test = k_fold_validation(dataset)
        
        rmses_train.append(rmse_train)
        rmses_test.append(rmse_test)

    
    print('training: ')
    print(rmses_train)
    print('testing: ')
    print(rmses_test)
    
    s_train = np.array(rmses_train)
    s_test = np.array(rmses_test)
    t = np.array(ps)
    
    fig, ax = plt.subplots()
    ax.plot(t, s_train, label = 'train RMSE')
    ax.plot(t, s_test, label = 'test RMSE')
    
    ax.set(xlabel='max(p)', ylabel='rmse',
           title='rmse for max ps')
    ax.grid()
    
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, shadow=True)
    plt.show()

    print()

    
def main():
    test_sinusoid()
    test_yacht()    
    

if __name__ == '__main__':
    main()