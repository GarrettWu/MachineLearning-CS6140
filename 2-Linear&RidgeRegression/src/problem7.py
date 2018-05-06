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

# center training set
def center(dataset_train):
    mean = np.mean(dataset_train, 0)
    return dataset_train - mean, mean

# center testing set
def center_test(dataset_test, mean):
    return dataset_test - mean

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

# implement ridge regression normal equation to calculate w
def normal_equation(X, y, lam):
    return np.linalg.pinv(X.T.dot(X) + lam * np.identity(X.shape[1])).dot(X.T).dot(y)
    
# k fold validation
def k_fold_validation(dataset, lam, folds=10):
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
        
        dataset_train, mean = center(dataset_train)
        dataset_test = center_test(dataset_test, mean)
        
        X_train = dataset_train[:, 0:-1]
        y_train = dataset_train[:, -1]
        
        X_test = dataset_test[:, 0:-1]
        y_test = dataset_test[:, -1]
        
        w = np.zeros(X_train.shape[1], dtype=np.float64)
        w = normal_equation(X_train, y_train, lam)
                
        rmse_train.append(rmse(X_train, y_train, w))
        rmse_test.append(rmse(X_test, y_test, w))
        

    return np.mean(rmse_train), np.mean(rmse_test)
    
    
def test_sinusoid(p):
    print('Sinusoid: p =', p)
    original_dataset = dataset_reader('sinData_Train.csv')
    lams = np.arange(0, 10.1, 0.2)
    dataset = add_p(original_dataset, p)
    
    rmses_train = []
    rmses_test = []
    
    for lam in lams:        
        rmse_train, rmse_test = k_fold_validation(dataset, lam)
        
        rmses_train.append(rmse_train)
        rmses_test.append(rmse_test)

    
    print('training: ')
    print(rmses_train)
    print('testing: ')
    print(rmses_test)
    
    s_train = np.array(rmses_train)
    s_test = np.array(rmses_test)
    t = lams
    
    fig, ax = plt.subplots()
    ax.plot(t, s_train, label = 'train RMSE')
    ax.plot(t, s_test, label = 'test RMSE')
    
    ax.set(xlabel='lambda', ylabel='rmse',
           title='rmse for lambdas')
    ax.grid()
    
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, shadow=True)
    plt.show()

    print()

    
def main():
    test_sinusoid(5)
    test_sinusoid(10)
    test_sinusoid(50)    

if __name__ == '__main__':
    main()