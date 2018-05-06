#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:54:13 2018

@author: Garrett
"""

# Mathieu Blondel, October 2010
# License: BSD 3 clause

import numpy as np
import pandas as pd
from numpy import linalg

# reader
def dataset_reader(file):
    return np.array(pd.read_csv(file, header=None), dtype=np.float64)

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, gamma=0.01):
    diff = x - y
    res = np.exp( -gamma * diff.dot(diff) )
    return res

# normalize X data using z-score and then add x0
def normalize(X):
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    X_norm = (X - mean) / std
    X_norm = add_x0(X_norm)
    
    return X_norm, mean, std

# normalize X testing data using mean and deviation of training data, then add x0
def test_normalize(X, mean, std):
    X_norm = (X - mean) / std
    X_norm = add_x0(X_norm)
    
    return X_norm

# add x0 to data
def add_x0(X):
    return np.column_stack([np.ones([X.shape[0], 1]), X])


class Perceptron(object):

    def __init__(self, T=1):
        self.T = T

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for t in range(self.T):
            for i in range(n_samples):
                if self.predict(X[i])[0] != y[i]:
                    self.w += y[i] * X[i]
                    self.b += y[i]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.project(X))

class KernelPerceptron(object):

    def __init__(self, kernel=linear_kernel, T=1):
        self.kernel = kernel
        self.T = T

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        for t in range(self.T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]


    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        return np.sign(self.project(X))

# evaluate the accuracy
def evaluate(y, y_hat):  
    class_id = 1
    
    accuracy = (y == y_hat).sum() / y.size
    
    true_positives = np.logical_and(y_hat == class_id, y == class_id).sum()
    predict_positives = (y_hat == class_id).sum()
    actual_positives = (y == class_id).sum()
    
    precision = true_positives / predict_positives
    recall = true_positives / actual_positives
    
    return accuracy, precision, recall


def main():
    dataset = dataset_reader('twoSpirals.csv')
    np.random.shuffle(dataset)
    
    end = int(0.9*len(dataset))
    dataset = dataset[0:end]
    
    X_train = dataset[:,0:-1]
    y_train = dataset[:,-1]
    
#    X_train, mean, std = normalize(X_train)
    
    
    clf = KernelPerceptron(gaussian_kernel, T=20)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_train)
    
    accuracy, pre, recall = evaluate(y_train, y_hat)
    print(accuracy)
    

if __name__ == '__main__':
    main()