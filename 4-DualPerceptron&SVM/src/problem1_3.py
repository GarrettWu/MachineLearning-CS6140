#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:07:17 2018

inspired by
https://gist.github.com/mblondel/656147#file-perceptron-py

@author: Garrett
"""

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

# normalize X testing data using mean and deviation of training data, then add x0
def test_normalize(X, mean, std):
    X_norm = (X - mean) / std
    X_norm = add_x0(X_norm)
    
    return X_norm

# add x0 to data
def add_x0(X):
    return np.column_stack([np.ones([X.shape[0], 1]), X])

# linear kernel
def kernel_linear(x, y):
    return x.dot(y)

# rbf kernel
def kernel_rbf(x, y, gamma):
    diff = x - y
    res = np.exp( -gamma * diff.dot(diff) )
    return res

class KernelPerceptron(object):
    def __init__(self, gamma, kernel=kernel_rbf, max_itr=20000):
        self.kernel = kernel
        self.max_itr = max_itr
        self.gamma = gamma

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j], gamma=self.gamma)

        T = int(np.ceil(self.max_itr / len(y)))
        for t in range(T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors, only them are needed to predict
        sv_logic = self.alpha > 1e-5
        self.alpha = self.alpha[sv_logic]
        self.sv_X = X[sv_logic]
        self.sv_y = y[sv_logic]

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv_X in zip(self.alpha, self.sv_y, self.sv_X):
                s += a * sv_y * self.kernel(X[i], sv_X, gamma=self.gamma)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.project(X))
    
# evaluate the accuracy
def evaluate(y, y_hat, class_id=1):      
    accuracy = (y == y_hat).sum() / y.size
    
    true_positives = np.logical_and(y_hat == class_id, y == class_id).sum()
    predict_positives = (y_hat == class_id).sum()
    actual_positives = (y == class_id).sum()
    
    precision = true_positives / predict_positives
    recall = true_positives / actual_positives
    
    return accuracy, precision, recall

# k fold validation dual perceptron
def k_fold_validation_rbf(dataset, gamma, folds=10):
    # shuffle the data
    np.random.shuffle(dataset)
    
    end = 0
    size = math.floor(dataset.shape[0] / folds)
    
    accuracies_train = []
    precisions_train = []
    recalls_train = []
    
    accuracies_test = []
    precisions_test = []
    recalls_test = []
        
            
    for k in range(folds):
        start = end
        end = start + size
        
        dataset_test = dataset[start: end]
        dataset_train = np.vstack([dataset[0: start], dataset[end: ]])
                
        X_train = dataset_train[:, 0:-1]
        y_train = dataset_train[:, -1]
#        X_train, mean, std = normalize(X_train)
        
        X_test = dataset_test[:, 0:-1]
        y_test = dataset_test[:, -1]
#        X_test = test_normalize(X_test, mean, std)
        
        clf = KernelPerceptron(gamma)
        clf.fit(X_train, y_train)

        y_hat_train = clf.predict(X_train)
        
        accuracy, precision, recall = evaluate(y_train, y_hat_train)
        accuracies_train.append(accuracy)
        precisions_train.append(precision)
        recalls_train.append(recall)
        
        y_hat_test = clf.predict(X_test)
        
        accuracy, precision, recall = evaluate(y_test, y_hat_test)
        accuracies_test.append(accuracy)
        precisions_test.append(precision)
        recalls_test.append(recall)

#    print(accuracies_train)
    print('Evaluation for training data:')
    print('Mean Accuracy: ', np.mean(accuracies_train))
    print('Std Accuracy: ', np.std(accuracies_train))
    print('Mean Precision: ', np.mean(precisions_train))
    print('Std Precision: ', np.std(precisions_train))
    print('Mean Recall: ', np.mean(recalls_train))
    print('Std Recall: ', np.std(recalls_train))
    
    print('Evaluation for testing data:')
    print('Mean Accuracy: ', np.mean(accuracies_test))
    print('Std Accuracy: ', np.std(accuracies_test))
    print('Mean Precision: ', np.mean(precisions_test))
    print('Std Precision: ', np.std(precisions_test))
    print('Mean Recall: ', np.mean(recalls_test))
    print('Std Recall: ', np.std(recalls_test))
    print()
    
def test_rbf_perceptron(gamma):
    print('Dual Perceptron RBF: gamma = ', gamma)
    dataset = dataset_reader('twoSpirals.csv')
    k_fold_validation_rbf(dataset, gamma)

    print()  
    
def main():
    gamma = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    
    for g in gamma:
        test_rbf_perceptron(g)
        
    

if __name__ == '__main__':
    main()