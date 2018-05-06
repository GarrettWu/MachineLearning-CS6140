#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:21:16 2018

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

# sigmoid function
def sigmoid(a): 
    sig = 1 / (1 + np.exp(-a))

    return sig

# cost function of regression
def cost_function(X, y, w, lam):
    sig = sigmoid(X.dot(w))

    temp = y * np.log(sig) + (1-y) * np.log(1-sig)
    cost = - temp.sum() + lam / 2 * w[1:].dot(w[1:])

    return cost
    
# derivative vector of the cost function
def gradient(X, y, w, lam):
    sig = sigmoid(X.dot(w))
    
    return (sig-y).dot(X) + lam * np.hstack([[0], w[1:]])

def plot_cost(cost_sequence):
    # Data for plotting
    s = np.array(cost_sequence)
    t = np.arange(s.size)
    
    fig, ax = plt.subplots()
    ax.plot(t, s)
    
    ax.set(xlabel='iterations', ylabel='cost',
           title='cost trend')
    ax.grid()
    
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, shadow=True)
    plt.show()

# implement gradient descent to calculate w
def gradient_descent(X, y, w, learningrate, tolerance, lam, maxIteration=1000):
    cost_sequence = []
    
    last = float('inf')
    for i in range(maxIteration):
        w = w - learningrate * gradient(X, y, w, lam)
        cur = cost_function(X, y, w, lam)
        diff = last - cur
        last = cur
        cost_sequence.append(cur)
        if diff < tolerance:
            break
    
#    plot_cost(cost_sequence)
    return w

# predict y_hat using X and w
def predict(X, w):
    sig = sigmoid(X.dot(w))
    
    return np.around(sig)

# evaluate the prediction
def evaluate(y, y_hat):
    y = (y == 1)
    y_hat = (y_hat == 1)
    
    accuracy = (y == y_hat).sum() / y.size
    precision = (y & y_hat).sum() / y_hat.sum()
    recall = (y & y_hat).sum() / y.sum()
    
    return accuracy, precision, recall
    
# k fold validation
def k_fold_validation(dataset, learningrate, tolerance, lam=0, folds=10):
    np.random.shuffle(dataset)
    
    end = 0
    size = math.floor(dataset.shape[0] / folds)
    
    
    accuracies_train = []
    precisions_train = []
    recalls_train = []
    
    accuracies_test = []
    precisions_test = []
    recalls_test = []
    
    costs = []
        
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
        w = gradient_descent(X_train, y_train, w, learningrate, tolerance, lam)

#        print(w)
        y_hat_train = predict(X_train, w)
        
        accuracy, precision, recall = evaluate(y_train, y_hat_train)
        
        accuracies_train.append(accuracy)
        precisions_train.append(precision)
        recalls_train.append(recall)
        
        y_hat_test = predict(X_test, w)
        accuracy, precision, recall = evaluate(y_test, y_hat_test)
        
        accuracies_test.append(accuracy)
        precisions_test.append(precision)
        recalls_test.append(recall)
        
        costs.append(cost_function(X_train, y_train, w, lam))
        
    
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

    return np.mean(costs)
        
def test_spambase():
    print('Spambase:')
    dataset = dataset_reader('spambase.csv')
    
    # must stop on tolerance 1, otherwise overflow
    k_fold_validation(dataset, 0.1e-3, 1, 100) 
    
    print()
    
def test_breastcancer():
    print('Breast Cancer:')
    dataset = dataset_reader('breastcancer.csv')
    k_fold_validation(dataset, 0.1e-3, 0.3e-1, 100)
    
#    tolerances = [0.3e-1, 0.1, 0.3, 1, 3, 10]
#    cost_sequence = []
#    for tolerance in tolerances:
#        cost = k_fold_validation(dataset, 0.1e-3, tolerance)
#        cost_sequence.append(cost)
#    
#    plot_cost(cost_sequence)
    print()
    
def test_diabetes():
    print('Diabets:')
    dataset = dataset_reader('diabetes.csv')
    k_fold_validation(dataset, 0.3e-4, 0.1e-1, 100)
    print()

    
def main():
    test_spambase()
    test_breastcancer()
    test_diabetes()
    

if __name__ == '__main__':
    main()