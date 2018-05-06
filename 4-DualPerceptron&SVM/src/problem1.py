#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:07:17 2018

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

# single perceptron step
def perceptron_step(Xi, yi, w, learning_rate):
    y_hat_i = predict_i(Xi, w)
    flag = False
    if yi * y_hat_i <= 0:
        w = w + learning_rate * yi * Xi
        flag = True
    
    return w, flag

def perceptron(X, y, learning_rate, max_iteration):
    
    n = y.size
    i = 0;
    w = np.zeros(X.shape[1])
    
    last_change = -1
    
    for j in range(max_iteration):
        w, flag = perceptron_step(X[i], y[i], w, learning_rate)
        
        if flag == True:
            last_change = i
        else:
            if last_change == i:
#                print(j)
                break
        
        i += 1
        if i == n:
            i = 0
    
    return w
    
# predict a single y_hat_i using Xi and w
def predict_i(Xi, w):
    y_hat_i = np.sign(Xi.dot(w))
    
    return y_hat_i

# predict y_hat using X and w
def predict(X, w):
    y_hat = np.sign(X.dot(w))
    
    return y_hat

# linear kernel
def kernel_linear(x, y):
    return x.dot(y)

# rbf kernel
def kernel_rbf(x, y, gamma):
    diff = x - y
    return np.exp( -gamma * diff.dot(diff) )

# single step of dual perceptron
def dual_perceptron_step(X, y, i, alpha):
    y_hat_i = dual_predict_i(X, y, X[i], alpha)
    flag = False
    if y[i] * y_hat_i <= 0:
        alpha[i] += 1
        flag = True
    
    return alpha, flag

# run dual perceptron to get alpha
def dual_perceptron(X, y, max_iteration):
    n = y.size
    i = 0;
    alpha = np.zeros(n)
    
    last_change = -1
    
    for j in range(max_iteration):
        alpha, flag = dual_perceptron_step(X, y, i, alpha)
        
        if flag == True:
            last_change = i
        else:
            if last_change == i:
#                print(j)
                break
        
        i += 1
        if i == n:
            i = 0
    
    return alpha

# predict a single yi through Xi
def dual_predict_i(X, y, Xi, alpha):
    s = (alpha * y * kernel_linear(X, Xi)).sum()
    y_hat_i = np.sign(s)
    
    return y_hat_i


# predict whole X
def dual_predict(X_train, y_train, X_test, alpha):
    y_hat = np.zeros(X_test.shape[0])
    
    for i in range(y_hat.size):
        y_hat[i] = dual_predict_i(X_train, y_train, X_test[i], alpha)
    
    return y_hat

# evaluate the accuracy, precision and recall
def evaluate(y, y_hat):  
    class_id = 1
    
    accuracy = (y == y_hat).sum() / y.size
    
    true_positives = np.logical_and(y_hat == class_id, y == class_id).sum()
    predict_positives = (y_hat == class_id).sum()
    actual_positives = (y == class_id).sum()
    
    precision = true_positives / predict_positives
    recall = true_positives / actual_positives
    
    return accuracy, precision, recall
    
# k fold validation
def k_fold_validation(dataset, learning_rate, max_iteration=100000, folds=10):
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
        
        left = dataset[0: start]
        right = dataset[end: ]
        dataset_train = np.vstack([left, right])
        
        X_train = dataset_train[:, 0:-1]
        y_train = dataset_train[:, -1]
        X_train, mean, std = normalize(X_train)
        
        X_test = dataset_test[:, 0:-1]
        y_test = dataset_test[:, -1]
        X_test = test_normalize(X_test, mean, std)
        
        w = perceptron(X_train, y_train, learning_rate, max_iteration)
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

# k fold validation dual perceptron
def k_fold_validation_dual(dataset, max_iteration=100000, folds=10):
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
        
        left = dataset[0: start]
        right = dataset[end: ]
        dataset_train = np.vstack([left, right])
        
        X_train = dataset_train[:, 0:-1]
        y_train = dataset_train[:, -1]
        X_train, mean, std = normalize(X_train)
        
        X_test = dataset_test[:, 0:-1]
        y_test = dataset_test[:, -1]
        X_test = test_normalize(X_test, mean, std)
        
        alpha = dual_perceptron(X_train, y_train, max_iteration)
#        print(w)

        y_hat_train = dual_predict(X_train, y_train, X_train, alpha)
        
        accuracy, precision, recall = evaluate(y_train, y_hat_train)
        
        accuracies_train.append(accuracy)
        precisions_train.append(precision)
        recalls_train.append(recall)
        
        y_hat_test = dual_predict(X_train, y_train, X_test, alpha)
        accuracy, precision, recall = evaluate(y_test, y_hat_test)
        
        accuracies_test.append(accuracy)
        precisions_test.append(precision)
        recalls_test.append(recall)

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


def test_perceptron():
    print('Perceptron:')
    dataset = dataset_reader('perceptronData.csv')
    k_fold_validation(dataset, 1)
    print()

def test_dual_perceptron():
    print('Dual Perceptron:')
    dataset = dataset_reader('perceptronData.csv')
    k_fold_validation_dual(dataset)
    print()
    
def test_dual_perceptron_sprirals():
    print('Dual Perceptron Two Sprirals:')
    dataset = dataset_reader('twoSpirals.csv')
    k_fold_validation_dual(dataset)
    print()
    
def main():
#    test_perceptron()
#    test_dual_perceptron()
    test_dual_perceptron_sprirals()
    

if __name__ == '__main__':
    main()