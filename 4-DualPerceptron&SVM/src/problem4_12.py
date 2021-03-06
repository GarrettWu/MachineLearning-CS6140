#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 22:15:43 2018

@author: Garrett
"""

import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
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

# evaluate the accuracy
def evaluate(y, y_hat, class_id=1):      
    accuracy = np.equal(y, y_hat).sum() / y.size
    
    true_positives = np.logical_and(y_hat == class_id, y == class_id).sum()
    
#    print(y_hat, class_id)
    predict_positives = np.equal(y_hat, class_id).sum()
    actual_positives = np.equal(y, class_id).sum()
    
#    print(predict_positives)
    precision = true_positives / predict_positives
    recall = true_positives / actual_positives
    
    return accuracy, precision, recall
    

def m_fold_validation(X, y, C, gamma, m=5):
    end = 0
    size = math.floor(y.size / m)
    accuracies = []
    
    for i in range(m):
        start = end
        end = start + size
        
        X_val = X[start:end]
        y_val = y[start:end]
        X_train = np.vstack([X[:start], X[end:]])
        y_train = np.hstack([y[:start], y[end:]])
        
#        clf = SVC(C=C, gamma=gamma)
        clf = SVC(C=C, kernel='linear')
        clf.fit(X_train, y_train)
        
        accuracy = clf.score(X_val, y_val)
        accuracies.append(accuracy)
        
    return np.mean(accuracies)      

def predict(probs):
    y_hat = np.argmax(probs, axis=1)+1
        
    return y_hat
 
    
def get_probs(X, clfs):
    probs = np.zeros([len(X), len(clfs)])
    for i in range(len(clfs)):
        clf = clfs[i]
        probs[:,i] = clf.predict_proba(X)[:, 1]
        
    return probs
    
# k fold validation
def k_fold_validation(dataset, folds=10, m=5):
    np.random.shuffle(dataset)
    
    end = 0
    size = math.floor(dataset.shape[0] / folds)
    
    accuracies_train = []
    precisions_train = []
    recalls_train = []
    
    accuracies_test = []
    precisions_test = []
    recalls_test = []
    
    C_grid = []
    gamma_grid = [1]
    for i in [-4, -1, 2, 5, 8]:
        C_grid.append(2 ** i)
#    for i in [-15, -10, -5, 0, 5]:
#        gamma_grid.append(2 ** i)
        
    class_id = 3
    for k in range(folds):
        start = end
        end = start + size
        dataset_test = dataset[start: end]
        
        left = dataset[0: start]
        right = dataset[end: ]
        dataset_train = np.vstack([left, right])
        
        X_train = dataset_train[:, 1:]
        y_train = dataset_train[:, 0]
        X_train, mean, std = normalize(X_train)
        
        X_test = dataset_test[:, 1:]
        y_test = dataset_test[:, 0]
        X_test = test_normalize(X_test, mean, std)
        
        clfs = []
        for class_id in range(1, 4):
            y_train_c = np.equal(y_train, class_id).astype(int)
            
            max_acc = -1
            for C in C_grid:
                for gamma in gamma_grid:
                    acc = m_fold_validation(X_train, y_train_c, C, gamma)
                    if acc > max_acc:
                        max_acc = acc
                        C_op = C
                        gamma_op = gamma

            print("Class ID: ", class_id)
            print("Max Accuracy: ", max_acc)
            print("Fold: ", k)
            print("C optimal: ", C_op)
            print("gamma optimal: ", gamma_op)
        
#            clf = SVC(C=C_op, gamma=gamma_op, probability=True)
            clf = SVC(C=C_op, kernel='linear', probability=True)
            clf.fit(X_train, y_train_c)
            
            clfs.append(clf)
        
        
        probs_train = get_probs(X_train, clfs)
        y_hat_train = predict(probs_train)

        accuracy, precision, recall = evaluate(y_train, y_hat_train, class_id=class_id)
        
        accuracies_train.append(accuracy)
        precisions_train.append(precision)
        recalls_train.append(recall)
        
        probs_test = get_probs(X_test, clfs)
        y_hat_test = predict(probs_test)
        
        accuracy, precision, recall = evaluate(y_test, y_hat_test, class_id=class_id)
        
        accuracies_test.append(accuracy)
        precisions_test.append(precision)
        recalls_test.append(recall)
        
        probas_ = probs_test[:,class_id-1]
        y_test_c = np.equal(y_test, class_id).astype(int)
        fpr, tpr, thresholds = roc_curve(y_test_c, probas_)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (k, roc_auc))
                
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
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
    
def test_wine():
    print('Wine:')
    dataset = dataset_reader('wine.data')
#    print(dataset)
    k_fold_validation(dataset)
    
    print()
    
def main():
    test_wine()
    

if __name__ == '__main__':
    main()