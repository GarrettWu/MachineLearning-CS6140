#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 22:15:43 2018

@author: Garrett

Used build in one vs rest to classify.
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
        
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_train, y_train)
        
        accuracy = clf.score(X_val, y_val)
        accuracies.append(accuracy)
        
    return np.mean(accuracies)      
        
    
def test_ocr():
    print('OCR:')
    dataset_train = dataset_reader('optdigits.tra')
    dataset_test = dataset_reader('optdigits.tes')
    
    np.random.shuffle(dataset_train)
    np.random.shuffle(dataset_test)
    
    C_grid = []
    gamma_grid = []
    for i in [-4, -1, 2, 5, 8]:
        C_grid.append(2 ** i)
    for i in [-15, -10, -5, 0, 5]:
        gamma_grid.append(2 ** i)
        
        
    X_train = dataset_train[:, :-1]
    y_train = dataset_train[:, -1]
    
    X_test = dataset_test[:, :-1]
    y_test = dataset_test[:, -1]
    
    max_acc = -1
    for C in C_grid:
        for gamma in gamma_grid:
            acc = m_fold_validation(X_train, y_train, C, gamma)
#                print(C, gamma, acc)
            if acc > max_acc:
                max_acc = acc
                C_op = C
                gamma_op = gamma

    print("Max Accuracy: ", max_acc)
    print("C optimal: ", C_op)
    print("gamma optimal: ", gamma_op)
    
    clf = SVC(C=C_op, gamma=gamma_op, probability=True)
    clf.fit(X_train, y_train)
    
    y_hat_train = clf.predict(X_train)
    y_hat_test = clf.predict(X_test)
    probas_ = clf.predict_proba(X_test)
    
    for class_id in range(10):
        print("Class ID: ", class_id)
        
        accuracy_train, precision_train, recall_train = evaluate(y_train, y_hat_train, class_id=class_id)
        
        accuracy_test, precision_test, recall_test = evaluate(y_test, y_hat_test, class_id=class_id)
        
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, class_id], pos_label=class_id)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC (AUC = %0.2f)' % (roc_auc))
                    
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        
        print('Evaluation for training data:')
        print('Accuracy: ', accuracy_train)
        print('Precision: ', precision_train)
        print('Recall: ', recall_train)
        
        print('Evaluation for testing data:')
        print('Accuracy: ', accuracy_test)
        print('Precision: ', precision_test)
        print('Recall: ', recall_test)
    print()
    
def main():
    test_ocr()
    

if __name__ == '__main__':
    main()