#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:31:05 2018

@author: Garrett
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVC

# reader
def dataset_reader(file):
    return np.array(pd.read_csv(file, header=None), dtype=np.float64)

def test_perceptron():
    data = dataset_reader('perceptronData.csv')
#    data = dataset_reader('twoSpirals.csv')
    n = len(data)
    
    train = []

    X = data[:, 0:-1]
    y = data[:, -1]
    
    clf = linear_model.Perceptron(max_iter=10000, tol=None)
    clf.fit(X, y)
    
    print('accuracy:', clf.score(X, y))

def test_svm():
    data = dataset_reader('twoSpirals.csv')
    X = data[:, 0:-1]
    y = data[:, -1]
    
    clf = SVC(gamma=0.25)
    clf.fit(X, y)
    
    print('accuracy:', clf.score(X, y))

def main():
#    test_perceptron()
    test_svm()
    

if __name__ == '__main__':
    main()