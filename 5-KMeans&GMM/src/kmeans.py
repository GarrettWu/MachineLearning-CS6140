#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:30:15 2018

@author: Garrett
"""

import utils
import numpy as np

class KMeans(object):
    # init the kmeans class, maxIter: max iterations, tol: tolerance, 
    # n_init: run n times with different random seeds, return the best one
    def __init__(self, n_clusters, maxIter=300, tol=0.0001, n_init=10):
        self.n_clusters = n_clusters
        self.maxIter = maxIter
        self.tol = tol
        self.n_init=n_init
        
    # fit model to data
    def fit(self, X):   
        self.X = X.copy()
        
        list_cluster_centers_ = []
        list_labels_ = []
        list_sse_ = []
        
        for n in range(self.n_init):
            self.cluster_centers_ = np.random.permutation(self.X)[0:self.n_clusters].copy()
            self.labels_ = np.zeros(self.X.shape[0], dtype=int)
            
            sse_last = -np.inf
            for i in range(self.maxIter):
                self.labels_ = self.predict(self.X)
                self.cluster_centers_ = self.__get_centers()
                
                self.sse_ = utils.sum_of_squared_errors(self.X, self.labels_, self.cluster_centers_)
                if (self.sse_ - sse_last) < self.tol:
                    break
                sse_last = self.sse_
            
            list_cluster_centers_.append(self.cluster_centers_)
            list_labels_.append(self.labels_)
            list_sse_.append(self.sse_)
        
        best = np.argmin(list_sse_)
        self.cluster_centers_ = list_cluster_centers_[best]
        self.labels_ = list_labels_[best]
        self.sse_ = list_sse_[best]
        
        return self
        
    # get prediction clusters of X
    def predict(self, X):
        y = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            squared_errors = np.zeros(self.n_clusters)
            for j in range(self.n_clusters):
                squared_errors[j] = utils.squared_error(X[i], self.cluster_centers_[j])
            
            y[i] = np.argmin(squared_errors)
            
        return y
    
    # get centers by averaging X in the cluster
    def __get_centers(self):
        for i in range(self.n_clusters):
            logic = (self.labels_ == i)
            X_cluster_i = self.X[logic]
            
            self.cluster_centers_[i] = np.mean(X_cluster_i, axis=0)
        
        return self.cluster_centers_
        
        