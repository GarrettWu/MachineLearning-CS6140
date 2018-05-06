#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:15:54 2018

@author: Garrett
"""

import utils
import numpy as np
from kmeans import KMeans


class GaussianMixture(object):
    # number of components, max iterations, tolerance, numbers of run different seeds
    def __init__(self, n_components, maxIter=100, tol=0.001, n_init=1):
        self.n_components = n_components 
        self.maxIter = maxIter 
        self.tol = tol
        self.n_init=n_init
        
    # fit model to data
    def fit(self, X):
        self.X = X.copy()
        
        list_means_ = []
        list_covariances_ = []
        list_weights_ = []
        
        list_lower_bound_ = []
        list_n_iter_ = []
        
        for n in range(self.n_init):
            self.__init_parameters()
            
            lower_bound_last = -np.inf
            self.n_iter_ = self.maxIter
            for i in range(self.maxIter):
                self.__e_step()
                self.__m_step()
#                print('self.lower_bound_', self.lower_bound_)
                if (self.lower_bound_ - lower_bound_last) < self.tol:
                    
                    self.n_iter_ = i+1
                    break
                lower_bound_last = self.lower_bound_
            
            list_means_.append(self.means_)
            list_covariances_.append(self.covariances_)
            list_weights_.append(self.weights_)
            
            list_lower_bound_.append(self.lower_bound_)
            list_n_iter_.append(self.n_iter_)
        
        best = np.argmin(list_lower_bound_)
        self.means_ = list_means_[best]
        self.covariances_ = list_covariances_[best]
        self.weights_ = list_weights_[best]
        self.lower_bound_ = list_lower_bound_[best]
        self.n_iter_ = list_n_iter_[best]
        
        return self
        
    def predict(self, X):
        N = X.shape[0]
        k = self.n_components
        
        gamma = np.zeros([N, k])
        for i in range(N):
            
            p = np.zeros(k)
            for j in range(k):
                p[j] = self.weights_[j] * utils.multivariate_gaussian(self.X[i], self.means_[j], self.covariances_[j])
                 
            sp = p.sum()
            for j in range(k):
                gamma[i, j] = p[j] / sp
                        
#        print(gamma)
        return np.argmax(gamma, axis=1)
        
    # init parameters by running kmeans algorithms
    def __init_parameters(self):
        N = self.X.shape[0]
        n_features = self.X.shape[1]
        
        kmeans = KMeans(n_clusters=self.n_components, n_init=5)
        kmeans.fit(self.X)
        
        # mu, means for each component
        self.means_ = kmeans.cluster_centers_
        # sigma, covariances for each component
        self.covariances_ = np.zeros([self.n_components, n_features, n_features])
        # pi, weights for each component
        self.weights_ = np.zeros(self.n_components)
        for k in range(self.n_components):
            logic = (kmeans.labels_ == k)
            Nk = logic.sum()
            
            # otherwise error
            if Nk > 1:
                Xk = self.X[logic]
                self.covariances_[k] = np.cov(Xk.T)
            
            self.weights_[k] = Nk / N
        
        # gamma(Znk)
        self.gamma = np.zeros([N, self.n_components])
        # log_likelihood 
        self.lower_bound_ = -np.inf
        
        return self
        
    # evaluate gamma(Znk) with theta and calculate lower_bound_
    def __e_step(self):
#        print(self.gamma)
        
        N = self.X.shape[0]
        k = self.n_components
        
        self.lower_bound_ = 0
        for i in range(N):
            
            p = np.zeros(k)
            for j in range(k):
                p[j] = self.weights_[j] * utils.multivariate_gaussian(self.X[i], self.means_[j], self.covariances_[j])
#                print('x, mean, cov:  ', self.X[i], self.means_[j], self.covariances_[j])
#                print('self.weights_[j]  ', self.weights_[j])
#                print('utils.multivariate_gaussian(self.X[i], self.means_[j], self.covariances_[j])  ', utils.multivariate_gaussian(self.X[i], self.means_[j], self.covariances_[j]))
#                
#                print('pij  ',i, '  ', j, '  ',  p[j])
                
            sp = p.sum()
            for j in range(k):
                self.gamma[i, j] = p[j] / sp
                
            self.lower_bound_ += np.log(sp)
        
#        print('e step self.gamma:  ', self.gamma)
        return self
    
    # evaluate theta with gamma(Znk)
    def __m_step(self):
#        print(self.means_, self.covariances_, self.weights_)
        
        N = self.X.shape[0]
        k = self.n_components
        Nk = self.gamma.sum(axis=0)
        
        self.means_ = self.gamma.T.dot(self.X) / Nk[:, None]
        
        for j in range(k):
            
            s = 0
            for i in range(N):
                diff = self.X[i] - self.means_[j]
                s += self.gamma[i, j] * np.outer(diff, diff)
        
            self.covariances_[j] = s / Nk[j]
        
        self.weights_ = Nk / N
#        print('m step means, covs, weights:  ', self.means_, self.covariances_, self.weights_)
        return self
            
     