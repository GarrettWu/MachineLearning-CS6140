#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 22:50:55 2018

@author: Garrett
"""

import utils
import numpy as np

# one layer neural network with stachostic gradient descent and logistic activation
class AdaBoost(object):
    class WeakClassifier(object):
        def __init__(self, ada_boost):
            self.ada_boost = ada_boost
            self.inv = 1
            
        def fit(self):
            if self.ada_boost.decision_stumps == 'optimal':
                self.__optimal()
            else:
                self.__random()
                
            return self
            
        def predict(self, X):
            X = np.atleast_2d(X)
            y_hat = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                x = X[i]
                y_hat[i] = 1 if (x[self.feature] < self.threshold) else -1
                y_hat[i] = y_hat[i] * self.inv
                
            return y_hat
            
        def error(self):
            y_hat = self.predict(self.ada_boost.X)
            y = self.ada_boost.y
            Dt = self.ada_boost.Dt
            
            return (Dt[y != y_hat]).sum()
            
            
        def __optimal(self):
            feature_thresholds = self.ada_boost.feature_thresholds
            
            max_measure = -1
            max_feature = 0
            max_threshold = 0
            max_inv = 1
            max_err = 0
            
            for f in range(len(feature_thresholds)):
                thresholds = feature_thresholds[f]
                self.feature = f
                for th in range(len(thresholds)):
                    self.threshold = thresholds[th]
                    err = self.error()
                    
                    measure = np.abs(0.5 - err)
                    if (measure > max_measure):
                        max_measure = measure
                        max_feature = self.feature
                        max_threshold = self.threshold
                        
                        if err > 0.5:
                            err = 1 - err
                            max_inv = -1
                        max_err = err
            
            self.feature = max_feature
            self.threshold = max_threshold
            self.inv = max_inv
            self.err = max_err
            
        def __random(self):
            feature_thresholds = self.ada_boost.feature_thresholds
            
            self.feature = np.random.randint(0, len(feature_thresholds))
            self.threshold = feature_thresholds[self.feature][np.random.randint(0, len(feature_thresholds[self.feature]))]
            self.inv = [-1, 1][np.random.randint(0, 2)]
            
            self.err = self.error()
            
    
    def __init__(self, n_estimators=20, learning_rate=1.0, decision_stumps='optimal'):
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.decision_stumps=decision_stumps
        
    def fit(self, X, y, X_test, y_test):
        self.X = X.copy()
        self.y = y.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        
        self.n_instances = self.X.shape[0]
        
        self.h = []
        self.Dt = np.ones(self.n_instances) / self.n_instances
        self.alpha = np.zeros(self.n_estimators)
        
        self.feature_thresholds = self.__feature_thresholds()
        
        self.local_errs = []
        self.train_errs = []
        self.test_errs = []
        
        for t in range(self.n_estimators):
            ht = self.WeakClassifier(self).fit()
            self.h.append(ht)
            self.y_hat = self.h[t].predict(self.X)
            
            gamma = self.__gamma()
            self.alpha[t] = self.__alpha(gamma)
            self.Dt = self.__update_Dt(t)
            
            self.local_errs.append(ht.err)
            self.train_errs.append(self.__error(self.X, self.y, t+1))
            self.test_errs.append(self.__error(self.X_test, self.y_test, t+1))
            
        return self
        
        
    def predict(self, X, T=20):
        X = np.atleast_2d(X)
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i]
            s = 0
            for t in range(T):
                s += self.h[t].predict(x) * self.alpha[t]
                
            y_hat[i] = np.sign(s)
            
        return y_hat
            
    def __alpha(self, gamma):
#        return np.log((1 + gamma) / (1 - gamma))
        return 0.5 * np.log((1-gamma) / gamma) 
    
    def __gamma(self):
#        return (self.Dt * self.y * self.y_hat).sum()
        return (self.Dt * (self.y != self.y_hat)).sum()
    
    def __update_Dt(self, t):
#        print('alpha t: ', self.alpha[t])
        self.Dt = self.Dt * np.exp( - self.alpha[t] * self.y * self.y_hat)
        s = self.Dt.sum()
        self.Dt = self.Dt / s
        
#        print(self.Dt)
        return self.Dt
    
    def __feature_thresholds(self):
        feature_thresholds = []
        
        n_features = self.X.shape[1]
        for f in range(n_features):
            thresholds = []
            
            X_feature = self.X[:, f].copy()
            X_feature.sort()
            
            thresholds.append(X_feature[0] - 0.0001)
            for i in range(len(X_feature)-1):
                if X_feature[i] != X_feature[i+1]:
                    thresholds.append((X_feature[i] + X_feature[i+1]) / 2)
            thresholds.append(X_feature[-1] + 0.0001)
            
            feature_thresholds.append(thresholds)
        
        return feature_thresholds
    
    def __error(self, X, y, T):
        y_hat = self.predict(X, T)
        acc = utils.evaluate(y, y_hat)
        err = 1 - acc
        
        return err
            
        
        