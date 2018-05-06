#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:37:21 2018

@author: Garrett
"""

import utils
import numpy as np

# one layer neural network with stachostic gradient descent and logistic activation
class OneLayerNeuralNetwork(object):
    def __init__(self, n_neurons=3, learning_rate=0.3, max_iter=200, tol=0.0001, momentum=0.9):
        self.n_neurons=n_neurons
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        self.tol=tol
        self.momentum=momentum
        
    def fit(self, X, Y):
        self.X=X.copy()
        self.Y=Y.copy()
        self.N = X.shape[0]
        self.n_input = X.shape[1]
        self.n_output = Y.shape[1]
        
        # ReLU init
        self.w0 = np.random.rand(self.n_input, self.n_neurons) * np.sqrt(2/self.n_input)
        self.w1 = np.random.rand(self.n_neurons, self.n_output) * np.sqrt(2/self.n_neurons)
        self.b0 = np.random.rand() * np.sqrt(2/self.n_input)
        self.b1 = np.random.rand() * np.sqrt(2/self.n_input)
        
        self.layer0 = np.zeros(self.n_input)
        self.layer1 = np.zeros(self.n_neurons)
        self.layer2 = np.zeros(self.n_output)
        
        last_sse = np.inf
        for i in range(self.max_iter):
            
            sse = 0
            for j in range(X.shape[0]):
                o = self.__forward_propagate(self.X[j])
                self.__back_propagate(self.Y[j])
                sse += utils.squared_error(o, self.Y[j])
            
            print(i, '  ', sse)
            if (last_sse - sse) < self.tol:
                break
            
            last_sse = sse
        
    def predict(self, X):
        Y_prob = np.zeros([X.shape[0], self.n_output])
        for i in range(X.shape[0]):
            Y_prob[i] = self.__forward_propagate(X[i])
        
        print(Y_prob)
        return Y_prob.round()
    
    
    def __forward_propagate(self, x):
        self.layer0 = x
        self.layer1 = utils.sigmoid(self.layer0.dot(self.w0) + self.b0)
        self.layer2 = utils.sigmoid(self.layer1.dot(self.w1) + self.b1)
        
#        self.layer1 = utils.sigmoid(self.layer0.dot(self.w0))
#        self.layer2 = utils.sigmoid(self.layer1.dot(self.w1))
        
        return self.layer2
        
    def __back_propagate(self, y):
        o2 = self.layer2
        t2 = y
        
        delta2 = (t2 - o2) * o2 * (1 - o2)
        delta_w1 = self.learning_rate * np.outer(self.layer1, delta2)
        delta_b1 = self.learning_rate * delta2
        
        o1 = self.layer1
        delta1 = o1 * (1 - o1) * delta2.dot(self.w1.T)
        delta_w0 = self.learning_rate * np.outer(self.layer0, delta1)
        delta_b0 = self.learning_rate * delta1
        
        self.w1 += delta_w1
        self.w0 += delta_w0
        self.b1 += delta_b1
        self.b0 += delta_b0
        
        