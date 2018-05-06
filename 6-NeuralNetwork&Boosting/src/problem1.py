#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:35:19 2018

@author: Garrett
"""

from neural_network import OneLayerNeuralNetwork
import numpy as np

def test(n_neurons=3):
    X = np.eye(8)
    Y = X.copy()
    
    nn = OneLayerNeuralNetwork(n_neurons=n_neurons, learning_rate=1, max_iter=10000, tol=0.0001)
    nn.fit(X, Y)
    
    print(nn.predict(X))

def main():
#    test(1)
    test(2)
#    test(3) 
#    test(4)
    
if __name__ == '__main__': 
    main()