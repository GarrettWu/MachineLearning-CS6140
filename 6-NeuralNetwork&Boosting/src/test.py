#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:38:38 2018

@author: Garrett
"""

import network

import utils
import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.eye(8)
Y = X.copy()
#Y = np.arange(8)

#clf = MLPClassifier(hidden_layer_sizes=(10,), verbose=True, max_iter=1000)
clf = MLPClassifier(hidden_layer_sizes=(3,), activation='logistic', solver='sgd', momentum=0, batch_size=1,
                    verbose=True, max_iter=100000, learning_rate_init=0.01, tol = 0.000001)
#
#clf = network.Network([3])
#
clf.fit(X, Y)

print(clf.predict(X))
print(clf.intercepts_)
print(clf.predict_proba(X))

