#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:53:43 2018

@author: Garrett
"""

from kmeans import KMeans
#from sklearn.cluster import KMeans

from gmm import GaussianMixture
import numpy as np

X = np.array([[2, 2], [3, 4], [1, 0], [101, 2], [102, 4], [100, 0]])
kmeans = KMeans(n_clusters=2).fit(X)
#print(kmeans.labels_)
#print(kmeans.predict(np.array([[0, 0], [4, 4]])))
#print(kmeans.cluster_centers_)

gmm = GaussianMixture(n_components=2).fit(X)
print('gmm predict  ', gmm.predict(X))
#print(gmm.predict(np.array([[0, 0], [4, 4]])))
print('gmm.means_  ', gmm.means_)
print('gmm.covariances_  ', gmm.covariances_)
print('gmm.n_iter', gmm.n_iter_)