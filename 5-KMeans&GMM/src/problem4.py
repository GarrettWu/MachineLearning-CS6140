#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:49:26 2018

@author: Garrett
"""

import utils
import numpy as np
from gmm import GaussianMixture
#from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

def test(file, max_n_components, n_classes):
    print('GaussianMixture for set: '+file)
    
    dataset = utils.dataset_reader(file)
    
    X, y = utils.data_process(dataset)
        
    list_sse = []
    list_nmi = []
    for n_components in range(1, max_n_components+1):
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X)
        
        y_hat = gmm.predict(X)
        sse = utils.sum_of_squared_errors(X, y_hat, gmm.means_)
        nmi = utils.normalized_mutual_information(y, n_classes, y_hat, n_components)
    
        print('{0:2d} components, SSE: {1:.2f}, NMI: {2:.4f}'.format(n_components, sse, nmi))
#        print('iterations: ', gmm.n_iter_)
#        print(gmm.means_, gmm.covariances_, gmm.weights_)
#        print(gmm.lower_bound_)
        list_sse.append(sse)
        list_nmi.append(nmi)
    
    utils.plot_measure_vs_k('SSE', list_sse, range(1, max_n_components+1))
    utils.plot_measure_vs_k('NMI', list_nmi, range(1, max_n_components+1))
    
def main():
    test('dermatologyData.csv', 20, 6) 
    
#    test('vowelsData.csv', 20, 11) 
#    
#    test('glassData.csv', 20, 6) 
#    
#    test('ecoliData.csv', 20, 5) 
#
#    test('yeastData.csv', 20, 9)
#    
#    test('soybeanData.csv', 20, 15)
    
if __name__ == '__main__': 
    main()