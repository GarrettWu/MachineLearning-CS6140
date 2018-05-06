#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:20:34 2018

@author: Garrett
"""

import utils
import numpy as np
#from sklearn.cluster import KMeans
from kmeans import KMeans

def test(file, max_n_clusters, n_classes):
    print('K-Means for set: '+file)
    
    dataset = utils.dataset_reader(file)
    
    X, y = utils.data_process(dataset)
    
    list_sse = []
    list_nmi = []
    for n_clusters in range(1, max_n_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        
        nmi = utils.normalized_mutual_information(y, n_classes, kmeans.labels_, n_clusters)
    
        print('{0:2d} clusters, SSE: {1:.2f}, NMI: {2:.4f}'.format(n_clusters, kmeans.sse_, nmi))
        
        list_sse.append(kmeans.sse_)
        list_nmi.append(nmi)
    
    utils.plot_measure_vs_k('SSE', list_sse, range(1, max_n_clusters+1))
    utils.plot_measure_vs_k('NMI', list_nmi, range(1, max_n_clusters+1))
    
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