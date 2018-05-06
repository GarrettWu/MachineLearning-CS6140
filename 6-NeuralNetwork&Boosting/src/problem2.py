#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 22:50:43 2018

@author: Garrett
"""
import utils
import numpy as np
from ada_boost import AdaBoost

def test(file):
    dataset = utils.dataset_reader(file)
    X_train, y_train, X_test, y_test = utils.data_process(dataset)
    
    n_estimators = 30
    
    print('AdaBoost Optimal for:', file)
    ada_boost = AdaBoost(n_estimators=n_estimators)
    
    ada_boost.fit(X_train, y_train, X_test, y_test)
    
    utils.plot_error_vs_t('local error', ada_boost.local_errs, ada_boost.n_estimators)
    utils.plot_error_vs_t('train error', ada_boost.train_errs, ada_boost.n_estimators)
    utils.plot_error_vs_t('test error', ada_boost.test_errs, ada_boost.n_estimators)
    
    print('AdaBoost Random for:', file)
    ada_boost_random = AdaBoost(n_estimators=n_estimators, decision_stumps='random')
    
    ada_boost_random.fit(X_train, y_train, X_test, y_test)
    
    utils.plot_error_vs_t('local error', ada_boost_random.local_errs, ada_boost_random.n_estimators)
    utils.plot_error_vs_t('train error', ada_boost_random.train_errs, ada_boost_random.n_estimators)
    utils.plot_error_vs_t('test error', ada_boost_random.test_errs, ada_boost_random.n_estimators)
    
    

    

def main():
    test('diabetes.csv')
#    test('breastcancer.csv')
#    test('spambase.csv')

    
if __name__ == '__main__': 
    main()