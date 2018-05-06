#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 23:52:56 2018

@author: Garrett
"""
import numpy as np

# sizes to work with
class_size = 20
voc_size = 61188
working_sizes = [100, 500, 1000, 5000, 7500, 10000, 12500, 25000, 50000, 61188]  

# reader
def dataset_reader():
    return np.loadtxt('train.data', dtype = np.int), np.loadtxt('train.label', dtype = np.int), \
            np.loadtxt('test.data', dtype = np.int), np.loadtxt('test.label', dtype = np.int)
        

# create occurency matrix for multivariat bernoulli model
# return theta matrix, i for class, j for word; x_id order in theta, y prob array
def bernoulli_matrix(data_train, label_train, voc_size, class_size):
    m = label_train.size
    theta_matrix = np.zeros([class_size, voc_size])
    y_array = np.zeros([class_size, 1])
    
    # y occurence
    for i in range(m):
        y_class = label_train[i]
        y_array[y_class-1] += 1
        
    n = data_train.shape[0]
    
    # read input data to occurence matrix and sort
    x_map = np.row_stack([np.arange(voc_size) + 1, np.zeros(voc_size)])
    for i in range(n):
        document_id = data_train[i][0]
        word_id = data_train[i][1]
        word_occ = data_train[i][2]
        y_class = label_train[document_id-1]
        
        theta_matrix[y_class-1][word_id-1] += 1
        x_map[1][word_id-1] += word_occ
        
        
    theta_matrix = np.row_stack([x_map, theta_matrix])
    
    theta_matrix = np.flip(theta_matrix[:, theta_matrix[1, :].argsort()], 1)
    
    # seperate x id and values
    x_id = theta_matrix[0, :]
    theta_matrix = theta_matrix[2:, :]
    
    # get prob matrix
    for i in range(class_size):
        theta_matrix[i] = (theta_matrix[i] + 1) / (y_array[i] + 2)
        
    y_sum = y_array.sum()
    y_array /= y_sum
    
    return theta_matrix, x_id, y_array

# create occurency matrix for multinomial model
# return theta matrix, i for class, j for word; x_id order in theta, y prob array
def multinomial_matrix(data_train, label_train, voc_size, class_size, working_size):
    m = label_train.size
    theta_matrix = np.zeros([class_size, voc_size])
    y_array = np.zeros([class_size, 1])
    y_sum = np.zeros([class_size, 1])
        
    # y occurence
    for i in range(m):
        y_class = label_train[i]
        y_array[y_class-1] += 1
        
    n = data_train.shape[0]
    
    # read input data to occurence matrix and sort
    x_map = np.row_stack([np.arange(voc_size) + 1, np.zeros(voc_size)])
    for i in range(n):
        document_id = data_train[i][0]
        word_id = data_train[i][1]
        word_occ = data_train[i][2]
        y_class = label_train[document_id-1]
        
        theta_matrix[y_class-1][word_id-1] += word_occ
        x_map[1][word_id-1] += word_occ
        
    for i in range(class_size):
        y_sum[i] = theta_matrix[i].sum()
    
    theta_matrix = np.row_stack([x_map, theta_matrix])
    
    theta_matrix = np.flip(theta_matrix[:, theta_matrix[1, :].argsort()], 1)
        
    # seperate x id and values
    x_id = theta_matrix[0, :]
    theta_matrix = theta_matrix[2:, :]
    
    # get prob matrix
    for i in range(class_size):
        theta_matrix[i] = (theta_matrix[i] + 1) / (y_sum[i] + working_size)
        
    y_array /= y_array.sum()
    
    return theta_matrix, x_id, y_array
        
# get dictionary of {x_id: current_index}, limit to working vocabulary size
def get_x_indices(x_id, working_size):
    x_indices = dict()
    for i in range(min(x_id.size, working_size)):
        x_indices[x_id[i]] = i
        
    return x_indices

# limit theta_matrix to working vocabulary size
def theta_matrix_working(theta_matrix, working_size):
    return theta_matrix.view()[:,:working_size]

# transfer raw to test matrix
def get_x_test(x_indices, data_test, test_size):
    x_test = np.zeros([test_size, len(x_indices)])
    for i in range(data_test.shape[0]):
        document_id = data_test[i][0]
        word_id = data_test[i][1]
        
        if word_id in x_indices:
            x_test[document_id-1][x_indices[word_id]] = 1
        
    return x_test

# predict single y_hat using X and log theta, theta_0 is log(1-theta)
def predict(x, theta_matrix, theta_matrix_0, y_prob):  
    max_p = -np.inf
    max_i = 0
    for i in range(y_prob.size):
        py = y_prob[i]
        pxy = 0
        for j in range(theta_matrix.shape[1]):
            if x[j] == 1:
                pxy += theta_matrix[i][j]
            else:
                pxy += theta_matrix_0[i][j]
        
        p = py + pxy
        
        if p > max_p:
            max_p = p
            max_i = i
        
        
    return max_i+1

# predict y_hat array, transfer theta and y_prob to log
def get_y_hat(x_test, theta_matrix, y_prob):
    theta_matrix_0 = 1 - theta_matrix
    theta_matrix = np.log(theta_matrix)
    theta_matrix_0 = np.log(theta_matrix_0)
    y_prob = np.log(y_prob)
    
    y_hat = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        x = x_test[i]
        y_hat[i] = predict(x, theta_matrix, theta_matrix_0, y_prob)
        
    return y_hat

# transfer raw to test matrix, multinomial
def get_x_test_multinomial(x_indices, data_test, test_size):
    x_test = np.zeros([test_size, len(x_indices)])
    for i in range(data_test.shape[0]):
        document_id = data_test[i][0]
        word_id = data_test[i][1]
        word_occ = data_test[i][2]
        
        if word_id in x_indices:
            x_test[document_id-1][x_indices[word_id]] = word_occ
        
    return x_test

# predict single y_hat using X and log theta, multinomial
def predict_multinomial(x, theta_matrix, y_prob):
    max_p = -np.inf
    max_i = 0
    for i in range(y_prob.size):
        py = y_prob[i]
        pxy = 0
        for j in range(theta_matrix.shape[1]):
            pxy += theta_matrix[i][j]* x[j]
        
        p = py + pxy
        
        if p > max_p:
            max_p = p
            max_i = i
        
        
    return max_i+1

# predict y_hat array, transfer theta and y_prob to log
def get_y_hat_multinomial(x_test, theta_matrix, y_prob):
    theta_matrix = np.log(theta_matrix)
    y_prob = np.log(y_prob)
    
    y_hat = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        x = x_test[i]
        y_hat[i] = predict_multinomial(x, theta_matrix, y_prob)
        
    return y_hat


# train and predict bernoulli models, output y_hat to file      
def train_bernoulli():
    print('Train Bernoulli:')
    
    data_train, label_train, data_test, label_test = dataset_reader()
    
    class_size = 20
    voc_size = 61188
    
    for working_size in working_sizes:
    
        theta_matrix, x_id, y_prob = bernoulli_matrix(data_train, label_train, voc_size, class_size)
    
    
        x_indices = get_x_indices(x_id, working_size)
        theta_matrix = theta_matrix_working(theta_matrix, working_size)
        
        x_test = get_x_test(x_indices, data_test, label_test.size)
        y_hat = get_y_hat(x_test, theta_matrix, y_prob)
    
#        np.savetxt('ber_y_hat'+ str(working_size) + '.csv', y_hat, fmt = '%d')
        print(working_size)

# train and predict multinomail models, output y_hat to file 
def train_multinomial():
    print('Train Multinomial:')
    
    data_train, label_train, data_test, label_test = dataset_reader()
    
    for working_size in working_sizes:
            
        theta_matrix, x_id, y_prob = multinomial_matrix(data_train, label_train, voc_size, class_size, working_size)
        x_indices = get_x_indices(x_id, working_size)
        theta_matrix = theta_matrix_working(theta_matrix, working_size)
    
        x_test = get_x_test_multinomial(x_indices, data_test, label_test.size)
        y_hat = get_y_hat_multinomial(x_test, theta_matrix, y_prob)
        
#        np.savetxt('multi_y_hat'+str(working_size)+'.csv', y_hat, fmt = '%d')
    
        print(working_size)
    
    
def main():
    train_bernoulli()
    train_multinomial()
    

if __name__ == '__main__':
    main()