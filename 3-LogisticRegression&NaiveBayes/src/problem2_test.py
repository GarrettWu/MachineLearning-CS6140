#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 22:28:04 2018

@author: Garrett
"""

import math
import numpy as np
import matplotlib.pyplot as plt
 
# sizes to work with
class_size = 20
voc_size = 61188
working_sizes = [100, 500, 1000, 5000, 7500, 10000, 12500, 25000, 50000, 61188]       

# evaluate the accuracy
def get_accuracy(y, y_hat):    
    accuracy = (y == y_hat).sum() / y.size
    
    return accuracy

# evaluate precision and recall for specific class
def get_precision_recall(y, y_hat, class_id):
    true_positives = np.logical_and(y_hat == class_id, y == class_id).sum()
    
    predict_positives = (y_hat == class_id).sum()
    actual_positives = (y == class_id).sum()
#    false_negative = (y_hat != class_id & y == class_id).sum()
#    false_positive = (y_hat == class_id & y != class_id).sum()
    
    precision = true_positives / predict_positives
    recall = true_positives / actual_positives
    
    return precision, recall
    
# evaluate ber or multi   
def evaluate(prefix):
    y = np.loadtxt('test.label')
    
    accuracies = []
    for working_size in working_sizes:
        file = 'test/' + prefix + '_y_hat' + str(working_size) + '.csv'
        y_hat = np.loadtxt(file)
        
        accuracies.append(get_accuracy(y, y_hat))
        
           
    # choose voc size 25000 to evaluate precision and recall
    precisions = []
    recalls = []         
    working_size = 25000
    for c in range(class_size):
        precision, recall = get_precision_recall(y, y_hat, c+1)
        precisions.append(precision)
        recalls.append(recall)
        
    return accuracies, precisions, recalls

def plot_accuracy(accuracies_ber, accuracies_multi):   
    
    fig, ax = plt.subplots()
    
    ax.plot(working_sizes, accuracies_ber, label='Bernoulli')
    ax.plot(working_sizes, accuracies_multi, label='Multinomial')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracies')
    ax.set_xlabel('Vocabulary Sizes')
    ax.set_title('Accuracies by Vocabulary Size')
    ax.legend()
    
    plt.show()

def plot_accuracy_bar(accuracies_ber, accuracies_multi):
    
    ind = np.arange(len(working_sizes))  # the x locations for the groups
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    ax.bar(ind - width/2, accuracies_ber, width, 
                    color='SkyBlue', label='Bernoulli')
    ax.bar(ind + width/2, accuracies_multi, width, 
                    color='IndianRed', label='Multinomial')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracies')
    ax.set_xlabel('Vocabulary Sizes')
    ax.set_title('Accuracies by Vocabulary Size')
    ax.set_xticks(ind)
    ax.set_xticklabels(working_sizes)
    ax.legend()
    
    plt.show()

# plot precisions for start to end classes 
def plot_precision(precisions_ber, precisions_multi, start, end):
    precisions_ber = precisions_ber[start:end]
    precisions_multi = precisions_multi[start:end]
    
    ind = np.arange(end-start)  # the x locations for the groups
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    ax.bar(ind - width/2, precisions_ber, width, 
                    color='SkyBlue', label='Bernoulli')
    ax.bar(ind + width/2, precisions_multi, width, 
                    color='IndianRed', label='Multinomial')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0, 1)
    ax.set_ylabel('Precisions')
    ax.set_xlabel('Classes')
    ax.set_title('Precisions by Class of Vocabulary Size 25000')
    ax.set_xticks(ind)
    ax.set_xticklabels(np.arange(start, end) + 1)
    ax.legend()
    
    plt.show()
    
# plot recalls for start to end classes
def plot_recall(recalls_ber, recalls_multi, start, end):
    recalls_ber = recalls_ber[start:end]
    recalls_multi = recalls_multi[start:end]
    
    ind = np.arange(end-start)  # the x locations for the groups
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    ax.bar(ind - width/2, recalls_ber, width, 
                    color='SkyBlue', label='Bernoulli')
    ax.bar(ind + width/2, recalls_multi, width, 
                    color='IndianRed', label='Multinomial')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0, 1)
    ax.set_ylabel('Recalls')
    ax.set_xlabel('Classes')
    ax.set_title('Recalls by Class of Vocabulary Size 25000')
    ax.set_xticks(ind)
    ax.set_xticklabels(np.arange(start, end) + 1)
    ax.legend()
    
    plt.show()
        
def main():
    accuracies_ber, precisions_ber, recalls_ber = evaluate('ber')
    accuracies_multi, precisions_multi, recalls_multi = evaluate('multi')
    
    plot_accuracy(accuracies_ber, accuracies_multi)
    
    divide = 4
    for i in range(divide):
        size = class_size // divide
        start = i * size
        end = start + size
        plot_precision(precisions_ber, precisions_multi, start, end)
    
    for i in range(divide):
        size = class_size // divide
        start = i * size
        end = start + size
        plot_recall(recalls_ber, recalls_multi, start, end)

if __name__ == '__main__':
    main()
    
