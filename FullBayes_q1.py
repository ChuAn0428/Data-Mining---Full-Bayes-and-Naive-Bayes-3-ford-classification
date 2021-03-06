#!/usr/bin/env python
# -*- coding: utf-8 -*-
###########################
# CSCI 573 Data Mining - Full Bayes and Naive Bayes Algorithms 
# Author: Chu-An Tsai
# 11/15/2019
# Question #1 - Full Bayes version
###########################

import numpy as np
import sys
 
# Get arguments from user input
script = sys.argv[0]
filename = sys.argv[1]
data = np.loadtxt(filename, delimiter=",", dtype=str)

#############################################################################
# Local testing
# dataset fold(1,2) training and fold(3) testing
#data = np.loadtxt("iris_train1_f1f2.txt.shuffled",delimiter=",", dtype=str)
# dataset fold(1,3) training and fold(2) testing
#data = np.loadtxt("iris_train2_f1f3.txt.shuffled",delimiter=",", dtype=str)
# dataset fold(2,3) training and fold(1) testing
#data = np.loadtxt("iris_train3_f2f3.txt.shuffled",delimiter=",", dtype=str)
#############################################################################

N = len(data)
class_num = np.unique(data[:,-1])
k = len(class_num)
#print(data)
data = np.array(data)
priors = [[] for i in range(k)]
means = [[] for i in range(k)]
covs = [[] for i in range(k)]
class_member = [[] for i in range(k)]

# Separate data points by its class and calculate the
# prior probibilities, means, and covariance matrices
for i in range(len(class_num)):
    class_group = data[data[:, 4] == class_num[i]][:, :4].astype(np.float)
    class_member[i] = class_group
    #print('Class:', class_num[i])
    """
    ### round to 2
    print('Prior probability:', round(len(class_group)/N, 2))
    print('Mean:\n', np.mean(class_group, axis=0).round(2))
    print('Covariance matrix:\n', np.cov(class_group.T).round(2))
    priors[i] = round(len(class_group)/N, 2)
    means[i] = np.mean(class_group, axis=0).round(2)
    covs[i] = np.cov(class_group.T, ddof=0).round(2)
    """
    ### normal
    #print('Prior probability:', len(class_group)/N)
    #print('Mean:\n', np.mean(class_group, axis=0))
    #print('Covariance matrix:\n', np.cov(class_group.T))
    priors[i] = len(class_group)/N
    means[i] = np.mean(class_group, axis=0)
    covs[i] = np.cov(class_group.T, ddof=0)
        
np.savez('modelfile.npz', p = priors, m = means, c = covs) 
print('\nThe model file \"modelfile.npz\" has been created.')

'''
### testing
model = np.load('modelfile.npz')
print('The model file contains:')
print('prior probibilities:\n',model['p'])
print('means:\n',model['m'])
print('covariance matrices:\n',model['c'])
'''
