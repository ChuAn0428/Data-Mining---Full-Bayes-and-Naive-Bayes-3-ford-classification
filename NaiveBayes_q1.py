#!/usr/bin/env python
# -*- coding: utf-8 -*-
###########################
# CSCI 573 Data Mining - Full Bayes and Naive Bayes Algorithms 
# Author: Chu-An Tsai
# 11/15/2019
# Question #1 - Naive Bayes version
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
temp_covs = [[] for i in range(k)]
covs_m = [[] for i in range(k)]
class_member = [[] for i in range(k)]
covs = [[] for i in range(k)]

# Separate data points by its class and calculate the
# prior probibilities, means, and covariance matrices
for i in range(k):
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
    temp_covs[i] = np.cov(class_group.T).round(2)
       # The naive assumption corresponds to setting all the covariances to zero   
    covs[i] = np.zeros((4,4))
    for j in range(k+1):
        covs[i][j][j] = round(temp_covs[i][j][j], 2)
    """
    ### normal
    #print('Prior probability:', len(class_group)/N)
    #print('Mean:\n', np.mean(class_group, axis=0))
    priors[i] = len(class_group)/N
    means[i] = np.mean(class_group, axis=0)
    # The naive assumption corresponds to setting all the covariances to zero
    temp_covs[i] = np.cov(class_group.T, ddof=0)
    covs_m[i] = np.zeros((4,4))
    # calculate covariance matrix
    for j in range(k+1):
        covs_m[i][j][j] = temp_covs[i][j][j]
          
np.savez('modelfile_naive.npz', p = priors, m = means, c = covs_m)    
print('\nThe model file \"modelfile_naive.npz\" has been created.')

'''
### testing
model = np.load('modelfile_naive.npz')
print('The model file contains:')
print('prior probibilities:\n',model['p'])
print('means:\n',model['m'])
print('covariance matrices:\n',model['c'])
'''

