#!/usr/bin/env python
# -*- coding: utf-8 -*-
###########################
# CSCI 573 Data Mining - Full Bayes and Naive Bayes Algorithms 
# Author: Chu-An Tsai
# 11/15/2019
# Question #2 - Naive Bayes version
###########################

import numpy as np
import sys

# Get arguments from user input
script = sys.argv[0]
modelfile = sys.argv[1]
testfile = sys.argv[2]
model = np.load(modelfile)
priors = model['p']
means = model['m']
covs = model['c']
test_data1 = np.loadtxt(testfile, delimiter=",", dtype=str)
test_data2 = np.loadtxt(testfile, delimiter=",", usecols=(0,1,2,3))
actual_label = np.loadtxt(testfile, delimiter=",", dtype=np.str, usecols=(4))

##############################################################################
### Local testing
# Load model file
#model = np.load('modelfile_naive.npz')
#priors = model['p']
#means = model['m']
#covs = model['c']
# Load testing data
# dataset fold(1,2) training fold(3) testing
#test_data1 = np.loadtxt("iris_test1_f3.txt.shuffled",delimiter=",", dtype=str)
#test_data2 = np.loadtxt("iris_test1_f3.txt.shuffled",delimiter=",", usecols=(0,1,2,3))
#actual_label = np.loadtxt('iris_test1_f3.txt.shuffled', delimiter=",", dtype=np.str, usecols=(4))
# dataset fold(1,3) training fold(2) testing
#test_data1 = np.loadtxt("iris_test2_f2.txt.shuffled",delimiter=",", dtype=str)
#test_data2 = np.loadtxt("iris_test2_f2.txt.shuffled",delimiter=",", usecols=(0,1,2,3))
#actual_label = np.loadtxt('iris_test2_f2.txt.shuffled', delimiter=",", dtype=np.str, usecols=(4))
# dataset fold(2,3) training fold(1) testing
#test_data1 = np.loadtxt("iris_test3_f1.txt.shuffled",delimiter=",", dtype=str)
#test_data2 = np.loadtxt("iris_test3_f1.txt.shuffled",delimiter=",", usecols=(0,1,2,3))
#actual_label = np.loadtxt('iris_test3_f1.txt.shuffled', delimiter=",", dtype=np.str, usecols=(4))
##############################################################################

testdata = test_data2.copy()
outputdata = test_data1.copy()
outputdata[:,-1] = 0

for i in range (len(actual_label)):
    if (actual_label[i] == 'Iris-setosa'):
        actual_label[i] = 0
    elif (actual_label[i] == 'Iris-versicolor'):
        actual_label[i] = 1 
    else:
        actual_label[i] = 2         
actual_label = actual_label.astype(np.int)

# compute posterior probibilities
'''
# Full Bayes
def prob(x, means, covs):
    return (1./((np.sqrt(2*np.pi))**(len(covs[0]))*np.sqrt(np.linalg.det(covs))))*(np.exp(-(1./2)*((x-means).T)@(np.linalg.inv(covs))@(x-means)))
'''
# Naive Bayes
def prob1(x, means, covs):
    return (1./((np.sqrt(2*np.pi))*np.sqrt(covs[0][0])))*(np.exp(-(((x[0]-means[0])**2)/(2*covs[0][0]))))

def prob2(x, means, covs):
    return (1./((np.sqrt(2*np.pi))*np.sqrt(covs[1][1])))*(np.exp(-(((x[1]-means[1])**2)/(2*covs[1][1]))))

def prob3(x, means, covs):
    return (1./((np.sqrt(2*np.pi))*np.sqrt(covs[2][2])))*(np.exp(-(((x[2]-means[2])**2)/(2*covs[2][2]))))

def prob4(x, means, covs):
    return (1./((np.sqrt(2*np.pi))*np.sqrt(covs[3][3])))*(np.exp(-(((x[3]-means[3])**2)/(2*covs[3][3]))))

for i in range(len(testdata)):
    '''
    # Full Bayes
    p_c1_x = prob(testdata[i],means[0], covs[0])*priors[0]
    p_c2_x = prob(testdata[i],means[1], covs[1])*priors[1]
    p_c3_x = prob(testdata[i],means[2], covs[2])*priors[1]
    '''
    # Naive Bayes
    p_c1_x = prob1(testdata[i],means[0], covs[0])*prob2(testdata[i],means[0], covs[0])*prob3(testdata[i],means[0], covs[0])*prob4(testdata[i],means[0], covs[0])*priors[0]
    p_c2_x = prob1(testdata[i],means[1], covs[1])*prob2(testdata[i],means[1], covs[1])*prob3(testdata[i],means[1], covs[1])*prob4(testdata[i],means[1], covs[1])*priors[1]
    p_c3_x = prob1(testdata[i],means[2], covs[2])*prob2(testdata[i],means[2], covs[2])*prob3(testdata[i],means[2], covs[2])*prob4(testdata[i],means[2], covs[2])*priors[2]
   
    if p_c1_x > p_c2_x and p_c1_x > p_c3_x:
        outputdata[i][4] = 0
    if p_c2_x > p_c1_x and p_c2_x > p_c3_x:
        outputdata[i][4] = 1
    if p_c3_x > p_c1_x and p_c3_x > p_c2_x:
        outputdata[i][4] = 2
 
# compute confusion matrix       
predicted_label = outputdata[:,-1].astype(np.int)
class_num = np.unique(test_data1[:,-1])
con_mat = np.zeros((len(class_num), len(class_num)))
for i, j in zip(predicted_label, actual_label):
    con_mat[i][j] += 1
print('\nIndicate class:')
print('Iris-setosa     -> 1')
print('Iris-versicolor -> 2')
print('Iris-virginica  -> 3')
print('\nConfusion matrix:')
print('                 Actual')
print('               1   2   3')
print('predicted  1',con_mat[0])
print('           2',con_mat[1])
print('           3',con_mat[2])

# compute accuracy, precision, recall, and F-score
# add up the predicted class 1,2,3 (row0,1,2)
prow1 = con_mat[0][0] + con_mat[0][1] + con_mat[0][2] 
prow2 = con_mat[1][0] + con_mat[1][1] + con_mat[1][2]
prow3 = con_mat[2][0] + con_mat[2][1] + con_mat[2][2] 
# add up the actual class 1,2,3 (column0,1,2)
acol1 = con_mat[0][0] + con_mat[1][0] + con_mat[2][0] 
acol2 = con_mat[0][1] + con_mat[1][1] + con_mat[2][1] 
acol3 = con_mat[0][2] + con_mat[1][2] + con_mat[2][2] 

total = acol1 + acol2 + acol3
# precision for each class and average
prec1 = con_mat[0][0]/prow1  
prec2 = con_mat[1][1]/prow2 
prec3 = con_mat[2][2]/prow3 
prec_average = (prec1 + prec2 + prec3)/float(len(class_num))
# the class-specific accuracy = precision
acc1 = prec1 
acc2 = prec2
acc3 = prec3
# overall accuracy
acc_average = (con_mat[0][0]+con_mat[1][1]+con_mat[2][2])/float(len(actual_label))

# recall for each class and average
recall1 = con_mat[0][0]/acol1
recall2 = con_mat[1][1]/acol2
recall3 = con_mat[2][2]/acol3
recall_average = (recall1 + recall2 + recall3)/float(len(class_num))

# F-score for each class and average
fscore1 = 2*con_mat[0][0]/(acol1+prow1)
fscore2 = 2*con_mat[1][1]/(acol2+prow2)
fscore3 = 2*con_mat[2][2]/(acol3+prow3)
fscore_average = (fscore1 + fscore2 + fscore3)/float(len(class_num))
'''
# print
print('\nFor class Iris-setosa:')
print(acc1,prec1,recall1,fscore1)
print('Accuracy:',acc1)
print('Precision:',prec1)
print('Recall:',recall1)
print('F-score:',fscore1)
print('\nFor class Iris-versicolor:')
print('Accuracy:',acc2)
print('Precision:',prec2)
print('Recall:',recall2)
print('F-score:',fscore2)
print('\nFor class Iris-virginica:')
print('Accuracy:',acc3)
print('Precision:',prec3)
print('Recall:',recall3)
print('F-score:',fscore3)
print('\nOverall accuracy:',acc_average)
print('Average precision:',prec_average)
print('Average recall:',recall_average)
print('Overall F-measure:',fscore_average)
'''
a = [acc1,prec1,recall1,fscore1]
b = [acc2,prec2,recall2,fscore2]
c = [acc3,prec3,recall3,fscore3]
d = [acc_average,prec_average,recall_average,fscore_average]
print('\nClassification Report:')
print('Class: accuracy | precision | recall | f1-score')
print('  1 :',a)
print('  2 :',b)
print('  3 :',c)
print(' Avg:',d)
