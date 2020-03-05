#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:45:21 2019

@author: shreya
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import mode
from PCA import PCA
from LDA import LDA

'''getting user input for type of classification'''
opt = int(input("Select 1.KNN 2.KNN+PCA 3.KNN+LDA: "))

x = loadmat('data.mat')
t = x['face'] #acess the images
t = t.reshape((504,600))

''' dividing into train and test for data.mat -- 2 images train, 1 image test'''

c = 2 # number of classes
d = 504 # number of features
ntrain = 300
ntest = 100
n = 150 # no.of train samples per class

k = 1

x1 = np.matrix(np.delete(t,np.arange(2,600,3),axis=1))


x_train = x1[:,100:400]
x_test = x1[:,0:100]
label =  np.zeros((400,1))

for i in range(0,200):
    count = 0
    for j in range(0,2):
        if j==0 or j==1:
            label[2*i+count] = count
            count = count+1  

label_train = label[100:400]
label_test = label[0:100]

if opt==1:
    x_train,x_test = x_train,x_test
elif opt==2: 
    x_train,x_test = PCA(x_train,x_test)
    d,_ = x_train.shape
elif opt ==3:
    x_train,x_test = LDA(x_train,x_test,c,n)
    d,_ = x_train.shape



            
'''getting k nearest neighbours'''
solution = np.zeros((ntest,1))

for i in range(0,ntest):
    knn = np.zeros((k,1))
    knn_label = np.zeros((k,1))
    
    knn[:] = np.inf
    knn_label[:] = np.inf
    
    for j in range(0,ntrain):
        norm = np.linalg.norm(x_test[:,i]-x_train[:,j])
        val_max = max(knn)
        argmax = np.argmax(knn)
        if norm<val_max:
            knn[argmax] = norm
            knn_label[argmax] = label_train[j]
    solution[i],_ = mode(knn_label[:]) # choosing the label that is most frequent, majority voting

accuracy = 0.0
for z in range(0,ntest):
   if solution[z]== label_test[z]:
       accuracy = accuracy + 1

accuracy = accuracy / ntest;
print("The accuracy is: ")
print(accuracy)

