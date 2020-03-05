#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 23:03:50 2019

@author: shreya
"""

import numpy as np
from scipy.io import loadmat
from PCA import PCA 
from LDA import LDA

opt = int(input("Select 1.Bayes 2.Bayes+PCA 3.Bayes+LDA: "))
x = loadmat('data.mat')
t = x['face'] #acess the images
t = t.reshape((504,600))

''' dividing into train and test for data.mat -- 2 images train, 1 image test'''

c = 2 # number of classes
d = 504 # number of features
ntrain = 300
ntest = 100
n = 150 # no.of train samples per class

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
    
''' estimating mean and variance from train data'''

mean_train = np.matrix(np.zeros((d,c)),dtype = complex)
for k in range(0,c):
    for l in range(0,n):
        mean_train[:,k] = mean_train[:,k]+ x_train[:,k+2*l]
    mean_train[:,k] = (1/n)*mean_train[:,k]

cov_train = np.zeros((d,d,c),dtype = complex)
cov_inv = np.zeros((d,d,c),dtype = complex)
for a in range(0,c):
    for b in range(0,n):
        cov_train[:,:,a] = cov_train[:,:,a]+((x_train[:,a+2*b]-mean_train[:,a])*((x_train[:,a+2*b]-mean_train[:,a]).T))
    cov_train[:,:,a] = (1/n)*cov_train[:,:,a] 
    cov_train[:,:,a] = cov_train[:,:,a]+ 1*np.eye(d)
    
    cov_inv[:,:,a] = np.linalg.inv(cov_train[:,:,a])

''' calculate discriminant'''

W = np.zeros((d, d, c),dtype = complex)
w = np.matrix(np.zeros((d, c)),dtype = complex)
w0 = np.zeros((c,1),dtype = complex)

for m in range(0,c):
    W[:,:,m] = (-1/2) * cov_inv[:, :, m]
    w[:,m] = cov_inv[:, :, m] * mean_train[:,m]
    w0[m] = (-1/2) *((( mean_train[:, m]).T* cov_inv[:,:,m]*mean_train[:,m])+np.log(np.linalg.det(cov_train[:,:,m])))
    

''' assign label based on discriminant'''    
solution = np.zeros((ntest,1))
for u in range(0, ntest):
    max_g = (x_test[:,u].T * W[:,:,1] * x_test[:,u]) + ((w[:,1].T) * x_test[:,u]) + w0[1]
    for v in range(0,c):
        g = (x_test[:,u].T * W[:,:,v] * x_test[:,u]) +(( w[:,v].T) * x_test[:,u]) + w0[v]
#        print(g)
        if (g >= max_g):
           max_g = g
           solution[u] = v

''' accuracy'''
accuracy = 0.0
for z in range(0,ntest):
   if solution[z]== label_test[z]:
       accuracy = accuracy + 1

accuracy = accuracy / ntest;
print("The accuracy is: ")
print(accuracy)
            