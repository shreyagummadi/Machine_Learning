#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:55:31 2019

@author: shreya
"""

import numpy as np
from scipy.io import loadmat
from scipy.stats import mode
from PCA import PCA
from LDA import LDA

data = int(input("Select 1.data.mat 2.pose.mat 3.illumination.mat: "))
opt = int(input("Select 1.KNN 2.KNN+PCA 3.KNN+LDA: "))


''' selecting the dataset'''
''' dividing into train and test'''
if data ==1:
    x = loadmat('data.mat')
    t = x['face'] #acess the images
    t = np.matrix(t.reshape((504,600)))
    c = 200 # number of classes
    d = 504
    ntrain = 400
    ntest = 200
    n = 2 # no.of train samples per class
    x_train = np.matrix(np.zeros((d,ntrain)),dtype = complex)
    x_test = np.matrix(np.zeros((d,ntest)),dtype= complex)
    label_train = np.zeros((ntrain,1))
    label_test = np.zeros((ntest,1))
    for i in range(0,c):
        count = 0
        for j in range(0,3):
            if j==0 or j==1:
                x_train[:,2*i+count] =  t[:,3*i+j]
                label_train[2*i+count] = i
                count = count+1
            else:
                x_test[:,i] = t[:,3*i+j]
                label_test[i] = i

elif data==2:
    x = loadmat('pose.mat')
    t = x['pose']
    d = 1920
    c = 68
    percent = 0.6 #percent of data for training
    n = round(percent*13) # no.of train samples per class
    ntrain = n*68
    ntest = (13-n)*68
    x_train = np.matrix(np.zeros((d,ntrain)),dtype = complex)
    x_test = np.matrix(np.zeros((d,ntest)),dtype= complex)
    label_train = np.zeros((ntrain,1))
    label_test = np.zeros((ntest,1))
    for i in range(0,c):
        for j in range(0,n):
            x_train[:,n*i+j] = np.reshape(t[:,:,j,i],(d,1))
            label_train[n*i+j] = i
        for j in range(0,(13-n)):
            x_test[:,(13-n)*i+j] = np.reshape(t[:,:,n+j,i],(d,1))
            label_test[(13-n)*i+j]= i

elif data==3:
    x = loadmat('illumination.mat')
    t = x['illum']
    c = 68
    d  =1920
    percent = 0.6
    n = round(percent*21)
    ntrain = n*68
    ntest = (21-n)*68
    x_train = np.matrix(np.zeros((d,ntrain)),dtype = complex)
    x_test = np.matrix(np.zeros((d,ntest)),dtype= complex)
    label_train = np.zeros((ntrain,1))
    label_test = np.zeros((ntest,1))
    for i in range(0,c):
        for j in range(0,n):
            x_train[:,n*i+j] = np.reshape(t[:,j,i],(d,1))
            label_train[n*i+j] = i
        for j in range(0,(21-n)):
            x_test[:,(21-n)*i+j] = np.reshape(t[:,n+j,i],(d,1))
            label_test[(21-n)*i+j]= i


''' dimensionality reduction or original data'''
if opt==1:
    x_train,x_test = x_train,x_test
elif opt==2: 
    x_train,x_test = PCA(x_train,x_test)
    d,_ = x_train.shape
elif opt ==3:
    x_train,x_test = LDA(x_train,x_test,c,n)
    d,_ = x_train.shape
    
    
k = 1 # k nearest neighbours
   
''' getting k nearest neighbours'''
solution = np.zeros((ntest,1))

for i in range(0,ntest):
    knn = np.zeros((k,1))
    knn_label = np.zeros((k,1))
    
    knn[:] = np.inf
    knn_label[:] = np.inf
    
    knn[0] = np.linalg.norm(x_test[:,i]-x_train[:,0])
    knn_label[0]=label_train[0]
    
    for j in range(1,ntrain):
        norm = np.linalg.norm(x_test[:,i]-x_train[:,j])
        val_max = np.max(knn)
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