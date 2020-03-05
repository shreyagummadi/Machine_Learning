#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 01:01:16 2019

@author: shreya
"""
import numpy as np
def PCA(x_train,x_test):
    h,f = np.linalg.eig(np.cov(x_train)) # w-->eigenvalue v-->eigenvector
    idx = h.argsort()[::-1]   
    h= np.matrix(h[idx])
    f = f[:,idx]
    
    m = 0
    flag = np.inf
    i = 0
    while(flag !=0):
        if(np.sum(h[:,0:i])/np.sum(h)>0.95):
            flag = 0
        else:
            m = m+1
            i = i+1
    
    f = np.matrix(f[:,0:m])
    
    x_train = f.T*x_train
    x_test = f.T*x_test
    return x_train,x_test