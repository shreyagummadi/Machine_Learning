#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 01:05:46 2019

@author: shreya
"""

import numpy as np

def LDA(x_train,x_test,c,n):
    x1 = x_train
    d1,N = x1.shape
    mean = np.matrix(np.zeros((d1,c)),dtype = complex)
    sb = np.matrix(np.zeros((d1,d1),dtype=complex))
    sw = np.matrix(np.zeros((d1,d1),dtype=complex))
    mean_all = np.matrix(np.zeros((d1,1)),dtype=complex)
    # for 2 class classification problem
    if c == 2:
        for m in range(0,c):
            for l in range(0,n):
                mean[:,m] = mean[:,m]+ x1[:,m+2*l]
            mean[:,m] = (1/n)*mean[:,m]
            
        for i in range(0,N):
            mean_all = mean_all+x1[:,i]
        mean_all = (1/N)*mean_all
        
        for i in range(0,c): 
            for j in range(0,n):
                s = (x1[:,i+2*j]-mean[:,i])*((x1[:,i+2*j]-mean[:,i]).T)
            s = s+0.05*np.eye((d1))
            sw = sw+s
        
        for i in range(0,c):
            sb = sb+2*((mean[:,i]-mean_all)*(mean[:,i]-mean_all).T)
    # for all other cases
    elif c!=2:
        for m in range(c):
            for l in range(n):
                mean[:,m] = mean[:,m]+x1[:,n*m+l]
            mean[:,m]=(1/n)*mean[:,m]
        mean_all = np.zeros((d1,1))
        for i in range(0,N):
            mean_all = mean_all+x1[:,i]
        mean_all = (1/N)*mean_all
        sw = np.zeros((d1,d1),dtype=complex)
        for i in range(0,c): 
            for j in range(0,n):
                s = (x1[:,n*i+j]-mean[:,i])*((x1[:,n*i+j]-mean[:,i]).T)
            s = s+0.05*np.eye((d1))
            sw = sw+s
        
        sb = np.zeros((d1,d1),dtype=complex)
        for i in range(0,c):
            sb = sb+2*((mean[:,i]-mean_all)*(mean[:,i]-mean_all).T)
        
    
    w,v = np.linalg.eig(np.linalg.inv(sw)*sb) # w-->eigenvalue v-->eigenvector
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]
    
    v = v[:,0:c-1]
    x_train = v.T*x_train
    x_test = v.T*x_test
    return x_train, x_test