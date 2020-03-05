# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Sun Dec  8 12:39:01 2019

@author: Shreya
"""

from scipy.io import loadmat
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

x1 = loadmat('clustering_data2.mat')
colmap = {1: 'red', 2: 'green', 3: 'blue',4:'yellow',5:'black',6:'magenta',7:'cyan',8:'orange'}
m = {1:'+',2:'o',3:'^',4:'s',5:'*',6:'x',7:'p',8:'H'}
data = x1['X']

d = pd.DataFrame({'x':data[0],'y':data[1]})

kmeans = KMeans(n_clusters=4)
pred = kmeans.fit(d)
labels = kmeans.predict(d)
centroids = kmeans.cluster_centers_
colors = map(lambda x: colmap[x+1], labels)
plt.figure(1)
plt.scatter(d['x'], d['y'],color = list(colors), alpha=0.5, edgecolor='k')
plt.show()

#plt.figure(2)
#m1 = map(lambda x:m[x+1],labels)
#m1 = list(m1)
#for i in range(len(d)):
#   plt.scatter(d['x'][i], d['y'][i], color = 'r', marker = m1[i])
#plt.show()