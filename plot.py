# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:46:25 2017

@author: wangs
"""
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from matplotlib.lines import Line2D
from numpy import *
import numpy as np

def plotData(dataSet):
    ''' (array) -> figure

    Plot a figure of dataSet
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    labels = array(dataSet[:,2])
    idx_1 = np.where(dataSet[:,2]==-1)
    p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], marker='o', color='g', label=1, s=50)
    idx_2 = where(dataSet[:,2]==1)
    p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], marker='x', color='r', label=2, s=50)
    plt.legend(loc = 'upper right')
    plt.show()

def makeline(dataSet,w):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    labels = array(dataSet[:,2])
    idx_1 = where(dataSet[:,2]==-1)
    p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], 
        marker='o', color='g', label=1, s=50)
    idx_2 = where(dataSet[:,2]==1)
    p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], 
        marker='x', color='r', label=2, s=50)
    print(w.shape)
    plt.legend(loc = 'upper right')
    plt.show()