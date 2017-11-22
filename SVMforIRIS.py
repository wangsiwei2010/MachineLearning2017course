# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:00:38 2017

@author: wangs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.lines import Line2D
from numpy import *
import plot
from cvxopt import solvers,matrix
import svm



iris = datasets.load_iris() 
X = iris.data[0:100, 0:2]  
y = iris.target[0:100]
y[y == 0] = -1;
iris_train = np.column_stack((X,y))
w = svm.svm(X,y)
print (X)
print (y)
print (w)
plot.plotData(iris_train)
##pla.train(iris_train)
##train(iris_train)


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
    p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], marker='o', color='g', label=1, s=20)
    idx_2 = where(dataSet[:,2]==1)
    p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], marker='x', color='r', label=2, s=20)
    plt.legend(loc = 'upper right')
    plt.show()
    
    
