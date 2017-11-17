# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:00:38 2017

@author: wangs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from matplotlib.lines import Line2D
from numpy import *
import plot
from cvxopt import solvers,matrix



iris = datasets.load_iris() 
X = iris.data[:, 1:3]  # 只取前两维特征，二维作图
y = iris.target
iris_train = np.column_stack((X,y))
iris_train = iris_train[0:100,:]  ##get the trian samples
print (X)
print (y)
print (iris_train)
plot.plotData(iris_train)
##pla.train(iris_train)
##train(iris_train)

def svm(pts, labels):
    """
    Support Vector Machine using CVXOPT in Python. This example is
    mean to illustrate how SVMs work.
    """
    n = len(pts[0])

    # x is a column vector [w b]^T

    # set up P
    P = matrix(0.0, (n+1,n+1))
    for i in range(n):
        P[i,i] = 1.0

    # q^t x
    # set up q
    q = matrix(0.0,(n+1,1))
    q[-1] = 1.0

    m = len(pts)
    # set up h
    h = matrix(-1.0,(m,1))

    # set up G
    G = matrix(0.0, (m,n+1))
    for i in range(m):
        G[i,:n] = -labels[i] * pts[i]
        G[i,n] = -labels[i]

    x = solvers.qp(P,q,G,h)['x']

    return P, q, h, G, x
   

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
    idx_1 = np.where(dataSet[:,2]==0)
    p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], marker='o', color='g', label=1, s=20)
    idx_2 = where(dataSet[:,2]==1)
    p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], marker='x', color='r', label=2, s=20)
    plt.legend(loc = 'upper right')
    plt.show()
    
    
