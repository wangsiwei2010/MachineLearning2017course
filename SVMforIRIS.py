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


iris = datasets.load_iris()
X = iris.data[:, :2]  # 只取前两维特征，二维作图
iris_train[:,0:2]=X
y = iris.target
iris_train[:,2] = y
print (X)
print (y)
print (iris_train)

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
    idx_1 = np.where(dataSet[:,2]==1)
    p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], marker='o', color='g', label=1, s=20)
    idx_2 = where(dataSet[:,2]==-1)
    p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], marker='x', color='r', label=2, s=20)
    plt.legend(loc = 'upper right')
    plt.show()
    