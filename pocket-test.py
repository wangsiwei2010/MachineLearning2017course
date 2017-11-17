# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:12:22 2017

@author: wangs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from matplotlib.lines import Line2D
from numpy import *
import pocket as pocket
import plot as pl



iris = datasets.load_iris() 
X = iris.data[0:100, 0:2]  # 只取前两维特征，二维作图
Y = iris.target[0:100]
Y[Y == 0] = -1;
trainset = np.column_stack((X,Y))
pl.plotData(trainset)
W = pocket.Pocket_Algo(X,Y,1,50)
pl.makeline(trainset,W)
print(W)