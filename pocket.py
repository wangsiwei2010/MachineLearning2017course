# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:05:58 2017

@author: wangs
"""

import numpy as np
from numpy.linalg import *
from numpy import *
import random

def Is_correct(W, x, y):  
    if (np.inner(W,x)*y > 0):  
        return True  
    else:  
        return False  
  
def Get_Err_Num(X, Y, W):  
    (dataNum, featureNum) = X.shape  
    Sum = 0  
    for i in range(0, featureNum-1):  
        if not Is_correct(W, X[i,:], Y[i]):  
            Sum += 1  
    return Sum  
  
def Pocket_Algo(X, Y, eta, itertime):
    numLines = X.shape[0]
    numFeatures = X.shape[1]
    one = ones((1,numLines))
    X = np.column_stack((X,one.T))
    w = zeros((1, numFeatures+1))         # initialize weights
    W = zeros((1, numFeatures+1))   
    ErrNum_p = Get_Err_Num(X, Y, W)     
    ErrNum = 0  
    iter = 0  
    # print 'ErrNum_p: ' + str(ErrNum_p)  


    while iter<=itertime:
        i = random.randint(0,numLines-1)
        if ((Y[i] * np.inner(w,X[i,:])) <= 0):
            iter += 1
            w = w + eta * Y[i] * X[i,:]
            ErrNum = Get_Err_Num(X, Y, w)
            if ErrNum < ErrNum_p:  
                W = w  
                ErrNum_p = ErrNum
    print(ErrNum_p)
    return w