# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:04:23 2017

@author: wangs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy import *
from cvxopt import solvers,matrix

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
    ##q[-1] = 1.0

    m = len(pts)
    # set up h
    h = matrix(-1.0,(m,1))

    # set up G
    G = matrix(0.0, (m,n+1))
    for i in range(m):
        y = labels[i];
        G[i,:n] = -(pts[i]*labels[i] );
        G[i,n] = -int(y);

    x = solvers.qp(P,q,G,h)['x']

    return x