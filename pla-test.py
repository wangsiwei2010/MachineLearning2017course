# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:55:16 2017

@author: wangs
"""

import pla as pla
import plot as pl

data = pla.makeLinearSeparableData([4,3],100)
pl.plotData(data)
w = pla.train(data,True)
print(w)
