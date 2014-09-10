# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 09:12:59 2014

@author: manu
"""

from numpy import *
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import random
from pylab import *

# setup data
x = np.linspace(0, 10, 9)
y = np.sin(x)
xi = np.linspace(0, 10, 101)

# use fitpack2 method
ius = InterpolatedUnivariateSpline(x, y)
yi = ius(xi)

M=360
ent=zeros((360,2))
anglelist=list(xrange(0,360))
for i in range(4,M):
    x=sort(random.sample(anglelist,i))#linspace(0,360,i)
    y=randn(i)
    xi=linspace(0,360,M)
    ius = InterpolatedUnivariateSpline(x, y)
    yi = ius(xi)
    ent[i,0]=entropy(y,round(sqrt(i)))
    ent[i,1]=entropy(yi,round(sqrt(M)))

plot(range(360),(ent[:,0]))
plot(range(360),(ent[:,1]))